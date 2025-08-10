from __future__ import annotations

import os
import csv
from typing import Tuple

import numpy as np

from .common import make_env, set_env_preference
from borearl import constants as const


def _select_action_index(env, density_delta: int, conifer_fraction: float) -> int:
    densities = env.DENSITY_ACTIONS
    mixes = env.CONIFER_FRACTIONS
    if density_delta not in densities:
        raise ValueError(f"density_delta {density_delta} not in DENSITY_ACTIONS {densities}")
    # pick the closest available species mix to requested value
    conifer_idx = int(np.argmin(np.abs(np.array(mixes) - conifer_fraction)))
    density_idx = densities.index(density_delta)
    action = density_idx * len(mixes) + conifer_idx
    return action


def _rollout_fixed_action(env, action_index: int, fixed_preference: float) -> Tuple[float, float, list[dict]]:
    set_env_preference(env, fixed_preference)
    obs, info = env.reset()
    done, truncated = False, False
    total_carb, total_thaw = 0.0, 0.0
    rows: list[dict] = []
    step_idx = 0
    while not (done or truncated):
        obs, reward_vec, done, truncated, step_info = env.step(action_index)
        carb = float(step_info.get('raw_carbon_component', reward_vec[0]))
        thaw = float(step_info.get('raw_thaw_component', reward_vec[1]))
        rows.append({
            'step': step_idx,
            'action': int(action_index),
            'carbon': carb,
            'thaw': thaw,
            'scalarized': float(fixed_preference * carb + (1.0 - fixed_preference) * thaw),
        })
        total_carb += carb
        total_thaw += thaw
        step_idx += 1
    return total_carb, total_thaw, rows


def run_counterfactual_sensitivity(num_rng_samples: int = 100, fixed_preference: float = 0.5, output_dir: str = "logs"):
    os.makedirs(output_dir, exist_ok=True)
    env = make_env()
    set_env_preference(env, fixed_preference)

    fixed_seed = 12345
    env.reset(seed=fixed_seed)

    # Evaluate all actions from the same state
    action_rewards = []
    for a in range(env.action_space.n):
        env.reset(seed=fixed_seed)
        set_env_preference(env, fixed_preference)
        _, reward_vec, _, _, step_info = env.step(a)
        action_rewards.append({
            'action': a,
            'carbon': float(step_info.get('raw_carbon_component', reward_vec[0])),
            'thaw': float(step_info.get('raw_thaw_component', reward_vec[1])),
            'scalarized': float(fixed_preference * step_info.get('raw_carbon_component', reward_vec[0]) + (1.0 - fixed_preference) * step_info.get('raw_thaw_component', reward_vec[1])),
        })

    # Fixed action: middle index
    fixed_action = env.action_space.n // 2
    rng_rewards = []
    for i in range(num_rng_samples):
        seed = 1000 + i
        env.reset(seed=seed)
        set_env_preference(env, fixed_preference)
        _, reward_vec, _, _, step_info = env.step(fixed_action)
        rng_rewards.append({
            'seed': seed,
            'carbon': float(step_info.get('raw_carbon_component', reward_vec[0])),
            'thaw': float(step_info.get('raw_thaw_component', reward_vec[1])),
            'scalarized': float(fixed_preference * step_info.get('raw_carbon_component', reward_vec[0]) + (1.0 - fixed_preference) * step_info.get('raw_thaw_component', reward_vec[1])),
        })

    def summarize(vals):
        v = np.array(vals, dtype=np.float32)
        return float(np.mean(v)), float(np.std(v)), float(np.max(v) - np.min(v))

    act_c_mean, act_c_std, _ = summarize([d['carbon'] for d in action_rewards])
    act_t_mean, act_t_std, _ = summarize([d['thaw'] for d in action_rewards])
    act_s_mean, act_s_std, _ = summarize([d['scalarized'] for d in action_rewards])
    rng_c_mean, rng_c_std, _ = summarize([d['carbon'] for d in rng_rewards])
    rng_t_mean, rng_t_std, _ = summarize([d['thaw'] for d in rng_rewards])
    rng_s_mean, rng_s_std, _ = summarize([d['scalarized'] for d in rng_rewards])

    print("\nCounterfactual sensitivity (one-year):")
    print(f"  Actions@fixed RNG -> carbon std={act_c_std:.3f}, thaw std={act_t_std:.3f}, scalarized std={act_s_std:.3f}")
    print(f"  RNG@fixed action -> carbon std={rng_c_std:.3f}, thaw std={rng_t_std:.3f}, scalarized std={rng_s_std:.3f}")
    print("  Ratio (action/rng) std -> carbon={:.2f}, thaw={:.2f}, scalarized={:.2f}".format(
        act_c_std/max(rng_c_std,1e-6), act_t_std/max(rng_t_std,1e-6), act_s_std/max(rng_s_std,1e-6)))

    csv_path = os.path.join(output_dir, 'counterfactual_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['type','id','carbon','thaw','scalarized'])
        for d in action_rewards:
            w.writerow(['action', d['action'], d['carbon'], d['thaw'], d['scalarized']])
        for d in rng_rewards:
            w.writerow(['rng', d['seed'], d['carbon'], d['thaw'], d['scalarized']])
    print(f"Saved sensitivity results to '{csv_path}'")


def run_baselines(output_dir: str = 'logs', fixed_preference: float = 0.5):
    os.makedirs(output_dir, exist_ok=True)
    env = make_env()
    # Zero density baseline (conifer mix irrelevant; choose 0.5)
    zero_action = _select_action_index(env.unwrapped, 0, 0.5)
    z_c, z_t, z_rows = _rollout_fixed_action(env, zero_action, fixed_preference)
    z_path = os.path.join(output_dir, 'baseline_zero_density.csv')
    with open(z_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step','action','carbon','thaw','scalarized'])
        w.writeheader()
        w.writerows(z_rows)
    print(f"Saved zero-density baseline to '{z_path}' (totals: carbon={z_c:.3f}, thaw={z_t:.3f})")

    # +100 density with 0.5 species mix baseline
    plus_action = _select_action_index(env.unwrapped, 100, 0.5)
    p_c, p_t, p_rows = _rollout_fixed_action(env, plus_action, fixed_preference)
    p_path = os.path.join(output_dir, 'baseline_plus100_density_0p5mix.csv')
    with open(p_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step','action','carbon','thaw','scalarized'])
        w.writeheader()
        w.writerows(p_rows)
    print(f"Saved +100 density baseline to '{p_path}' (totals: carbon={p_c:.3f}, thaw={p_t:.3f})")

    # Counterfactual experiment
    run_counterfactual_sensitivity(num_rng_samples=const.COUNTERFACTUAL_SAMPLES_DEFAULT,
                                   fixed_preference=fixed_preference,
                                   output_dir=output_dir)


