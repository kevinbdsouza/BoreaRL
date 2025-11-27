from __future__ import annotations

import os
import csv
from typing import Tuple, Dict, Any

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


def _rollout_fixed_action(env, action_index: int, fixed_preference: float, seed: int | None = None) -> Tuple[float, float, list[dict]]:
    set_env_preference(env, fixed_preference)
    # Ensure identical initial conditions and weather by passing an explicit seed when provided
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
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
            'preference_weight': float(fixed_preference),
            'baseline_type': str(getattr(env.unwrapped, 'baseline_type', '')),
        })
        total_carb += carb
        total_thaw += thaw
        step_idx += 1
    return total_carb, total_thaw, rows


def run_counterfactual_sensitivity(
    num_rng_samples: int = 100,
    fixed_preference: float = 0.5,
    output_dir: str = "logs",
    num_action_eval_seeds: int = 10,
    num_rng_eval_actions: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    prev_phase = os.environ.get('BOREARL_PHASE')
    os.environ['BOREARL_PHASE'] = 'baseline'
    
    # Ensure the environment config has the proper CSV output directory
    env_config = {'csv_output_dir': output_dir}
    
    try:
        env = make_env(env_config)
        # Mark baseline environment as evaluation to prevent step counting
        current_env = env
        while hasattr(current_env, 'env'):
            setattr(current_env, 'in_evaluation', True)
            current_env = current_env.env
        setattr(current_env, 'in_evaluation', True)
    finally:
        if prev_phase is not None:
            os.environ['BOREARL_PHASE'] = prev_phase
        else:
            os.environ.pop('BOREARL_PHASE', None)
    set_env_preference(env, fixed_preference)

    fixed_seed = 12345
    env.reset(seed=fixed_seed)

    # Evaluate all actions, averaging over multiple seeds to reduce dependence on a single RNG state
    action_sums = {
        a: {'carbon': 0.0, 'thaw': 0.0, 'scalarized': 0.0}
        for a in range(env.action_space.n)
    }
    for s_i in range(num_action_eval_seeds):
        seed = fixed_seed + s_i
        for a in range(env.action_space.n):
            env.reset(seed=seed)
            set_env_preference(env, fixed_preference)
            _, reward_vec, _, _, step_info = env.step(a)
            c = float(step_info.get('raw_carbon_component', reward_vec[0]))
            t = float(step_info.get('raw_thaw_component', reward_vec[1]))
            s = float(fixed_preference * c + (1.0 - fixed_preference) * t)
            action_sums[a]['carbon'] += c
            action_sums[a]['thaw'] += t
            action_sums[a]['scalarized'] += s
    action_rewards = [
        {
            'action': a,
            'carbon': action_sums[a]['carbon'] / max(1, num_action_eval_seeds),
            'thaw': action_sums[a]['thaw'] / max(1, num_action_eval_seeds),
            'scalarized': action_sums[a]['scalarized'] / max(1, num_action_eval_seeds),
        }
        for a in range(env.action_space.n)
    ]

    # RNG sensitivity: average over multiple actions to reduce dependence on a single action choice
    n_actions = env.action_space.n
    if num_rng_eval_actions >= n_actions:
        sampled_actions = list(range(n_actions))
    else:
        # Evenly spaced unique actions
        sampled_actions = sorted(set(np.linspace(0, n_actions - 1, num=num_rng_eval_actions, dtype=int).tolist()))
        if len(sampled_actions) == 0:
            sampled_actions = [n_actions // 2]
    rng_rewards = []
    for i in range(num_rng_samples):
        seed = 1000 + i
        # For each seed, evaluate the selected actions from the same initial state and average
        carbon_vals = []
        thaw_vals = []
        scalar_vals = []
        for a in sampled_actions:
            env.reset(seed=seed)
            set_env_preference(env, fixed_preference)
            _, reward_vec, _, _, step_info = env.step(a)
            c = float(step_info.get('raw_carbon_component', reward_vec[0]))
            t = float(step_info.get('raw_thaw_component', reward_vec[1]))
            s = float(fixed_preference * c + (1.0 - fixed_preference) * t)
            carbon_vals.append(c)
            thaw_vals.append(t)
            scalar_vals.append(s)
        rng_rewards.append({
            'seed': seed,
            'carbon': float(np.mean(carbon_vals)) if len(carbon_vals) > 0 else 0.0,
            'thaw': float(np.mean(thaw_vals)) if len(thaw_vals) > 0 else 0.0,
            'scalarized': float(np.mean(scalar_vals)) if len(scalar_vals) > 0 else 0.0,
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
    print(
        f"  Actions (avg over {num_action_eval_seeds} seeds) -> carbon std={act_c_std:.3f}, "
        f"thaw std={act_t_std:.3f}, scalarized std={act_s_std:.3f}"
    )
    print(
        f"  RNG (avg over {len(sampled_actions)} actions) -> carbon std={rng_c_std:.3f}, "
        f"thaw std={rng_t_std:.3f}, scalarized std={rng_s_std:.3f}"
    )
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
    prev_phase = os.environ.get('BOREARL_PHASE')
    os.environ['BOREARL_PHASE'] = 'baseline'
    
    # Ensure the environment config has the proper CSV output directory
    env_config = {'csv_output_dir': output_dir}
    
    try:
        env = make_env(env_config)
        # Mark baseline environment as evaluation to prevent step counting
        current_env = env
        while hasattr(current_env, 'env'):
            setattr(current_env, 'in_evaluation', True)
            current_env = current_env.env
        setattr(current_env, 'in_evaluation', True)
    finally:
        if prev_phase is not None:
            os.environ['BOREARL_PHASE'] = prev_phase
        else:
            os.environ.pop('BOREARL_PHASE', None)
    # Zero density baseline (conifer mix irrelevant; choose 0.5)
    zero_action = _select_action_index(env.unwrapped, 0, 0.5)
    setattr(env.unwrapped, 'baseline_type', 'zero')
    z_c, z_t, z_rows = _rollout_fixed_action(env, zero_action, fixed_preference)
    z_path = os.path.join(output_dir, 'baseline_zero_density.csv')
    with open(z_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step','action','carbon','thaw','scalarized','preference_weight','baseline_type'])
        w.writeheader()
        w.writerows(z_rows)
    print(f"Saved zero-density baseline to '{z_path}' (totals: carbon={z_c:.3f}, thaw={z_t:.3f})")

    # +100 density with 0.5 species mix baseline
    plus_action = _select_action_index(env.unwrapped, 100, 0.5)
    setattr(env.unwrapped, 'baseline_type', 'plus100_0p5mix')
    p_c, p_t, p_rows = _rollout_fixed_action(env, plus_action, fixed_preference)
    p_path = os.path.join(output_dir, 'baseline_plus100_density_0p5mix.csv')
    with open(p_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['step','action','carbon','thaw','scalarized','preference_weight','baseline_type'])
        w.writeheader()
        w.writerows(p_rows)
    print(f"Saved +100 density baseline to '{p_path}' (totals: carbon={p_c:.3f}, thaw={p_t:.3f})")

    # Counterfactual experiment
    run_counterfactual_sensitivity(num_rng_samples=const.COUNTERFACTUAL_SAMPLES_DEFAULT,
                                   fixed_preference=fixed_preference,
                                   output_dir=output_dir)


def run_baseline_pair_for_seed(
    env_config: dict | None,
    seed: int,
    fixed_preference: float,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Run the two fixed-action baselines (zero-density, +100 density @ 0.5 mix) using the
    exact same initial condition and weather seed as the provided seed. Returns totals
    for logging. The environment's own CSV logging will capture per-step/episode rows.
    """
    prev_phase = os.environ.get('BOREARL_PHASE')
    os.environ['BOREARL_PHASE'] = 'baseline'
    
    # Ensure the environment config has the proper CSV output directory
    if env_config is None:
        env_config = {}
    if output_dir is not None:
        env_config = dict(env_config)  # Create a copy to avoid modifying the original
        env_config['csv_output_dir'] = output_dir
    
    try:
        env = make_env(env_config)
        # Mark baseline environment as evaluation to prevent step counting
        current_env = env
        while hasattr(current_env, 'env'):
            setattr(current_env, 'in_evaluation', True)
            current_env = current_env.env
        setattr(current_env, 'in_evaluation', True)
    finally:
        if prev_phase is not None:
            os.environ['BOREARL_PHASE'] = prev_phase
        else:
            os.environ.pop('BOREARL_PHASE', None)
    # Zero density baseline (conifer mix irrelevant; choose 0.5)
    zero_action = _select_action_index(env.unwrapped, 0, 0.5)
    setattr(env.unwrapped, 'baseline_type', 'zero')
    z_c, z_t, _ = _rollout_fixed_action(env, zero_action, fixed_preference, seed=seed)

    # +100 density with 0.5 species mix baseline
    plus_action = _select_action_index(env.unwrapped, 100, 0.5)
    setattr(env.unwrapped, 'baseline_type', 'plus100_0p5mix')
    p_c, p_t, _ = _rollout_fixed_action(env, plus_action, fixed_preference, seed=seed)

    result = {
        'seed': int(seed),
        'preference': float(fixed_preference),
        'zero_density': {
            'carbon': float(z_c),
            'thaw': float(z_t),
            'scalarized': float(fixed_preference * z_c + (1.0 - fixed_preference) * z_t),
        },
        '+100_density_0p5mix': {
            'carbon': float(p_c),
            'thaw': float(p_t),
            'scalarized': float(fixed_preference * p_c + (1.0 - fixed_preference) * p_t),
        },
    }
    return result



class HeuristicAgent:
    """Base class for heuristic agents."""
    def __init__(self, env):
        self.env = env
    
    def predict(self, obs, deterministic=True):
        raise NotImplementedError

class TargetDensityAgent(HeuristicAgent):
    """
    Maintains a target density by thinning if too high and planting if too low.
    Maintains a mixed species composition (0.5 conifer).
    """
    def __init__(self, env, target_density=1000, tolerance=100):
        super().__init__(env)
        self.target_density = target_density
        self.tolerance = tolerance
        
    def predict(self, obs, deterministic=True):
        # obs[2] is normalized density (density / 1500)
        current_density = obs[2] * 1500.0
        
        action_density_idx = 2 # No change
        
        if current_density < self.target_density - self.tolerance:
            # Plant
            if current_density < self.target_density - self.tolerance - 50:
                action_density_idx = 4 # +100
            else:
                action_density_idx = 3 # +50
        elif current_density > self.target_density + self.tolerance:
            # Thin
            if current_density > self.target_density + self.tolerance + 50:
                action_density_idx = 0 # -100
            else:
                action_density_idx = 1 # -50
                
        # Always target 0.5 mix
        action_mix_idx = 2 
        
        action = action_density_idx * 5 + action_mix_idx
        return action, None

class ConiferRestorationAgent(HeuristicAgent):
    """
    Aggressively plants conifers to maximize carbon, similar to industrial forestry.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def predict(self, obs, deterministic=True):
        # Always plant +50 density, target 1.0 conifer
        # Unless density is already high (>1500), then do nothing
        current_density = obs[2] * 1500.0
        
        if current_density < 1500:
            action_density_idx = 3 # +50
        else:
            action_density_idx = 2 # 0
            
        action_mix_idx = 4 # 1.0 conifer
        
        action = action_density_idx * 5 + action_mix_idx
        return action, None
