from __future__ import annotations

import os
import numpy as np
import json
from datetime import datetime
import torch
import gymnasium as gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from morl_baselines.single_policy.esr.eupg import EUPG

# Ensure environment is registered on import
import borearl.env.forest_env  # noqa: F401
from borearl import constants as const
from ..utils.profiling import profiler
from ..utils.plotting import plot_profiling_statistics


def _make_env(env_config: dict | None = None):
    # Register the environment lazily via class import side effect
    return gym.make("ForestEnv-v0", config=env_config, disable_env_checker=True)


def _set_env_preference(env, pref: float):
    """Helper to set the environment's preference weight consistently."""
    try:
        unwrapped = env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, 'set_preference_weight'):
            unwrapped.set_preference_weight(pref)
        else:
            setattr(unwrapped, 'current_preference_weight', pref)
    except Exception:
        pass


def train_morl(total_timesteps=500_000, use_wandb=True, site_specific: bool | None = None):
    profiler.start_timer('total_training')

    env_config = None
    try:
        # Build env config using centralized defaults
        site_flag = const.SITE_SPECIFIC_DEFAULT if site_specific is None else bool(site_specific)
        env_config = dict(site_specific=site_flag)
    except Exception:
        env_config = None

    env = _make_env(env_config)
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env

    def scalarization(reward_vector, weights=None):
        # ESR: use the env-provided (and logged) preference weight each episode.
        import numpy as np
        import torch
        # Pull latest preference weight from the environment to align scalarization
        try:
            pref = float(getattr(unwrapped_env, 'current_preference_weight', 0.5))
            dynamic_weights = np.array([pref, 1.0 - pref], dtype=np.float32)
        except Exception:
            dynamic_weights = np.array([0.5, 0.5], dtype=np.float32)
        # Always use the dynamic weights derived from the environment to achieve ESR
        weights = dynamic_weights
        if isinstance(reward_vector, torch.Tensor):
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=reward_vector.dtype, device=reward_vector.device)
            if reward_vector.ndim == 2:
                return torch.matmul(reward_vector, weights)
            else:
                return torch.dot(reward_vector, weights)
        return np.dot(reward_vector, weights)

    # EUPG expects a fixed weight vector but our scalarization closure reads the
    # environment preference dynamically, so the exact value passed here is less
    # important; still provide a reasonable default.
    if use_wandb:
        model = EUPG(
            env=env,
            scalarization=scalarization,
            weights=np.array([0.5, 0.5]),
            gamma=0.99,
            learning_rate=0.001,
            log=True,
            project_name="Forest-MORL",
            experiment_name="EUPG-Forest",
        )
    else:
        model = EUPG(
            env=env,
            scalarization=scalarization,
            weights=np.array([0.5, 0.5]),
            gamma=0.99,
            learning_rate=0.001,
            log=False,
        )

    # Hook into the training loop to sample a weight per episode.
    # EUPG's high-level train API doesn't expose callbacks, so we rely on the
    # environment randomizing `current_preference_weight` on reset and then
    # we update the model's internal weight before each rollout. This keeps
    # the scalarization aligned with the env's displayed weight.
    model.train(total_timesteps=total_timesteps)

    policy = model.get_policy_net()
    # Ensure models directory exists and save model there
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(models_dir, "eupg_forest_manager.pth"))

    total_training_time = profiler.end_timer('total_training')
    print(f"Total training time: {total_training_time:.3f} seconds")
    profiler.print_summary()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiling_data_file = f"profiling_data_{timestamp}.json"
    profiler.save_profiling_data(profiling_data_file)
    plot_profiling_statistics(profiling_data_file)


def evaluate_morl_policy(model_path="models/eupg_forest_manager.pth", n_eval_episodes=50, use_wandb=True, site_specific: bool | None = None):
    env_config = None
    try:
        site_flag = const.SITE_SPECIFIC_DEFAULT if site_specific is None else bool(site_specific)
        env_config = dict(site_specific=site_flag)
    except Exception:
        env_config = None
    env = _make_env(env_config)
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env

    if use_wandb:
        model = EUPG(
            env=env,
            scalarization=lambda r, w=None: np.dot(r, w or [0.5, 0.5]),
            weights=np.array([0.5, 0.5]),
            gamma=0.99,
            learning_rate=0.001,
            log=True,
            project_name="Forest-MORL-Eval",
            experiment_name="EUPG-Forest-Eval",
        )
    else:
        model = EUPG(
            env=env,
            scalarization=lambda r, w=None: np.dot(r, w or [0.5, 0.5]),
            weights=np.array([0.5, 0.5]),
            gamma=0.99,
            learning_rate=0.001,
            log=False,
        )

    policy = model.get_policy_net()
    import torch
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    venv = MOSyncVectorEnv([lambda: _make_env(env_config) for _ in range(1)])

    eval_weights = np.array([
        [0.0, 1.0],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.9, 0.1],
        [1.0, 0.0],
    ])

    results = {'weights': [], 'carbon_objectives': [], 'thaw_objectives': [], 'scalarized_rewards': []}

    for weight in eval_weights:
        carbon_rewards, thaw_rewards, scalarized_rewards = [], [], []
        for _ in range(n_eval_episodes):
            venv.set_attr("current_preference_weight", weight[0])
            obs, info = venv.reset()
            terminated, truncated = False, False
            episode_carbon, episode_thaw = 0.0, 0.0
            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                acc_reward = torch.zeros((obs_tensor.shape[0], 2), dtype=torch.float32)
                with torch.no_grad():
                    logits = policy.forward(obs_tensor, acc_reward=acc_reward)
                    action = int(torch.argmax(logits, dim=-1).item())
                import numpy as _np
                obs, reward_vector, terminated, truncated, info = venv.step(_np.array([action], dtype=_np.int64))
                # reward_vector shape: (num_envs, reward_dim)
                episode_carbon += float(reward_vector[0][0])
                episode_thaw += float(reward_vector[0][1])
            scalarized_rewards.append(np.dot(weight, [episode_carbon, episode_thaw]))
            carbon_rewards.append(episode_carbon)
            thaw_rewards.append(episode_thaw)
        results['weights'].append(weight)
        results['carbon_objectives'].append(np.mean(carbon_rewards))
        results['thaw_objectives'].append(np.mean(thaw_rewards))
        results['scalarized_rewards'].append(np.mean(scalarized_rewards))

    # Pareto front plot is produced by caller or separate utility if needed
    return results



def run_counterfactual_sensitivity(num_rng_samples: int = 100, fixed_preference: float = 0.5, output_dir: str = "plots"):
    """
    Compare action-induced reward spread vs RNG-induced spread for a one-year rollout.
    - Action spread: same initial state and weather (identical seed), evaluate all actions.
    - RNG spread: fixed action, vary seeds.
    Prints summary statistics and saves a small CSV.
    """
    import os
    import numpy as np
    import csv
    os.makedirs(output_dir, exist_ok=True)

    env = _make_env()
    # Fix preference to avoid mixing scalarization effects
    _set_env_preference(env, fixed_preference)

    # Use a fixed seed to create the same start state and weather
    fixed_seed = 12345
    obs, info = env.reset(seed=fixed_seed)

    # Evaluate all actions from the same state
    action_rewards = []
    for a in range(env.action_space.n):
        env.reset(seed=fixed_seed)
        _set_env_preference(env, fixed_preference)
        _, reward_vec, _, _, step_info = env.step(a)
        action_rewards.append({
            'action': a,
            'carbon': float(step_info.get('raw_carbon_component', reward_vec[0])),
            'thaw': float(step_info.get('raw_thaw_component', reward_vec[1])),
            'scalarized': float(fixed_preference * step_info.get('raw_carbon_component', reward_vec[0]) + (1.0 - fixed_preference) * step_info.get('raw_thaw_component', reward_vec[1])),
        })

    # Pick a representative action: middle index
    fixed_action = env.action_space.n // 2
    rng_rewards = []
    for i in range(num_rng_samples):
        seed = 1000 + i
        env.reset(seed=seed)
        _set_env_preference(env, fixed_preference)
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

    act_c_mean, act_c_std, act_c_range = summarize([d['carbon'] for d in action_rewards])
    act_t_mean, act_t_std, act_t_range = summarize([d['thaw'] for d in action_rewards])
    act_s_mean, act_s_std, act_s_range = summarize([d['scalarized'] for d in action_rewards])

    rng_c_mean, rng_c_std, rng_c_range = summarize([d['carbon'] for d in rng_rewards])
    rng_t_mean, rng_t_std, rng_t_range = summarize([d['thaw'] for d in rng_rewards])
    rng_s_mean, rng_s_std, rng_s_range = summarize([d['scalarized'] for d in rng_rewards])

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

