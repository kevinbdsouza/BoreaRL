from __future__ import annotations

import os
import inspect
import numpy as np
import torch
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

from . import AGENTS
from .common import (
    make_env, save_run_config, build_preliminary_config, build_dynamic_scalarization,
    default_eval_weights,
)
from borearl import constants as const
from ..utils.profiling import profiler
from ..utils.plotting import plot_profiling_statistics


def train(algorithm: str = 'eupg', total_timesteps: int = 500_000, use_wandb: bool = True, site_specific: bool | None = None):
    profiler.start_timer('total_training')

    algo_key = str(algorithm).strip().lower()

    # Build env config
    env_config = None
    try:
        site_flag = False if site_specific is None else bool(site_specific)
        env_config = dict(site_specific=site_flag)
        if algo_key != 'eupg':
            env_config['use_fixed_preference'] = False
    except Exception:
        env_config = None

    env = make_env(env_config)
    unwrapped_env = getattr(env, 'unwrapped', env)

    # Save preliminary config
    try:
        pre_config = build_preliminary_config(unwrapped_env, algo_key, total_timesteps)
        output_dir = pre_config['environment']['csv_output_dir'] or 'logs'
        os.makedirs(output_dir, exist_ok=True)
        from .common import save_yaml
        save_yaml(os.path.join(output_dir, 'config.yaml'), pre_config)
        print(f"Saved preliminary training config to '{os.path.join(output_dir, 'config.yaml')}'")
    except Exception as e:
        print(f"Warning: failed to save preliminary config: {e}")

    # Create model via agent module
    if algo_key not in AGENTS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from eupg, pcn, chm, gpi_ls.")
    agent_mod = AGENTS[algo_key]
    model = agent_mod.create(env, unwrapped_env, use_wandb)

    # Save final configuration
    save_run_config(env, algo_key, model, total_timesteps)

    # Train with signature robustness
    try:
        train_sig = inspect.signature(model.train)
        if 'total_timesteps' in train_sig.parameters:
            model.train(total_timesteps=total_timesteps)
        elif 'n_timesteps' in train_sig.parameters:
            model.train(n_timesteps=total_timesteps)
        else:
            model.train(total_timesteps)
    except Exception:
        model.train(total_timesteps=total_timesteps)

    # Save trained model
    try:
        models_dir = os.path.join('models')
        os.makedirs(models_dir, exist_ok=True)
        fname = getattr(agent_mod, 'default_model_filename')()
        if getattr(agent_mod, 'supports_single_policy_eval')() and hasattr(model, 'get_policy_net'):
            policy = model.get_policy_net()
            torch.save(policy.state_dict(), os.path.join(models_dir, fname))
        else:
            # Coverage methods: persist policy set if helper is provided
            if hasattr(agent_mod, 'save_policy_set'):
                agent_mod.save_policy_set(model, os.path.join(models_dir, fname))
    except Exception:
        pass

    total_training_time = profiler.end_timer('total_training')
    print(f"Total training time: {total_training_time:.3f} seconds")
    profiler.print_summary()
    # Save and plot profiling
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(getattr(unwrapped_env, 'csv_output_dir', 'logs') or 'logs')
    os.makedirs(output_dir, exist_ok=True)
    profiling_data_file = os.path.join(output_dir, f"profiling_data_{timestamp}.json")
    profiler.save_profiling_data(profiling_data_file)
    plot_profiling_statistics(profiling_data_file)


def evaluate(
    algorithm: str = 'eupg',
    model_path: str | None = None,
    n_eval_episodes: int = 50,
    use_wandb: bool = True,
    site_specific: bool | None = None,
    config_overrides: dict | None = None,
):
    algo_key = str(algorithm).strip().lower()

    # Build env config with overrides
    env_config: dict | None = {}
    try:
        if config_overrides:
            allowed_keys = {
                'site_specific', 'include_site_params_in_obs', 'site_weather_seed', 'deterministic_temp_noise',
                'remove_age_jitter', 'use_fixed_site_initials', 'csv_logging_enabled', 'csv_output_dir',
                'site_overrides', 'standardize_rewards', 'reward_ema_beta',
                'use_fixed_preference', 'eupg_default_weights',
            }
            for k in allowed_keys:
                if k in config_overrides:
                    env_config[k] = config_overrides[k]
        if site_specific is not None and 'site_specific' not in env_config:
            env_config['site_specific'] = bool(site_specific)
        if 'site_specific' in env_config:
            env_config['site_specific'] = bool(env_config['site_specific'])
    except Exception:
        env_config = None

    env = make_env(env_config)
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env

    # Agent constructor
    if algo_key not in AGENTS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from eupg, pcn, chm, gpi_ls.")
    agent_mod = AGENTS[algo_key]

    # Set weights and hyperparameters from config_overrides when present
    selected_weights = np.array(config_overrides.get('weights', const.EUPG_DEFAULT_WEIGHTS)) if config_overrides else np.array(const.EUPG_DEFAULT_WEIGHTS)
    selected_net_arch = config_overrides.get('net_arch', const.EUPG_NET_ARCH_DEFAULT) if config_overrides else const.EUPG_NET_ARCH_DEFAULT
    selected_gamma = float(config_overrides.get('gamma', const.EUPG_GAMMA_DEFAULT)) if config_overrides else const.EUPG_GAMMA_DEFAULT
    selected_lr = float(config_overrides.get('learning_rate', const.EUPG_LEARNING_RATE_DEFAULT)) if config_overrides else const.EUPG_LEARNING_RATE_DEFAULT

    # Build model for inference
    model = agent_mod.create(env, unwrapped_env, use_wandb)
    # Load model params
    try:
        if agent_mod.supports_single_policy_eval():
            policy = model.get_policy_net()
            if model_path and os.path.exists(model_path):
                state = torch.load(model_path)
                policy.load_state_dict(state)
            policy.eval()
        else:
            if model_path and os.path.exists(model_path) and hasattr(agent_mod, 'load_policy_set'):
                # load_policy_set may return a new model instance
                loaded = agent_mod.load_policy_set(model, model_path)
                if loaded is not None:
                    model = loaded
    except Exception:
        pass

    venv = MOSyncVectorEnv([lambda: make_env(env_config) for _ in range(1)])
    eval_weights = default_eval_weights(env_config)

    results = {'weights': [], 'carbon_objectives': [], 'thaw_objectives': [], 'scalarized_rewards': []}

    for weight in eval_weights:
        carbon_rewards, thaw_rewards, scalarized_rewards = [], [], []
        for _ in range(n_eval_episodes):
            obs, info = venv.reset()
            venv.set_attr("current_preference_weight", float(weight[0]))
            terminated, truncated = False, False
            episode_carbon, episode_thaw = 0.0, 0.0
            acc_reward = torch.zeros((1, 2), dtype=torch.float32)
            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                with torch.no_grad():
                    if hasattr(model, 'get_policy_net'):
                        logits = model.get_policy_net().forward(obs_tensor, acc_reward=acc_reward)
                    else:
                        logits = model.policy_forward(obs_tensor, acc_reward=acc_reward)  # type: ignore
                    action_tensor = torch.distributions.Categorical(logits=logits).sample()
                    action = int(action_tensor.item())
                import numpy as _np
                obs, reward_vector, terminated, truncated, info = venv.step(_np.array([action], dtype=_np.int64))
                if hasattr(terminated, '__len__'):
                    terminated = bool(terminated[0])
                if hasattr(truncated, '__len__'):
                    truncated = bool(truncated[0])
                episode_carbon += float(reward_vector[0][0])
                episode_thaw += float(reward_vector[0][1])
                acc_reward = acc_reward + torch.as_tensor(reward_vector, dtype=torch.float32)
            scalarized_rewards.append(np.dot(weight, [episode_carbon, episode_thaw]))
            carbon_rewards.append(episode_carbon)
            thaw_rewards.append(episode_thaw)
        results['weights'].append(weight)
        results['carbon_objectives'].append(np.mean(carbon_rewards))
        results['thaw_objectives'].append(np.mean(thaw_rewards))
        results['scalarized_rewards'].append(np.mean(scalarized_rewards))

    return results


