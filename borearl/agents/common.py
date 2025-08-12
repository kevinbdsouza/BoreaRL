from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Tuple

import numpy as np
import torch
import gymnasium as gym
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

# Ensure environment is registered on import
import borearl.env.forest_env  # noqa: F401
from borearl import constants as const
from ..utils.profiling import profiler
from ..utils.plotting import plot_profiling_statistics


def yaml_format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(c in text for c in [":", "#", "{", "}", "[", "]", ","]) or text.strip() != text or text == "":
        return '"' + text.replace('"', '\\"') + '"'
    return text


def yaml_dump(obj: Any, indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    if isinstance(obj, dict):
        for key, val in obj.items():
            key_str = str(key)
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key_str}:")
                lines.extend(yaml_dump(val, indent + 2))
            else:
                lines.append(f"{pad}{key_str}: {yaml_format_scalar(val)}")
    elif isinstance(obj, list):
        for val in obj:
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(yaml_dump(val, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_format_scalar(val)}")
    else:
        lines.append(f"{pad}{yaml_format_scalar(obj)}")
    return lines


def make_env(env_config: dict | None = None):
    # Inject backend defaults if caller did not specify
    if env_config is None:
        env_config = {}
    else:
        env_config = dict(env_config)
    import os as _os
    # Allow environment variables to override defaults
    backend_env = _os.environ.get('BOREARL_PHYSICS_BACKEND')
    fast_mode_env = _os.environ.get('BOREARL_FAST_MODE')
    jit_iters_env = _os.environ.get('BOREARL_JIT_SOLVER_MAX_ITERS')
    stab_interval_env = _os.environ.get('BOREARL_STABILITY_UPDATE_INTERVAL_STEPS')

    env_config.setdefault('physics_backend', backend_env or const.PHYSICS_BACKEND_DEFAULT)
    env_config.setdefault('fast_mode', (str(fast_mode_env).lower() in ('1', 'true', 'yes')) if fast_mode_env is not None else const.FAST_MODE_DEFAULT)
    if env_config.get('physics_backend', const.PHYSICS_BACKEND_DEFAULT) == 'numba':
        env_config.setdefault('jit_solver_max_iters', int(jit_iters_env) if jit_iters_env else const.JIT_SOLVER_MAX_ITERS_DEFAULT)
        env_config.setdefault('stability_update_interval_steps', int(stab_interval_env) if stab_interval_env else const.STABILITY_UPDATE_INTERVAL_STEPS_DEFAULT)
    return gym.make("ForestEnv-v0", config=env_config, disable_env_checker=True)


def set_env_preference(env, pref: float):
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


def save_run_config(env, agent_name: str, model: Any, total_timesteps: int):
    # If a vectorized env is provided, introspect a single underlying env instance
    unwrapped = env
    try:
        # MOSyncVectorEnv API: has attribute env_fns or get_attr
        if hasattr(env, 'get_attr'):
            # Try to fetch attribute from the first sub-env
            vals = env.get_attr('eupg_default_weights')
            if isinstance(vals, (list, tuple)) and len(vals) > 0:
                sentinel_weights = vals[0]
            else:
                sentinel_weights = None
            # Grab a single real env for shape/introspection if needed
            try:
                first_envs = env.get_attr('observation_space')
                if first_envs:
                    unwrapped = env
            except Exception:
                pass
        # Fallback to unwrapped attribute for single env
        if hasattr(unwrapped, 'unwrapped'):
            unwrapped = unwrapped.unwrapped
    except Exception:
        unwrapped = getattr(env, 'unwrapped', env)
    env_conf = {
        'site_specific': bool(getattr(unwrapped, 'site_specific', False)),
        'include_site_params_in_obs': bool(getattr(unwrapped, 'include_site_params_in_obs', False)),
        'observation_size': int(getattr(unwrapped, 'observation_space', getattr(env, 'observation_space')).shape[0]),
        'reward_dim': int(getattr(unwrapped, 'reward_space', getattr(unwrapped, 'reward_space')).shape[0]),
        'standardize_rewards': bool(getattr(unwrapped, 'enable_reward_standardization', False)),
        'reward_ema_beta': float(getattr(unwrapped, 'reward_ema_beta', 0.99)),
        'site_weather_seed': int(getattr(unwrapped, 'site_weather_seed', 0)),
        'deterministic_temp_noise': bool(getattr(unwrapped, 'deterministic_temp_noise', False)),
        'remove_age_jitter': bool(getattr(unwrapped, 'remove_age_jitter', False)),
        # Read directly from env attribute populated during ForestEnv.__init__
        'use_fixed_site_initials': bool(getattr(unwrapped, 'use_fixed_site_initials', False)),
        'csv_logging_enabled': bool(getattr(unwrapped, 'csv_logging_enabled', True)),
        'csv_output_dir': str(getattr(unwrapped, 'csv_output_dir', 'logs')),
        'use_fixed_preference': bool(getattr(unwrapped, 'use_fixed_preference', False)),
        # Load directly from env (env guarantees attribute is set in __init__)
        'eupg_default_weights': list(getattr(unwrapped, 'eupg_default_weights', const.EUPG_DEFAULT_WEIGHTS)),
    }
    if bool(getattr(unwrapped, 'site_specific', False)):
        env_conf['site_overrides'] = dict(getattr(unwrapped, 'site_overrides', {}))

    try:
        weights = getattr(model, 'weights', None)
        weights_list = list(weights.tolist()) if hasattr(weights, 'tolist') else (list(weights) if weights is not None else None)
    except Exception:
        weights_list = None

    agent_conf = {
        'algorithm': str(agent_name).lower(),
        'gamma': float(getattr(model, 'gamma', const.EUPG_GAMMA_DEFAULT)),
        'learning_rate': float(getattr(model, 'learning_rate', const.EUPG_LEARNING_RATE_DEFAULT)),
        'buffer_size': int(getattr(model, 'buffer_size', 0)),
        'net_arch': list(getattr(model, 'net_arch', const.EUPG_NET_ARCH_DEFAULT)),
        'weights': weights_list,
        'device': str(getattr(model, 'device', 'auto')),
        'seed': int(getattr(model, 'seed', 0)) if getattr(model, 'seed', None) is not None else None,
        'log': bool(getattr(model, 'log', True)),
        'log_every': int(getattr(model, 'log_every', 1000)),
        'project_name': str(getattr(model, 'project_name', '')),
        'experiment_name': str(getattr(model, 'experiment_name', '')),
    }

    train_conf = {
        'total_timesteps': int(total_timesteps),
    }

    config_dict = {
        'environment': env_conf,
        'agent': agent_conf,
        'training': train_conf,
    }

    output_dir = env_conf['csv_output_dir'] or 'logs'
    os.makedirs(output_dir, exist_ok=True)
    yaml_text = "\n".join(yaml_dump(config_dict)) + "\n"
    path_latest = os.path.join(output_dir, 'config.yaml')
    with open(path_latest, 'w') as f:
        f.write(yaml_text)
    print(f"Saved training config to '{path_latest}'")


def build_preliminary_config(unwrapped_env, agent_name: str, total_timesteps: int):
    pre_env_conf = {
        'site_specific': bool(getattr(unwrapped_env, 'site_specific', False)),
        'include_site_params_in_obs': bool(getattr(unwrapped_env, 'include_site_params_in_obs', False)),
        'observation_size': int(getattr(unwrapped_env, 'observation_space', getattr(unwrapped_env, 'observation_space')).shape[0]),
        'reward_dim': int(getattr(unwrapped_env, 'reward_space', getattr(unwrapped_env, 'reward_space')).shape[0]),
        'standardize_rewards': bool(getattr(unwrapped_env, 'enable_reward_standardization', False)),
        'reward_ema_beta': float(getattr(unwrapped_env, 'reward_ema_beta', 0.99)),
        'site_weather_seed': int(getattr(unwrapped_env, 'site_weather_seed', 0)),
        'deterministic_temp_noise': bool(getattr(unwrapped_env, 'deterministic_temp_noise', False)),
        'remove_age_jitter': bool(getattr(unwrapped_env, 'remove_age_jitter', False)),
        # Read directly from env attribute populated during ForestEnv.__init__
        'use_fixed_site_initials': bool(getattr(unwrapped_env, 'use_fixed_site_initials', False)),
        'csv_logging_enabled': bool(getattr(unwrapped_env, 'csv_logging_enabled', True)),
        'csv_output_dir': str(getattr(unwrapped_env, 'csv_output_dir', 'logs')),
        'use_fixed_preference': bool(getattr(unwrapped_env, 'use_fixed_preference', False)),
        # Load directly from env (env guarantees attribute is set in __init__)
        'eupg_default_weights': list(getattr(unwrapped_env, 'eupg_default_weights')),
    }
    if bool(getattr(unwrapped_env, 'site_specific', False)):
        pre_env_conf['site_overrides'] = dict(getattr(unwrapped_env, 'site_overrides', {}))

    pre_agent_conf = {
        'algorithm': str(agent_name).lower(),
        'gamma': float(const.EUPG_GAMMA_DEFAULT),
        'learning_rate': float(const.EUPG_LEARNING_RATE_DEFAULT),
        'net_arch': list(const.EUPG_NET_ARCH_DEFAULT),
        'weights': list(const.EUPG_DEFAULT_WEIGHTS),
        'device': 'auto',
        'seed': None,
        'log': True,
        'log_every': 1000,
        'project_name': 'Forest-MORL',
        'experiment_name': f'{agent_name.upper()}-Forest',
    }
    pre_train_conf = {'total_timesteps': int(total_timesteps)}
    pre_config_dict = {
        'environment': pre_env_conf,
        'agent': pre_agent_conf,
        'training': pre_train_conf,
    }
    return pre_config_dict


def build_dynamic_scalarization(unwrapped_env):
    def scalarization(reward_vector, weights=None):
        import numpy as _np
        import torch as _torch
        try:
            pref = float(getattr(unwrapped_env, 'current_preference_weight', 0.5))
            dynamic_weights = _np.array([pref, 1.0 - pref], dtype=_np.float32)
        except Exception:
            dynamic_weights = _np.array([0.5, 0.5], dtype=_np.float32)
        weights = dynamic_weights
        if isinstance(reward_vector, _torch.Tensor):
            if not isinstance(weights, _torch.Tensor):
                weights = _torch.tensor(weights, dtype=reward_vector.dtype, device=reward_vector.device)
            if reward_vector.ndim == 2:
                return _torch.matmul(reward_vector, weights)
            else:
                return _torch.dot(reward_vector, weights)
        return _np.dot(reward_vector, weights)
    return scalarization


def default_eval_weights(env_config: dict | None) -> np.ndarray:
    try:
        use_fixed_pref_cfg = bool(env_config.get('use_fixed_preference', const.USE_FIXED_PREFERENCE_DEFAULT)) if isinstance(env_config, dict) else const.USE_FIXED_PREFERENCE_DEFAULT
    except Exception:
        use_fixed_pref_cfg = const.USE_FIXED_PREFERENCE_DEFAULT
    if use_fixed_pref_cfg:
        env_default_w = None
        try:
            if isinstance(env_config, dict) and 'eupg_default_weights' in env_config:
                w = env_config['eupg_default_weights']
                if isinstance(w, (list, tuple)) and len(w) == 2:
                    env_default_w = [float(w[0]), float(w[1])]
        except Exception:
            env_default_w = None
        if env_default_w is None:
            env_default_w = [float(const.EUPG_DEFAULT_WEIGHTS[0]), float(const.EUPG_DEFAULT_WEIGHTS[1])]
        return np.array([env_default_w], dtype=np.float32)
    return np.array([
        [0.0, 1.0], [0.33, 0.67], [0.67, 0.33], [1.0, 0.0],
    ], dtype=np.float32)


def save_yaml(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write("\n".join(yaml_dump(data)) + "\n")


def _parse_scalar_from_yaml_token(token: str):
    text = token.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('\\"', '"')
    try:
        if "." in text or "e" in text or "E" in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def load_simple_yaml(path: str) -> dict:
    """Load YAML configuration robustly.

    Tries PyYAML's safe loader first (if available) and falls back to a
    minimal parser for simple YAML. This avoids brittle parsing errors
    with nested lists and mappings.
    """
    # Prefer PyYAML if present
    try:
        import yaml  # type: ignore
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else (data or {})
    except Exception:
        pass

    # Fallback: minimal parser (supports a subset of YAML)
    with open(path, 'r') as f:
        lines = [ln.rstrip("\n") for ln in f]

    root: dict | list = {}
    stack: list[tuple[int, dict | list, str | None]] = [(-1, root, None)]

    def current_container_for_indent(indent: int):
        while stack and stack[-1][0] >= indent:
            stack.pop()
        return stack[-1][1]

    for raw in lines:
        if not raw.strip() or raw.strip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        container = current_container_for_indent(indent)
        line = raw.strip()

        if line.startswith('-'):
            # Ensure we have a list container to add to
            if not isinstance(container, list):
                # If parent is a dict, convert the last inserted key to a list
                parent_indent, parent_container, _ = stack[-1]
                if isinstance(parent_container, dict) and parent_container:
                    last_key = list(parent_container.keys())[-1]
                    if not isinstance(parent_container[last_key], list):
                        parent_container[last_key] = []
                    container = parent_container[last_key]
                else:
                    raise ValueError("YAML format unexpected: list item without preceding key")

            item_content = line[1:].strip()
            if not item_content:
                new_item: dict = {}
                container.append(new_item)
                stack.append((indent, new_item, None))
            else:
                container.append(_parse_scalar_from_yaml_token(item_content))
        else:
            if ':' not in line:
                continue
            key, rest = line.split(':', 1)
            key = key.strip()
            value_part = rest.strip()
            if value_part == "":
                if not isinstance(container, dict):
                    raise ValueError("YAML format unexpected: mapping entry inside list without '-' item")
                # Defer type selection; create a placeholder and decide on next token
                container[key] = {}
                stack.append((indent, container[key], key))
            else:
                if not isinstance(container, dict):
                    raise ValueError("YAML format unexpected: key-value inside list without '-' item")
                container[key] = _parse_scalar_from_yaml_token(value_part)

    return root



