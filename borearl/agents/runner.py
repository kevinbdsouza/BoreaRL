from __future__ import annotations

import os
import inspect
import csv
from datetime import datetime
import numpy as np
import torch
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv

from . import AGENTS
from .common import (
    make_env, save_run_config, build_preliminary_config, build_dynamic_scalarization,
    default_eval_weights, load_simple_yaml,
)
from borearl import constants as const
from ..utils.profiling import profiler
from ..utils.plotting import plot_profiling_statistics
from .baseline import run_baseline_pair_for_seed


def train(
    algorithm: str = 'eupg',
    total_timesteps: int = 500_000,
    use_wandb: bool = True,
    site_specific: bool | None = None,
    run_dir_name: str | None = None,
):
    profiler.start_timer('total_training')

    algo_key = str(algorithm).strip().lower()

    # Establish a stable run ID for this process to coordinate CSV and W&B behavior
    run_id = os.environ.get("BOREARL_RUN_ID")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["BOREARL_RUN_ID"] = run_id
    # Unify CSV filenames across any env instances created by upstream libs
    os.environ.setdefault("BOREARL_CSV_RUN_ID", run_id)
    # Encourage W&B to reuse a single run instead of spawning multiple
    if use_wandb:
        os.environ["WANDB_PROJECT"] = "Forest-MORL"
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"
        # Set the base directory for wandb, not the run directory itself
        os.environ["WANDB_DIR"] = os.path.abspath(os.getcwd())

        # Optional: avoid extra console noise
        os.environ.setdefault("WANDB_SILENT", "true")
        # Prefer online unless explicitly set to offline/disabled by user
        wandb_mode = str(os.environ.get("WANDB_MODE", "")).lower()
        wandb_disabled = str(os.environ.get("WANDB_DISABLED", "")).lower() == "true"
        if wandb_disabled or wandb_mode in ("offline", "dryrun", "disabled"):
            print("Warning: W&B is in offline/disabled mode (WANDB_MODE/WANDB_DISABLED). Metrics will not appear online.")
        else:
            os.environ.setdefault("WANDB_MODE", "online")
    # Prepare central per-run directory under logs/
    logs_base_dir = os.path.join('logs')
    os.makedirs(logs_base_dir, exist_ok=True)
    run_dir = os.path.join(logs_base_dir, (run_dir_name if run_dir_name else run_id))
    os.makedirs(run_dir, exist_ok=True)

    # Detect resume mode based on presence of run_id.txt, model file, and config.yaml
    resume_mode = False
    existing_run_id_path = os.path.join(run_dir, 'run_id.txt')
    if os.path.exists(existing_run_id_path):
        try:
            with open(existing_run_id_path, 'r') as f:
                existing_run_id = f.read().strip()
            if existing_run_id:
                run_id = existing_run_id
                resume_mode = True
        except Exception:
            resume_mode = False
    # Persist run id inside the run directory (overwrites with same value when resuming)
    with open(os.path.join(run_dir, 'run_id.txt'), 'w') as f:
        f.write(run_id + "\n")
    # Expose run directory and run id to subprocesses/utilities
    os.environ["BOREARL_RUN_DIR"] = os.path.abspath(run_dir)
    os.environ["BOREARL_RUN_ID"] = run_id
    os.environ["BOREARL_CSV_RUN_ID"] = run_id

    # Compute previous progress if resuming
    prev_global_steps = 0
    prev_episodes = 0
    if resume_mode:
        try:
            step_csv = os.path.join(run_dir, f"step_metrics_{run_id}.csv")
            if os.path.exists(step_csv):
                with open(step_csv, 'r') as f:
                    # subtract header
                    prev_global_steps = max(0, sum(1 for _ in f) - 1)
        except Exception:
            prev_global_steps = 0
        try:
            ep_csv = os.path.join(run_dir, f"episode_metrics_{run_id}.csv")
            if os.path.exists(ep_csv):
                with open(ep_csv, 'r') as f:
                    prev_episodes = max(0, sum(1 for _ in f) - 1)
        except Exception:
            prev_episodes = 0
        # Set the global step so logging and step caps continue from previous
        os.environ["BOREARL_GLOBAL_STEP_COUNT"] = str(prev_global_steps)

    # Enforce a hard global cap on total steps across env instances (safety)
    # If resuming: cap at previous steps + extra requested timesteps; otherwise just total_timesteps
    max_total_steps = int(total_timesteps) + (prev_global_steps if resume_mode else 0)
    os.environ["BOREARL_MAX_TOTAL_STEPS"] = str(max_total_steps)

    # Build env config
    env_config: dict = {}
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if resume_mode and os.path.exists(cfg_path):
        try:
            loaded_cfg = load_simple_yaml(cfg_path)
            if isinstance(loaded_cfg, dict) and 'environment' in loaded_cfg:
                env_config.update(loaded_cfg['environment'] or {})
        except Exception:
            env_config = {}
    # Apply CLI overrides or defaults
    site_flag = False if site_specific is None else bool(site_specific)
    if 'site_specific' not in env_config:
        env_config['site_specific'] = site_flag
    # Route all CSV logs to the central run directory
    env_config['csv_output_dir'] = run_dir
    # Keep CSV logging enabled during training
    env_config['csv_logging_enabled'] = True
    if algo_key != 'eupg':
        env_config['use_fixed_preference'] = False

    # Mark phase for downstream logging hooks before creating env so filenames are set correctly
    os.environ["BOREARL_PHASE"] = "train"
    env = make_env(env_config)
    unwrapped_env = getattr(env, 'unwrapped', env)

    # Save preliminary config for fresh runs only (do not overwrite when resuming)
    if not resume_mode:
        pre_config = build_preliminary_config(unwrapped_env, algo_key, total_timesteps)
        from .common import save_yaml
        save_yaml(os.path.join(run_dir, 'config.yaml'), pre_config)
        print(f"Saved preliminary training config to '{os.path.join(run_dir, 'config.yaml')}'")

    # Create model via agent module
    if algo_key not in AGENTS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from eupg, pcn, chm, gpi_ls.")
    agent_mod = AGENTS[algo_key]
    # Phase already set above
    # Pass through the agent hyperparameters
    if resume_mode and os.path.exists(cfg_path):
        loaded_cfg = {}
        try:
            loaded_cfg = load_simple_yaml(cfg_path)
        except Exception:
            loaded_cfg = {}
        pre_agent = (loaded_cfg.get('agent', {}) if isinstance(loaded_cfg, dict) else {})
    else:
        pre_agent = locals().get('pre_config', {}).get('agent', {}) if isinstance(locals().get('pre_config', {}), dict) else {}
    model = agent_mod.create(
        env,
        unwrapped_env,
        use_wandb,
        weights=pre_agent.get('weights'),
        gamma=pre_agent.get('gamma'),
        learning_rate=pre_agent.get('learning_rate'),
        net_arch=pre_agent.get('net_arch'),
    )

    # If resuming, load the saved model parameters before continuing training
    if resume_mode:
        try:
            fname = getattr(agent_mod, 'default_model_filename')()
            model_path = os.path.join(run_dir, fname)
            if os.path.exists(model_path):
                if getattr(agent_mod, 'supports_single_policy_eval')() and hasattr(model, 'get_policy_net'):
                    state = torch.load(model_path, map_location="cpu")
                    model.get_policy_net().load_state_dict(state)
                elif hasattr(agent_mod, 'load_policy_set'):
                    loaded = agent_mod.load_policy_set(model, model_path)
                    if loaded is not None:
                        model = loaded
        except Exception:
            pass

    # If resuming, also advance episode counter so CSVs continue numbering
    if resume_mode:
        try:
            uw = unwrapped_env
            while hasattr(uw, 'env'):
                uw = uw.env
            if hasattr(uw, 'episode_count'):
                setattr(uw, 'episode_count', int(prev_episodes))
        except Exception:
            pass

    # Prefer episode-based step axis for training metrics, if W&B is enabled
    if use_wandb:
        import wandb  # type: ignore
        # Ensure a consistent project/name and STABLE run id so eval can resume the same run
        if wandb.run is None:
            init_kwargs = {
                "project": os.environ.get("WANDB_PROJECT"),
                "name": f"{algo_key.upper()}-Forest",
                "id": run_id,
                "resume": "allow",
            }
            if os.environ.get("WANDB_ENTITY"):
                init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
            wandb.init(**init_kwargs)

        # Convenience: print URL if available
        if wandb.run is not None and getattr(wandb.run, "url", None):
            print(f"W&B run URL: {wandb.run.url}")

    # Save final configuration only for fresh runs (avoid overwriting original training config)
    if not resume_mode:
        save_run_config(env, algo_key, model, total_timesteps)

    # Train with signature robustness
    train_sig = inspect.signature(model.train)
    # During resume, treat total_timesteps as "extra timesteps"; otherwise, it's absolute
    model.train(total_timesteps=total_timesteps)

    # Save trained model into the run directory
    models_dir = run_dir
    os.makedirs(models_dir, exist_ok=True)
    fname = getattr(agent_mod, 'default_model_filename')()
    saved_model_path = None
    if getattr(agent_mod, 'supports_single_policy_eval')() and hasattr(model, 'get_policy_net'):
        policy = model.get_policy_net()
        saved_model_path = os.path.join(models_dir, fname)
        torch.save(policy.state_dict(), saved_model_path)
    else:
        # Coverage methods: persist policy set if helper is provided
        if hasattr(agent_mod, 'save_policy_set'):
            saved_model_path = os.path.join(models_dir, fname)
            agent_mod.save_policy_set(model, saved_model_path)

    # DO NOT finish the wandb run here. Let the training process exit, leaving the
    # run in a "crashed" state, which the evaluation process will then resume.

    total_training_time = profiler.end_timer('total_training')
    print(f"Total training time: {total_training_time:.3f} seconds")
    profiler.print_summary()
    # Save and plot profiling
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir)
    os.makedirs(output_dir, exist_ok=True)
    profiling_data_file = os.path.join(output_dir, f"profiling_data_{timestamp}.json")
    profiler.save_profiling_data(profiling_data_file)
    # Save profiling plots without blocking
    plot_profiling_statistics(profiling_data_file, show=False)

    return {"run_dir": run_dir, "model_path": saved_model_path}


def evaluate(
    algorithm: str = 'eupg',
    model_path: str | None = None,
    n_eval_episodes: int = 50,
    use_wandb: bool = True,
    site_specific: bool | None = None,
    config_overrides: dict | None = None,
    run_dir_name: str | None = None,
):
    algo_key = str(algorithm).strip().lower()

    # Resolve the run directory
    logs_base_dir = os.path.join('logs')
    env_run_dir = os.environ.get("BOREARL_RUN_DIR")
    if run_dir_name:
        run_dir = os.path.join(logs_base_dir, run_dir_name)
    elif env_run_dir:
        run_dir = env_run_dir
    else:
        # Fallbacks when no explicit run dir is provided:
        # 1) Back-compat: if logs/run_id.txt exists, use its value as the run dir name
        # 2) Otherwise, choose the most recently modified subdirectory of logs/ that contains a run_id.txt
        # 3) If none found, fall back to logs/ (will error below if run_id.txt missing)
        fallback_id_path = os.path.join(logs_base_dir, 'run_id.txt')
        if os.path.exists(fallback_id_path):
            try:
                with open(fallback_id_path, 'r') as f:
                    fallback_id = f.read().strip()
                run_dir = os.path.join(logs_base_dir, fallback_id)
            except Exception:
                run_dir = logs_base_dir
        else:
            try:
                candidates: list[tuple[float, str]] = []
                for entry in os.listdir(logs_base_dir):
                    entry_path = os.path.join(logs_base_dir, entry)
                    if os.path.isdir(entry_path):
                        rid_path = os.path.join(entry_path, 'run_id.txt')
                        if os.path.exists(rid_path):
                            candidates.append((os.path.getmtime(rid_path), entry_path))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    run_dir = candidates[0][1]
                else:
                    run_dir = logs_base_dir
            except Exception:
                run_dir = logs_base_dir
    os.makedirs(run_dir, exist_ok=True)

    # Load the run_id from the central run directory
    run_id_path = os.path.join(run_dir, 'run_id.txt')
    if not os.path.exists(run_id_path):
        raise FileNotFoundError(f"run_id.txt not found in run directory: {run_dir}")
    with open(run_id_path, 'r') as f:
        run_id = f.read().strip()
    if not run_id:
        raise FileNotFoundError("run_id.txt is empty")
    print(f"Using run directory '{run_dir}' with run_id: {run_id}")

    # FORCE the environment variables to use this exact run_id for all downstream processes
    os.environ["BOREARL_RUN_ID"] = run_id
    os.environ["BOREARL_CSV_RUN_ID"] = run_id
    os.environ["BOREARL_RUN_DIR"] = os.path.abspath(run_dir)

    if use_wandb:
        # FORCE wandb to use the exact same run_id as training and a single directory
        os.environ["WANDB_PROJECT"] = "Forest-MORL"
        os.environ["WANDB_RUN_ID"] = run_id
        # Use "must" to fail loudly if the run cannot be found
        os.environ["WANDB_RESUME"] = "must"
        os.environ["WANDB_DIR"] = os.path.abspath(os.getcwd())
        os.environ.setdefault("WANDB_SILENT", "true")

        wandb_mode = str(os.environ.get("WANDB_MODE", "")).lower()
        if "disabled" in wandb_mode or "offline" in wandb_mode:
            print("Warning: W&B is in offline/disabled mode.")
        else:
            os.environ.setdefault("WANDB_MODE", "online")

    # Ensure evaluation is not affected by any training step caps
    if "BOREARL_MAX_TOTAL_STEPS" in os.environ:
        os.environ.pop("BOREARL_MAX_TOTAL_STEPS", None)

    # Build env config with overrides
    env_config: dict | None = {}
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
        # Also look inside nested 'environment' section (format of config.yaml)
        if 'environment' in config_overrides and isinstance(config_overrides['environment'], dict):
            nested_env = config_overrides['environment']
            for k in allowed_keys:
                if k in nested_env and k not in env_config:
                    env_config[k] = nested_env[k]
    if site_specific is not None and 'site_specific' not in env_config:
        env_config['site_specific'] = bool(site_specific)
    if 'site_specific' in env_config:
        env_config['site_specific'] = bool(env_config['site_specific'])
    # Keep CSV logging enabled for evaluation but avoid empty CSVs (handled lazily in env)
    env_config['csv_logging_enabled'] = True

    # Mark phase for downstream logging hooks before creating env so filenames are set correctly
    os.environ["BOREARL_PHASE"] = "eval"
    # Ensure all CSVs and artifacts are written under the run directory
    env_config['csv_output_dir'] = run_dir
    env = make_env(env_config)
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env

    # Agent constructor
    if algo_key not in AGENTS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from eupg, pcn, chm, gpi_ls.")
    agent_mod = AGENTS[algo_key]

    # Build model for inference, applying optional overrides from config
    selected_weights = None
    selected_net_arch = None
    selected_gamma = None
    selected_lr = None
    if config_overrides:
        if 'weights' in config_overrides["agent"]:
            selected_weights = np.array(config_overrides["agent"]['weights'])
        if 'net_arch' in config_overrides["agent"]:
            selected_net_arch = config_overrides["agent"]['net_arch']
        if 'gamma' in config_overrides["agent"]:
            selected_gamma = float(config_overrides["agent"]['gamma'])
        if 'learning_rate' in config_overrides["agent"]:
            selected_lr = float(config_overrides["agent"]['learning_rate'])

    # Suppress agent's internal W&B logging during evaluation; we will resume and log manually
    # Phase already set above

    model = agent_mod.create(
        env,
        unwrapped_env,
        False,  # Disable wandb in agent to prevent duplicate runs
        weights=selected_weights,
        gamma=selected_gamma,
        learning_rate=selected_lr,
        net_arch=selected_net_arch,
    )

    # Load model params
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

    venv = MOSyncVectorEnv([lambda: make_env(env_config) for _ in range(1)])
    eval_weights = default_eval_weights(env_config)

    results = {'weights': [], 'carbon_objectives': [], 'thaw_objectives': [], 'scalarized_rewards': []}

    try:
        episodes_per_weight = int(n_eval_episodes)
        
        for weight_idx, weight in enumerate(eval_weights, start=1):
            carbon_rewards, thaw_rewards, scalarized_rewards = [], [], []
            episodes_for_this_weight = episodes_per_weight
            # Determine evaluation mode
            site_specific_run = bool(getattr(unwrapped_env, 'site_specific', False))
            # Precompute a representative seed per weight (used for site-specific baseline)
            first_episode_seed = int(1000003 * weight_idx)

            for episode_num in range(episodes_for_this_weight):
                venv.set_attr("current_preference_weight", float(weight[0]))
                # Derive a deterministic per-episode seed so baselines and agent share initial conditions/weather
                per_episode_seed = int(1000003 * weight_idx + episode_num)
                obs, info = venv.reset(seed=per_episode_seed)
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
                        if bool(const.EVAL_USE_ARGMAX_ACTIONS):
                            action = int(torch.argmax(logits, dim=1).item())
                        else:
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
                episodes_completed = (weight_idx - 1) * episodes_per_weight + episode_num + 1
                
                # Log each episode individually to W&B
                if use_wandb:
                    try:
                        import wandb  # type: ignore
                        payload = {
                            'eval_episode': int(episodes_completed),
                            'weight_carbon': float(weight[0]),
                            'weight_thaw': float(weight[1]),
                            'carbon_objective': float(episode_carbon),
                            'thaw_objective': float(episode_thaw),
                            'scalarized_reward': float(np.dot(weight, [episode_carbon, episode_thaw])),
                        }
                        wandb.log(payload, commit=True)
                    except Exception:
                        pass

                # Run baseline policies for the same seed and preference (exclude counterfactual here),
                # but do not log baseline results to W&B.
                # Generalist mode: run per episode to match varying seeds.
                # Site-specific mode: skip here; will run once per weight after loop.
                if not site_specific_run:
                    _ = run_baseline_pair_for_seed(
                        env_config=env_config,
                        seed=per_episode_seed,
                        fixed_preference=float(weight[0]),
                        output_dir=str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir),
                    )
                        
            # Site-specific mode: run baseline once per weight (identical across episodes)
            if site_specific_run:
                _ = run_baseline_pair_for_seed(
                    env_config=env_config,
                    seed=first_episode_seed,
                    fixed_preference=float(weight[0]),
                    output_dir=str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir),
                )

            # Record aggregated results for this weight if any episodes were completed
            if len(carbon_rewards) > 0:
                mean_carbon = np.mean(carbon_rewards)
                mean_thaw = np.mean(thaw_rewards)
                mean_scalarized = np.mean(scalarized_rewards)
                results['weights'].append(weight)
                results['carbon_objectives'].append(mean_carbon)
                results['thaw_objectives'].append(mean_thaw)
                results['scalarized_rewards'].append(mean_scalarized)
    finally:
        try:
            venv.close()
        except Exception:
            pass

    # Save an evaluation summary CSV for convenience
    output_dir = str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir)
    os.makedirs(output_dir, exist_ok=True)
    eval_csv_path = os.path.join(output_dir, f"eval_summary_{run_id}.csv")
    with open(eval_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["weight_carbon", "weight_thaw", "carbon_objective", "thaw_objective", "scalarized_reward"])
        for w, c, t, s in zip(results['weights'], results['carbon_objectives'], results['thaw_objectives'], results['scalarized_rewards']):
            writer.writerow([float(w[0]), float(w[1]), float(c), float(t), float(s)])
    print(f"Evaluation summary saved to '{eval_csv_path}'")

    # Finish the W&B run after evaluation is complete
    if use_wandb:
        import wandb  # type: ignore
        if wandb.run is not None:
            print(f"Finishing W&B run: {wandb.run.id}")
            wandb.finish()

    return results


