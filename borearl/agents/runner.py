from __future__ import annotations

import os
import inspect
import csv
from datetime import datetime
import numpy as np
import torch
from mo_gymnasium.wrappers.vector import MOSyncVectorEnv
from typing import Optional

from . import AGENTS
from .common import (
    make_env, save_run_config, build_preliminary_config, build_dynamic_scalarization,
    default_eval_weights, load_simple_yaml, get_action_from_model,
)
from borearl import constants as const
from ..utils.profiling import profiler
from ..utils.plotting import plot_profiling_statistics
from .baseline import run_baseline_pair_for_seed


def _evaluate_model_periodic(model, env_config, agent_mod, run_dir, run_id, current_step, n_eval_episodes=10):
    """
    Evaluate the current model checkpoint and log results to CSV.
    
    Args:
        model: The current model to evaluate
        env_config: Environment configuration
        agent_mod: Agent module for evaluation
        run_dir: Directory to save results
        run_id: Run ID for file naming
        current_step: Current training step number
        n_eval_episodes: Number of episodes per weight for evaluation
    """
    print(f"Performing periodic evaluation at step {current_step}...")
    
    # Create evaluation environment with CSV logging disabled to prevent interference with main training
    eval_env_config = env_config.copy()
    eval_env_config['csv_logging_enabled'] = False  # Disable CSV logging for periodic eval
    
    # Store the original global step count to restore it later
    original_global_step = os.environ.get('BOREARL_GLOBAL_STEP_COUNT', '0')
    
    # Create evaluation environment - use a completely separate environment instance
    # to avoid interfering with the main training step counter
    eval_env = make_env(eval_env_config)
    unwrapped_eval_env = eval_env
    seen = set()
    while hasattr(unwrapped_eval_env, "env"):
        if id(unwrapped_eval_env) in seen or unwrapped_eval_env.env is unwrapped_eval_env:
            break
        seen.add(id(unwrapped_eval_env))
        unwrapped_eval_env = unwrapped_eval_env.env
    
    # Set evaluation flag to prevent step counting during evaluation
    # Find the monitored environment and set the flag
    monitored_env = None
    if hasattr(model, 'env'):
        monitored_env = model.env
        if hasattr(monitored_env, 'in_evaluation'):
            monitored_env.in_evaluation = True
    
    # Get evaluation weights
    eval_weights = default_eval_weights(env_config)
    
    # Prepare CSV file for logging
    csv_path = os.path.join(run_dir, f"periodic_eval_{run_id}.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['step_number', 'lambda_carbon', 'lambda_thaw', 'episode_seed', 'avg_carbon_reward', 'avg_thaw_reward']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:
            writer.writeheader()
        
        # Evaluate for each weight
        for weight_idx, weight in enumerate(eval_weights):
            episodes_for_this_weight = n_eval_episodes
            
            for episode_num in range(episodes_for_this_weight):
                # Set preference weight
                unwrapped_eval_env.current_preference_weight = float(weight[0])
                
                # Derive deterministic seed
                per_episode_seed = int(1000003 * weight_idx + episode_num)
                obs, info = eval_env.reset(seed=per_episode_seed)
                
                terminated, truncated = False, False
                episode_carbon, episode_thaw = 0.0, 0.0
                acc_reward = torch.zeros((1, 2), dtype=torch.float32)
                
                # Run episode
                while not (terminated or truncated):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    action = get_action_from_model(model, obs_tensor, acc_reward, weight)
                    
                    # Use scalar action for non-vector environment
                    obs, reward_vector, terminated, truncated, info = eval_env.step(int(action))
                    
                    if hasattr(terminated, '__len__'):
                        terminated = bool(terminated[0])
                    if hasattr(truncated, '__len__'):
                        truncated = bool(truncated[0])
                    
                    # Treat reward as 1-D vector (2 objectives)
                    episode_carbon += float(reward_vector[0])
                    episode_thaw += float(reward_vector[1])
                    acc_reward = acc_reward + torch.as_tensor(reward_vector, dtype=torch.float32)  # broadcasts fine
                
                # Log results to CSV
                writer.writerow({
                    'step_number': current_step,
                    'lambda_carbon': float(weight[0]),
                    'lambda_thaw': float(weight[1]),
                    'episode_seed': per_episode_seed,
                    'avg_carbon_reward': episode_carbon,
                    'avg_thaw_reward': episode_thaw
                })
    
    # Clean up
    eval_env.close()
    
    # Clear evaluation flag to resume step counting
    if monitored_env and hasattr(monitored_env, 'in_evaluation'):
        monitored_env.in_evaluation = False
    
    # Restore the original global step count to prevent interference with main training
    os.environ['BOREARL_GLOBAL_STEP_COUNT'] = original_global_step
    
    print(f"Periodic evaluation completed at step {current_step}. Results saved to {csv_path}")


def _train_with_periodic_saving(model, unwrapped_env, total_timesteps, agent_mod, run_dir, save_interval=100, eval_interval=1000, n_eval_episodes=10):
    """
    Custom training function that saves the model every save_interval episodes (only when scalarized_episodic_return improves) 
    and evaluates every eval_interval steps.
    
    Args:
        model: The model to train
        unwrapped_env: The unwrapped environment to monitor episode count
        total_timesteps: Total timesteps for training
        agent_mod: Agent module for saving functionality
        run_dir: Directory to save models
        save_interval: Save model every N episodes (default: 100)
        eval_interval: Evaluate model every N steps (default: 1000)
        n_eval_episodes: Number of episodes per weight for evaluation (default: 10)
    """
    # Define a custom exception to signal the end of training
    class TrainingComplete(Exception):
        pass
    
    # Create models directory
    models_dir = run_dir
    os.makedirs(models_dir, exist_ok=True)
    
    # Track the last episode count when we saved
    last_saved_episode = 0
    last_eval_step = 0
    best_scalarized_return = float('-inf')  # Track the best metric seen so far
    
    # Get run_id for evaluation logging
    run_id = os.environ.get("BOREARL_RUN_ID", "unknown")
    
    # Get environment config for evaluation
    env_config = {}
    # Extract environment config from the unwrapped environment
    if hasattr(unwrapped_env, 'site_specific'):
        env_config['site_specific'] = unwrapped_env.site_specific
    if hasattr(unwrapped_env, 'eupg_default_weights'):
        env_config['eupg_default_weights'] = unwrapped_env.eupg_default_weights
    if hasattr(unwrapped_env, 'use_fixed_preference'):
        env_config['use_fixed_preference'] = unwrapped_env.use_fixed_preference
    # Ensure CSV output directory is set correctly
    if hasattr(unwrapped_env, 'csv_output_dir'):
        env_config['csv_output_dir'] = unwrapped_env.csv_output_dir
    else:
        env_config['csv_output_dir'] = run_dir
    
    def get_latest_scalarized_return() -> Optional[float]:
        """
        Get the latest scalarized reward from the training state JSON file.
        Falls back to CSV parsing for backward compatibility.
        """
        try:
            # Get run_id from environment variables for consistency
            current_run_id = os.environ.get("BOREARL_RUN_ID", run_id)
            
            # Try to read from the training state JSON file
            state_file = os.path.join(run_dir, f"training_state_{current_run_id}.json")
            if os.path.exists(state_file):
                import json
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    if "latest_scalarized_return" in state:
                        return float(state["latest_scalarized_return"])
            
            # Fallback to CSV parsing for backward compatibility
            csv_pattern = os.path.join(run_dir, f"episode_metrics_{current_run_id}.csv")
            if not os.path.exists(csv_pattern):
                return None
            
            import csv
            with open(csv_pattern, 'r') as f:
                reader = csv.DictReader(f)
                last = None
                for last in reader: pass
                if last and "total_scalarized_reward" in last:
                    return float(last["total_scalarized_reward"])
            return None
            
        except Exception as e:
            print(f"Warning: Could not read scalarized reward from any source: {e}")
            return None
    
    def save_training_state(scalarized_return: float, episode_num: int):
        """
        Save training state to JSON file for robust state management.
        Only saves when there's an improvement in scalarized return.
        """
        try:
            import json
            from datetime import datetime
            
            # Get run_id from environment variables for consistency
            current_run_id = os.environ.get("BOREARL_RUN_ID", run_id)
            
            # Save to the training state JSON file
            state_file = os.path.join(run_dir, f"training_state_{current_run_id}.json")
            state = {
                "latest_scalarized_return": scalarized_return,
                "last_episode": episode_num,
                "last_updated": datetime.now().isoformat(),
                "run_id": current_run_id
            }
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                    
        except Exception as e:
            print(f"Warning: Could not save training state: {e}")
    
    def save_model_checkpoint(episode_num):
        """Save model checkpoint if it's time to save and metric has improved"""
        nonlocal last_saved_episode, best_scalarized_return
        
        if episode_num >= last_saved_episode + save_interval:
            try:
                # Get the current scalarized return from the environment
                current_scalarized_return = unwrapped_env.get_current_episode_scalarized_reward()
                
                # If we can't get the metric from environment, try getting it from the wrapper
                if current_scalarized_return is None and hasattr(model, 'env') and hasattr(model.env, 'last_episode_scalarized_reward'):
                    current_scalarized_return = model.env.last_episode_scalarized_reward
                
                # If we still can't get the metric, try reading from JSON
                if current_scalarized_return is None:
                    current_scalarized_return = get_latest_scalarized_return()
                
                # If we still can't get the metric, save anyway (fallback behavior)
                if current_scalarized_return is None:
                    print(f"Warning: Could not read scalarized_episodic_return, saving checkpoint anyway at episode {episode_num}")
                    should_save = True
                else:
                    # Only save if the metric has improved
                    should_save = current_scalarized_return > best_scalarized_return
                    if should_save:
                        print(f"Metric improved from {best_scalarized_return:.6f} to {current_scalarized_return:.6f}, saving checkpoint")
                        best_scalarized_return = current_scalarized_return
                        # Save the improved state
                        save_training_state(current_scalarized_return, episode_num)
                    else:
                        print(f"Metric {current_scalarized_return:.6f} not better than {best_scalarized_return:.6f}, skipping checkpoint")
                
                if should_save:
                    # Create checkpoint filename (fixed name, gets overwritten)
                    base_fname = getattr(agent_mod, 'default_model_filename')()
                    checkpoint_path = os.path.join(models_dir, base_fname)
                    
                    # Save the model
                    if getattr(agent_mod, 'supports_single_policy_eval')():
                        if hasattr(model, 'save'):
                            # PCN agent - use torch.save directly (more reliable)
                            try:
                                import torch
                                # Temporarily remove the wrapper to avoid pickling issues
                                original_env = model.env
                                model.env = unwrapped_env
                                torch.save(model, checkpoint_path)
                                model.env = original_env  # Restore the wrapper
                                print(f"Saved model checkpoint at episode {episode_num}: {checkpoint_path}")
                            except Exception as e:
                                print(f"Error saving model checkpoint at episode {episode_num}: {e}")
                        elif hasattr(model, 'get_policy_net'):
                            # EUPG agent - save policy network
                            try:
                                policy = model.get_policy_net()
                                if policy is not None:
                                    torch.save(policy.state_dict(), checkpoint_path)
                                    print(f"Saved model checkpoint at episode {episode_num}: {checkpoint_path}")
                                else:
                                    print(f"Warning: Policy network is None for episode {episode_num}")
                            except Exception as e:
                                print(f"Error saving model checkpoint at episode {episode_num}: {e}")
                        elif hasattr(model, 'model'):
                            # Fallback for other agents with model attribute
                            try:
                                torch.save(model.model.state_dict(), checkpoint_path)
                                print(f"Saved model checkpoint at episode {episode_num}: {checkpoint_path}")
                            except Exception as e:
                                print(f"Error saving model checkpoint at episode {episode_num}: {e}")
                        else:
                            print(f"Warning: Could not save model checkpoint - no save method found")
                    else:
                        # Coverage methods: persist policy set if helper is provided
                        if hasattr(agent_mod, 'save_policy_set'):
                            agent_mod.save_policy_set(model, checkpoint_path)
                            print(f"Saved model checkpoint at episode {episode_num}: {checkpoint_path}")
                    
                    last_saved_episode = episode_num
                    
            except Exception as e:
                print(f"Warning: Failed to save model checkpoint at episode {episode_num}: {e}")
    
    # Create a custom environment wrapper that monitors episode completion and steps
    class EpisodeMonitorWrapper:
        def __init__(self, env, save_callback, eval_callback, max_steps, model_ref=None):
            self.env = env
            self.save_callback = save_callback
            self.eval_callback = eval_callback
            self.max_steps = max_steps
            self.model_ref = model_ref  # Reference to the model for PCN goal conditioning
            self.original_reset = env.reset
            self.original_step = env.step
            self.step_count = 0
            self.in_evaluation = False  # Flag to track if we're in evaluation mode
            
        def reset(self, *args, **kwargs):
            # Get the scalarized reward BEFORE reset clears the episode data
            if hasattr(self.env, 'episode_count'):
                current_episode = self.env.episode_count
                # Store the current episode's scalarized reward before it gets cleared
                if hasattr(self.env, 'get_current_episode_scalarized_reward'):
                    self.last_episode_scalarized_reward = self.env.get_current_episode_scalarized_reward()
            
            result = self.original_reset(*args, **kwargs)
            
            # >>> Sync PCN goal-conditioning with env preference (generalist λ) <<<
            try:
                # Get the preference weight from the environment after reset
                pref = float(getattr(self.env, 'current_preference_weight', 0.5))
                if self.model_ref is not None and hasattr(self.model_ref, 'set_desired_return_and_horizon'):
                    target_return = np.array([
                        pref * const.MAX_CARBON_RETURN,
                        (1.0 - pref) * const.MAX_THAW_RETURN,
                    ], dtype=np.float32)
                    target_horizon = const.EPISODE_LENGTH_YEARS
                    self.model_ref.set_desired_return_and_horizon(target_return, target_horizon)
            except Exception as e:
                print(f"Warning: could not sync PCN desired return from env preference: {e}")
            # ---------------------------------------------------------------
            
            # Check if we need to save after reset (episode count is incremented in reset)
            if hasattr(self.env, 'episode_count'):
                self.save_callback(self.env.episode_count)
            return result
        
        def step(self, *args, **kwargs):
            result = self.original_step(*args, **kwargs)
            
            # Only count steps if we're not in evaluation mode
            if not self.in_evaluation:
                self.step_count += 1
                # Check if we need to evaluate
                self.eval_callback(self.step_count)
                
                # **FIX:** Instead of just returning True, raise an exception to halt training
                if self.step_count >= self.max_steps:
                    raise TrainingComplete(f"Total timesteps limit of {self.max_steps} reached.")
            
            return result
        
        def __getattr__(self, name):
            return getattr(self.env, name)
    
    def evaluate_callback(current_step):
        """Evaluate model if it's time to evaluate"""
        nonlocal last_eval_step
        if current_step >= last_eval_step + eval_interval:
            try:
                _evaluate_model_periodic(model, env_config, agent_mod, run_dir, run_id, current_step, n_eval_episodes)
                last_eval_step = current_step
            except Exception as e:
                print(f"Warning: Failed to perform periodic evaluation at step {current_step}: {e}")
    
    # Wrap the environment to monitor episodes and steps
    monitored_env = EpisodeMonitorWrapper(unwrapped_env, save_model_checkpoint, evaluate_callback, total_timesteps, model)
    
    # Replace the environment in the model if possible
    if hasattr(model, 'env'):
        model.env = monitored_env
    
    print(f"Starting checkpointing every {save_interval} episodes (only when scalarized_episodic_return improves) and evaluation every {eval_interval} steps...")
    
    # **FIX:** Wrap the training call in a try...except block to catch our custom exception
    try:
        # Start the training process
        # Handle different agent types with different training signatures
        import inspect
        train_sig = inspect.signature(model.train)
        if 'eval_env' in train_sig.parameters and 'ref_point' in train_sig.parameters:
            # PCN agent requires eval_env and ref_point
            from borearl.agents.common import make_env
            eval_env = make_env(env_config)  # Pass the same config used for training
            # Use configurable reference point for hypervolume calculation
            ref_point = np.array(const.PCN_REFERENCE_POINT)
            
            # Calculate the total episode budget based on the timestep limit.
            max_episodes = max(1, total_timesteps // const.EPISODE_LENGTH_YEARS)
            
            # Split the episode budget between PCN's two sequential training phases.
            episodes_per_phase = max(1, max_episodes // 2)

            # Call train with both a timestep limit and a correctly allocated episode limit.
            # The wrapper will ensure the timestep limit is respected.
            model.train(
                total_timesteps=total_timesteps, 
                eval_env=eval_env, 
                ref_point=ref_point,
                # Pass half of the episode budget to each phase.
                num_er_episodes=episodes_per_phase,
                num_step_episodes=episodes_per_phase
            )
        else:
            # Standard training interface (EUPG, etc.)
            model.train(total_timesteps=total_timesteps)
    
    except TrainingComplete as e:
        print(f"Training successfully halted: {e}")
    
    # Manually trigger a final save after training halts
    final_episode = getattr(unwrapped_env, 'episode_count', 0)
    save_model_checkpoint(final_episode)


def train(
    algorithm: str = 'eupg',
    total_timesteps: int = 500_000,
    use_wandb: bool = True,
    site_specific: bool | None = None,
    run_dir_name: str | None = None,
    save_interval: int = 100,
    eval_interval: int = 1000,
    n_eval_episodes: int = 10,
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
        with open(existing_run_id_path, 'r') as f:
            existing_run_id = f.read().strip()
        if existing_run_id:
            run_id = existing_run_id
            resume_mode = True
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
        step_csv = os.path.join(run_dir, f"step_metrics_{run_id}.csv")
        if os.path.exists(step_csv):
            with open(step_csv, 'r') as f:
                # subtract header
                prev_global_steps = max(0, sum(1 for _ in f) - 1)
        ep_csv = os.path.join(run_dir, f"episode_metrics_{run_id}.csv")
        if os.path.exists(ep_csv):
            with open(ep_csv, 'r') as f:
                prev_episodes = max(0, sum(1 for _ in f) - 1)
        # Set the global step so logging and step caps continue from previous
        os.environ["BOREARL_GLOBAL_STEP_COUNT"] = str(prev_global_steps)

    # Enforce a hard global cap on total steps across env instances (safety)
    # If resuming: cap at previous steps + extra requested timesteps; otherwise just total_timesteps
    max_total_steps = int(total_timesteps) + (prev_global_steps if resume_mode else 0)
    os.environ["BOREARL_MAX_TOTAL_STEPS"] = str(max_total_steps)
    
    # Also set a stricter limit for the current run
    os.environ["BOREARL_CURRENT_RUN_STEPS"] = str(total_timesteps)

    # Build env config
    env_config: dict = {}
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if resume_mode and os.path.exists(cfg_path):
        loaded_cfg = load_simple_yaml(cfg_path)
        if isinstance(loaded_cfg, dict) and 'environment' in loaded_cfg:
            env_config.update(loaded_cfg['environment'] or {})
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
    # Optional vectorized training envs (disabled for some algorithms)
    num_envs = 1
    num_envs = max(1, int(os.environ.get('BOREARL_NUM_ENVS', '1')))
    # EUPG from morl-baselines expects non-vectorized envs; enforce single env
    if algo_key == 'eupg':
        num_envs = 1
    if num_envs > 1:
        # Build a sample env for configuration/introspection
        sample_env = make_env(env_config)
        sample_unwrapped = getattr(sample_env, 'unwrapped', sample_env)
        env = MOSyncVectorEnv([lambda: make_env(env_config) for _ in range(num_envs)])
        unwrapped_env = sample_unwrapped
    else:
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
        run_dir_name=run_dir_name,
    )

    # If resuming, load the saved model parameters before continuing training
    if resume_mode:
        try:
            fname = getattr(agent_mod, 'default_model_filename')()
            model_path = os.path.join(run_dir, fname)
            if os.path.exists(model_path):
                if getattr(agent_mod, 'supports_single_policy_eval')() and hasattr(model, 'get_policy_net'):
                    state = torch.load(model_path, map_location="cpu", weights_only=False)
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
            seen = set()
            while hasattr(uw, 'env'):
                if id(uw) in seen or uw.env is uw:
                    break
                seen.add(id(uw))
                uw = uw.env
            if hasattr(uw, 'episode_count'):
                setattr(uw, 'episode_count', int(prev_episodes))
        except Exception:
            pass

    # Prefer episode-based step axis for training metrics, if W&B is enabled
    if use_wandb:
        import wandb  # type: ignore
        os.environ.pop("WANDB_DISABLED", None)  # <<< ensure not disabled

        if wandb.run is None:
            wandb_run_name = run_dir_name if run_dir_name else f"{algo_key.upper()}-Forest"
            init_kwargs = {
                "project": os.environ.get("WANDB_PROJECT", "Forest-MORL"),
                "name": wandb_run_name,
                "id": run_id,
                "resume": "allow",
                "dir": os.path.abspath(os.environ.get("WANDB_DIR", os.getcwd())),
            }
            if os.environ.get("WANDB_ENTITY"):
                init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
            wandb.init(**init_kwargs)

        # Convenience: print URL if available
        if wandb.run is not None and getattr(wandb.run, "url", None):
            print(f"W&B run URL: {wandb.run.url}")

        # <<< define a consistent step axis and map your metrics to it
        # Simple approach: just log with step numbers
        # Test log to verify W&B is working
        wandb.log({"test": 1.0}, step=0)

    # Save final configuration only for fresh runs (avoid overwriting original training config)
    if not resume_mode:
        save_run_config(env, algo_key, model, total_timesteps)

    # Train with signature robustness
    train_sig = inspect.signature(model.train)
    # During resume, treat total_timesteps as "extra timesteps"; otherwise, it's absolute
    _train_with_periodic_saving(model, unwrapped_env, total_timesteps, agent_mod, run_dir, save_interval, eval_interval, n_eval_episodes)

    # Save trained model into the run directory
    models_dir = run_dir
    os.makedirs(models_dir, exist_ok=True)
    fname = getattr(agent_mod, 'default_model_filename')()
    saved_model_path = None
    if getattr(agent_mod, 'supports_single_policy_eval')():
        if hasattr(model, 'get_policy_net'):
            # EUPG agent - save policy network
            policy = model.get_policy_net()
            if policy is not None:
                saved_model_path = os.path.join(models_dir, fname)
                torch.save(policy.state_dict(), saved_model_path)
        elif hasattr(model, 'model'):
            # PCN agent - save the model
            saved_model_path = os.path.join(models_dir, fname)
            torch.save(model.model.state_dict(), saved_model_path)
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
    plot_profiling_statistics(profiling_data_file, show=False, output_dir=output_dir)

    try:
        env.close()
    except Exception:
        pass
    return {"run_dir": run_dir, "model_path": saved_model_path}


def evaluate(
    algorithm: str = 'eupg',
    model_path: str | None = None,
    n_eval_episodes: int = 50,
    use_wandb: bool = False,  # Disable wandb for evaluation
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
            with open(fallback_id_path, 'r') as f:
                fallback_id = f.read().strip()
            run_dir = os.path.join(logs_base_dir, fallback_id)
        else:
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

    # Wandb disabled for evaluation to prevent hanging issues
    use_wandb = False

    # Ensure evaluation is not affected by any training step caps
    if "BOREARL_MAX_TOTAL_STEPS" in os.environ:
        os.environ.pop("BOREARL_MAX_TOTAL_STEPS", None)
    # Reset the global step counter for evaluation
    os.environ["BOREARL_GLOBAL_STEP_COUNT"] = "0"

    # Build env config with overrides
    env_config: dict | None = {}
    if config_overrides:
        allowed_keys = {
            'site_specific', 'include_site_params_in_obs', 'site_weather_seed', 'deterministic_temp_noise',
            'remove_age_jitter', 'use_fixed_site_initials', 'csv_logging_enabled', 'csv_output_dir',
            'site_overrides', 'standardize_rewards', 'reward_ema_beta',
            'use_fixed_preference', 'eupg_default_weights',
        }
        # First look inside nested 'environment' section (format of config.yaml)
        if 'environment' in config_overrides and isinstance(config_overrides['environment'], dict):
            nested_env = config_overrides['environment']
            for k in allowed_keys:
                if k in nested_env:
                    env_config[k] = nested_env[k]
        # Then check top-level keys (for backward compatibility)
        for k in allowed_keys:
            if k in config_overrides and k not in env_config:
                env_config[k] = config_overrides[k]
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
    seen = set()
    while hasattr(unwrapped_env, 'env'):
        if id(unwrapped_env) in seen or unwrapped_env.env is unwrapped_env:
            break
        seen.add(id(unwrapped_env))
        unwrapped_env = unwrapped_env.env

    # Agent constructor
    if algo_key not in AGENTS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from eupg, pcn, chm, gpi_ls.")
    agent_mod = AGENTS[algo_key]

    # Build model for inference, applying optional overrides from config
    selected_weights = selected_net_arch = selected_gamma = selected_lr = None
    if config_overrides and isinstance(config_overrides, dict):
        agent_over = config_overrides.get("agent")
        if isinstance(agent_over, dict):
            if "weights" in agent_over:      selected_weights = np.array(agent_over["weights"])
            if "net_arch" in agent_over:     selected_net_arch = agent_over["net_arch"]
            if "gamma" in agent_over:        selected_gamma = float(agent_over["gamma"])
            if "learning_rate" in agent_over:selected_lr = float(agent_over["learning_rate"])

    # Suppress agent's internal W&B logging during evaluation; we will resume and log manually
    # Phase already set above

    model = agent_mod.create(
        env,
        unwrapped_env,
        False,  # Disable wandb in agent to prevent duplicate runs
        weights=None,  # Don't pass weights for evaluation - we'll load the trained model
        gamma=selected_gamma,
        learning_rate=selected_lr,
        net_arch=selected_net_arch,
        run_dir_name=run_dir_name,
    )

    # Load model params - ensure we always use a trained model for evaluation
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model file not found for evaluation. "
            f"Expected model path: {model_path}. "
            f"Please ensure the model was trained and saved before evaluation."
        )
    
    # Verify we have a valid model path before proceeding
    print(f"Loading trained model from: {model_path}")
    
    if agent_mod.supports_single_policy_eval():
        if hasattr(agent_mod, 'load_policy_set'):
            # For agents like PCN that save the entire model object
            loaded = agent_mod.load_policy_set(model, model_path)
            if loaded is not None:
                model = loaded
                print(f"Successfully loaded PCN model from {model_path}")
            else:
                raise RuntimeError(f"Failed to load PCN model from {model_path}")
        else:
            # For agents like EUPG that use get_policy_net
            policy = model.get_policy_net()
            if policy is not None:  # Check if get_policy_net returns a valid policy
                state = torch.load(model_path, weights_only=False)
                policy.load_state_dict(state)
                policy.eval()
                print(f"Successfully loaded EUPG model weights from {model_path}")
            else:
                raise RuntimeError(f"Failed to get policy network for EUPG model")
    else:
        if hasattr(agent_mod, 'load_policy_set'):
            # load_policy_set may return a new model instance
            loaded = agent_mod.load_policy_set(model, model_path)
            if loaded is not None:
                model = loaded
                print(f"Successfully loaded model from {model_path}")
            else:
                raise RuntimeError(f"Failed to load model from {model_path}")
        else:
            raise RuntimeError(f"Agent {algorithm} does not support model loading for evaluation")

    venv = MOSyncVectorEnv([lambda: make_env(env_config) for _ in range(1)])
    eval_weights = default_eval_weights(env_config)

    results = {'weights': [], 'carbon_objectives': [], 'thaw_objectives': [], 'scalarized_rewards': []}
    # Parallel list to results['weights'] holding averaged baseline metrics per weight
    baseline_means_per_weight: list[dict] = []

    try:
        episodes_per_weight = int(n_eval_episodes)
        
        for weight_idx, weight in enumerate(eval_weights, start=1):
            # Accumulators for baseline metrics across episodes for this weight
            baseline_buffers = {
                'zero': {'carbon': [], 'thaw': [], 'scalarized': []},
                'plus': {'carbon': [], 'thaw': [], 'scalarized': []},
            }
            carbon_rewards, thaw_rewards, scalarized_rewards = [], [], []
            episodes_for_this_weight = episodes_per_weight
            # Determine evaluation mode
            site_specific_run = bool(getattr(unwrapped_env, 'site_specific', False))
            # Precompute a representative seed per weight (used for site-specific baseline)
            first_episode_seed = int(1000003 * weight_idx)

            for episode_num in range(episodes_for_this_weight):
                # Set preference weight on vector env if supported, otherwise on underlying envs
                if hasattr(venv, "set_attr"):
                    venv.set_attr("current_preference_weight", float(weight[0]))
                else:
                    # Fallback: set on underlying envs after reset
                    try:
                        for e in getattr(venv, "envs", []):
                            setattr(getattr(e, "unwrapped", e), "current_preference_weight", float(weight[0]))
                    except Exception:
                        pass
                # Also set on the unwrapped env
                unwrapped_env.current_preference_weight = float(weight[0])
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
                    action = get_action_from_model(model, obs_tensor, acc_reward, weight)
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
                
                # Wandb logging disabled for evaluation

                # Run baseline policies for the same seed and preference (exclude counterfactual here),
                # but do not log baseline results to W&B.
                # Generalist mode: run per episode to match varying seeds.
                # Site-specific mode: skip here; will run once per weight after loop.
                if not site_specific_run:
                    b = run_baseline_pair_for_seed(
                        env_config=env_config,
                        seed=per_episode_seed,
                        fixed_preference=float(weight[0]),
                        output_dir=str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir),
                    )
                    # Aggregate baseline results per episode for this weight
                    baseline_buffers['zero']['carbon'].append(float(b['zero_density']['carbon']))
                    baseline_buffers['zero']['thaw'].append(float(b['zero_density']['thaw']))
                    baseline_buffers['zero']['scalarized'].append(float(b['zero_density']['scalarized']))
                    baseline_buffers['plus']['carbon'].append(float(b['+100_density_0p5mix']['carbon']))
                    baseline_buffers['plus']['thaw'].append(float(b['+100_density_0p5mix']['thaw']))
                    baseline_buffers['plus']['scalarized'].append(float(b['+100_density_0p5mix']['scalarized']))
                        
            # Site-specific mode: run baseline once per weight (identical across episodes)
            if site_specific_run:
                b = run_baseline_pair_for_seed(
                    env_config=env_config,
                    seed=first_episode_seed,
                    fixed_preference=float(weight[0]),
                    output_dir=str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir),
                )
                baseline_buffers['zero']['carbon'].append(float(b['zero_density']['carbon']))
                baseline_buffers['zero']['thaw'].append(float(b['zero_density']['thaw']))
                baseline_buffers['zero']['scalarized'].append(float(b['zero_density']['scalarized']))
                baseline_buffers['plus']['carbon'].append(float(b['+100_density_0p5mix']['carbon']))
                baseline_buffers['plus']['thaw'].append(float(b['+100_density_0p5mix']['thaw']))
                baseline_buffers['plus']['scalarized'].append(float(b['+100_density_0p5mix']['scalarized']))

            # Record aggregated results for this weight if any episodes were completed
            if len(carbon_rewards) > 0:
                mean_carbon = np.mean(carbon_rewards)
                mean_thaw = np.mean(thaw_rewards)
                mean_scalarized = np.mean(scalarized_rewards)
                results['weights'].append(weight)
                results['carbon_objectives'].append(mean_carbon)
                results['thaw_objectives'].append(mean_thaw)
                results['scalarized_rewards'].append(mean_scalarized)
                # Compute averaged baseline metrics for this weight
                def _avg(vals: list[float]) -> float:
                    return float(np.mean(vals)) if len(vals) > 0 else 0.0
                baseline_means_per_weight.append({
                    'zero': {
                        'carbon': _avg(baseline_buffers['zero']['carbon']),
                        'thaw': _avg(baseline_buffers['zero']['thaw']),
                        'scalarized': _avg(baseline_buffers['zero']['scalarized']),
                    },
                    'plus': {
                        'carbon': _avg(baseline_buffers['plus']['carbon']),
                        'thaw': _avg(baseline_buffers['plus']['thaw']),
                        'scalarized': _avg(baseline_buffers['plus']['scalarized']),
                    },
                })
    finally:
        venv.close()

    # Save an evaluation summary CSV aggregated from eval episode CSV by weight
    output_dir = str(getattr(unwrapped_env, 'csv_output_dir', run_dir) or run_dir)
    os.makedirs(output_dir, exist_ok=True)
    eval_epi_path = os.path.join(output_dir, f"eval_episode_{run_id}.csv")
    eval_csv_path = os.path.join(output_dir, f"eval_summary_{run_id}.csv")
    rows_to_write: list[list[float]] = []
    if os.path.exists(eval_epi_path):
        try:
            import collections
            grouped = collections.defaultdict(lambda: {"carbon": [], "thaw": [], "scalarized": []})
            with open(eval_epi_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        w = float(row.get('preference_weight', row.get('weight_carbon', 0.0)))
                        c = float(row.get('total_carbon_reward', 0.0))
                        t = float(row.get('total_thaw_reward', 0.0))
                        s = float(row.get('total_scalarized_reward', 0.0))
                        grouped[w]["carbon"].append(c)
                        grouped[w]["thaw"].append(t)
                        grouped[w]["scalarized"].append(s)
                    except Exception:
                        continue
            for w, vals in sorted(grouped.items(), key=lambda kv: kv[0]):
                avg_c = float(np.mean(vals["carbon"])) if vals["carbon"] else 0.0
                avg_t = float(np.mean(vals["thaw"])) if vals["thaw"] else 0.0
                avg_s = float(np.mean(vals["scalarized"])) if vals["scalarized"] else 0.0
                rows_to_write.append([w, 1.0 - w, avg_c, avg_t, avg_s])
        except Exception:
            rows_to_write = []
    # Fallback to in-memory results if episode CSV not available
    if not rows_to_write and results['weights']:
        for w, c, t, s in zip(results['weights'], results['carbon_objectives'], results['thaw_objectives'], results['scalarized_rewards']):
            rows_to_write.append([float(w[0]), float(w[1]), float(c), float(t), float(s)])
    with open(eval_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["weight_carbon", "weight_thaw", "carbon_objective", "thaw_objective", "scalarized_reward"])
        for row in rows_to_write:
            writer.writerow(row)
    # Save a baseline summary CSV aggregated from baseline episode CSV by weight and baseline_type
    baseline_csv_path = os.path.join(output_dir, f"baseline_summary_{run_id}.csv")
    episode_baseline_path = os.path.join(output_dir, f"baseline_episode_{run_id}.csv")
    aggregated_rows: list[list[float]] = []
    header = [
        "weight_carbon", "weight_thaw", "baseline_type",
        "avg_carbon", "avg_thaw", "avg_scalarized",
        "count_episodes",
    ]
    if os.path.exists(episode_baseline_path):
        try:
            import collections
            grouped = collections.defaultdict(lambda: {"carbon": [], "thaw": [], "scalarized": []})
            with open(episode_baseline_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        w = float(row.get('preference_weight', 0.0))
                        bt = str(row.get('baseline_type', '') or '')
                        # Use totals at episode level
                        c = float(row.get('total_carbon_reward', 0.0))
                        t = float(row.get('total_thaw_reward', 0.0))
                        s = float(row.get('total_scalarized_reward', 0.0))
                        key = (w, bt)
                        grouped[key]["carbon"].append(c)
                        grouped[key]["thaw"].append(t)
                        grouped[key]["scalarized"].append(s)
                    except Exception:
                        continue
            for (w, bt), vals in grouped.items():
                avg_c = float(np.mean(vals["carbon"])) if vals["carbon"] else 0.0
                avg_t = float(np.mean(vals["thaw"])) if vals["thaw"] else 0.0
                avg_s = float(np.mean(vals["scalarized"])) if vals["scalarized"] else 0.0
                count = int(len(vals["carbon"]))
                aggregated_rows.append([w, 1.0 - w, bt, avg_c, avg_t, avg_s, count])
        except Exception:
            pass
    with open(baseline_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in aggregated_rows:
            writer.writerow(row)

    # Wandb disabled for evaluation - no cleanup needed

    return results


