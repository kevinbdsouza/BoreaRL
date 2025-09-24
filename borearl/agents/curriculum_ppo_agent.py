# agents/curriculum_ppo_agent.py
from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from borearl import constants as const
from .ppo_gated import build_plant_masks, ActorCriticGated, RolloutBuffer, RolloutBatch

# =========================
# 1) Site Selection Head
# =========================
class SiteSelectionHead(nn.Module):
    """
    Neural network head that decides whether to select a site for training
    based on initial episode observations.
    """
    def __init__(self, site_obs_dim: int):
        super().__init__()
        self.site_evaluator = nn.Sequential(
            nn.Linear(site_obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # Site selection logits
        )
    
    def forward(self, site_obs: torch.Tensor) -> torch.Tensor:
        """Returns probability of selecting the site"""
        logits = self.site_evaluator(site_obs)
        return torch.sigmoid(logits)


# =========================
# 2) Vanilla Actor-Critic (for fallback)
# =========================
class VanillaAC(nn.Module):
    """Vanilla Actor-Critic network (fallback when plant gate is disabled)"""
    def __init__(self, obs_dim: int, act_dim: int, net_arch: Optional[list[int]] = None):
        super().__init__()
        hidden = []
        last = obs_dim
        for k in (net_arch or [64, 64]):
            hidden += [nn.Linear(last, k), nn.Tanh()]
            last = k
        
        self.body = nn.Sequential(*hidden)
        self.pi = nn.Linear(last, act_dim)
        self.v = nn.Linear(last, 1)
    
    def act(self, obs):
        h = self.body(obs)
        logits = self.pi(h)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = self.v(h).squeeze(-1)
        return a, logp, v, {}
        
    def evaluate_actions(self, obs, actions):
        h = self.body(obs)
        logits = self.pi(h)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        v = self.v(h).squeeze(-1)
        return v, logp, entropy, {}


# =========================
# 3) Site Feature Extractor
# =========================
def extract_site_features(obs: torch.Tensor, unwrapped_env, target_dim: int = 20) -> torch.Tensor:
    """
    Extract site-level features from the initial observation.
    These are features that characterize the site's potential for thaw optimization.
    """
    # Get observation structure from environment
    # Assume obs is flattened - we need to know the structure to extract features
    
    # For now, extract key features we know are important for thaw potential
    # This will need to be adapted based on the actual observation structure
    
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    
    # Extract features based on observation indices
    # Note: These indices will need to be determined from the actual env obs structure
    site_features = []
    
    try:
        # Climate features (if available in obs)
        if hasattr(unwrapped_env, 'last_latitude_deg'):
            site_features.append(torch.tensor([unwrapped_env.last_latitude_deg], device=obs.device))
        if hasattr(unwrapped_env, 'last_mean_temp_c'):
            site_features.append(torch.tensor([unwrapped_env.last_mean_temp_c], device=obs.device))
        if hasattr(unwrapped_env, 'last_temp_amplitude_c'):
            site_features.append(torch.tensor([unwrapped_env.last_temp_amplitude_c], device=obs.device))
        if hasattr(unwrapped_env, 'last_growing_season_days'):
            site_features.append(torch.tensor([unwrapped_env.last_growing_season_days], device=obs.device))
            
        # Initial state features (from observation)
        # These indices will need to be determined from actual obs structure
        if len(obs.shape) > 1:
            batch_size = obs.shape[0]
        else:
            batch_size = 1
            obs = obs.unsqueeze(0)
            
        # Use a subset of the observation as site features, ensuring target_dim size
        obs_subset_size = min(target_dim, obs.shape[1])
        obs_subset = obs[:, :obs_subset_size] if obs.shape[1] >= obs_subset_size else obs
        
        if site_features:
            # Stack additional features if we have them
            additional_features = torch.stack(site_features, dim=1).expand(batch_size, -1)
            site_obs = torch.cat([obs_subset, additional_features], dim=1)
        else:
            site_obs = obs_subset
            
        # Ensure we have exactly target_dim features by padding or truncating
        if site_obs.shape[1] < target_dim:
            # Pad with zeros
            padding = torch.zeros(batch_size, target_dim - site_obs.shape[1], device=obs.device)
            site_obs = torch.cat([site_obs, padding], dim=1)
        elif site_obs.shape[1] > target_dim:
            # Truncate
            site_obs = site_obs[:, :target_dim]
            
    except Exception:
        # Fallback: create features with the right size
        site_obs = obs[:, :min(target_dim, obs.shape[1])] if obs.shape[1] >= min(target_dim, obs.shape[1]) else obs
        if site_obs.shape[1] < target_dim:
            padding = torch.zeros(batch_size, target_dim - site_obs.shape[1], device=obs.device)
            site_obs = torch.cat([site_obs, padding], dim=1)
        elif site_obs.shape[1] > target_dim:
            site_obs = site_obs[:, :target_dim]
    
    return site_obs


# =========================
# 4) Curriculum PPO Agent
# =========================
class CurriculumPPO(nn.Module):
    """
    Curriculum PPO agent with adaptive episode selection for curriculum learning.
    Uses a two-level decision process:
    1. Episode Selection: Decide whether to train on this episode (curriculum learning)
    2. Action Selection: If selected, use gated PPO for actions
    """
    
    def __init__(
        self,
        env,
        unwrapped_env,
        net_arch: Optional[list[int]] = None,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 2048,
        batch_size: int = 64,
        update_epochs: int = 10,
        use_plant_gate: bool = True,
        curriculum_threshold: float = 0.5,
        curriculum_lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.env = env
        self.unwrapped_env = unwrapped_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.use_plant_gate = use_plant_gate
        self.curriculum_threshold = curriculum_threshold
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.n
        
        # Curriculum episode selection network
        # Store the actual site obs dim for consistent feature extraction
        self.site_obs_dim = min(obs_dim, 20)  # Use subset of obs for site features
        self.episode_selector = SiteSelectionHead(self.site_obs_dim).to(self.device)
        # Note: episode_selector is used for inference only, not trained
        
        # Action selection policy (same as PPO gated)
        if use_plant_gate:
            PLANT_MASK, NOPLANT_MASK = build_plant_masks(unwrapped_env)
            self.action_policy = ActorCriticGated(obs_dim, act_dim, net_arch, PLANT_MASK, NOPLANT_MASK)
        else:
            self.action_policy = self._build_vanilla_ac(obs_dim, act_dim, net_arch)
            
        self.action_policy = self.action_policy.to(self.device)
        self.action_optimizer = torch.optim.Adam(self.action_policy.parameters(), lr=learning_rate)
        
        # Tracking statistics
        self.curriculum_stats = {
            'episodes_selected': 0,
            'episodes_skipped': 0,
            'total_episodes': 0,
            'selected_thaw_rewards': [],
            'skipped_thaw_rewards': [],
            'selection_accuracy': []
        }
        
        # Environment state
        self.n_envs = getattr(env, "num_envs", 1)
        self.obs = None
        self.current_episode_selected = True
        
    def _build_vanilla_ac(self, obs_dim, act_dim, net_arch):
        """Build vanilla actor-critic (fallback when plant gate is disabled)"""
        return VanillaAC(obs_dim, act_dim, net_arch)
    
    def should_select_episode(self, initial_obs: torch.Tensor) -> Tuple[bool, float]:
        """
        Decide whether to select this episode for training (curriculum learning).
        Returns (select, selection_probability)
        """
        with torch.no_grad():
            site_obs = extract_site_features(initial_obs, self.unwrapped_env, self.site_obs_dim)
            selection_prob = self.episode_selector(site_obs).item()
            select = selection_prob > self.curriculum_threshold
            return select, selection_prob
    
    def train(self, total_timesteps: int):
        """
        Training loop with curriculum episode selection + action selection.
        """
        device = self.device
        buffer = RolloutBuffer(
            int(np.prod(self.env.observation_space.shape)), 
            self.rollout_steps, 
            device
        )
        
        global_step = 0
        episode_count = 0
        
        # Reset environment
        obs, info = self.env.reset()
        self.obs = self._to_tensor(obs, device).view(self.n_envs, -1)
        
        while global_step < total_timesteps:
            # === ROLLOUT PHASE ===
            buffer.reset()
            
            for step in range(self.rollout_steps):
                if global_step >= total_timesteps:
                    break
                
                # Check if this is the start of a new episode
                if step == 0 or info.get('terminated', False) or info.get('truncated', False):
                    # Site selection decision
                    select_episode, selection_prob = self.should_select_episode(self.obs)
                    self.current_episode_selected = select_episode
                    episode_count += 1
                    self.curriculum_stats['total_episodes'] += 1
                    
                    if select_episode:
                        self.curriculum_stats['episodes_selected'] += 1
                    else:
                        self.curriculum_stats['episodes_skipped'] += 1
                
                if not self.current_episode_selected:
                    # Skip this episode - advance environment without learning
                    with torch.no_grad():
                        # Take random action to advance environment
                        random_action = torch.randint(0, self.env.action_space.n, (self.n_envs,), device=device)
                        o, r, terminated, truncated, info = self.env.step(random_action.cpu().numpy())
                        
                        if terminated or truncated:
                            # Episode ended, track skipped episode rewards
                            if 'final_info' in info and info['final_info'] is not None:
                                episode_info = info['final_info'][0] if isinstance(info['final_info'], list) else info['final_info']
                                if 'avg_thaw_reward' in episode_info:
                                    self.curriculum_stats['skipped_thaw_rewards'].append(episode_info['avg_thaw_reward'])
                            
                            obs, info = self.env.reset()
                        else:
                            obs = o
                        
                        self.obs = self._to_tensor(obs, device).view(self.n_envs, -1)
                        global_step += 1  # Increment step counter even for skipped episodes
                        continue
                
                # Normal training step for selected episodes
                with torch.no_grad():
                    if hasattr(self.action_policy, 'act'):
                        a, logp, v, aux = self.action_policy.act(self.obs)
                    else:
                        # Fallback for vanilla AC
                        a, logp, v, aux = self.action_policy.act(self.obs)
                
                # Environment step
                o, r, terminated, truncated, info = self.env.step(a.cpu().numpy())
                
                # Store in buffer
                if isinstance(r, (list, np.ndarray)) and len(r) > 1:
                    # Multi-objective reward - use scalarized version
                    pref_weight = getattr(self.unwrapped_env, 'current_preference_weight', 0.5)
                    reward_scalar = pref_weight * r[0] + (1.0 - pref_weight) * r[1]
                else:
                    reward_scalar = float(r[0] if isinstance(r, (list, np.ndarray)) else r)
                
                buffer.store(
                    self.obs.clone().flatten(),
                    a,
                    logp,
                    torch.tensor(reward_scalar, dtype=torch.float32, device=device),
                    torch.tensor(terminated or truncated, dtype=torch.float32, device=device),
                    v
                )
                
                # Track selected episode rewards
                if terminated or truncated:
                    if 'final_info' in info and info['final_info'] is not None:
                        episode_info = info['final_info'][0] if isinstance(info['final_info'], list) else info['final_info']
                        if 'avg_thaw_reward' in episode_info:
                            self.curriculum_stats['selected_thaw_rewards'].append(episode_info['avg_thaw_reward'])
                    
                    obs, info = self.env.reset()
                else:
                    obs = o
                
                self.obs = self._to_tensor(obs, device).view(self.n_envs, -1)
                global_step += 1
                
                if buffer.ready():
                    break
            
            # If we've reached the timestep limit, break out of training
            if global_step >= total_timesteps:
                break
                
            # If buffer is not ready and we haven't reached the limit, continue collecting
            if not buffer.ready():
                continue
                
            # === TRAINING PHASE ===
            # Bootstrap last value
            with torch.no_grad():
                if hasattr(self.action_policy, 'act'):
                    _, _, last_v, _ = self.action_policy.act(self.obs)
                else:
                    _, _, last_v, _ = self.action_policy.act(self.obs)
            
            batch = buffer.compute_returns_advantages(last_v.detach(), self.gamma, self.gae_lambda)
            
            # Update action policy (same as regular PPO)
            self._update_action_policy(batch)
            
            # Update curriculum policy
            self._update_curriculum_policy()
            
            # Print progress
            if episode_count % 100 == 0:
                self._print_progress(episode_count, global_step)
        
        return self
    
    def _update_action_policy(self, batch):
        """Update the action selection policy (same as PPO)"""
        n = self.rollout_steps
        idx = torch.randperm(n, device=self.device)
        
        for _ in range(self.update_epochs):
            for start in range(0, n, self.batch_size):
                mb_idx = idx[start:start + self.batch_size]
                
                obs_b = batch.obs[mb_idx]
                act_b = batch.actions[mb_idx]
                old_logp_b = batch.logp[mb_idx]
                ret_b = batch.returns[mb_idx]
                adv_b = batch.advantages[mb_idx]
                
                if hasattr(self.action_policy, 'evaluate_actions'):
                    v_pred, logp_b, ent_b, aux = self.action_policy.evaluate_actions(obs_b, act_b)
                else:
                    v_pred, logp_b, ent_b, aux = self.action_policy.evaluate_actions(obs_b, act_b)
                
                # Policy loss (clipped)
                ratio = torch.exp(logp_b - old_logp_b)
                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_b
                pg_loss = -torch.min(unclipped, clipped).mean()
                
                # Value loss (clipped)
                v_clipped = batch.values[mb_idx] + (v_pred - batch.values[mb_idx]).clamp(-self.clip_coef, self.clip_coef)
                v_loss_unclipped = (v_pred - ret_b).pow(2)
                v_loss_clipped = (v_clipped - ret_b).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                
                # Entropy bonus
                entropy_loss = ent_b.mean()
                if aux and "gate_entropy" in aux:
                    entropy_loss = entropy_loss + aux["gate_entropy"].mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                self.action_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.action_policy.parameters(), self.max_grad_norm)
                self.action_optimizer.step()
    
    def _update_curriculum_policy(self):
        """Update curriculum policy based on outcomes"""
        if len(self.curriculum_stats['selected_thaw_rewards']) < 10:
            return  # Need some data first
        
        # Simple reward-based update: adjust curriculum threshold based on performance
        selected_rewards = np.array(self.curriculum_stats['selected_thaw_rewards'][-50:])  # Last 50
        skipped_rewards = np.array(self.curriculum_stats['skipped_thaw_rewards'][-50:]) if self.curriculum_stats['skipped_thaw_rewards'] else np.array([])
        
        if len(selected_rewards) > 0 and len(skipped_rewards) > 0:
            selection_quality = np.mean(selected_rewards) - np.mean(skipped_rewards)
            
            # If selected episodes perform better, become more selective (curriculum learning)
            # If selected episodes perform worse, become less selective
            if selection_quality > 0:
                self.curriculum_threshold *= 0.999  # Be slightly more selective
            else:
                self.curriculum_threshold *= 1.001  # Be slightly less selective
            
            # Keep threshold in reasonable bounds
            self.curriculum_threshold = np.clip(self.curriculum_threshold, 0.1, 0.9)
    
    def _print_progress(self, episode_count, global_step):
        """Print training progress including curriculum stats"""
        total_eps = self.curriculum_stats['total_episodes']
        selected_eps = self.curriculum_stats['episodes_selected']
        skipped_eps = self.curriculum_stats['episodes_skipped']
        
        selection_rate = selected_eps / max(total_eps, 1)
        
        selected_rewards = self.curriculum_stats['selected_thaw_rewards']
        skipped_rewards = self.curriculum_stats['skipped_thaw_rewards']
        
        avg_selected_thaw = np.mean(selected_rewards[-50:]) if len(selected_rewards) >= 50 else (np.mean(selected_rewards) if selected_rewards else 0.0)
        avg_skipped_thaw = np.mean(skipped_rewards[-50:]) if len(skipped_rewards) >= 50 else (np.mean(skipped_rewards) if skipped_rewards else 0.0)
        
        print(f"Episode {episode_count}, Step {global_step}")
        print(f"  Curriculum Selection Rate: {selection_rate:.3f} (threshold: {self.curriculum_threshold:.3f})")
        print(f"  Selected Episodes Thaw Reward: {avg_selected_thaw:.4f}")
        print(f"  Skipped Episodes Thaw Reward: {avg_skipped_thaw:.4f}")
        print(f"  Curriculum Advantage: {avg_selected_thaw - avg_skipped_thaw:.4f}")
    
    def _to_tensor(self, x, device):
        """Convert input to tensor"""
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)
    
    def save(self, path: str):
        """Save the model state"""
        state_dict = {
            'episode_selector': self.episode_selector.state_dict(),
            'action_policy': self.action_policy.state_dict(),
            'action_optimizer': self.action_optimizer.state_dict(),
            'curriculum_stats': self.curriculum_stats,
            'curriculum_threshold': self.curriculum_threshold,
            'site_obs_dim': self.site_obs_dim,
        }
        torch.save(state_dict, path)
    
    def load(self, path: str):
        """Load the model state"""
        try:
            loaded_obj = torch.load(path, map_location=self.device, weights_only=False)
            
            # Check if this is a curriculum PPO object, state dict, or something else
            if isinstance(loaded_obj, CurriculumPPO):
                # The entire model was saved - copy its state
                print("Loading from saved CurriculumPPO object")
                self.episode_selector.load_state_dict(loaded_obj.episode_selector.state_dict())
                
                # Check if the saved action policy type matches current type
                saved_policy = loaded_obj.action_policy
                current_policy = self.action_policy
                
                # Check policy compatibility by examining state dict keys
                saved_state_dict = saved_policy.state_dict()
                saved_keys = set(saved_state_dict.keys())
                
                # VanillaAC has keys like: body.0.weight, body.0.bias, pi.weight, pi.bias, v.weight, v.bias
                # ActorCriticGated has keys like: body.0.weight, body.0.bias, gated_head.gate.weight, etc.
                
                has_gated_keys = any('gated_head' in key for key in saved_keys)
                has_vanilla_keys = any(key.startswith('pi.') for key in saved_keys)
                
                if has_vanilla_keys and not has_gated_keys:
                    # Saved model uses VanillaAC, but current model expects ActorCriticGated
                    print("Detected VanillaAC policy in saved model, but current model expects ActorCriticGated")
                    print("Recreating action policy as VanillaAC for compatibility")
                    
                    # Recreate as VanillaAC
                    obs_dim = int(np.prod(self.env.observation_space.shape))
                    act_dim = self.env.action_space.n
                    net_arch = [128, 64]  # Use saved net_arch if available
                    self.action_policy = self._build_vanilla_ac(obs_dim, act_dim, net_arch).to(self.device)
                    self.action_optimizer = torch.optim.Adam(self.action_policy.parameters(), lr=3e-4)
                    self.use_plant_gate = False  # Update flag to match loaded model
                
                elif has_gated_keys and not has_vanilla_keys:
                    # Saved model uses ActorCriticGated
                    print("Detected ActorCriticGated policy in saved model")
                
                else:
                    print(f"Warning: Could not determine policy type from keys: {list(saved_keys)[:10]}...")
                
                # Now load the state dict
                try:
                    self.action_policy.load_state_dict(saved_state_dict)
                    print("Successfully loaded action policy state")
                except Exception as policy_load_error:
                    print(f"Failed to load action policy: {policy_load_error}")
                    # Try to copy the saved policy directly
                    self.action_policy = saved_policy.to(self.device)
                    self.action_optimizer = torch.optim.Adam(self.action_policy.parameters(), lr=3e-4)
                    print("Used saved policy directly")
                
                self.curriculum_stats = loaded_obj.curriculum_stats
                self.curriculum_threshold = loaded_obj.curriculum_threshold
                self.site_obs_dim = loaded_obj.site_obs_dim
                print("Successfully loaded CurriculumPPO model state")
                
            elif isinstance(loaded_obj, dict):
                # Check for expected keys
                if 'episode_selector' in loaded_obj and 'action_policy' in loaded_obj:
                    # This is a proper curriculum PPO state dict save
                    print("Loading from state dictionary")
                    self.episode_selector.load_state_dict(loaded_obj['episode_selector'])
                    self.action_policy.load_state_dict(loaded_obj['action_policy'])
                    if 'action_optimizer' in loaded_obj:
                        self.action_optimizer.load_state_dict(loaded_obj['action_optimizer'])
                    if 'curriculum_stats' in loaded_obj:
                        self.curriculum_stats = loaded_obj['curriculum_stats']
                    if 'curriculum_threshold' in loaded_obj:
                        self.curriculum_threshold = loaded_obj['curriculum_threshold']
                    if 'site_obs_dim' in loaded_obj:
                        self.site_obs_dim = loaded_obj['site_obs_dim']
                else:
                    # This might be just the action policy state dict
                    print(f"Warning: Loading state dict with keys: {list(loaded_obj.keys())}")
                    # Try to load it as action policy only
                    self.action_policy.load_state_dict(loaded_obj)
            else:
                # If loaded_obj is not a dict or CurriculumPPO, it might be something else
                raise ValueError(f"Unexpected loaded object type: {type(loaded_obj)}")
                
        except Exception as e:
            print(f"Error loading curriculum PPO model: {e}")
            print(f"Loaded object type: {type(loaded_obj) if 'loaded_obj' in locals() else 'unknown'}")
            if 'loaded_obj' in locals() and isinstance(loaded_obj, dict):
                print(f"State dict keys: {list(loaded_obj.keys())}")
            raise
    
    def predict(self, obs, deterministic: bool = False):
        """Predict action for evaluation - always use action policy without site selection"""
        with torch.no_grad():
            obs_tensor = self._to_tensor(obs, self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Always use the action policy (no site selection during evaluation)
            if hasattr(self.action_policy, 'act'):
                a, logp, v, aux = self.action_policy.act(obs_tensor)
            else:
                # Fallback for vanilla AC
                a, logp, v, aux = self.action_policy.act(obs_tensor)
            
            if deterministic:
                # For deterministic evaluation, use the mode of the distribution
                with torch.no_grad():
                    if hasattr(self.action_policy, 'evaluate_actions'):
                        v_pred, logp_b, ent_b, aux = self.action_policy.evaluate_actions(obs_tensor, a)
                    else:
                        v_pred, logp_b, ent_b, aux = self.action_policy.evaluate_actions(obs_tensor, a)
                    
                    # Get logits and take argmax for deterministic action
                    h = self.action_policy.body(obs_tensor) if hasattr(self.action_policy, 'body') else obs_tensor
                    if hasattr(self.action_policy, 'pi'):
                        logits = self.action_policy.pi(h)
                    else:
                        # For gated policy, need to handle differently
                        logits = None
                        # Just use the sampled action for now
                    
                    if logits is not None:
                        a = torch.argmax(logits, dim=1)
            
            return a.cpu().numpy()
    
    def act(self, obs, acc_reward=None, **kwargs):
        """Act method for evaluation compatibility"""
        action = self.predict(obs, deterministic=False)
        return action[0] if isinstance(action, np.ndarray) and action.size > 0 else action
    
    def get_policy_net(self):
        """Return the action policy for compatibility with evaluation system"""
        return self.action_policy


# =========================
# 5) Module-level helpers
# =========================
def create(
    env,
    unwrapped_env,
    use_wandb: bool,
    *,
    weights=None,
    gamma=None,
    learning_rate=None,
    net_arch=None,
    run_dir_name=None,
    use_plant_gate: bool = True,
    curriculum_threshold: float = 0.5,
):
    """Create curriculum PPO agent"""
    agent = CurriculumPPO(
        env=env,
        unwrapped_env=unwrapped_env,
        net_arch=net_arch,
        gamma=float(gamma) if gamma is not None else 0.99,
        learning_rate=float(learning_rate) if learning_rate is not None else 3e-4,
        use_plant_gate=bool(use_plant_gate),
        curriculum_threshold=float(curriculum_threshold),
    )
    return agent

def default_model_filename() -> str:
    return "curriculum_ppo_forest_manager.pth"

def supports_single_policy_eval() -> bool:
    """Curriculum PPO supports single policy evaluation"""
    return True

def load_policy_set(model, model_path: str):
    """Load the curriculum PPO model from saved state"""
    try:
        model.load(model_path)
        return model
    except Exception as e:
        print(f"Failed to load curriculum PPO model from {model_path}: {e}")
        return None
