# agents/ppo_gated.py
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

# =========================
# 1) Masks from your env
# =========================
def build_plant_masks(unwrapped_env) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Returns (PLANT_MASK, NOPLANT_MASK) over the *flat* action set [A].
    By convention 'plant' means actions whose density delta > 0.
    """
    A = unwrapped_env.action_space.n
    # infer factorization a = density_idx * K + species_idx (if applicable)
    K = getattr(unwrapped_env, 'num_species_levels', None)
    if K is None:
        # fallback: assume species grid from env constants if present
        K = len(getattr(unwrapped_env, 'CONIFER_FRACTIONS', [0.0, 0.25, 0.5, 0.75, 1.0]))
    densities = getattr(unwrapped_env, 'DENSITY_ACTIONS', None)
    if densities is None:
        # fallback to constants if your project exposes them
        densities = const.DENSITY_ACTIONS  # e.g. [-100, 0, +100]

    plant = np.zeros(A, dtype=bool)
    for a in range(A):
        d_idx = a // K
        delta_density = float(densities[d_idx])
        plant[a] = (delta_density > 0)

    PLANT_MASK = torch.tensor(plant, dtype=torch.bool)
    NOPLANT_MASK = ~PLANT_MASK
    assert PLANT_MASK.any() and NOPLANT_MASK.any(), "Masks degenerate; check action encoding."
    return PLANT_MASK, NOPLANT_MASK


# =========================
# 2) Gated action head
# =========================
def _masked_entropy_from_logits(masked_logits: torch.Tensor) -> torch.Tensor:
    # masked_logits already include -inf on disallowed actions in each branch
    probs = torch.softmax(masked_logits, dim=-1)
    logp = torch.log_softmax(masked_logits, dim=-1)
    return -(probs * logp).sum(-1)

class GatedActionHead(nn.Module):
    """
    Produces the *marginal* logits over the original action set by
    log-sum-exp over the two branches z∈{no-plant, plant}.
    """
    def __init__(self, in_dim: int, n_actions: int,
                 plant_mask: torch.BoolTensor, noplant_mask: torch.BoolTensor):
        super().__init__()
        self.n_actions = int(n_actions)
        self.register_buffer("plant_mask", plant_mask, persistent=False)
        self.register_buffer("noplant_mask", noplant_mask, persistent=False)
        self.gate = nn.Linear(in_dim, 1)            # Bernoulli logits g(s)
        self.cond_plant = nn.Linear(in_dim, n_actions)
        self.cond_noplant = nn.Linear(in_dim, n_actions)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g = self.gate(h).squeeze(-1)                    # [B]
        log_p1 = -F.softplus(-g)                        # log σ(g)
        log_p0 = -F.softplus( g)                        # log σ(-g)

        lp = self.cond_plant(h)                         # [B, A]
        ln = self.cond_noplant(h)                       # [B, A]

        neg_inf = torch.finfo(lp.dtype).min
        lp = lp.masked_fill(~self.plant_mask, neg_inf)
        ln = ln.masked_fill(~self.noplant_mask, neg_inf)

        logits_joint_plant = lp + log_p1.unsqueeze(-1)  # [B, A]
        logits_joint_nopl  = ln + log_p0.unsqueeze(-1)  # [B, A]
        logits_marginal = torch.logaddexp(logits_joint_plant, logits_joint_nopl)

        p1 = torch.sigmoid(g)
        aux = {
            "p_plant": p1,                                   # [B]
            "gate_entropy": -(p1*torch.log(p1.clamp_min(1e-8))
                              + (1-p1)*torch.log((1-p1).clamp_min(1e-8))),
            "cond_entropy_plant": _masked_entropy_from_logits(lp),
            "cond_entropy_noplant": _masked_entropy_from_logits(ln),
        }
        return logits_marginal, aux


# =========================
# 3) Actor-Critic policy
# =========================
class ActorCriticGated(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, net_arch: Optional[list[int]],
                 plant_mask: torch.BoolTensor, noplant_mask: torch.BoolTensor):
        super().__init__()
        hidden = []
        last = obs_dim
        for k in (net_arch or [64, 64]):
            hidden += [nn.Linear(last, k), nn.Tanh()]
            last = k
        self.body = nn.Sequential(*hidden)
        self.gated_head = GatedActionHead(last, act_dim, plant_mask, noplant_mask)
        self.v = nn.Linear(last, 1)

    def distribution(self, obs: torch.Tensor):
        h = self.body(obs)
        logits, aux = self.gated_head(h)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, aux, h

    def act(self, obs: torch.Tensor):
        dist, aux, h = self.distribution(obs)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = self.v(h).squeeze(-1)
        return a, logp, v, aux

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, aux, h = self.distribution(obs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        v = self.v(h).squeeze(-1)
        return v, logp, entropy, aux


# =========================
# 4) Rollout buffer
# =========================
@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor

class RolloutBuffer:
    def __init__(self, obs_dim: int, size: int, device: torch.device):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size,), dtype=torch.long, device=device)
        self.logp = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((size,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.max_size = size
        self.device = device

    def store(self, obs, action, logp, reward, done, value):
        i = self.ptr
        self.obs[i].copy_(obs)
        self.actions[i] = action
        self.logp[i] = logp
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.ptr += 1

    def ready(self) -> bool:
        return self.ptr >= self.max_size

    def compute_returns_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        adv = torch.zeros_like(self.rewards, device=self.device)
        last_adv = 0.0
        for t in reversed(range(self.max_size)):
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * last_value * mask - self.values[t]
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            adv[t] = last_adv
            last_value = self.values[t]
        returns = adv + self.values
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = RolloutBatch(
            obs=self.obs, actions=self.actions, logp=self.logp,
            returns=returns, advantages=adv, values=self.values
        )
        return batch


# =========================
# 5) PPO Agent
# =========================
class PPO(nn.Module):
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
        add_gate_entropy_bonus: bool = True,
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
        self.add_gate_entropy_bonus = add_gate_entropy_bonus
        self.use_plant_gate = use_plant_gate

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = env.action_space.n

        if use_plant_gate:
            PLANT_MASK, NOPLANT_MASK = build_plant_masks(unwrapped_env)
            policy = ActorCriticGated(obs_dim, act_dim, net_arch, PLANT_MASK, NOPLANT_MASK)
        else:
            # Simple non-gated actor-critic for comparison
            policy = self._build_vanilla_ac(obs_dim, act_dim, net_arch)

        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # vectorized env support (n_envs=1 is fine)
        self.n_envs = getattr(env, "num_envs", 1)
        self.obs = None  # filled on first reset

    def _build_vanilla_ac(self, obs_dim, act_dim, net_arch):
        hidden = []
        last = obs_dim
        for k in (net_arch or [64, 64]):
            hidden += [nn.Linear(last, k), nn.Tanh()]
            last = k
        body = nn.Sequential(*hidden)
        pi = nn.Linear(last, act_dim)
        v = nn.Linear(last, 1)

        class AC(nn.Module):
            def __init__(self, body, pi, v):
                super().__init__()
                self.body, self.pi, self.v = body, pi, v
            def distribution(self, obs):
                h = self.body(obs)
                return torch.distributions.Categorical(logits=self.pi(h)), {}, h
            def act(self, obs):
                dist, _, h = self.distribution(obs)
                a = dist.sample()
                return a, dist.log_prob(a), self.v(h).squeeze(-1), {}
            def evaluate_actions(self, obs, actions):
                dist, _, h = self.distribution(obs)
                return self.v(h).squeeze(-1), dist.log_prob(actions), dist.entropy(), {}
        return AC(body, pi, v)

    # ---- helpers ----
    @staticmethod
    def _to_tensor(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x).to(device)

    def _scalarize_rewards(self, reward_vec, lambda_weight: Optional[float]) -> float:
        # reward_vec shape could be (2,) or nested when vectorized env
        r_c, r_t = float(reward_vec[0]), float(reward_vec[1])
        lam = float(lambda_weight) if lambda_weight is not None else 0.5
        return lam * r_c + (1.0 - lam) * r_t

    def _current_lambda(self) -> Optional[float]:
        # try vector env API first
        try:
            if hasattr(self.env, "get_attr"):
                vals = self.env.get_attr("current_preference_weight")
                if vals and isinstance(vals, (list, tuple)) and len(vals) > 0:
                    return float(vals[0])
        except Exception:
            pass
        # fall back to unwrapped single env
        if hasattr(self.unwrapped_env, "current_preference_weight"):
            return float(getattr(self.unwrapped_env, "current_preference_weight"))
        return None

    # ---- public API ----
    def get_policy_net(self):
        return self.policy

    def train(self, total_timesteps: int):
        device = self.device
        # initial reset
        if self.obs is None:
            obs, _ = self.env.reset()
            self.obs = self._to_tensor(obs, device).view(self.n_envs, -1)

        steps_collected = 0
        obs_dim = self.obs.shape[-1]
        buffer = RolloutBuffer(obs_dim, self.rollout_steps, device)

        while steps_collected < total_timesteps:
            # ===== collect rollouts =====
            buffer.ptr = 0
            for _ in range(self.rollout_steps):
                with torch.no_grad():
                    a, logp, v, aux = self.policy.act(self.obs)
                # env expects numpy int actions, handle vectorized vs single
                a_np = a.detach().cpu().numpy()
                if self.n_envs == 1:
                    act_to_env = int(a_np.item())
                else:
                    act_to_env = a_np

                next_obs, reward_vec, terminated, truncated, info = self.env.step(act_to_env)
                done_flag = (terminated if isinstance(terminated, (bool, np.bool_))
                             else bool(terminated[0])) or (truncated if isinstance(truncated, (bool, np.bool_))
                             else bool(truncated[0]))

                lam = self._current_lambda()
                if self.n_envs > 1:
                    # assume same λ across envs; take first
                    lam = lam if lam is not None else 0.5
                    r_scalar = [self._scalarize_rewards(rv, lam) for rv in reward_vec]
                    r_scalar = float(np.asarray(r_scalar, dtype=np.float32).mean())
                else:
                    r_scalar = self._scalarize_rewards(reward_vec, lam)

                # store
                buffer.store(self.obs.view(-1), a.view(-1), logp.view(-1),
                             torch.tensor(r_scalar, device=device),
                             float(done_flag),
                             v.view(-1))

                # advance
                self.obs = self._to_tensor(next_obs, device).view(self.n_envs, -1)
                steps_collected += 1
                if done_flag:
                    # reset (keep λ sampling logic inside env)
                    o, _ = self.env.reset()
                    self.obs = self._to_tensor(o, device).view(self.n_envs, -1)

                if buffer.ready():
                    break

            # bootstrap last value
            with torch.no_grad():
                _, _, last_v, _ = self.policy.act(self.obs)
            batch = buffer.compute_returns_advantages(last_v.detach(), self.gamma, self.gae_lambda)

            # ===== PPO update =====
            n = self.rollout_steps
            idx = torch.randperm(n, device=device)
            for _ in range(self.update_epochs):
                for start in range(0, n, self.batch_size):
                    mb_idx = idx[start:start + self.batch_size]

                    obs_b = batch.obs[mb_idx]
                    act_b = batch.actions[mb_idx]
                    old_logp_b = batch.logp[mb_idx]
                    ret_b = batch.returns[mb_idx]
                    adv_b = batch.advantages[mb_idx]

                    v_pred, logp_b, ent_b, aux = self.policy.evaluate_actions(obs_b, act_b)

                    # policy loss (clipped)
                    ratio = torch.exp(logp_b - old_logp_b)
                    unclipped = ratio * adv_b
                    clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_b
                    pg_loss = -torch.min(unclipped, clipped).mean()

                    # value loss (clipped)
                    v_clipped = batch.values[mb_idx] + (v_pred - batch.values[mb_idx]).clamp(-self.clip_coef, self.clip_coef)
                    v_loss_unclipped = (v_pred - ret_b).pow(2)
                    v_loss_clipped = (v_clipped - ret_b).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # entropy bonus
                    entropy_loss = ent_b.mean()
                    if self.add_gate_entropy_bonus and "gate_entropy" in aux:
                        entropy_loss = entropy_loss + aux["gate_entropy"].mean()

                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

        return self

# =========================
# 6) Module-level helpers
# =========================
def create(
    env,
    unwrapped_env,
    use_wandb: bool,
    *,
    weights=None,             # ignored for PPO; λ comes from the env
    gamma=None,
    learning_rate=None,
    net_arch=None,
    run_dir_name=None,
    use_plant_gate: bool = True,
):
    agent = PPO(
        env=env,
        unwrapped_env=unwrapped_env,
        net_arch=net_arch,
        gamma=float(gamma) if gamma is not None else 0.99,
        learning_rate=float(learning_rate) if learning_rate is not None else 3e-4,
        use_plant_gate=bool(use_plant_gate),
    )
    return agent

def default_model_filename() -> str:
    return "ppo_gated_forest_manager.pth"

def supports_single_policy_eval() -> bool:
    return True

def load_policy_set(model, path: str):
    """For compatibility with your loader hooks; load state_dict into policy."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"PPO model not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.get_policy_net().load_state_dict(state)
    return model
