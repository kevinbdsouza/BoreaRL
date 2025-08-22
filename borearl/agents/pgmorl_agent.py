# agents/pgmorl_agent.py
from __future__ import annotations

import os
import inspect
import numpy as np
import torch

try:
    from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
except Exception:  # pragma: no cover - fallback path
    from morl_baselines.multi_policy.pgmorl import PGMORL  # type: ignore

from .common import build_dynamic_scalarization
from borearl.env.forest_env import ForestEnv
from borearl import constants as const


def _safe_construct(cls, **kwargs):
    try:
        return cls(**kwargs)
    except TypeError:
        sig = inspect.signature(cls)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)


class _PGMORLAdapter:
    """
    Normalizes API to your framework:
      * `.act(obs, ...)`
      * `.set_eval_weight(lambda)`
      * persistence helpers
    """
    def __init__(self, inner: PGMORL, unwrapped_env):
        self.inner = inner
        self._unwrapped_env = unwrapped_env
        self._eval_lambda: float = 0.5
        self._active_policy = None

    def train(self, total_timesteps: int, **kwargs):
        return self.inner.train(total_timesteps=total_timesteps, **kwargs)

    def set_eval_weight(self, lam: float):
        self._eval_lambda = float(lam)
        self._active_policy = self._select_policy_for_lambda(self._eval_lambda)

    def select_policy(self, lam: float):
        self.set_eval_weight(lam)

    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor, acc_reward=None, eval_mode: bool = True) -> int:
        if self._active_policy is None:
            self._active_policy = self._select_policy_for_lambda(self._eval_lambda)

        pol = self._active_policy
        x = obs_tensor
        if x.ndim == 2 and x.shape[0] == 1:
            x_np = x.squeeze(0).cpu().numpy()
        else:
            x_np = x.view(-1).cpu().numpy()

        if hasattr(pol, "predict"):  # SB3-style
            act, _ = pol.predict(x_np, deterministic=eval_mode)
            return int(act)

        if hasattr(pol, "forward"):
            out = pol.forward(torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0))
            if isinstance(out, (tuple, list)):
                out = out[0]
            return int(torch.argmax(out, dim=-1).item())

        if hasattr(pol, "policy_forward"):
            logits = pol.policy_forward(torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0))
            if bool(const.EVAL_USE_ARGMAX_ACTIONS):
                return int(torch.argmax(logits, dim=-1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())

        raise RuntimeError("PGMORLAdapter: could not infer how to act with the active policy.")

    def state_dict(self):
        if hasattr(self.inner, "state_dict"):
            return self.inner.state_dict()
        return {}

    def load_state_dict(self, state):
        if hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(state)

    # ---- internal helpers ----
    def _discover_policy_bank(self):
        policies = None
        lambdas = None
        for name in ("policies", "policy_set", "archive", "pareto_policies"):
            if hasattr(self.inner, name):
                policies = getattr(self.inner, name)
                break
        for name in ("lambdas", "weights", "thetas", "ws"):
            if hasattr(self.inner, name):
                lambdas = getattr(self.inner, name)
                break
        if isinstance(policies, dict):
            lambdas = list(policies.keys()) if lambdas is None else lambdas
            policies = list(policies.values())
        if policies is None or len(policies) == 0:
            return None, None
        return policies, lambdas

    def _select_policy_for_lambda(self, lam: float):
        for name in ("select_policy", "best_response"):
            if hasattr(self.inner, name):
                try:
                    res = getattr(self.inner, name)(float(lam))
                    if res is not None:
                        return res
                    for cand in ("current_policy", "active_policy", "best_policy"):
                        if hasattr(self.inner, cand):
                            return getattr(self.inner, cand)
                except TypeError:
                    pass
        policies, lambdas = self._discover_policy_bank()
        if policies is not None and len(policies) > 0:
            if lambdas is not None and len(lambdas) == len(policies):
                ws = np.asarray(lambdas, dtype=np.float32).reshape(-1)
                idx = int(np.argmin(np.abs(ws - lam)))
            else:
                idx = 0
            return policies[idx]
        if hasattr(self.inner, "policy"):
            return getattr(self.inner, "policy")
        raise RuntimeError("PGMORLAdapter: no policy available for evaluation.")


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
):
    # Dynamic scalarization for plug-and-play with your generalist env.
    scalarization = build_dynamic_scalarization(unwrapped_env)
    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    sel_arch = list(net_arch) if net_arch is not None else const.EUPG_NET_ARCH_DEFAULT

    inner = _safe_construct(
        PGMORL,
        env=env,
        scalarization=scalarization,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        net_arch=sel_arch,
        log=bool(use_wandb),
        project_name=("Forest-MORL" if use_wandb else ""),
        experiment_name=(run_dir_name if run_dir_name else "PGMORL-Forest"),
    )
    return _PGMORLAdapter(inner, unwrapped_env)


def default_model_filename() -> str:
    return "pgmorl_policy_set.pt"


def supports_single_policy_eval() -> bool:
    return False


def save_policy_set(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    inner = getattr(model, "inner", model)
    if hasattr(inner, "save") and callable(inner.save):
        inner.save(path)
        return
    if hasattr(inner, "state_dict"):
        try:
            torch.save(inner.state_dict(), path)
            return
        except Exception as e:
            print(f"PGMORL save warning (state_dict): {e}")
    try:
        torch.save(inner, path)
    except Exception as e:
        if "pickle" in str(e).lower():
            print(f"PGMORL save skipped (unpicklable): {e}")
        else:
            raise


def load_policy_set(model, path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"PGMORL model file not found: {path}")
    inner = getattr(model, "inner", model)
    if hasattr(inner, "load") and callable(inner.load):
        inner.load(path)
        return model
    torch.serialization.add_safe_globals([PGMORL, ForestEnv])
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and hasattr(inner, "load_state_dict"):
        inner.load_state_dict(loaded)
        return model
    if isinstance(loaded, PGMORL):
        return _PGMORLAdapter(loaded, getattr(model, "_unwrapped_env", None))
    try:
        model.inner = loaded
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load PGMORL policy set: {e}")
