# agents/envelope_agent.py
from __future__ import annotations

import os
import inspect
import numpy as np
import torch

try:
    from morl_baselines.multi_policy.envelope.envelope import Envelope
except Exception:  # pragma: no cover - fallback path
    from morl_baselines.multi_policy.envelope import Envelope  # type: ignore

from .common import build_dynamic_scalarization
from borearl.env.forest_env import ForestEnv
from borearl import constants as const


# ---- small helper: instantiate even if the upstream class signature differs ----
def _safe_construct(cls, **kwargs):
    try:
        return cls(**kwargs)
    except TypeError:
        sig = inspect.signature(cls)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)


class _EnvelopeAdapter:
    """
    Thin adapter to:
      * standardize `.act(obs, ...)` for your get_action_from_model
      * choose the best policy for a requested λ at eval
      * keep references for (optional) coordination with env preference
    """
    def __init__(self, inner: Envelope, unwrapped_env):
        self.inner = inner
        self._unwrapped_env = unwrapped_env
        self._eval_lambda: float = 0.5
        self._active_policy = None  # cache selected policy/module

    # ----- training passthrough -----
    def train(self, total_timesteps: int, **kwargs):
        # Envelope has its own loop; just forward
        return self.inner.train(total_timesteps=total_timesteps, **kwargs)

    # ----- eval λ selection -----
    def set_eval_weight(self, lam: float):
        self._eval_lambda = float(lam)
        self._active_policy = self._select_policy_for_lambda(self._eval_lambda)

    def select_policy(self, lam: float):
        """Alias used by some of your helpers."""
        self.set_eval_weight(lam)

    # ----- unified action API -----
    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor, acc_reward=None, eval_mode: bool = True) -> int:
        """
        Returns a single int action for your non-vector env path.
        """
        # Pick/refresh active policy
        if self._active_policy is None:
            self._active_policy = self._select_policy_for_lambda(self._eval_lambda)

        # Try common policy interfaces (SB3-like or custom):
        pol = self._active_policy
        x = obs_tensor
        if x.ndim == 2 and x.shape[0] == 1:
            x_np = x.squeeze(0).cpu().numpy()
        else:
            x_np = x.view(-1).cpu().numpy()

        # 1) Stable-Baselines style
        if hasattr(pol, "predict"):
            act, _ = pol.predict(x_np, deterministic=eval_mode)
            return int(act)

        # 2) DQN-ish: q-network with .forward -> logits/Qs
        if hasattr(pol, "forward"):
            out = pol.forward(torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0))
            if isinstance(out, (tuple, list)):
                out = out[0]
            a = int(torch.argmax(out, dim=-1).item())
            return a

        # 3) Generic categorical logits
        if hasattr(pol, "policy_forward"):
            logits = pol.policy_forward(torch.as_tensor(x_np, dtype=torch.float32).unsqueeze(0))
            if bool(const.EVAL_USE_ARGMAX_ACTIONS):
                return int(torch.argmax(logits, dim=-1).item())
            return int(torch.distributions.Categorical(logits=logits).sample().item())

        raise RuntimeError("EnvelopeAdapter: could not infer how to act with the active policy.")

    # ----- persistence helpers for your saver loader -----
    def state_dict(self):
        if hasattr(self.inner, "state_dict"):
            return self.inner.state_dict()
        return {}

    def load_state_dict(self, state):
        if hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(state)

    # ----- internals -----
    def _discover_policy_bank(self):
        """
        Try to find (policies, lambdas) inside the Envelope object.
        We look for common attribute names and fall back gracefully.
        """
        policies = None
        lambdas = None

        # common attributes seen across MORL-baselines variants
        for name in ("policies", "policy_set", "archive", "pareto_policies"):
            if hasattr(self.inner, name):
                policies = getattr(self.inner, name)
                break

        for name in ("lambdas", "weights", "thetas", "ws"):
            if hasattr(self.inner, name):
                lambdas = getattr(self.inner, name)
                break

        # normalize to lists
        if isinstance(policies, dict):
            # sometimes {lambda: policy}
            lambdas = list(policies.keys()) if lambdas is None else lambdas
            policies = list(policies.values())

        if policies is None or len(policies) == 0:
            # If the library exposes a selection API, try to use it on the fly
            return None, None
        # lambdas may be None (then we just pick index 0)
        return policies, lambdas

    def has_any_policy(self) -> bool:
        """Check if any policy is available in the archive."""
        # Check if archive exists and has policies
        if hasattr(self.inner, "archive"):
            archive = self.inner.archive
            if hasattr(archive, "__len__"):
                return len(archive) > 0
            if hasattr(archive, "policies"):
                return len(archive.policies) > 0
        # Check if policies attribute exists directly
        if hasattr(self.inner, "policies"):
            return len(self.inner.policies) > 0
        return False

    def _select_policy_for_lambda(self, lam: float):
        # Check if any policy is available first
        if not self.has_any_policy():
            raise RuntimeError(
                "EnvelopeAdapter: no policy available for evaluation. "
                "Train longer or lower warmup thresholds so at least one policy enters the archive."
            )

        # Try direct selection APIs first:
        for name in ("select_policy", "best_response"):
            if hasattr(self.inner, name):
                try:
                    res = getattr(self.inner, name)(float(lam))
                    # Some APIs set an internal "current" policy and return None;
                    # some return a policy object. Handle both.
                    if res is not None:
                        return res
                    # try to fetch exposed current policy
                    for cand in ("current_policy", "active_policy", "best_policy"):
                        if hasattr(self.inner, cand):
                            return getattr(self.inner, cand)
                except TypeError:
                    pass

        # Fallback: nearest-lambda pick from stored bank
        policies, lambdas = self._discover_policy_bank()
        if policies is not None and len(policies) > 0:
            if lambdas is not None and len(lambdas) == len(policies):
                ws = np.asarray(lambdas, dtype=np.float32).reshape(-1)
                idx = int(np.argmin(np.abs(ws - lam)))
            else:
                idx = 0
            return policies[idx]

        # last resort: if there is a single policy object inside
        if hasattr(self.inner, "policy"):
            return getattr(self.inner, "policy")

        raise RuntimeError("EnvelopeAdapter: no policy available for evaluation.")


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
    # Pragmatic default: dynamic scalarization lets it "just work" with your
    # generalist env. For strict coverage, set a fixed-λ scalarization per policy.
    scalarization = build_dynamic_scalarization(unwrapped_env)

    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    sel_arch = list(net_arch) if net_arch is not None else const.EUPG_NET_ARCH_DEFAULT

    inner = _safe_construct(
        Envelope,
        env=env,
        scalarization=scalarization,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        net_arch=sel_arch,
        log=bool(use_wandb),
        project_name=("Forest-MORL" if use_wandb else ""),
        experiment_name=(run_dir_name if run_dir_name else "Envelope-Forest"),
    )
    return _EnvelopeAdapter(inner, unwrapped_env)


def default_model_filename() -> str:
    return "envelope_policy_set.tar"


def supports_single_policy_eval() -> bool:
    # Envelope learns a policy set; we'll pick per-λ at eval
    return False


def save_policy_set(model, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    inner = getattr(model, "inner", model)

    # Try library-native save first, but handle missing attributes
    if hasattr(inner, "save") and callable(inner.save):
        try:
            # Set experiment_name if it doesn't exist (required by some versions)
            if not hasattr(inner, 'experiment_name'):
                inner.experiment_name = "envelope_forest"
            
            # The envelope save method expects save_dir and filename separately
            save_dir = os.path.dirname(path)
            filename = os.path.basename(path)
            # Remove .tar extension if present since the envelope agent will add it
            if filename.endswith('.tar'):
                filename = filename[:-4]
            inner.save(save_dir=save_dir, filename=filename)
            return
        except Exception as e:
            print(f"Envelope save warning (native save): {e}")
            # If native save fails, try state_dict save instead

    # Next, state_dict
    if hasattr(inner, "state_dict") and callable(inner.state_dict):
        try:
            state_dict = inner.state_dict()
            if state_dict:  # Only save if state_dict is not empty
                torch.save(state_dict, path)
                return
        except Exception as e:
            print(f"Envelope save warning (state_dict): {e}")

    # Fallback: whole object (guard pickling)
    try:
        torch.save(inner, path)
    except Exception as e:
        if "pickle" in str(e).lower():
            print(f"Envelope save skipped (unpicklable): {e}")
        else:
            raise


def load_policy_set(model, path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Envelope model file not found: {path}")
    inner = getattr(model, "inner", model)

    # Prefer library-native load
    if hasattr(inner, "load") and callable(inner.load):
        try:
            inner.load(path)
            return model
        except Exception as e:
            # Try with weights_only=False for PyTorch 2.6+ compatibility
            if "weights_only" in str(e) or "UnpicklingError" in str(e):
                print(f"Retrying load with weights_only=False: {e}")
                # We'll handle this in the torch.load fallback below
            else:
                raise

    # Torch 2.6+ safety for class pickles
    torch.serialization.add_safe_globals([Envelope, ForestEnv])

    try:
        # Try with weights_only=True first (PyTorch 2.6+ default)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if "weights_only" in str(e) or "UnpicklingError" in str(e):
            # Fallback to weights_only=False for older format files
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise

    if isinstance(loaded, dict) and hasattr(inner, "load_state_dict"):
        inner.load_state_dict(loaded)
        return model
    if isinstance(loaded, Envelope):
        return _EnvelopeAdapter(loaded, getattr(model, "_unwrapped_env", None))
    # If we get here, best effort: assign attributes
    try:
        model.inner = loaded
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Envelope policy set: {e}")


def _maybe_set_envelope_eval_weight(model, weight):
    """Select the best policy for the requested lambda (weight[0])."""
    lam = float(weight[0]) if hasattr(weight, "__len__") else float(weight)
    # Prefer an explicit API on the adapter if present
    if hasattr(model, "set_eval_weight"):
        model.set_eval_weight(lam)
    elif hasattr(model, "select_policy"):
        model.select_policy(lam)  # some impls expose select_policy(λ)
    elif hasattr(model, "best_response"):
        try:
            model.best_response(lam)  # in-place selection in some impls
        except TypeError:
            pass
