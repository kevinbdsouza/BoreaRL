from __future__ import annotations

try:
    from morl_baselines.multi_policy.chm.chm import CHM
except Exception:  # pragma: no cover - fallback
    from morl_baselines.multi_policy.chm import CHM  # type: ignore

import os
import torch

from .common import build_dynamic_scalarization
from borearl import constants as const


def create(
    env,
    unwrapped_env,
    use_wandb: bool,
    *,
    weights=None,
    gamma=None,
    learning_rate=None,
    net_arch=None,
):
    scalarization = build_dynamic_scalarization(unwrapped_env)
    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    sel_arch = list(net_arch) if net_arch is not None else const.EUPG_NET_ARCH_DEFAULT
    return CHM(
        env=env,
        scalarization=scalarization,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        net_arch=sel_arch,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name="CHM-Forest" if use_wandb else "",
    )


def default_model_filename() -> str:
    # CHM manages a policy set; we won't save a single policy file by default
    return "chm_policy_set.pt"


def supports_single_policy_eval() -> bool:
    # CHM is a coverage method; to evaluate across weights, we select/best policy per weight
    return False


def save_policy_set(model, path: str) -> None:
    """Persist the CHM policy set to disk.

    Tries, in order:
    - model.save(path) if available
    - torch.save(model.state_dict(), path) if state_dict exists
    - torch.save(model, path) as a last resort (object pickle)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        if hasattr(model, "save") and callable(model.save):
            model.save(path)
            return
    except Exception:
        pass
    try:
        state = model.state_dict()  # type: ignore[attr-defined]
        torch.save(state, path)
        return
    except Exception:
        pass
    torch.save(model, path)


def load_policy_set(model, path: str):
    """Load a CHM policy set from disk.

    Returns the loaded model when whole-object deserialization is used; otherwise
    returns the input model after loading its state.
    """
    try:
        if hasattr(model, "load") and callable(model.load):
            model.load(path)
            return model
    except Exception:
        pass
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and hasattr(model, "load_state_dict"):
            model.load_state_dict(state)  # type: ignore[attr-defined]
            return model
    except Exception:
        pass
    # Fallback: object was saved directly
    try:
        loaded = torch.load(path, map_location="cpu")
        return loaded
    except Exception:
        return model

