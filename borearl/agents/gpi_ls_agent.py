from __future__ import annotations

try:
    from morl_baselines.multi_policy.gpi_ls.gpi_ls import GPI_LS as GPILS
except Exception:  # pragma: no cover - fallback
    try:
        from morl_baselines.multi_policy.gpi_ls import GPI_LS as GPILS  # type: ignore
    except Exception:
        from morl_baselines.multi_policy.gpi_ls.gpi_ls import GPILS  # type: ignore

import os
import torch

from .common import build_dynamic_scalarization
from borearl import constants as const


def create(env, unwrapped_env, use_wandb: bool):
    scalarization = build_dynamic_scalarization(unwrapped_env)
    return GPILS(
        env=env,
        scalarization=scalarization,
        gamma=const.EUPG_GAMMA_DEFAULT,
        learning_rate=const.EUPG_LEARNING_RATE_DEFAULT,
        net_arch=const.EUPG_NET_ARCH_DEFAULT,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name="GPI-LS-Forest" if use_wandb else "",
    )


def default_model_filename() -> str:
    # GPI-LS may aggregate multiple policies; single file not guaranteed
    return "gpi_ls_policy_set.pt"


def supports_single_policy_eval() -> bool:
    return False


def save_policy_set(model, path: str) -> None:
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
    try:
        loaded = torch.load(path, map_location="cpu")
        return loaded
    except Exception:
        return model

