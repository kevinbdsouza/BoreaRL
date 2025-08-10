from __future__ import annotations

import numpy as np

try:
    from morl_baselines.multi_policy.pcn.pcn import PCN
except Exception:  # pragma: no cover - fallback import path
    from morl_baselines.multi_policy.pcn import PCN  # type: ignore

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
    sel_weights = np.array(weights) if weights is not None else np.array(const.EUPG_DEFAULT_WEIGHTS)
    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    sel_arch = list(net_arch) if net_arch is not None else const.EUPG_NET_ARCH_DEFAULT
    return PCN(
        env=env,
        scalarization=scalarization,
        weights=sel_weights,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        net_arch=sel_arch,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name="PCN-Forest" if use_wandb else "",
    )


def default_model_filename() -> str:
    return "pcn_forest_manager.pth"


def supports_single_policy_eval() -> bool:
    return True


