from __future__ import annotations

import numpy as np

try:
    from morl_baselines.multi_policy.pcn.pcn import PCN
except Exception:  # pragma: no cover - fallback import path
    from morl_baselines.multi_policy.pcn import PCN  # type: ignore

from .common import build_dynamic_scalarization
from borearl import constants as const


def create(env, unwrapped_env, use_wandb: bool):
    scalarization = build_dynamic_scalarization(unwrapped_env)
    return PCN(
        env=env,
        scalarization=scalarization,
        weights=np.array(const.EUPG_DEFAULT_WEIGHTS),
        gamma=const.EUPG_GAMMA_DEFAULT,
        learning_rate=const.EUPG_LEARNING_RATE_DEFAULT,
        net_arch=const.EUPG_NET_ARCH_DEFAULT,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name="PCN-Forest" if use_wandb else "",
    )


def default_model_filename() -> str:
    return "pcn_forest_manager.pth"


def supports_single_policy_eval() -> bool:
    return True


