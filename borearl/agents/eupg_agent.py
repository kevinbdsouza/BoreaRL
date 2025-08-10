from __future__ import annotations

import os
import numpy as np
from morl_baselines.single_policy.esr.eupg import EUPG

from .common import (
    make_env, build_dynamic_scalarization, save_run_config,
    build_preliminary_config, default_eval_weights,
)
from borearl import constants as const


def create(env, unwrapped_env, use_wandb: bool):
    scalarization = build_dynamic_scalarization(unwrapped_env)
    return EUPG(
        env=env,
        scalarization=scalarization,
        weights=np.array(const.EUPG_DEFAULT_WEIGHTS),
        gamma=const.EUPG_GAMMA_DEFAULT,
        learning_rate=const.EUPG_LEARNING_RATE_DEFAULT,
        net_arch=const.EUPG_NET_ARCH_DEFAULT,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name="EUPG-Forest" if use_wandb else "",
    )


def default_model_filename() -> str:
    return "eupg_forest_manager.pth"


def supports_single_policy_eval() -> bool:
    return True


