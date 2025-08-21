from __future__ import annotations

import os
import numpy as np
from morl_baselines.single_policy.esr.eupg import EUPG

from .common import (
    make_env, build_dynamic_scalarization, save_run_config,
    build_preliminary_config, default_eval_weights,
)
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
    run_dir_name=None,
):
    scalarization = build_dynamic_scalarization(unwrapped_env)
    sel_weights = np.array(weights) if weights is not None else np.array(const.EUPG_DEFAULT_WEIGHTS)
    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    sel_arch = list(net_arch) if net_arch is not None else const.EUPG_NET_ARCH_DEFAULT
    # Allow tighter logging cadence if upstream supports it
    # Use run_dir_name as experiment_name if provided, otherwise fall back to default
    experiment_name = run_dir_name if run_dir_name else "EUPG-Forest"
    
    kwargs = dict(
        env=env,
        scalarization=scalarization,
        weights=sel_weights,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        net_arch=sel_arch,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name=experiment_name if use_wandb else "",
    )
    # Some morl-baselines versions accept log_every
    kwargs["log_every"] = int(100)
    return EUPG(**kwargs)


def default_model_filename() -> str:
    return "eupg_forest_manager.pth"


def supports_single_policy_eval() -> bool:
    return True


