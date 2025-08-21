from __future__ import annotations

import os
import numpy as np
import torch

try:
    from morl_baselines.multi_policy.pcn.pcn import PCN
except Exception:  # pragma: no cover - fallback import path
    from morl_baselines.multi_policy.pcn import PCN  # type: ignore

from .common import build_dynamic_scalarization
from borearl import constants as const
from borearl.env.forest_env import ForestEnv


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
    # Expect a single (w1, w2) weight vector for PCN. If None, pick a neutral preference.
    if weights is not None:
        w = np.array(weights, dtype=float).reshape(-1)
        if w.shape != (2,):
            raise ValueError(f"PCN expects a single weight vector of shape (2,), got {w.shape}")
        
        # Normalize weights if they sum to a positive value
        if w.sum() > 0:
            w = w / w.sum()
        # Note: w is computed but not used in PCN creation - PCN uses its own goal sampling during training
    
    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    
    # Map net_arch to hidden_dim if provided, otherwise use default
    hidden_dim = 64  # PCN default
    if net_arch is not None and len(net_arch) > 0:
        # Use the first layer size as hidden_dim
        hidden_dim = int(net_arch[0])
    
    # PCN scaling_factor normalizes the conditioning inputs (desired_return, desired_horizon)
    # Based on expected ranges for the forest environment:
    # - Horizon: 50 years (episode length)
    # We use existing constants for scaling factors
    expected_carbon_return_range = const.MAX_TOTAL_CARBON  # Use existing carbon constant
    expected_thaw_return_range = const.MAX_THAW_DEGREE_DAYS_PER_YEAR  # Use existing thaw constant
    max_horizon = const.EPISODE_LENGTH_YEARS  # 50 years
    
    # scaling_factor = [carbon_scale, thaw_scale, horizon_scale]
    scaling_factor = np.array([
        expected_carbon_return_range,
        expected_thaw_return_range, 
        max_horizon
    ], dtype=float)
    
    return PCN(
        env=env,
        scaling_factor=scaling_factor,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        hidden_dim=hidden_dim,
        log=False,  # Disable PCN's internal W&B logging to avoid conflicts
        project_name="",  # Empty to prevent W&B initialization
        experiment_name="",  # Empty to prevent W&B initialization
    )


def default_model_filename() -> str:
    return "pcn_forest_manager.pth"


def supports_single_policy_eval() -> bool:
    return True


def load_policy_set(model, path: str):
    """Load a PCN policy from disk.
    
    PCN models are saved as whole objects, so we load the entire model.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"PCN model file not found: {path}")
    
    # Add PCN and ForestEnv classes to safe globals for PyTorch 2.6+ compatibility
    torch.serialization.add_safe_globals([PCN, ForestEnv])
    
    try:
        # PCN saves the entire model object, so we load it directly
        # Explicitly set weights_only=False to allow loading the full object pickle
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        
        # Verify that we loaded a valid PCN model
        if not isinstance(loaded, PCN):
            raise RuntimeError(f"Loaded object is not a PCN model: {type(loaded)}")
        
        return loaded
    except Exception as e:
        raise RuntimeError(f"Failed to load PCN model from {path}: {str(e)}")


