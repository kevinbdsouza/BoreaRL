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
from borearl.env.forest_env import ForestEnv


def create(
    env,
    unwrapped_env,
    use_wandb: bool,
    *,
    gamma=None,
    learning_rate=None,
    net_arch=None,
    run_dir_name=None,
):
    scalarization = build_dynamic_scalarization(unwrapped_env)
    sel_gamma = float(gamma) if gamma is not None else const.EUPG_GAMMA_DEFAULT
    sel_lr = float(learning_rate) if learning_rate is not None else const.EUPG_LEARNING_RATE_DEFAULT
    sel_arch = list(net_arch) if net_arch is not None else const.EUPG_NET_ARCH_DEFAULT
    return GPILS(
        env=env,
        scalarization=scalarization,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        net_arch=sel_arch,
        log=bool(use_wandb),
        project_name="Forest-MORL" if use_wandb else "",
        experiment_name=run_dir_name if run_dir_name else "GPI-LS-Forest",
    )


def default_model_filename() -> str:
    # GPI-LS may aggregate multiple policies; single file not guaranteed
    return "gpi_ls_policy_set.pt"


def supports_single_policy_eval() -> bool:
    return False


def save_policy_set(model, path: str) -> None:
    """Persist the GPI-LS policy set to disk.

    Tries, in order:
    - model.save(path) if available
    - torch.save(model.state_dict(), path) if state_dict exists
    - Skip saving if model contains unpicklable components (like scalarization functions)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if hasattr(model, "save") and callable(model.save):
        model.save(path)
        return
    
    # Try to save state_dict if available
    if hasattr(model, "state_dict"):
        try:
            state = model.state_dict()
            torch.save(state, path)
            return
        except Exception as e:
            print(f"Warning: Could not save model state_dict: {e}")
    
    # Try to save the entire model, but catch pickling errors
    try:
        torch.save(model, path)
        return
    except Exception as e:
        if "Can't pickle" in str(e) or "pickle" in str(e).lower():
            print(f"Warning: Could not save model due to unpicklable components (likely scalarization function): {e}")
            print("Skipping model save - this is expected for GPI-LS with dynamic scalarization")
            return
        else:
            # Re-raise if it's not a pickling error
            raise


def load_policy_set(model, path: str):
    """Load a GPI-LS policy set from disk.

    Returns the loaded model when whole-object deserialization is used; otherwise
    returns the input model after loading its state.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"GPI-LS model file not found: {path}")
    
    try:
        if hasattr(model, "load") and callable(model.load):
            model.load(path)
            return model
        
        # Add GPILS and ForestEnv classes to safe globals for PyTorch 2.6+ compatibility
        torch.serialization.add_safe_globals([GPILS, ForestEnv])
        
        # Try to load the saved data
        loaded_data = torch.load(path, map_location="cpu")
        
        # Check if we loaded a state_dict or the full model
        if isinstance(loaded_data, dict) and hasattr(model, "load_state_dict"):
            # We have a state_dict, load it into the existing model
            model.load_state_dict(loaded_data)
            return model
        elif isinstance(loaded_data, GPILS):
            # We have the full model object
            return loaded_data
        else:
            raise RuntimeError(f"Loaded data is neither a state_dict nor a GPI-LS model: {type(loaded_data)}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load GPI-LS model from {path}: {str(e)}")

