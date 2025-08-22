# borearl/agents/sec_pcn_agent.py
from __future__ import annotations
import os
import numpy as np
import torch
import gymnasium as gym

try:
    from morl_baselines.multi_policy.pcn.pcn import PCN
except Exception:  # pragma: no cover
    from morl_baselines.multi_policy.pcn import PCN  # type: ignore

from borearl import constants as const
from borearl.env.forest_env import ForestEnv  # safe-global for torch load

class SeasonPhaseObsWrapper(gym.ObservationWrapper):
    """
    Appends seasonal phase Fourier features to the observation:
      g = [sin(2πφ), cos(2πφ), sin(4πφ), cos(4πφ)]
    where φ = 0.5*(growth_day + fall_day)/DAYS_PER_YEAR.
    This is read-only and does not change env dynamics or logging.
    """
    def __init__(self, env: gym.Env, harmonics: int = 2):
        super().__init__(env)
        assert harmonics >= 1
        self.harmonics = int(harmonics)
        base: gym.spaces.Box = env.observation_space  # type: ignore
        extra = 2 * self.harmonics  # sin+cos for each harmonic
        low  = np.concatenate([base.low,  -np.ones(extra, dtype=np.float32)], axis=0)
        high = np.concatenate([base.high,  np.ones(extra, dtype=np.float32)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _phase(self) -> float:
        try:
            D = float(getattr(self.unwrapped, "DAYS_PER_YEAR", 365))
            p = getattr(self.unwrapped, "simulator", None)
            if p is None or not hasattr(p, "p"):
                return 0.5  # fallback mid-season
            growth = float(p.p.get("growth_day", 0.25*D))
            fall   = float(p.p.get("fall_day",   0.75*D))
            return 0.5 * (growth + fall) / D
        except Exception:
            return 0.5

    def observation(self, obs: np.ndarray) -> np.ndarray:
        phi = self._phase()
        feats = []
        for h in range(1, self.harmonics + 1):
            feats.append(np.sin(2*np.pi*h*phi))
            feats.append(np.cos(2*np.pi*h*phi))
        return np.concatenate([obs, np.asarray(feats, dtype=np.float32)], axis=0)


def _wrap_env_with_season_features(env: gym.Env, harmonics: int = 2) -> gym.Env:
    # Only wrap once
    if isinstance(env, SeasonPhaseObsWrapper):
        return env
    return SeasonPhaseObsWrapper(env, harmonics=harmonics)


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
    harmonics: int = 2,         # <= best default for this env
    hidden_dim_override: int | None = None,
    **_,
):
    """
    Drop-in PCN that augments observations with seasonal Fourier features.
    Signature accepts **kwargs to remain compatible with runner.py calls
    (e.g., use_plant_gate) without breaking other agents.
    """
    # Wrap env with seasonal features (no change to the underlying dynamics)
    env = _wrap_env_with_season_features(env, harmonics=harmonics)

    # Hyperparameters with robust defaults for this domain
    sel_gamma = float(gamma) if gamma is not None else 0.99
    sel_lr    = float(learning_rate) if learning_rate is not None else 3e-4

    # Map net_arch → hidden_dim if provided; otherwise prefer 128 for slightly richer features
    hidden_dim = 128 if hidden_dim_override is None else int(hidden_dim_override)
    if net_arch is not None and len(net_arch) > 0:
        hidden_dim = int(net_arch[0])

    # PCN scaling: (carbon, thaw, horizon) magnitudes for conditioning normalisation
    scaling_factor = np.array([
        float(const.MAX_TOTAL_CARBON),
        float(const.MAX_THAW_DEGREE_DAYS_PER_YEAR),
        float(const.EPISODE_LENGTH_YEARS),
    ], dtype=float)

    # IMPORTANT: keep logging disabled here; your runner handles W&B
    return PCN(
        env=env,
        scaling_factor=scaling_factor,
        gamma=sel_gamma,
        learning_rate=sel_lr,
        hidden_dim=hidden_dim,
        log=False,
        project_name="",
        experiment_name="",
    )


def default_model_filename() -> str:
    # keep distinct from vanilla PCN to avoid collisions
    return "sec_pcn_forest_manager.pth"


def supports_single_policy_eval() -> bool:
    return True


def load_policy_set(model, path: str):
    """Load a PCN object saved by runner's periodic saver."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"SEC-PCN model file not found: {path}")
    torch.serialization.add_safe_globals([PCN, ForestEnv])
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, PCN):
        raise RuntimeError(f"Loaded object is not a PCN model: {type(loaded)}")
    return loaded
