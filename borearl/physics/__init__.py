"""
Physics package for the BoreaRL project.

This package provides a modular energy-balance simulator and supporting
weather/climate utilities, demography, and configuration.
"""

from .energy_balance import ForestSimulator
from .constants import MIN_STEMS_HA, MAX_STEMS_HA
from .config import get_model_config

__all__ = [
    "ForestSimulator",
    "MIN_STEMS_HA",
    "MAX_STEMS_HA",
    "get_model_config",
]


