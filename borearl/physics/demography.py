from typing import Dict, Tuple, Optional
import numpy as np


def calculate_natural_demography(
    current_density: float,
    max_natural_density: float,
    natural_mortality_rate: float,
    natural_recruitment_rate: float,
    density_dependent_mortality: float,
    density_mortality_threshold: float,
    environmental_stress: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()
    base_mortality_fraction = rng.normal(natural_mortality_rate, natural_mortality_rate * 0.3)
    base_mortality_fraction = np.clip(base_mortality_fraction, 0.0, 0.1)
    density_ratio = current_density / max_natural_density
    if density_ratio > density_mortality_threshold:
        density_mortality_fraction = density_dependent_mortality * (density_ratio - density_mortality_threshold) / (1.0 - density_mortality_threshold)
    else:
        density_mortality_fraction = 0.0
    total_mortality_fraction = (base_mortality_fraction + density_mortality_fraction) * environmental_stress
    mortality_stems = current_density * total_mortality_fraction
    available_space = max(0.0, max_natural_density - current_density)
    recruitment_potential = max_natural_density * natural_recruitment_rate
    recruitment_fraction = rng.normal(natural_recruitment_rate, natural_recruitment_rate * 0.5)
    recruitment_fraction = np.clip(recruitment_fraction, 0.0, 0.05)
    recruitment_stems = min(recruitment_potential * recruitment_fraction, available_space)
    return mortality_stems, recruitment_stems


