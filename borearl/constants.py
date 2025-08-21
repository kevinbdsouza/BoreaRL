"""
Shared RL-level constants (separate from simulator physics constants).

Keep only environment/training-facing constants here. Physics constants remain
in the simulator configuration (energy_balance_rc.get_model_config).
"""

# Episode
EPISODE_LENGTH_YEARS = 50

# Actions
# Increase leverage of management moves while keeping action count small
DENSITY_ACTIONS = [-100, -50, 0, 50, 100]
CONIFER_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

HISTORY_WINDOW = 2

# Memory management constants
# Maximum number of entries to keep in history lists to prevent memory leaks
MAX_HISTORY_SIZE = 1000  # For carbon, disturbance, and management history
MAX_EPISODE_HISTORY_SIZE = 500  # For episode tracking lists (smaller since episodes are shorter)

# Normalization constants
MAX_TOTAL_CARBON = 50.0
MAX_GPP = 2.0
MAX_BIOMASS_CHANGE = 0.5
MAX_SOIL_CHANGE = 0.2
MAX_TOTAL_CHANGE = 0.7

# Climate normalization
LATITUDE_MIN = 50.0
LATITUDE_RANGE = 20.0
TEMP_MEAN_OFFSET = 10.0
TEMP_MEAN_RANGE = 20.0
TEMP_AMP_MAX = 30.0
DAYS_PER_YEAR = 365.0
MAX_GROWING_SEASON = 200.0

# Disturbance normalization
MAX_DROUGHT_INDEX = 100.0

# Derived
MAX_DENSITY_CHANGE = float(max(abs(x) for x in DENSITY_ACTIONS))

# Asymmetric thaw reward
WARMING_PENALTY_FACTOR = 5.0

# Contrast thaw calculation flag
# When True, use contrast thaw calculation instead of normalized asymmetric thaw
BOREARL_USE_CONTRAST_THAW = False

# HWP sales reward
MAX_HWP_SALES_PER_YEAR = 1.0
# Drop explicit HWP sales bonus from the reward shaping
HWP_SALE_REWARD_MULTIPLIER = 0.0

# Ineffective action penalties
INEFFECTIVE_THINNING_PENALTY = 0.5
INEFFECTIVE_PLANTING_PENALTY = 1.0

# Carbon stock limits (RL-level for penalties; hard caps enforced in simulator diag only)
MAX_BIOMASS_CARBON_LIMIT = 15.0
MAX_SOIL_CARBON_LIMIT = 20.0
CARBON_LIMIT_PENALTY = 0.5

# Density penalties
MAX_DENSITY_PENALTY = 1.0

# Thinning floor to avoid unrealistic collapses but still allow learning
SAFE_MIN_DENSITY_THINNING = 150

# Reward normalization constants
MAX_CARBON_CHANGE_PER_YEAR = 2.0
MAX_THAW_DEGREE_DAYS_PER_YEAR = 40.0

# PCN maximum return estimates for goal-conditioned evaluation
# These represent conservative estimates for 50-year episodes
MAX_CARBON_RETURN = 50.0  # Conservative estimate for 50-year episode
MAX_THAW_RETURN = 50.0    # Conservative estimate for 50-year episode

# PCN reference point for hypervolume calculation
# This defines the worst-case scenario used in hypervolume computation
PCN_REFERENCE_POINT = (-1.5, -1.5)  # Conservative reference point for forest environment

# Reward standardization defaults
# Set to False to log non-standardized rewards by default while keeping the
# option to enable per-run via env config `standardize_rewards=True`.
STANDARDIZE_REWARDS_DEFAULT = False
REWARD_EMA_BETA_DEFAULT = 0.99

# Stock bonus multiplier for carbon stock incentive
STOCK_BONUS_MULTIPLIER = 0.0

# Default ESR/EUPG scalarization weights (used for logging/eval defaults)
# Preference selection behavior for episodes
# When True, the environment will use the fixed preference vector in
# `EUPG_DEFAULT_WEIGHTS` for every episode instead of randomizing a
# new preference weight. The scalar preference weight used is the
# first element of `EUPG_DEFAULT_WEIGHTS`.
EUPG_DEFAULT_WEIGHTS = (0.5, 0.5)
USE_FIXED_PREFERENCE_DEFAULT = False
INCLUDE_SITE_PARAMS_IN_OBS_DEFAULT = True  # Generalist-only observation augmentation
USE_FIXED_SITE_INITIALS_DEFAULT = False   # If True, use fixed site defaults for initial state instead of sampling ranges

# EUPG hyperparameter defaults
EUPG_GAMMA_DEFAULT = 1
EUPG_LEARNING_RATE_DEFAULT = 0.001
EUPG_NET_ARCH_DEFAULT = [128, 64]

# Initial state ranges for environment reset
INITIAL_DENSITY_RANGE = (0, 600)  # stems/ha
INITIAL_CONIFER_FRACTION_RANGE = (0.2, 0.8)
INITIAL_BIOMASS_CARBON_RANGE = (1.0, 10.0)  # kg C/m²
INITIAL_SOIL_CARBON_RANGE = (2.0, 10.0)  # kg C/m²

# Site-specific mode defaults (env-level)
# The environment defaults site_specific=False if not provided; temperature noise determinism
# and age jitter toggles default to the site-specific flag within the environment.
SITE_WEATHER_SEED_DEFAULT = 123456

# Default site parameter overrides (midpoints of physics ranges)
# These are applied in site-specific mode if no explicit overrides are provided.
SITE_DEFAULT_OVERRIDES = {
    # Latitude and climate means
    'latitude_deg': 60.5,
    'T_annual_mean_offset': -7.5,
    'T_seasonal_amplitude': 22.5,
    'T_diurnal_amplitude': 6.0,
    'T_hour_peak_diurnal': 4.0,
    'mean_relative_humidity': 0.7,
    'u_ref': 3.0,
    # Surface/aerodynamics & soil
    'z0_can': 1.5,
    'z0_soil': 0.0125,
    'k_soil': 1.2,
    'SWC_max_mm': 150.0,
    'soil_stress_threshold': 0.45,
    'T_deep_boundary': 270.0,
    'k_ext_factor': 0.6,
    'k_snow_factor': 0.80,
    # Phenology (days) and rates
    'growth_day': 140,
    'fall_day': 270,
    'growth_rate': 0.115,
    'fall_rate': 0.115,
    'woody_area_index': 0.35,
    # Seasonal windows (days of year)
    'shoulder_1_start': 90,
    'shoulder_1_end': 150,
    'summer_day_start': 150,
    'summer_day_end': 250,
    'shoulder_2_start': 250,
    'shoulder_2_end': 300,
    'snow_season_end': 120,
    'snow_season_start': 280,
    # Weather stochasticity magnitudes (precip stays stochastic in site mode)
    'T_daily_noise_std': 1.5,  # will be set to 0.0 if deterministic_temp_noise=True
    # Precipitation climatology
    'rain_summer_prob': 0.15,
    'rain_summer_mm_day': 15.0,
    'rain_shoulder_prob': 0.10,
    'rain_shoulder_mm_day': 10.0,
    'snow_winter_prob': 0.225,
    'winter_snow_mm_day': 5.5,
    'temp_precip_summer_sensitivity': 0.07,
    'temp_precip_shoulder_sensitivity': 0.05,
    'temp_precip_summer_base_temp': 15.0,
    'temp_precip_shoulder_base_temp': 8.5,
    'rain_diurnal_sensitivity': 0.15,
    'rain_diurnal_threshold': 1.25,
    'max_diurnal_reduction': 0.55,
    'min_diurnal_amplitude': 1.15,
    # Carbon cycle & respiration
    'R_BASE_KG_M2_YR': 0.35,
    'R_BASE_SOIL_KG_M2_YR': 0.5,
    'Q10': 2.05,
    'LITTERFALL_FRACTION': 0.035,
    'LITTERFALL_SEASONALITY': 0.75,
    'GPP_SCALING_BIOMASS_THRESHOLD_kg_m2': 12.0,  # from config constant
    # Stand structure & optics
    'LAI_max_conifer': 4.0,
    'LAI_max_deciduous': 5.0,
    'alpha_can_base_conifer': 0.09,
    'alpha_can_base_deciduous': 0.175,
    # Demography
    'NATURAL_MORTALITY_RATE': 0.025,
    'NATURAL_RECRUITMENT_RATE': 0.01,
    'MAX_NATURAL_DENSITY': 1750,
    'DENSITY_DEPENDENT_MORTALITY': 0.035,
    'DENSITY_MORTALITY_THRESHOLD': 0.8,
    # Disturbances
    'FIRE_DROUGHT_THRESHOLD': 30,
    'FIRE_BASE_PROB': 0.0003,
    'INSECT_BASE_PROB': 0.035,
    'INSECT_MORTALITY_RATE': 0.035,
    # Fixed initial state defaults (used when env config sets use_fixed_site_initials=True)
    'initial_density': 300,
    'initial_conifer_fraction': 0.5,
    'initial_biomass_carbon': 5.5,
    'initial_soil_carbon': 6.0,
}

# Counterfactual sensitivity defaults
COUNTERFACTUAL_SAMPLES_DEFAULT = 5
COUNTERFACTUAL_PREF_DEFAULT = 0.5

# Evaluation behavior
# If True, actions during evaluation are selected via argmax over logits (greedy).
# If False, actions are sampled from the categorical distribution defined by logits.
EVAL_USE_ARGMAX_ACTIONS = False

# Physics backend defaults
# physics_backend: 'python' (default) or 'numba'
PHYSICS_BACKEND_DEFAULT = 'numba'
# Fast mode reduces time resolution and iterations for speed during training
FAST_MODE_DEFAULT = True
# Default JIT solver iterations (used only when backend is 'numba')
JIT_SOLVER_MAX_ITERS_DEFAULT = 4
# Update atmospheric stability every N sub-steps (>=1)
STABILITY_UPDATE_INTERVAL_STEPS_DEFAULT = 3


def generate_site_overrides_from_seed(seed: int) -> dict:
    """
    Generate site-specific parameter overrides by sampling from physics ranges.
    
    This function creates a new set of site overrides for each seed, ensuring
    that different seeds produce different site configurations in site-specific mode.
    
    Args:
        seed: Random seed for reproducible sampling
        
    Returns:
        Dictionary of site parameter overrides sampled from physics ranges
    """
    import numpy as np
    from .physics.config import get_model_config
    
    rng = np.random.default_rng(seed)
    config = get_model_config()
    
    # Sample from ranges to create site-specific overrides
    site_overrides = {}
    
    # Climate and latitude parameters
    lat_range = config['latitude_deg_range']
    site_overrides['latitude_deg'] = rng.uniform(low=lat_range[0], high=lat_range[1])
    
    # Temperature parameters
    temp_mean_range = config['T_annual_mean_offset_range']
    site_overrides['T_annual_mean_offset'] = rng.uniform(low=temp_mean_range[0], high=temp_mean_range[1])
    
    temp_amp_range = config['T_seasonal_amplitude_range']
    site_overrides['T_seasonal_amplitude'] = rng.uniform(low=temp_amp_range[0], high=temp_amp_range[1])
    
    temp_diurnal_range = config['T_diurnal_amplitude_range']
    site_overrides['T_diurnal_amplitude'] = rng.uniform(low=temp_diurnal_range[0], high=temp_diurnal_range[1])
    
    temp_peak_range = config['T_hour_peak_diurnal_range']
    site_overrides['T_hour_peak_diurnal'] = rng.uniform(low=temp_peak_range[0], high=temp_peak_range[1])
    
    # Humidity and wind
    humidity_range = config['mean_relative_humidity_range']
    site_overrides['mean_relative_humidity'] = rng.uniform(low=humidity_range[0], high=humidity_range[1])
    
    u_ref_range = config['u_ref_range']
    site_overrides['u_ref'] = rng.uniform(low=u_ref_range[0], high=u_ref_range[1])
    
    # Surface and soil parameters
    z0_can_range = config['z0_can_range']
    site_overrides['z0_can'] = rng.uniform(low=z0_can_range[0], high=z0_can_range[1])
    
    z0_soil_range = config['z0_soil_range']
    site_overrides['z0_soil'] = rng.uniform(low=z0_soil_range[0], high=z0_soil_range[1])
    
    k_soil_range = config['k_soil_range']
    site_overrides['k_soil'] = rng.uniform(low=k_soil_range[0], high=k_soil_range[1])
    
    swc_range = config['SWC_max_mm_range']
    site_overrides['SWC_max_mm'] = rng.uniform(low=swc_range[0], high=swc_range[1])
    
    stress_range = config['soil_stress_threshold_range']
    site_overrides['soil_stress_threshold'] = rng.uniform(low=stress_range[0], high=stress_range[1])
    
    deep_temp_range = config['T_deep_boundary_range']
    site_overrides['T_deep_boundary'] = rng.uniform(low=deep_temp_range[0], high=deep_temp_range[1])
    
    k_ext_range = config['k_ext_factor_range']
    site_overrides['k_ext_factor'] = rng.uniform(low=k_ext_range[0], high=k_ext_range[1])
    
    k_snow_range = config['k_snow_factor_range']
    site_overrides['k_snow_factor'] = rng.uniform(low=k_snow_range[0], high=k_snow_range[1])
    
    # Phenology parameters
    growth_day_range = config['growth_day_range']
    site_overrides['growth_day'] = int(rng.uniform(low=growth_day_range[0], high=growth_day_range[1]))
    
    fall_day_range = config['fall_day_range']
    site_overrides['fall_day'] = int(rng.uniform(low=fall_day_range[0], high=fall_day_range[1]))
    
    growth_rate_range = config['growth_rate_range']
    site_overrides['growth_rate'] = rng.uniform(low=growth_rate_range[0], high=growth_rate_range[1])
    
    fall_rate_range = config['fall_rate_range']
    site_overrides['fall_rate'] = rng.uniform(low=fall_rate_range[0], high=fall_rate_range[1])
    
    woody_range = config['woody_area_index_range']
    site_overrides['woody_area_index'] = rng.uniform(low=woody_range[0], high=woody_range[1])
    
    # Seasonal windows
    shoulder1_start_range = config['shoulder_1_start_range']
    site_overrides['shoulder_1_start'] = int(rng.uniform(low=shoulder1_start_range[0], high=shoulder1_start_range[1]))
    
    shoulder1_end_range = config['shoulder_1_end_range']
    site_overrides['shoulder_1_end'] = int(rng.uniform(low=shoulder1_end_range[0], high=shoulder1_end_range[1]))
    
    summer_start_range = config['summer_day_start_range']
    site_overrides['summer_day_start'] = int(rng.uniform(low=summer_start_range[0], high=summer_start_range[1]))
    
    summer_end_range = config['summer_day_end_range']
    site_overrides['summer_day_end'] = int(rng.uniform(low=summer_end_range[0], high=summer_end_range[1]))
    
    shoulder2_start_range = config['shoulder_2_start_range']
    site_overrides['shoulder_2_start'] = int(rng.uniform(low=shoulder2_start_range[0], high=shoulder2_start_range[1]))
    
    shoulder2_end_range = config['shoulder_2_end_range']
    site_overrides['shoulder_2_end'] = int(rng.uniform(low=shoulder2_end_range[0], high=shoulder2_end_range[1]))
    
    snow_end_range = config['snow_season_end_range']
    site_overrides['snow_season_end'] = int(rng.uniform(low=snow_end_range[0], high=snow_end_range[1]))
    
    snow_start_range = config['snow_season_start_range']
    site_overrides['snow_season_start'] = int(rng.uniform(low=snow_start_range[0], high=snow_start_range[1]))
    
    # Weather stochasticity
    temp_noise_range = config['T_daily_noise_std_range']
    site_overrides['T_daily_noise_std'] = rng.uniform(low=temp_noise_range[0], high=temp_noise_range[1])
    
    # Precipitation climatology
    rain_summer_prob_range = config['rain_summer_prob_range']
    site_overrides['rain_summer_prob'] = rng.uniform(low=rain_summer_prob_range[0], high=rain_summer_prob_range[1])
    
    rain_summer_mm_range = config['rain_summer_mm_day_range']
    site_overrides['rain_summer_mm_day'] = rng.uniform(low=rain_summer_mm_range[0], high=rain_summer_mm_range[1])
    
    rain_shoulder_prob_range = config['rain_shoulder_prob_range']
    site_overrides['rain_shoulder_prob'] = rng.uniform(low=rain_shoulder_prob_range[0], high=rain_shoulder_prob_range[1])
    
    rain_shoulder_mm_range = config['rain_shoulder_mm_day_range']
    site_overrides['rain_shoulder_mm_day'] = rng.uniform(low=rain_shoulder_mm_range[0], high=rain_shoulder_mm_range[1])
    
    snow_winter_prob_range = config['snow_winter_prob_range']
    site_overrides['snow_winter_prob'] = rng.uniform(low=snow_winter_prob_range[0], high=snow_winter_prob_range[1])
    
    winter_snow_range = config['winter_snow_mm_day_range']
    site_overrides['winter_snow_mm_day'] = rng.uniform(low=winter_snow_range[0], high=winter_snow_range[1])
    
    # Temperature-precipitation sensitivity
    temp_precip_summer_sens_range = config['temp_precip_summer_sensitivity_range']
    site_overrides['temp_precip_summer_sensitivity'] = rng.uniform(low=temp_precip_summer_sens_range[0], high=temp_precip_summer_sens_range[1])
    
    temp_precip_shoulder_sens_range = config['temp_precip_shoulder_sensitivity_range']
    site_overrides['temp_precip_shoulder_sensitivity'] = rng.uniform(low=temp_precip_shoulder_sens_range[0], high=temp_precip_shoulder_sens_range[1])
    
    temp_precip_summer_base_range = config['temp_precip_summer_base_temp_range']
    site_overrides['temp_precip_summer_base_temp'] = rng.uniform(low=temp_precip_summer_base_range[0], high=temp_precip_summer_base_range[1])
    
    temp_precip_shoulder_base_range = config['temp_precip_shoulder_base_temp_range']
    site_overrides['temp_precip_shoulder_base_temp'] = rng.uniform(low=temp_precip_shoulder_base_range[0], high=temp_precip_shoulder_base_range[1])
    
    # Rain diurnal patterns
    rain_diurnal_sens_range = config['rain_diurnal_sensitivity_range']
    site_overrides['rain_diurnal_sensitivity'] = rng.uniform(low=rain_diurnal_sens_range[0], high=rain_diurnal_sens_range[1])
    
    rain_diurnal_thresh_range = config['rain_diurnal_threshold_range']
    site_overrides['rain_diurnal_threshold'] = rng.uniform(low=rain_diurnal_thresh_range[0], high=rain_diurnal_thresh_range[1])
    
    max_diurnal_reduction_range = config['max_diurnal_reduction_range']
    site_overrides['max_diurnal_reduction'] = rng.uniform(low=max_diurnal_reduction_range[0], high=max_diurnal_reduction_range[1])
    
    min_diurnal_amp_range = config['min_diurnal_amplitude_range']
    site_overrides['min_diurnal_amplitude'] = rng.uniform(low=min_diurnal_amp_range[0], high=min_diurnal_amp_range[1])
    
    # Carbon cycle parameters
    r_base_range = config['R_BASE_KG_M2_YR_range']
    site_overrides['R_BASE_KG_M2_YR'] = rng.uniform(low=r_base_range[0], high=r_base_range[1])
    
    r_base_soil_range = config['R_BASE_SOIL_KG_M2_YR_range']
    site_overrides['R_BASE_SOIL_KG_M2_YR'] = rng.uniform(low=r_base_soil_range[0], high=r_base_soil_range[1])
    
    q10_range = config['Q10_range']
    site_overrides['Q10'] = rng.uniform(low=q10_range[0], high=q10_range[1])
    
    litterfall_frac_range = config['LITTERFALL_FRACTION_range']
    site_overrides['LITTERFALL_FRACTION'] = rng.uniform(low=litterfall_frac_range[0], high=litterfall_frac_range[1])
    
    litterfall_season_range = config['LITTERFALL_SEASONALITY_range']
    site_overrides['LITTERFALL_SEASONALITY'] = rng.uniform(low=litterfall_season_range[0], high=litterfall_season_range[1])
    
    # Stand structure parameters
    lai_conifer_range = config['LAI_max_conifer_range']
    site_overrides['LAI_max_conifer'] = rng.uniform(low=lai_conifer_range[0], high=lai_conifer_range[1])
    
    lai_deciduous_range = config['LAI_max_deciduous_range']
    site_overrides['LAI_max_deciduous'] = rng.uniform(low=lai_deciduous_range[0], high=lai_deciduous_range[1])
    
    alpha_conifer_range = config['alpha_can_base_conifer_range']
    site_overrides['alpha_can_base_conifer'] = rng.uniform(low=alpha_conifer_range[0], high=alpha_conifer_range[1])
    
    alpha_deciduous_range = config['alpha_can_base_deciduous_range']
    site_overrides['alpha_can_base_deciduous'] = rng.uniform(low=alpha_deciduous_range[0], high=alpha_deciduous_range[1])
    
    # Demography parameters
    mortality_range = config['NATURAL_MORTALITY_RATE_range']
    site_overrides['NATURAL_MORTALITY_RATE'] = rng.uniform(low=mortality_range[0], high=mortality_range[1])
    
    recruitment_range = config['NATURAL_RECRUITMENT_RATE_range']
    site_overrides['NATURAL_RECRUITMENT_RATE'] = rng.uniform(low=recruitment_range[0], high=recruitment_range[1])
    
    max_density_range = config['MAX_NATURAL_DENSITY_range']
    site_overrides['MAX_NATURAL_DENSITY'] = int(rng.uniform(low=max_density_range[0], high=max_density_range[1]))
    
    density_mortality_range = config['DENSITY_DEPENDENT_MORTALITY_range']
    site_overrides['DENSITY_DEPENDENT_MORTALITY'] = rng.uniform(low=density_mortality_range[0], high=density_mortality_range[1])
    
    density_threshold_range = config['DENSITY_MORTALITY_THRESHOLD_range']
    site_overrides['DENSITY_MORTALITY_THRESHOLD'] = rng.uniform(low=density_threshold_range[0], high=density_threshold_range[1])
    
    # Disturbance parameters
    fire_drought_range = config['FIRE_DROUGHT_THRESHOLD_range']
    site_overrides['FIRE_DROUGHT_THRESHOLD'] = int(rng.uniform(low=fire_drought_range[0], high=fire_drought_range[1]))
    
    fire_prob_range = config['FIRE_BASE_PROB_range']
    site_overrides['FIRE_BASE_PROB'] = rng.uniform(low=fire_prob_range[0], high=fire_prob_range[1])
    
    insect_prob_range = config['INSECT_BASE_PROB_range']
    site_overrides['INSECT_BASE_PROB'] = rng.uniform(low=insect_prob_range[0], high=insect_prob_range[1])
    
    insect_mortality_range = config['INSECT_MORTALITY_RATE_range']
    site_overrides['INSECT_MORTALITY_RATE'] = rng.uniform(low=insect_mortality_range[0], high=insect_mortality_range[1])
    
    # Fixed initial state defaults (used when env config sets use_fixed_site_initials=True)
    site_overrides['initial_density'] = rng.uniform(low=INITIAL_DENSITY_RANGE[0], high=INITIAL_DENSITY_RANGE[1])
    site_overrides['initial_conifer_fraction'] = rng.uniform(low=INITIAL_CONIFER_FRACTION_RANGE[0], high=INITIAL_CONIFER_FRACTION_RANGE[1])
    site_overrides['initial_biomass_carbon'] = rng.uniform(low=INITIAL_BIOMASS_CARBON_RANGE[0], high=INITIAL_BIOMASS_CARBON_RANGE[1])
    site_overrides['initial_soil_carbon'] = rng.uniform(low=INITIAL_SOIL_CARBON_RANGE[0], high=INITIAL_SOIL_CARBON_RANGE[1])
    
    return site_overrides


