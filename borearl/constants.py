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
EUPG_DEFAULT_WEIGHTS = (0.75, 0.25)
USE_FIXED_PREFERENCE_DEFAULT = True
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


