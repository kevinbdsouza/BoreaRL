from __future__ import annotations

import os
import csv
from datetime import datetime
from typing import Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import borearl.physics as ebm
from borearl.physics import MIN_STEMS_HA, MAX_STEMS_HA

from .. import constants as const
from ..utils.profiling import profiler


class ForestEnv(gym.Env):
    """
    A Gym environment for boreal forest management with comprehensive observation space.
    
    Memory Management:
    This environment implements automatic memory management to prevent memory leaks
    during long training runs. History lists are automatically limited to prevent
    unbounded growth. See constants.py for configurable limits:
    - MAX_HISTORY_SIZE: Maximum entries in carbon/disturbance/management history (default: 1000)
    - MAX_EPISODE_HISTORY_SIZE: Maximum entries in episode tracking lists (default: 500)
    """
    metadata = {'render_modes': []}

    # --- Constants ---
    EPISODE_LENGTH_YEARS = const.EPISODE_LENGTH_YEARS

    # History window for tracking recent events
    HISTORY_WINDOW = const.HISTORY_WINDOW

    # --- Realistic Initial Value Ranges for Boreal Forests ---
    INITIAL_DENSITY_RANGE = const.INITIAL_DENSITY_RANGE
    INITIAL_CONIFER_FRACTION_RANGE = const.INITIAL_CONIFER_FRACTION_RANGE
    INITIAL_BIOMASS_CARBON_RANGE = const.INITIAL_BIOMASS_CARBON_RANGE
    INITIAL_SOIL_CARBON_RANGE = const.INITIAL_SOIL_CARBON_RANGE

    # --- Action Mapping ---
    DENSITY_ACTIONS = const.DENSITY_ACTIONS
    CONIFER_FRACTIONS = const.CONIFER_FRACTIONS

    # --- Normalization Constants ---
    MAX_TOTAL_CARBON = const.MAX_TOTAL_CARBON
    MAX_GPP = const.MAX_GPP
    MAX_BIOMASS_CHANGE = const.MAX_BIOMASS_CHANGE
    MAX_SOIL_CHANGE = const.MAX_SOIL_CHANGE
    MAX_TOTAL_CHANGE = const.MAX_TOTAL_CHANGE

    # Climate normalization
    LATITUDE_MIN = const.LATITUDE_MIN
    LATITUDE_RANGE = const.LATITUDE_RANGE
    TEMP_MEAN_OFFSET = const.TEMP_MEAN_OFFSET
    TEMP_MEAN_RANGE = const.TEMP_MEAN_RANGE
    TEMP_AMP_MAX = const.TEMP_AMP_MAX
    DAYS_PER_YEAR = const.DAYS_PER_YEAR
    MAX_GROWING_SEASON = const.MAX_GROWING_SEASON

    # Disturbance normalization
    MAX_DROUGHT_INDEX = const.MAX_DROUGHT_INDEX

    # Management normalization
    MAX_DENSITY_CHANGE = const.MAX_DENSITY_CHANGE

    # --- Reward Normalization Constants ---
    MAX_CARBON_CHANGE_PER_YEAR = const.MAX_CARBON_CHANGE_PER_YEAR
    MAX_THAW_DEGREE_DAYS_PER_YEAR = const.MAX_THAW_DEGREE_DAYS_PER_YEAR

    # --- Asymmetric Thaw Reward Constants ---
    WARMING_PENALTY_FACTOR = const.WARMING_PENALTY_FACTOR

    # --- HWP Sales Reward ---
    MAX_HWP_SALES_PER_YEAR = const.MAX_HWP_SALES_PER_YEAR
    HWP_SALE_REWARD_MULTIPLIER = const.HWP_SALE_REWARD_MULTIPLIER

    # --- Ineffective Action Penalties ---
    INEFFECTIVE_THINNING_PENALTY = const.INEFFECTIVE_THINNING_PENALTY
    INEFFECTIVE_PLANTING_PENALTY = const.INEFFECTIVE_PLANTING_PENALTY

    # --- Carbon Stock Limits ---
    MAX_BIOMASS_CARBON_LIMIT = const.MAX_BIOMASS_CARBON_LIMIT
    MAX_SOIL_CARBON_LIMIT = const.MAX_SOIL_CARBON_LIMIT
    CARBON_LIMIT_PENALTY = const.CARBON_LIMIT_PENALTY

    # --- Density Penalties ---
    MAX_DENSITY_PENALTY = const.MAX_DENSITY_PENALTY

    # --- Stock Bonus Constants ---
    STOCK_BONUS_MULTIPLIER = const.STOCK_BONUS_MULTIPLIER

    # --- Context parameters for generalist agent (episode-level site params) ---
    # Only used when not in site-specific mode
    INCLUDE_SITE_PARAMS_IN_OBS = const.INCLUDE_SITE_PARAMS_IN_OBS_DEFAULT
    # Will be populated per-instance from physics config ranges
    OBS_PARAM_KEYS: list[str] = []

    def _calculate_observation_size(self) -> int:
        # Current state (4): year, density, mix, carbon
        current_state_size = 4
        # Climate info (6)
        climate_info_size = 6
        # Disturbance history (6)
        disturbance_history_size = 6
        # Carbon cycle details (7)
        carbon_cycle_size = 7
        # Management history (4)
        management_history_size = 4
        # Age distribution (10)
        age_distribution_size = 10
        # Carbon stocks (2)
        carbon_stocks_size = 2
        # Penalty values (3)
        penalty_size = 3
        # Preference weight (1)
        preference_size = 1
        # Episode-level parameter context (generalist only)
        param_context_size = (
            len(getattr(self, '_obs_param_keys', self.OBS_PARAM_KEYS))
            if (not self.site_specific and getattr(self, 'include_site_params_in_obs', self.INCLUDE_SITE_PARAMS_IN_OBS))
            else 0
        )

        total_size = (
            current_state_size
            + climate_info_size
            + disturbance_history_size
            + carbon_cycle_size
            + management_history_size
            + age_distribution_size
            + carbon_stocks_size
            + penalty_size
            + preference_size
            + param_context_size
        )
        return total_size

    def __init__(self, config: dict | None = None):
        super().__init__()

        # --- Configuration ---
        self.config = config or {}
        # Use-case flag: site-specific vs generalist
        # site_specific: fix weather seed, zero temp noise, remove age jitter, allow parameter overrides
        self.site_specific = bool(self.config.get('site_specific', False))
        # Instance-level toggle to include sampled site params in observations
        # Default behavior: when site_specific=True, force default to False unless explicitly overridden
        if 'include_site_params_in_obs' in self.config:
            self.include_site_params_in_obs = bool(self.config.get('include_site_params_in_obs'))
        else:
            self.include_site_params_in_obs = False if self.site_specific else bool(const.INCLUDE_SITE_PARAMS_IN_OBS_DEFAULT)
        self.site_weather_seed = int(self.config.get('site_weather_seed', const.SITE_WEATHER_SEED_DEFAULT))
        # Site overrides:
        # - In site-specific mode, start from defaults and allow user to override selectively
        # - In generalist mode, do NOT preload defaults; only use explicit user-provided overrides
        if self.site_specific:
            self.site_overrides = dict(const.SITE_DEFAULT_OVERRIDES)
            self.site_overrides.update(dict(self.config.get('site_overrides', {})))
        else:
            self.site_overrides = dict(self.config.get('site_overrides', {}))
        # By default, tie these toggles to the site_specific flag unless explicitly overridden
        self.deterministic_temp_noise = bool(self.config.get('deterministic_temp_noise', self.site_specific))
        self.remove_age_jitter = bool(self.config.get('remove_age_jitter', self.site_specific))
        # Physics backend and performance tuning defaults (config-overridable)
        self.physics_backend = str(self.config.get('physics_backend', const.PHYSICS_BACKEND_DEFAULT))
        self.fast_mode = bool(self.config.get('fast_mode', const.FAST_MODE_DEFAULT))
        # Use explicit defaults; downstream uses these primarily when JIT backend is active
        self.jit_solver_max_iters = int(self.config.get('jit_solver_max_iters', const.JIT_SOLVER_MAX_ITERS_DEFAULT))
        self.stability_update_interval_steps = int(
            self.config.get('stability_update_interval_steps', const.STABILITY_UPDATE_INTERVAL_STEPS_DEFAULT)
        )
        # Expose fixed-initials toggle as an attribute for external consumers (config export, etc.)
        self.use_fixed_site_initials = (
            bool(self.config.get('use_fixed_site_initials', const.USE_FIXED_SITE_INITIALS_DEFAULT))
            if self.site_specific else False
        )

        # --- Episode Tracking ---
        self.episode_count = 0
        self.current_episode_rewards = []
        self.current_episode_carbon_rewards = []
        self.current_episode_thaw_rewards = []
        self.current_episode_actions = []
        self.current_episode_states = []

        # --- Ineffective Action Tracking ---
        self.last_ineffective_thinning = False
        self.last_ineffective_planting = False

        # --- CSV Logging Setup ---
        self.csv_logging_enabled = self.config.get('csv_logging_enabled', True)
        self.csv_output_dir = self.config.get('csv_output_dir', 'logs')
        if self.csv_logging_enabled:
            os.makedirs(self.csv_output_dir, exist_ok=True)
            # Allow a stable run ID via env to consolidate logs
            run_id = os.environ.get('BOREARL_CSV_RUN_ID') or os.environ.get('BOREARL_RUN_ID')
            if not run_id:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Choose filenames based on phase to avoid mixing train/eval/baseline
            phase = str(os.environ.get('BOREARL_PHASE', 'train')).strip().lower()
            if phase == 'eval':
                step_name = f'eval_step_{run_id}.csv'
                episode_name = f'eval_episode_{run_id}.csv'
            elif phase == 'baseline':
                step_name = f'baseline_step_{run_id}.csv'
                episode_name = f'baseline_episode_{run_id}.csv'
            else:
                step_name = f'step_metrics_{run_id}.csv'
                episode_name = f'episode_metrics_{run_id}.csv'
            self.step_csv_path = os.path.join(self.csv_output_dir, step_name)
            self.episode_csv_path = os.path.join(self.csv_output_dir, episode_name)
            # Defer header creation until the first actual write to avoid empty CSVs

        # --- Action Space ---
        self._density_action_size = len(self.DENSITY_ACTIONS)
        self._conifer_fraction_size = len(self.CONIFER_FRACTIONS)
        self.action_space = spaces.Discrete(self._density_action_size * self._conifer_fraction_size)

        # Build dynamic list of parameter keys from physics config ranges
        cfg = ebm.get_model_config()
        self._obs_param_keys = sorted({k.replace('_range', '') for k in cfg.keys() if k.endswith('_range')})

        # Preference handling overrides (allow YAML to override constants)
        self.use_fixed_preference = bool(self.config.get('use_fixed_preference', const.USE_FIXED_PREFERENCE_DEFAULT))

        cfg_weights = self.config.get('eupg_default_weights', const.EUPG_DEFAULT_WEIGHTS)
        if isinstance(cfg_weights, (list, tuple)) and len(cfg_weights) == 2:
            self.eupg_default_weights = (float(cfg_weights[0]), float(cfg_weights[1]))
        else:
            self.eupg_default_weights = tuple(const.EUPG_DEFAULT_WEIGHTS)

        # --- Observation Space ---
        obs_size = self._calculate_observation_size()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        # --- Reward Space ---
        self.reward_space = spaces.Box(
            low=np.array([-2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # --- State Variables (initialized in reset) ---
        self.year = 0
        self.stem_density = 0
        self.conifer_fraction = 0.0
        self.biomass_carbon_kg_m2 = 0.0
        self.soil_carbon_kg_m2 = 0.0
        self.cumulative_thaw_dd = 0.0

        # --- History Tracking ---
        self.disturbance_history = {
            'fire_fractions': [],
            'insect_fractions': [],
            'drought_indices': [],
            'winter_temps': [],
            'summer_temps': [],
        }
        self.management_history = {
            'density_actions': [],
            'mix_actions': [],
            'density_changes': [],
            'mix_changes': [],
        }
        self.carbon_history = {
            'biomass_changes': [],
            'soil_changes': [],
            'total_changes': [],
            'gpp_values': [],
        }

        # --- Last Step Variables for Info ---
        self.last_mortality_stems = 0.0
        self.last_recruitment_stems = 0.0
        self.last_density_change = 0.0
        self.last_fire_mortality_fraction = 0.0
        self.last_insect_mortality_fraction = 0.0
        self.last_drought_index = 0.0

        # --- Carbon Limit Tracking ---
        self.last_biomass_limit_violation = False
        self.last_soil_limit_violation = False
        self.last_biomass_excess = 0.0
        self.last_soil_excess = 0.0
        self.last_limit_penalty = 0.0

        # --- Simulator ---
        self.simulator = None

        # --- Preference Weight for MORL ---
        self._preference_was_set_externally = False
        object.__setattr__(self, 'current_preference_weight', 0.5)

        # --- Unclipped stem density for termination checking ---
        self.unclipped_stem_density = 0

        # --- Reward normalization (EMA standardization) ---
        # Defaults come from constants, can be overridden via env `config`.
        self.enable_reward_standardization = self.config.get(
            'standardize_rewards', const.STANDARDIZE_REWARDS_DEFAULT
        )
        self.reward_ema_beta = self.config.get(
            'reward_ema_beta', const.REWARD_EMA_BETA_DEFAULT
        )
        self._reward_mean = np.zeros(2, dtype=np.float32)
        self._reward_var = np.ones(2, dtype=np.float32)
        self._reward_count = 0
        self._eps = 1e-6
        self.last_raw_reward_vector = np.zeros(2, dtype=np.float32)

    def _initialize_csv_files(self):
        step_headers = [
            'episode_no', 'step_no', 'year',
            'density_action_idx', 'density_change', 'species_action_idx', 'species_fraction',
            'stem_density_ha',
            'carbon_reward', 'thaw_reward', 'scalarized_reward',
            'asymmetric_thaw_reward', 'normalized_asymmetric_thaw', 'positive_flux_sum', 'negative_flux_sum',
            'normalized_carbon_change', 'base_stock_bonus', 'biomass_penalty', 'soil_penalty', 'total_carbon_limit_penalty', 'hwp_sale_reward', 'normalized_hwp_sales',
            'conifer_fraction', 'biomass_carbon_kg_m2', 'soil_carbon_kg_m2', 'carbon_stock_kg_m2',
            'latitude_deg', 'mean_temp_c', 'temp_amplitude_c', 'growing_season_days',
            'natural_mortality_stems', 'natural_recruitment_stems', 'natural_density_change',
            'fire_mortality_fraction', 'insect_mortality_fraction', 'final_drought_index',
            'conifer_seedling_stems', 'conifer_sapling_stems', 'conifer_young_stems', 'conifer_mature_stems', 'conifer_old_stems',
            'deciduous_seedling_stems', 'deciduous_sapling_stems', 'deciduous_young_stems', 'deciduous_mature_stems', 'deciduous_old_stems',
            'biomass_carbon_change', 'soil_carbon_change', 'total_carbon_change', 'gpp_kg_m2',
            'total_gpp_kg_m2', 'total_autotrophic_resp_kg_m2', 'total_soil_resp_kg_m2', 'total_litterfall_kg_m2',
            'natural_mortality_biomass', 'fire_biomass_loss', 'insect_biomass_loss', 'total_mortality_losses', 'carbon_loss_thinning', 'hwp_carbon_stored',
            'cumulative_thaw_dd',
            'ineffective_thinning', 'ineffective_planting',
            'biomass_limit_violation', 'soil_limit_violation', 'biomass_excess', 'soil_excess', 'limit_penalty',
            'max_density_penalty',
        ]
        episode_headers = [
            'episode_no', 'total_steps',
            'terminated', 'truncated',
            'total_carbon_reward', 'total_thaw_reward', 'total_scalarized_reward',
            'avg_carbon_reward', 'avg_thaw_reward',
            'carbon_reward_std', 'thaw_reward_std',
            'final_total_carbon', 'final_biomass_carbon', 'final_soil_carbon',
            'final_stem_density', 'final_conifer_fraction',
            'avg_total_carbon', 'avg_biomass_carbon', 'avg_soil_carbon',
            'avg_stem_density', 'avg_conifer_fraction',
            'net_carbon_gain', 'carbon_efficiency',
            'total_positive_flux', 'total_negative_flux',
            'preference_weight', 'baseline_type',
        ]
        with open(self.step_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(step_headers)
        with open(self.episode_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(episode_headers)

    def _log_step_metrics(self, action, reward_vector, info):
        if not self.csv_logging_enabled:
            return
        # Lazily create the CSV and write header on first use
        if not os.path.exists(self.step_csv_path) or os.path.getsize(self.step_csv_path) == 0:
            with open(self.step_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'episode_no', 'step_no', 'year',
                    'density_action_idx', 'density_change', 'species_action_idx', 'species_fraction',
                    'stem_density_ha',
                    'carbon_reward', 'thaw_reward', 'scalarized_reward',
                    'asymmetric_thaw_reward', 'normalized_asymmetric_thaw', 'positive_flux_sum', 'negative_flux_sum',
                    'normalized_carbon_change', 'base_stock_bonus', 'biomass_penalty', 'soil_penalty', 'total_carbon_limit_penalty', 'hwp_sale_reward', 'normalized_hwp_sales',
                    'conifer_fraction', 'biomass_carbon_kg_m2', 'soil_carbon_kg_m2', 'carbon_stock_kg_m2',
                    'latitude_deg', 'mean_temp_c', 'temp_amplitude_c', 'growing_season_days',
                    'natural_mortality_stems', 'natural_recruitment_stems', 'natural_density_change',
                    'fire_mortality_fraction', 'insect_mortality_fraction', 'final_drought_index',
                    'conifer_seedling_stems', 'conifer_sapling_stems', 'conifer_young_stems', 'conifer_mature_stems', 'conifer_old_stems',
                    'deciduous_seedling_stems', 'deciduous_sapling_stems', 'deciduous_young_stems', 'deciduous_mature_stems', 'deciduous_old_stems',
                    'biomass_carbon_change', 'soil_carbon_change', 'total_carbon_change', 'gpp_kg_m2',
                    'total_gpp_kg_m2', 'total_autotrophic_resp_kg_m2', 'total_soil_resp_kg_m2', 'total_litterfall_kg_m2',
                    'natural_mortality_biomass', 'fire_biomass_loss', 'insect_biomass_loss', 'total_mortality_losses', 'carbon_loss_thinning', 'hwp_carbon_stored',
                    'cumulative_thaw_dd',
                    'ineffective_thinning', 'ineffective_planting',
                    'biomass_limit_violation', 'soil_limit_violation', 'biomass_excess', 'soil_excess', 'limit_penalty',
                    'max_density_penalty',
                ])
        if np.isscalar(action) or (hasattr(action, 'size') and action.size == 1):
            # Robustly convert Python scalars, NumPy scalars, or 0-d arrays
            try:
                action_value = int(action.item()) if hasattr(action, 'item') else int(action)
            except Exception:
                action_value = int(action)
            density_action_idx = action_value // self._conifer_fraction_size
            species_action_idx = action_value % self._conifer_fraction_size
        else:
            action_value = int(action[0])
            density_action_idx = action_value // self._conifer_fraction_size
            species_action_idx = action_value % self._conifer_fraction_size
        density_change = self.DENSITY_ACTIONS[density_action_idx]
        species_fraction = self.CONIFER_FRACTIONS[species_action_idx]
        biomass_change = self.carbon_history['biomass_changes'][-1] if self.carbon_history['biomass_changes'] else 0.0
        soil_change = self.carbon_history['soil_changes'][-1] if self.carbon_history['soil_changes'] else 0.0
        total_change = self.carbon_history['total_changes'][-1] if self.carbon_history['total_changes'] else 0.0
        gpp = self.carbon_history['gpp_values'][-1] if self.carbon_history['gpp_values'] else 0.0
        # Scalarize using the actual preference used this episode to keep logs consistent
        eupg_weights = np.array(getattr(self, 'eupg_default_weights', const.EUPG_DEFAULT_WEIGHTS))
        pref = float(getattr(self, 'current_preference_weight', eupg_weights[0]))
        scalarized_reward = float(pref * reward_vector[0] + (1.0 - pref) * reward_vector[1])
        row = [
            self.episode_count, len(self.current_episode_rewards), self.year,
            density_action_idx, density_change, species_action_idx, species_fraction,
            info['stem_density_ha'],
            reward_vector[0], reward_vector[1], scalarized_reward,
            info['asymmetric_thaw_reward'], info['normalized_asymmetric_thaw'], info['positive_flux_sum'], info['negative_flux_sum'],
            info['normalized_carbon_change'], info['base_stock_bonus'], info['biomass_penalty'], info['soil_penalty'], info['total_carbon_limit_penalty'], info.get('hwp_sale_reward', 0.0), info.get('normalized_hwp_sales', 0.0),
            info['conifer_fraction'], info['biomass_carbon_kg_m2'], info['soil_carbon_kg_m2'], info['carbon_stock_kg_m2'],
            info['latitude_deg'], info['mean_temp_c'], info['temp_amplitude_c'], info['growing_season_days'],
            info['natural_mortality_stems'], info['natural_recruitment_stems'], info['natural_density_change'],
            info['fire_mortality_fraction'], info['insect_mortality_fraction'], info['final_drought_index'],
            info['conifer_seedling_stems'], info['conifer_sapling_stems'], info['conifer_young_stems'], info['conifer_mature_stems'], info['conifer_old_stems'],
            info['deciduous_seedling_stems'], info['deciduous_sapling_stems'], info['deciduous_young_stems'], info['deciduous_mature_stems'], info['deciduous_old_stems'],
            biomass_change, soil_change, total_change, gpp,
            info['total_gpp_kg_m2'], info['total_autotrophic_resp_kg_m2'], info['total_soil_resp_kg_m2'], info['total_litterfall_kg_m2'],
            info['natural_mortality_biomass'], info['fire_biomass_loss'], info['insect_biomass_loss'], info['total_mortality_losses'], info['carbon_loss_thinning'], info['hwp_carbon_stored'],
            info['cumulative_thaw_dd'],
            info['ineffective_thinning'], info['ineffective_planting'],
            info['biomass_limit_violation'], info['soil_limit_violation'], info['biomass_excess'], info['soil_excess'], info['limit_penalty'],
            info['max_density_penalty'],
        ]
        with open(self.step_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        # Optional: stream a small set of per-step metrics to W&B during training phase
        try:
            if str(os.environ.get('WANDB_DISABLED', '')).lower() != 'true' and str(os.environ.get('BOREARL_PHASE', '')).lower() == 'train':
                import wandb  # type: ignore
                # Use a monotonically increasing global step if present
                try:
                    global_step = int(os.environ.get('BOREARL_GLOBAL_STEP_COUNT', '0'))
                except Exception:
                    global_step = int(self.episode_count * 1000000 + len(self.current_episode_rewards))
                payload = {
                    'train/step': int(global_step),
                    'train_step/episode_no': int(self.episode_count),
                    'train_step/step_no': int(len(self.current_episode_rewards)),
                    'train_step/year': int(self.year),
                    'train_step/reward_carbon': float(reward_vector[0]),
                    'train_step/reward_thaw': float(reward_vector[1]),
                    'train_step/scalarized_reward': float(scalarized_reward),
                }
                # Map to the global step axis to show dense progress
                wandb.log(payload, step=int(global_step), commit=True)
        except Exception:
            pass

    def _log_episode_metrics(self, terminated, truncated):
        if not self.csv_logging_enabled:
            return
        stats = self.get_episode_statistics()
        if not stats:
            return
        # Lazily create the CSV and write header on first use
        if not os.path.exists(self.episode_csv_path) or os.path.getsize(self.episode_csv_path) == 0:
            with open(self.episode_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'episode_no', 'total_steps',
                    'terminated', 'truncated',
                    'total_carbon_reward', 'total_thaw_reward', 'total_scalarized_reward',
                    'avg_carbon_reward', 'avg_thaw_reward',
                    'carbon_reward_std', 'thaw_reward_std',
                    'final_total_carbon', 'final_biomass_carbon', 'final_soil_carbon',
                    'final_stem_density', 'final_conifer_fraction',
                    'avg_total_carbon', 'avg_biomass_carbon', 'avg_soil_carbon',
                    'avg_stem_density', 'avg_conifer_fraction',
                    'net_carbon_gain', 'carbon_efficiency',
                    'total_positive_flux', 'total_negative_flux',
                    'preference_weight', 'baseline_type', 'weather_seed',
                ])
        row = [
            stats['episode_number'], stats['total_steps'],
            terminated, truncated,
            stats['total_carbon_reward'], stats['total_thaw_reward'], stats['total_scalarized_reward'],
            stats['avg_carbon_reward'], stats['avg_thaw_reward'],
            stats['carbon_reward_std'], stats['thaw_reward_std'],
            stats['final_total_carbon'], stats['final_biomass_carbon'], stats['final_soil_carbon'],
            stats['final_stem_density'], stats['final_conifer_fraction'],
            stats['avg_total_carbon'], stats['avg_biomass_carbon'], stats['avg_soil_carbon'],
            stats['avg_stem_density'], stats['avg_conifer_fraction'],
            stats['net_carbon_gain'], stats['carbon_efficiency'],
            stats['total_positive_flux'], stats['total_negative_flux'],
            float(getattr(self, 'current_preference_weight', 0.0)),
            str(getattr(self, 'baseline_type', '')),
            getattr(self, '_episode_weather_seed_used', None),
        ]
        with open(self.episode_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        profiler.start_episode()
        self._in_reset = True
        super().reset(seed=seed)
        # Store the seed used for this episode (None if no seed was provided)
        self._episode_seed_used = seed
        self.episode_count += 1
        print(f"Episode {self.episode_count} started")
        self.current_episode_rewards = []
        self.current_episode_carbon_rewards = []
        self.current_episode_thaw_rewards = []
        self.current_episode_actions = []
        self.current_episode_states = []
        self.episode_positive_flux_sum = 0.0
        self.episode_negative_flux_sum = 0.0

        # Initial state selection:
        # - If site_specific and use_fixed_site_initials=True -> use fixed values from site_overrides/constants
        # - Else -> sample from env config ranges or global defaults (no site_overrides fallback for ranges)
        use_fixed_site_initials = bool(getattr(self, 'use_fixed_site_initials', False))

        if self.site_specific and use_fixed_site_initials:
            fixed_density = self.config.get('initial_density', self.site_overrides.get('initial_density'))
            fixed_conifer = self.config.get('initial_conifer_fraction', self.site_overrides.get('initial_conifer_fraction'))
            fixed_biomass = self.config.get('initial_biomass_carbon', self.site_overrides.get('initial_biomass_carbon'))
            fixed_soil = self.config.get('initial_soil_carbon', self.site_overrides.get('initial_soil_carbon'))
            # Use exact values (no sampling)
            density_range = (int(fixed_density), int(fixed_density))
            conifer_range = (float(fixed_conifer), float(fixed_conifer))
            biomass_range = (float(fixed_biomass), float(fixed_biomass))
            soil_range = (float(fixed_soil), float(fixed_soil))
        else:
            density_range = self.config.get('initial_density_range', self.INITIAL_DENSITY_RANGE)
            conifer_range = self.config.get('initial_conifer_fraction_range', self.INITIAL_CONIFER_FRACTION_RANGE)
            biomass_range = self.config.get('initial_biomass_carbon_range', self.INITIAL_BIOMASS_CARBON_RANGE)
            soil_range = self.config.get('initial_soil_carbon_range', self.INITIAL_SOIL_CARBON_RANGE)

        self.year = 0
        self.stem_density = self.np_random.integers(density_range[0], density_range[1] + 1)
        self.conifer_fraction = self.np_random.uniform(conifer_range[0], conifer_range[1])
        self.biomass_carbon_kg_m2 = self.np_random.uniform(biomass_range[0], biomass_range[1])
        self.soil_carbon_kg_m2 = self.np_random.uniform(soil_range[0], soil_range[1])
        self.cumulative_thaw_dd = 0.0

        self.disturbance_history = {
            'fire_fractions': [], 'insect_fractions': [], 'drought_indices': [], 'winter_temps': [], 'summer_temps': []
        }
        self.management_history = {
            'density_actions': [], 'mix_actions': [], 'density_changes': [], 'mix_changes': []
        }
        self.carbon_history = {
            'biomass_changes': [], 'soil_changes': [], 'total_changes': [], 'gpp_values': []
        }

        self.last_mortality_stems = 0.0
        self.last_recruitment_stems = 0.0
        self.last_density_change = 0.0
        self.last_fire_mortality_fraction = 0.0
        self.last_insect_mortality_fraction = 0.0
        self.last_drought_index = 0.0

        self.consecutive_max_density_steps = 0
        self.consecutive_carbon_penalty_steps = 0

        assert self.stem_density >= MIN_STEMS_HA, f"Initial stem density {self.stem_density} below minimum {MIN_STEMS_HA}"
        assert self.stem_density <= MAX_STEMS_HA, f"Initial stem density {self.stem_density} above maximum {MAX_STEMS_HA}"
        assert 0.0 <= self.conifer_fraction <= 1.0
        assert self.biomass_carbon_kg_m2 >= 0.0
        assert self.soil_carbon_kg_m2 >= 0.0

        if self.site_specific:
            self._weather_seed_used = self.site_weather_seed
        else:
            self._weather_seed_used = int(self.np_random.integers(0, 2**31 - 1))
        # Store the actual seed used for this episode (weather seed for physics, env seed for initial conditions)
        self._episode_weather_seed_used = self._weather_seed_used
        self.simulator = ebm.ForestSimulator(
            coniferous_fraction=self.conifer_fraction,
            stem_density=self.stem_density,
            weather_seed=self._weather_seed_used,
            site_specific=self.site_specific,
            site_overrides=self.site_overrides,
            deterministic_temp_noise=self.deterministic_temp_noise,
            remove_age_jitter=self.remove_age_jitter,
            physics_backend=self.physics_backend,
            fast_mode=self.fast_mode,
            jit_solver_max_iters=self.jit_solver_max_iters,
            stability_update_interval_steps=self.stability_update_interval_steps,
        )

        if getattr(self, 'use_fixed_preference'):
            # choose first weight from configured default weights
            w0 = float(getattr(self, 'eupg_default_weights')[0])
            self.current_preference_weight = w0
        elif options and 'preference' in options:
            self.current_preference_weight = options['preference'][0]
        elif self._preference_was_set_externally:
            # Keep the externally set preference weight (for evaluation)
            pass
        else:
            self.current_preference_weight = self.np_random.random()

        delattr(self, '_in_reset')
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        try:
            profiler.start_step()

            profiler.start_timer('action_processing')
            if np.isscalar(action) or (hasattr(action, 'size') and action.size == 1):
                try:
                    action_value = int(action.item()) if hasattr(action, 'item') else int(action)
                except Exception:
                    action_value = int(action)
                if not (0 <= action_value < self.action_space.n):
                    action_value = np.clip(action_value, 0, self.action_space.n - 1)
                density_action_idx = action_value // self._conifer_fraction_size
                conifer_fraction_idx = action_value % self._conifer_fraction_size
            else:
                action_value = int(action[0])
                density_action_idx = action_value // self._conifer_fraction_size
                conifer_fraction_idx = action_value % self._conifer_fraction_size

            if not (0 <= density_action_idx < len(self.DENSITY_ACTIONS)):
                density_action_idx = np.clip(density_action_idx, 0, len(self.DENSITY_ACTIONS) - 1)
            if not (0 <= conifer_fraction_idx < len(self.CONIFER_FRACTIONS)):
                conifer_fraction_idx = np.clip(conifer_fraction_idx, 0, len(self.CONIFER_FRACTIONS) - 1)

            delta_density = self.DENSITY_ACTIONS[density_action_idx]
            action_conifer_fraction = self.CONIFER_FRACTIONS[conifer_fraction_idx]

            # Lower floor so thinning actions are not frequently clipped
            safe_min_density = const.SAFE_MIN_DENSITY_THINNING
            max_thinning = self.stem_density - safe_min_density
            if delta_density < -max_thinning:
                delta_density = -max_thinning

            # Apply memory management to management history lists
            self.management_history['density_actions'].append(density_action_idx)
            self.management_history['density_actions'] = self._limit_history_size(self.management_history['density_actions'])
            
            self.management_history['mix_actions'].append(conifer_fraction_idx)
            self.management_history['mix_actions'] = self._limit_history_size(self.management_history['mix_actions'])
            
            self.management_history['density_changes'].append(delta_density)
            self.management_history['density_changes'] = self._limit_history_size(self.management_history['density_changes'])
            
            self.management_history['mix_changes'].append(action_conifer_fraction)
            self.management_history['mix_changes'] = self._limit_history_size(self.management_history['mix_changes'])
            original_density = self.stem_density
            profiler.end_timer('action_processing')

            profiler.start_timer('physics_simulation')
            annual_results = self.simulator.run_annual_cycle(
                current_stem_density=self.stem_density,
                current_biomass_carbon_kg_m2=self.biomass_carbon_kg_m2,
                current_soil_carbon_kg_m2=self.soil_carbon_kg_m2,
                density_change=delta_density,
                management_conifer_fraction=action_conifer_fraction,
                warming_penalty_factor=self.WARMING_PENALTY_FACTOR,
            )
            profiler.end_timer('physics_simulation')

            profiler.start_timer('state_updates')
            self.stem_density = annual_results['final_stem_density']
            self.biomass_carbon_kg_m2 = annual_results['final_biomass_carbon_kg_m2']
            self.soil_carbon_kg_m2 = annual_results['final_soil_carbon_kg_m2']
            self.conifer_fraction = annual_results['final_conifer_fraction']
            self.unclipped_stem_density = annual_results.get('unclipped_stem_density', self.stem_density)
            if self.stem_density >= MAX_STEMS_HA:
                self.consecutive_max_density_steps += 1
            else:
                self.consecutive_max_density_steps = 0

            asymmetric_thaw_reward = annual_results['asymmetric_thaw_reward']
            thaw_dd_year = annual_results['thaw_degree_days']
            net_carbon_change = annual_results['net_carbon_change_with_hwp']
            biomass_change = annual_results.get('biomass_carbon_change', 0.0)
            soil_change = annual_results.get('soil_carbon_change', 0.0)
            # Apply memory management to history lists
            self.carbon_history['biomass_changes'].append(biomass_change)
            self.carbon_history['biomass_changes'] = self._limit_history_size(self.carbon_history['biomass_changes'])
            
            self.carbon_history['soil_changes'].append(soil_change)
            self.carbon_history['soil_changes'] = self._limit_history_size(self.carbon_history['soil_changes'])
            
            self.carbon_history['total_changes'].append(net_carbon_change)
            self.carbon_history['total_changes'] = self._limit_history_size(self.carbon_history['total_changes'])
            
            self.carbon_history['gpp_values'].append(annual_results['total_gpp_kg_m2'])
            self.carbon_history['gpp_values'] = self._limit_history_size(self.carbon_history['gpp_values'])
            
            self.disturbance_history['fire_fractions'].append(annual_results['fire_mortality_fraction'])
            self.disturbance_history['fire_fractions'] = self._limit_history_size(self.disturbance_history['fire_fractions'])
            
            self.disturbance_history['insect_fractions'].append(annual_results['insect_mortality_fraction'])
            self.disturbance_history['insect_fractions'] = self._limit_history_size(self.disturbance_history['insect_fractions'])
            
            self.disturbance_history['drought_indices'].append(annual_results['final_drought_index'])
            self.disturbance_history['drought_indices'] = self._limit_history_size(self.disturbance_history['drought_indices'])
            self.last_mortality_stems = annual_results['natural_mortality_stems']
            self.last_recruitment_stems = annual_results['natural_recruitment_stems']
            self.last_density_change = annual_results['natural_mortality_stems'] - annual_results['natural_recruitment_stems']
            self.last_fire_mortality_fraction = annual_results['fire_mortality_fraction']
            self.last_insect_mortality_fraction = annual_results['insect_mortality_fraction']
            self.last_drought_index = annual_results['final_drought_index']
            self.last_total_gpp_kg_m2 = annual_results['total_gpp_kg_m2']
            self.last_total_autotrophic_resp_kg_m2 = annual_results['total_autotrophic_resp_kg_m2']
            self.last_total_soil_resp_kg_m2 = annual_results['total_soil_resp_kg_m2']
            self.last_total_litterfall_kg_m2 = annual_results['total_litterfall_kg_m2']
            self.last_natural_mortality_biomass = annual_results['natural_mortality_biomass']
            self.last_fire_biomass_loss = annual_results['fire_biomass_loss']
            self.last_insect_biomass_loss = annual_results['insect_biomass_loss']
            self.last_total_mortality_losses = annual_results['total_mortality_losses']
            self.last_carbon_loss_thinning = annual_results['carbon_loss_thinning']
            self.last_hwp_carbon_stored = annual_results['hwp_carbon_stored']
            self.last_total_carbon_stock_with_hwp = annual_results['total_carbon_stock_with_hwp']
            profiler.end_timer('state_updates')

            profiler.start_timer('reward_calculation')
            self.cumulative_thaw_dd += thaw_dd_year
            normalized_carbon_change = np.clip(net_carbon_change / self.MAX_CARBON_CHANGE_PER_YEAR, -1.0, 1.0)
            normalized_asymmetric_thaw = np.clip(
                asymmetric_thaw_reward / self.MAX_THAW_DEGREE_DAYS_PER_YEAR, -1.0, 1.0
            )
            total_carbon_stock = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
            stock_bonus = self.STOCK_BONUS_MULTIPLIER * np.clip(
                total_carbon_stock / self.MAX_TOTAL_CARBON, 0.0, 1.0
            )
            biomass_excess = annual_results.get('biomass_excess_before_clip', 0.0)
            soil_excess = annual_results.get('soil_excess_before_clip', 0.0)
            biomass_penalty = (biomass_excess / self.MAX_BIOMASS_CARBON_LIMIT) * self.CARBON_LIMIT_PENALTY
            soil_penalty = (soil_excess / self.MAX_SOIL_CARBON_LIMIT) * self.CARBON_LIMIT_PENALTY
            limit_penalty = biomass_penalty + soil_penalty
            stock_bonus -= limit_penalty
            self.last_biomass_limit_violation = biomass_excess > 0
            self.last_soil_limit_violation = soil_excess > 0
            self.last_biomass_excess = biomass_excess
            self.last_soil_excess = soil_excess
            self.last_limit_penalty = limit_penalty
            self.last_normalized_carbon_change = normalized_carbon_change
            self.last_stock_bonus = stock_bonus
            self.last_base_stock_bonus = self.STOCK_BONUS_MULTIPLIER * np.clip(
                total_carbon_stock / self.MAX_TOTAL_CARBON, 0.0, 1.0
            )
            self.last_biomass_penalty = biomass_penalty
            self.last_soil_penalty = soil_penalty
            self.last_asymmetric_thaw_reward = asymmetric_thaw_reward
            self.last_normalized_asymmetric_thaw = normalized_asymmetric_thaw
            self.last_positive_flux_sum = annual_results.get('positive_flux_sum', 0.0)
            self.last_negative_flux_sum = annual_results.get('negative_flux_sum', 0.0)
            # Remove explicit HWP sales bonus from shaping; keep for logging only
            hwp_sold_kg_m2 = annual_results.get('hwp_carbon_stored', 0.0)
            normalized_hwp_sales = np.clip(hwp_sold_kg_m2 / self.MAX_HWP_SALES_PER_YEAR, 0.0, 1.0)
            hwp_sale_reward = self.HWP_SALE_REWARD_MULTIPLIER*normalized_hwp_sales
            self.last_hwp_sale_reward = hwp_sale_reward
            self.last_normalized_hwp_sales = normalized_hwp_sales
            self.episode_positive_flux_sum += self.last_positive_flux_sum
            self.episode_negative_flux_sum += self.last_negative_flux_sum
            if limit_penalty > 0:
                self.consecutive_carbon_penalty_steps += 1
            else:
                self.consecutive_carbon_penalty_steps = 0
            raw_carbon_component = normalized_carbon_change + stock_bonus + hwp_sale_reward
            raw_thaw_component = normalized_asymmetric_thaw
            reward_vector = np.array([raw_carbon_component, raw_thaw_component], dtype=np.float32)
            # Save raw (pre-standardization) for logging/analysis
            self.last_raw_reward_vector = reward_vector.copy()
            # Standardize to roughly zero-mean, unit-ish variance for learning stability
            if self.enable_reward_standardization:
                if self._reward_count > 10:  # warmup before using stats
                    std = np.sqrt(self._reward_var + self._eps)
                    reward_vector = np.clip((reward_vector - self._reward_mean) / std, -5.0, 5.0)
                # Update EMA statistics with raw rewards
                self._reward_mean = (
                    self.reward_ema_beta * self._reward_mean + (1 - self.reward_ema_beta) * self.last_raw_reward_vector
                )
                diff = self.last_raw_reward_vector - self._reward_mean
                self._reward_var = (
                    self.reward_ema_beta * self._reward_var + (1 - self.reward_ema_beta) * (diff * diff)
                )
                self._reward_count += 1
            max_density_penalty = 0.0
            if self.stem_density >= MAX_STEMS_HA:
                max_density_penalty = self.MAX_DENSITY_PENALTY
                reward_vector[0] -= self.MAX_DENSITY_PENALTY
            self.last_max_density_penalty = max_density_penalty
            self.last_ineffective_thinning = False
            self.last_ineffective_planting = False
            self.last_thinning_penalty = 0.0
            self.last_planting_penalty = 0.0
            if delta_density < 0:
                age_dist = self.simulator.age_distribution
                available_old_trees = age_dist['conifer']['old'] + age_dist['deciduous']['old']
                requested_thinning = abs(delta_density)
                if available_old_trees == 0:
                    thinning_penalty = self.INEFFECTIVE_THINNING_PENALTY
                    self.last_ineffective_thinning = True
                elif available_old_trees < requested_thinning:
                    wasted_thinning = requested_thinning - available_old_trees
                    penalty_fraction = wasted_thinning / requested_thinning
                    thinning_penalty = penalty_fraction * self.INEFFECTIVE_THINNING_PENALTY
                    self.last_ineffective_thinning = True
                else:
                    thinning_penalty = 0.0
                if thinning_penalty > 0:
                    reward_vector[0] -= thinning_penalty
                    self.last_thinning_penalty = thinning_penalty
            if delta_density > 0:
                requested_planting = delta_density
                available_space = MAX_STEMS_HA - original_density
                if available_space <= 0:
                    planting_penalty = self.INEFFECTIVE_PLANTING_PENALTY
                    self.last_ineffective_planting = True
                elif available_space < requested_planting:
                    wasted_planting = requested_planting - available_space
                    penalty_fraction = wasted_planting / requested_planting
                    planting_penalty = penalty_fraction * self.INEFFECTIVE_PLANTING_PENALTY
                    self.last_ineffective_planting = True
                else:
                    planting_penalty = 0.0
                if planting_penalty > 0:
                    reward_vector[0] -= planting_penalty
                    self.last_planting_penalty = planting_penalty
            profiler.end_timer('reward_calculation')

            profiler.start_timer('termination_checks')
            self.year += 1
            truncated = self.year >= self.EPISODE_LENGTH_YEARS
            total_carbon = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
            ecological_failure = total_carbon < 1.0
            density_crash = self.unclipped_stem_density < MIN_STEMS_HA
            max_density_truncation = self.consecutive_max_density_steps >= 5
            terminated = ecological_failure
            # Global cap on total training steps across env instances (optional safety)
            max_steps_env = int(os.environ.get('BOREARL_MAX_TOTAL_STEPS', '0'))

            if max_steps_env > 0:
                total_steps_so_far = int(os.environ.get('BOREARL_GLOBAL_STEP_COUNT', '0'))
                total_steps_so_far += 1
                os.environ['BOREARL_GLOBAL_STEP_COUNT'] = str(total_steps_so_far)
                if total_steps_so_far >= max_steps_env:
                    truncated = True
            truncated = truncated or density_crash or max_density_truncation
            if terminated:
                reward_vector += np.array([-1.0, -1.0])
            profiler.end_timer('termination_checks')

            profiler.start_timer('episode_tracking')
            # Apply memory management to episode tracking lists
            self.current_episode_rewards.append(reward_vector)
            self.current_episode_rewards = self._limit_history_size(self.current_episode_rewards, max_size=const.MAX_EPISODE_HISTORY_SIZE)
            
            self.current_episode_carbon_rewards.append(reward_vector[0])
            self.current_episode_carbon_rewards = self._limit_history_size(self.current_episode_carbon_rewards, max_size=const.MAX_EPISODE_HISTORY_SIZE)
            
            self.current_episode_thaw_rewards.append(reward_vector[1])
            self.current_episode_thaw_rewards = self._limit_history_size(self.current_episode_thaw_rewards, max_size=const.MAX_EPISODE_HISTORY_SIZE)
            
            self.current_episode_actions.append(action)
            self.current_episode_actions = self._limit_history_size(self.current_episode_actions, max_size=const.MAX_EPISODE_HISTORY_SIZE)
            
            self.current_episode_states.append({
                'year': self.year,
                'stem_density': self.stem_density,
                'conifer_fraction': self.conifer_fraction,
                'biomass_carbon': self.biomass_carbon_kg_m2,
                'soil_carbon': self.soil_carbon_kg_m2,
                'total_carbon': self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2,
            })
            self.current_episode_states = self._limit_history_size(self.current_episode_states, max_size=const.MAX_EPISODE_HISTORY_SIZE)
            profiler.end_timer('episode_tracking')

            profiler.start_timer('episode_statistics')
            if terminated or truncated:
                episode_time = profiler.end_episode()
                self._log_episode_metrics(terminated, truncated)
            profiler.end_timer('episode_statistics')

            profiler.start_timer('csv_logging')
            info = self._get_info()
            self._log_step_metrics(action, reward_vector, info)
            profiler.end_timer('csv_logging')

            profiler.end_step()
            return self._get_obs(), reward_vector, terminated, truncated, info
        except Exception as e:
            print(f"Error in ForestEnv.step: {e}")
            profiler.end_step()
            reward_vector = np.array([-1.0, -1.0], dtype=np.float32)
            return self._get_obs(), reward_vector, True, False, self._get_info()

    def get_episode_statistics(self) -> Dict[str, float] | None:
        if not self.current_episode_rewards:
            return None
        episode_rewards = np.array(self.current_episode_rewards)
        episode_carbon_rewards = np.array(self.current_episode_carbon_rewards)
        episode_thaw_rewards = np.array(self.current_episode_thaw_rewards)
        total_carbons = [state['total_carbon'] for state in self.current_episode_states]
        biomass_carbons = [state['biomass_carbon'] for state in self.current_episode_states]
        soil_carbons = [state['soil_carbon'] for state in self.current_episode_states]
        stem_densities = [state['stem_density'] for state in self.current_episode_states]
        conifer_fractions = [state['conifer_fraction'] for state in self.current_episode_states]
        # Compute total scalarized reward using the episode's preference weight
        # Use the current preference weight that was set during evaluation
        pref = float(getattr(self, 'current_preference_weight', 0.5))
        total_scalarized_reward = float(np.sum(pref * episode_carbon_rewards + (1.0 - pref) * episode_thaw_rewards))

        stats = {
            'episode_number': self.episode_count,
            'total_steps': len(episode_rewards),
            'episode_length_years': self.year,
            'total_carbon_reward': float(np.sum(episode_carbon_rewards)),
            'total_thaw_reward': float(np.sum(episode_thaw_rewards)),
            'total_scalarized_reward': total_scalarized_reward,
            'avg_carbon_reward': float(np.mean(episode_carbon_rewards)),
            'avg_thaw_reward': float(np.mean(episode_thaw_rewards)),
            'carbon_reward_std': float(np.std(episode_carbon_rewards)),
            'thaw_reward_std': float(np.std(episode_thaw_rewards)),
            'final_total_carbon': float(total_carbons[-1]) if total_carbons else 0.0,
            'final_biomass_carbon': float(biomass_carbons[-1]) if biomass_carbons else 0.0,
            'final_soil_carbon': float(soil_carbons[-1]) if soil_carbons else 0.0,
            'final_stem_density': float(stem_densities[-1]) if stem_densities else 0.0,
            'final_conifer_fraction': float(conifer_fractions[-1]) if conifer_fractions else 0.0,
            'avg_total_carbon': float(np.mean(total_carbons)) if total_carbons else 0.0,
            'avg_biomass_carbon': float(np.mean(biomass_carbons)) if biomass_carbons else 0.0,
            'avg_soil_carbon': float(np.mean(soil_carbons)) if soil_carbons else 0.0,
            'avg_stem_density': float(np.mean(stem_densities)) if stem_densities else 0.0,
            'avg_conifer_fraction': float(np.mean(conifer_fractions)) if conifer_fractions else 0.0,
            'net_carbon_gain': float(total_carbons[-1] - total_carbons[0]) if len(total_carbons) > 1 else 0.0,
            'carbon_efficiency': float((total_carbons[-1] - total_carbons[0]) / len(episode_rewards)) if len(episode_rewards) > 0 and len(total_carbons) > 1 else 0.0,
            'total_positive_flux': float(self.episode_positive_flux_sum),
            'total_negative_flux': float(self.episode_negative_flux_sum),
            'net_asymmetric_reward': float(np.sum(episode_thaw_rewards)),
            'avg_asymmetric_thaw_reward': float(np.mean(episode_thaw_rewards)) if len(episode_thaw_rewards) > 0 else 0.0,
            'asymmetric_thaw_reward_std': float(np.std(episode_thaw_rewards)) if len(episode_thaw_rewards) > 0 else 0.0,
            'cumulative_asymmetric_thaw_reward': float(np.sum(episode_thaw_rewards)),
        }
        return stats



    def _normalize_param(self, key: str, value: float) -> float:
        """Normalize a simulator parameter to [0,1] using config ranges if available."""
        try:
            cfg = getattr(self.simulator, 'config', None)
            if cfg is not None:
                rng_key = f"{key}_range"
                if rng_key in cfg:
                    lo, hi = cfg[rng_key]
                    if hi > lo:
                        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        except Exception:
            pass
        # Fallback: best-effort clip
        return float(np.clip(value, 0.0, 1.0))

    def _get_obs(self):
        norm_year = self.year / self.EPISODE_LENGTH_YEARS
        norm_density = (self.stem_density - MIN_STEMS_HA) / (MAX_STEMS_HA - MIN_STEMS_HA)
        total_carbon = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
        norm_carbon = total_carbon / self.MAX_TOTAL_CARBON
        lat_norm = (self.simulator.p['latitude_deg'] - self.LATITUDE_MIN) / self.LATITUDE_RANGE
        temp_mean_norm = (self.simulator.p['T_annual_mean_offset'] + self.TEMP_MEAN_OFFSET) / self.TEMP_MEAN_RANGE
        temp_amp_norm = self.simulator.p['T_seasonal_amplitude'] / self.TEMP_AMP_MAX
        growth_day_norm = self.simulator.p['growth_day'] / self.DAYS_PER_YEAR
        fall_day_norm = self.simulator.p['fall_day'] / self.DAYS_PER_YEAR
        growing_season_norm = (self.simulator.p['fall_day'] - self.simulator.p['growth_day']) / self.MAX_GROWING_SEASON
        fire_history = self._get_history_window(self.disturbance_history['fire_fractions'], 2)
        insect_history = self._get_history_window(self.disturbance_history['insect_fractions'], 2)
        drought_history = self._get_history_window(self.disturbance_history['drought_indices'], 2)
        drought_history = [np.clip(d / self.MAX_DROUGHT_INDEX, 0.0, 1.0) for d in drought_history]
        last_biomass_change = self.carbon_history['biomass_changes'][-1] if self.carbon_history['biomass_changes'] else 0.0
        last_soil_change = self.carbon_history['soil_changes'][-1] if self.carbon_history['soil_changes'] else 0.0
        last_total_change = self.carbon_history['total_changes'][-1] if self.carbon_history['total_changes'] else 0.0
        last_natural_mortality_biomass = getattr(self, 'last_natural_mortality_biomass', 0.0)
        last_litterfall = getattr(self, 'last_total_litterfall_kg_m2', 0.0)
        last_carbon_loss_thinning = getattr(self, 'last_carbon_loss_thinning', 0.0)
        last_hwp_carbon_stored = getattr(self, 'last_hwp_carbon_stored', 0.0)
        recent_biomass_change = (last_biomass_change + self.MAX_BIOMASS_CHANGE) / (2 * self.MAX_BIOMASS_CHANGE)
        recent_soil_change = (last_soil_change + self.MAX_SOIL_CHANGE) / (2 * self.MAX_SOIL_CHANGE)
        recent_total_change = (last_total_change + self.MAX_TOTAL_CHANGE) / (2 * self.MAX_TOTAL_CHANGE)
        recent_natural_mortality = last_natural_mortality_biomass / self.MAX_BIOMASS_CHANGE
        recent_litterfall = last_litterfall / self.MAX_GPP
        recent_thinning_loss = (last_carbon_loss_thinning + self.MAX_BIOMASS_CHANGE) / (2 * self.MAX_BIOMASS_CHANGE)
        recent_hwp_stored = last_hwp_carbon_stored / self.MAX_BIOMASS_CHANGE
        max_density_idx = len(self.DENSITY_ACTIONS) - 1
        last_density_action = self.management_history['density_actions'][-1] if self.management_history['density_actions'] else 0
        last_mix_action = self.management_history['mix_actions'][-1] if self.management_history['mix_actions'] else 0
        last_density_change = self.management_history['density_changes'][-1] if self.management_history['density_changes'] else 0.0
        last_mix_change = self.management_history['mix_changes'][-1] if self.management_history['mix_changes'] else 0.0
        recent_density_action = last_density_action / max_density_idx if max_density_idx > 0 else 0.0
        recent_mix_action = last_mix_action / max(1, (len(self.CONIFER_FRACTIONS) - 1))
        recent_density_change = (last_density_change + self.MAX_DENSITY_CHANGE) / (2 * self.MAX_DENSITY_CHANGE)
        recent_mix_change = last_mix_change
        age_dist = self.simulator.age_distribution
        total_conifer = sum(age_dist['conifer'].values())
        total_deciduous = sum(age_dist['deciduous'].values())
        total_stems = total_conifer + total_deciduous
        conifer_age_fractions = []
        deciduous_age_fractions = []
        if total_stems > 0:
            for age_class in ['seedling', 'sapling', 'young', 'mature', 'old']:
                conifer_fraction = age_dist['conifer'][age_class] / total_stems
                deciduous_fraction = age_dist['deciduous'][age_class] / total_stems
                conifer_age_fractions.append(conifer_fraction)
                deciduous_age_fractions.append(deciduous_fraction)
        else:
            conifer_age_fractions = [0.0] * 5
            deciduous_age_fractions = [0.0] * 5
        norm_biomass_stock = self.biomass_carbon_kg_m2 / self.MAX_TOTAL_CARBON
        norm_soil_stock = self.soil_carbon_kg_m2 / self.MAX_TOTAL_CARBON
        last_biomass_penalty = getattr(self, 'last_biomass_penalty', 0.0)
        last_soil_penalty = getattr(self, 'last_soil_penalty', 0.0)
        last_max_density_penalty = getattr(self, 'last_max_density_penalty', 0.0)
        norm_biomass_penalty = last_biomass_penalty / self.CARBON_LIMIT_PENALTY if self.CARBON_LIMIT_PENALTY > 0 else 0.0
        norm_soil_penalty = last_soil_penalty / self.CARBON_LIMIT_PENALTY if self.CARBON_LIMIT_PENALTY > 0 else 0.0
        norm_max_density_penalty = last_max_density_penalty / self.MAX_DENSITY_PENALTY if self.MAX_DENSITY_PENALTY > 0 else 0.0
        preference_weight = self.current_preference_weight
        base_obs = [
            np.clip(norm_year, 0.0, 1.0),
            np.clip(norm_density, 0.0, 1.0),
            np.clip(self.conifer_fraction, 0.0, 1.0),
            np.clip(norm_carbon, 0.0, 1.0),
            np.clip(lat_norm, 0.0, 1.0),
            np.clip(temp_mean_norm, 0.0, 1.0),
            np.clip(temp_amp_norm, 0.0, 1.0),
            np.clip(growth_day_norm, 0.0, 1.0),
            np.clip(fall_day_norm, 0.0, 1.0),
            np.clip(growing_season_norm, 0.0, 1.0),
            *fire_history,
            *insect_history,
            *drought_history,
            np.clip(recent_biomass_change, 0.0, 1.0),
            np.clip(recent_soil_change, 0.0, 1.0),
            np.clip(recent_total_change, 0.0, 1.0),
            np.clip(recent_natural_mortality, 0.0, 1.0),
            np.clip(recent_litterfall, 0.0, 1.0),
            np.clip(recent_thinning_loss, 0.0, 1.0),
            np.clip(recent_hwp_stored, 0.0, 1.0),
            np.clip(recent_density_action, 0.0, 1.0),
            np.clip(recent_mix_action, 0.0, 1.0),
            np.clip(recent_density_change, 0.0, 1.0),
            np.clip(recent_mix_change, 0.0, 1.0),
            *[np.clip(f, 0.0, 1.0) for f in conifer_age_fractions],
            *[np.clip(f, 0.0, 1.0) for f in deciduous_age_fractions],
            np.clip(norm_biomass_stock, 0.0, 1.0),
            np.clip(norm_soil_stock, 0.0, 1.0),
            np.clip(norm_biomass_penalty, 0.0, 1.0),
            np.clip(norm_soil_penalty, 0.0, 1.0),
            np.clip(norm_max_density_penalty, 0.0, 1.0),
            preference_weight,
        ]
        obs = np.asarray(base_obs, dtype=np.float32)

        # Append episode parameter context for generalist runs
        if not self.site_specific and getattr(self, 'include_site_params_in_obs', self.INCLUDE_SITE_PARAMS_IN_OBS):
            try:
                p = self.simulator.p if self.simulator is not None else {}
                ctx_vals = []
                keys = getattr(self, '_obs_param_keys', self.OBS_PARAM_KEYS)
                for k in keys:
                    v = float(p.get(k, 0.0)) if isinstance(p, dict) else 0.0
                    ctx_vals.append(self._normalize_param(k, v))
                if ctx_vals:
                    obs = np.concatenate([obs, np.asarray(ctx_vals, dtype=np.float32)], axis=0)
            except Exception:
                # Ensure observation length matches declared space by appending zeros
                keys = getattr(self, '_obs_param_keys', self.OBS_PARAM_KEYS)
                zeros_ctx = np.zeros(len(keys), dtype=np.float32)
                obs = np.concatenate([obs, zeros_ctx], axis=0)
        return obs

    def _get_history_window(self, history_list, window_size):
        if len(history_list) >= window_size:
            return history_list[-window_size:]
        else:
            return history_list + [0.0] * (window_size - len(history_list))

    def _limit_history_size(self, history_list, max_size=None):
        """
        Limit the size of a history list to prevent memory leaks.
        Keeps the most recent entries and discards older ones.
        
        Args:
            history_list: The list to limit
            max_size: Maximum number of entries to keep (default: uses constants)
            
        Returns:
            The limited history list
        """
        if max_size is None:
            max_size = const.MAX_HISTORY_SIZE
        if len(history_list) > max_size:
            return history_list[-max_size:]
        return history_list

    def _get_recent_average(self, history_list, window_size):
        if len(history_list) == 0:
            return 0.0
        recent_values = history_list[-window_size:] if len(history_list) >= window_size else history_list
        return sum(recent_values) / len(recent_values) if recent_values else 0.0

    def _get_info(self):
        if not self.simulator:
            return {
                "year": self.year,
                "error": "Simulator not initialized",
                "stem_density_ha": self.stem_density,
                "conifer_fraction": self.conifer_fraction,
                "biomass_carbon_kg_m2": self.biomass_carbon_kg_m2,
                "soil_carbon_kg_m2": self.soil_carbon_kg_m2,
                "carbon_stock_kg_m2": self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2,
                "cumulative_thaw_dd": self.cumulative_thaw_dd,
                "hwp_sale_reward": 0.0,
                "normalized_hwp_sales": 0.0,
                "raw_carbon_component": self.last_raw_reward_vector[0] if hasattr(self, 'last_raw_reward_vector') else 0.0,
                "raw_thaw_component": self.last_raw_reward_vector[1] if hasattr(self, 'last_raw_reward_vector') else 0.0,
            }
        total_carbon = self.biomass_carbon_kg_m2 + self.soil_carbon_kg_m2
        age_dist = self.simulator.age_distribution
        age_info = {}
        for species in ['conifer', 'deciduous']:
            for age_class in ['seedling', 'sapling', 'young', 'mature', 'old']:
                key = f"{species}_{age_class}_stems"
                age_info[key] = age_dist[species][age_class]
        return {
            "year": self.year,
            "stem_density_ha": self.stem_density,
            "conifer_fraction": self.conifer_fraction,
            "biomass_carbon_kg_m2": self.biomass_carbon_kg_m2,
            "soil_carbon_kg_m2": self.soil_carbon_kg_m2,
            "carbon_stock_kg_m2": total_carbon,
            "cumulative_thaw_dd": self.cumulative_thaw_dd,
            "latitude_deg": self.simulator.p['latitude_deg'],
            "mean_temp_c": self.simulator.p['T_annual_mean_offset'],
            "temp_amplitude_c": self.simulator.p['T_seasonal_amplitude'],
            "growing_season_days": self.simulator.p['fall_day'] - self.simulator.p['growth_day'],
            "natural_mortality_stems": self.last_mortality_stems,
            "natural_recruitment_stems": self.last_recruitment_stems,
            "natural_density_change": self.last_density_change,
            "fire_mortality_fraction": self.last_fire_mortality_fraction,
            "insect_mortality_fraction": self.last_insect_mortality_fraction,
            "final_drought_index": self.last_drought_index,
            **age_info,
            "total_gpp_kg_m2": getattr(self, 'last_total_gpp_kg_m2', 0.0),
            "total_autotrophic_resp_kg_m2": getattr(self, 'last_total_autotrophic_resp_kg_m2', 0.0),
            "total_soil_resp_kg_m2": getattr(self, 'last_total_soil_resp_kg_m2', 0.0),
            "total_litterfall_kg_m2": getattr(self, 'last_total_litterfall_kg_m2', 0.0),
            "natural_mortality_biomass": getattr(self, 'last_natural_mortality_biomass', 0.0),
            "fire_biomass_loss": getattr(self, 'last_fire_biomass_loss', 0.0),
            "insect_biomass_loss": getattr(self, 'last_insect_biomass_loss', 0.0),
            "total_mortality_losses": getattr(self, 'last_total_mortality_losses', 0.0),
            "carbon_loss_thinning": getattr(self, 'last_carbon_loss_thinning', 0.0),
            "hwp_carbon_stored": getattr(self, 'last_hwp_carbon_stored', 0.0),
            "total_carbon_stock_with_hwp": getattr(self, 'last_total_carbon_stock_with_hwp', total_carbon),
            "disturbance_history_length": len(self.disturbance_history['fire_fractions']),
            "management_history_length": len(self.management_history['density_actions']),
            "carbon_history_length": len(self.carbon_history['biomass_changes']),
            "ineffective_thinning": self.last_ineffective_thinning,
            "ineffective_planting": self.last_ineffective_planting,
            "thinning_penalty": getattr(self, 'last_thinning_penalty', 0.0),
            "planting_penalty": getattr(self, 'last_planting_penalty', 0.0),
            "biomass_limit_violation": self.last_biomass_limit_violation,
            "soil_limit_violation": self.last_soil_limit_violation,
            "biomass_excess": self.last_biomass_excess,
            "soil_excess": self.last_soil_excess,
            "limit_penalty": self.last_limit_penalty,
            "normalized_carbon_change": getattr(self, 'last_normalized_carbon_change', 0.0),
            "stock_bonus": getattr(self, 'last_stock_bonus', 0.0),
            "base_stock_bonus": getattr(self, 'last_base_stock_bonus', 0.0),
            "biomass_penalty": getattr(self, 'last_biomass_penalty', 0.0),
            "soil_penalty": getattr(self, 'last_soil_penalty', 0.0),
            "total_carbon_limit_penalty": getattr(self, 'last_limit_penalty', 0.0),
            "hwp_sale_reward": getattr(self, 'last_hwp_sale_reward', 0.0),
            "normalized_hwp_sales": getattr(self, 'last_normalized_hwp_sales', 0.0),
            "max_density_penalty": getattr(self, 'last_max_density_penalty', 0.0),
            "asymmetric_thaw_reward": getattr(self, 'last_asymmetric_thaw_reward', 0.0),
            "normalized_asymmetric_thaw": getattr(self, 'last_normalized_asymmetric_thaw', 0.0),
            "positive_flux_sum": getattr(self, 'last_positive_flux_sum', 0.0),
            "negative_flux_sum": getattr(self, 'last_negative_flux_sum', 0.0),
            "raw_carbon_component": getattr(self, 'last_raw_reward_vector', np.array([0.0, 0.0]))[0],
            "raw_thaw_component": getattr(self, 'last_raw_reward_vector', np.array([0.0, 0.0]))[1],
        }

    def close(self):
        pass

    def set_preference_weight(self, weight):
        self.current_preference_weight = weight
        self._preference_was_set_externally = True

    def __setattr__(self, name, value):
        if name == 'current_preference_weight':
            if hasattr(self, '_preference_was_set_externally') and not hasattr(self, '_in_reset'):
                self._preference_was_set_externally = True
        super().__setattr__(name, value)


try:
    gym.register(
        id="ForestEnv-v0",
        entry_point="borearl.env.forest_env:ForestEnv",
        max_episode_steps=const.EPISODE_LENGTH_YEARS,
    )
except Exception:
    pass


