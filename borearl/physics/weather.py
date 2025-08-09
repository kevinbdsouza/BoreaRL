from typing import Any, Dict, Tuple
import numpy as np


def esat_kPa(T: float) -> float:
    return 0.6108 * np.exp(17.27 * (T - 273.15) / ((T - 273.15) + 237.3))


def delta_svp_kPa_per_K(T: float) -> float:
    es = esat_kPa(T)
    return 4098.0 * es / ((T - 273.15) + 237.3) ** 2


def calculate_temperature_dependent_precipitation(T_for_precip: float, base_precip_mm: float, season_type: str, p: Dict) -> float:
    T_celsius = T_for_precip - 273.15
    if season_type == 'summer':
        base_temp = p.get('temp_precip_summer_base_temp', 15.0)
        sensitivity = p.get('temp_precip_summer_sensitivity', 0.07)
        temp_factor = np.exp(sensitivity * (T_celsius - base_temp))
        temp_factor = np.clip(temp_factor, 0.5, 2.5)
    elif season_type == 'shoulder':
        base_temp = p.get('temp_precip_shoulder_base_temp', 8.0)
        sensitivity = p.get('temp_precip_shoulder_sensitivity', 0.05)
        temp_factor = np.exp(sensitivity * (T_celsius - base_temp))
        temp_factor = np.clip(temp_factor, 0.6, 2.0)
    elif season_type == 'winter':
        sensitivity = p.get('temp_precip_winter_sensitivity', 0.1)
        if T_celsius < 0:
            temp_factor = np.exp(sensitivity * T_celsius)
            temp_factor = np.clip(temp_factor, 0.1, 1.0)
        else:
            temp_factor = np.exp(0.03 * T_celsius)
            temp_factor = np.clip(temp_factor, 0.8, 1.5)
    else:
        temp_factor = 1.0
    return base_precip_mm * temp_factor


def calculate_rain_adjusted_diurnal_amplitude(base_diurnal_amplitude: float, rain_mm_day: float, p: Dict) -> float:
    rain_diurnal_sensitivity = p.get('rain_diurnal_sensitivity', 0.15)
    rain_diurnal_threshold = p.get('rain_diurnal_threshold', 1.0)
    max_diurnal_reduction = p.get('max_diurnal_reduction', 0.6)
    if rain_mm_day < rain_diurnal_threshold:
        return base_diurnal_amplitude
    rain_factor = min(rain_mm_day * rain_diurnal_sensitivity, max_diurnal_reduction)
    adjusted_amplitude = base_diurnal_amplitude * (1.0 - rain_factor)
    min_amplitude = p.get('min_diurnal_amplitude', 1.0)
    return max(adjusted_amplitude, min_amplitude)


def calculate_latitude_dependent_parameters(latitude_deg: float, rng: np.random.Generator) -> Dict[str, float]:
    base_temp_at_60N = -2.0
    temp_latitude_slope = -0.6
    base_T_annual_mean = base_temp_at_60N + temp_latitude_slope * (latitude_deg - 60.0)
    T_annual_mean_offset = base_T_annual_mean + rng.uniform(-2.0, 2.0)
    base_amplitude_at_60N = 22.0
    amplitude_latitude_slope = 0.3
    base_T_seasonal_amplitude = base_amplitude_at_60N + amplitude_latitude_slope * (latitude_deg - 60.0)
    T_seasonal_amplitude = base_T_seasonal_amplitude + rng.uniform(-3.0, 3.0)

    def day_length(day_of_year: float, latitude_deg: float) -> float:
        declination = -23.45 * np.cos(2 * np.pi * (day_of_year + 10) / 365.0)
        lat_rad = np.deg2rad(latitude_deg)
        decl_rad = np.deg2rad(declination)
        cos_sunset = -np.tan(lat_rad) * np.tan(decl_rad)
        cos_sunset = np.clip(cos_sunset, -1.0, 1.0)
        sunset_hour_angle = np.arccos(cos_sunset)
        return 24.0 * sunset_hour_angle / np.pi

    spring_equinox = 80
    fall_equinox = 266
    spring_day_length = day_length(spring_equinox, latitude_deg)
    fall_day_length = day_length(fall_equinox, latitude_deg)

    base_growth_day = 140
    growth_latitude_delay = 2.0
    growth_day = base_growth_day + growth_latitude_delay * max(0, latitude_deg - 60.0)
    temp_adjustment = max(0, -T_annual_mean_offset) * 2.0
    growth_day += temp_adjustment
    growth_day += rng.uniform(-5.0, 5.0)

    base_fall_day = 260
    fall_latitude_advance = 1.5
    fall_day = base_fall_day - fall_latitude_advance * max(0, latitude_deg - 60.0)
    temp_adjustment = max(0, -T_annual_mean_offset) * 1.5
    fall_day -= temp_adjustment
    fall_day += rng.uniform(-5.0, 5.0)

    min_growing_season = 60.0
    if fall_day - growth_day < min_growing_season:
        fall_day = growth_day + min_growing_season + rng.uniform(0, 10)
    fall_day = min(fall_day, 320)

    return {
        'T_annual_mean_offset': T_annual_mean_offset,
        'T_seasonal_amplitude': T_seasonal_amplitude,
        'growth_day': growth_day,
        'fall_day': fall_day,
        'latitude_deg': latitude_deg,
    }


def calculate_snow_season_dates(p: Dict[str, Any], rng: np.random.Generator) -> Tuple[int, int]:
    T_offset = p['T_annual_mean_offset']
    T_amplitude = p['T_seasonal_amplitude']
    cos_crossing = T_offset / T_amplitude
    if abs(cos_crossing) > 1.0:
        return 280, 120
    day_angle_crossing = np.arccos(cos_crossing)
    spring_day = 1 + (day_angle_crossing * 365.0) / (2 * np.pi)
    fall_day = 1 + ((2 * np.pi - day_angle_crossing) * 365.0) / (2 * np.pi)
    spring_variation = rng.uniform(-3, 3)
    fall_variation = rng.uniform(-3, 3)
    snow_season_end = int(np.clip(spring_day + spring_variation, 100, 150))
    snow_season_start = int(np.clip(fall_day + fall_variation, 260, 320))
    if snow_season_end > snow_season_start:
        snow_season_end, snow_season_start = snow_season_start, snow_season_end
        snow_season_end = int(np.clip(snow_season_end, 100, 150))
        snow_season_start = int(np.clip(snow_season_start, 260, 320))
    return snow_season_start, snow_season_end


def verify_temperature_snow_consistency(p: Dict[str, Any]):
    snow_season_start = p.get('snow_season_start')
    snow_season_end = p.get('snow_season_end')
    T_offset = p.get('T_annual_mean_offset')
    T_amplitude = p.get('T_seasonal_amplitude')
    if any(param is None for param in [snow_season_start, snow_season_end, T_offset, T_amplitude]):
        return
    day_angle_start = 2 * np.pi * (snow_season_start - 1) / 365.0
    T_at_start = 273.15 + T_offset - T_amplitude * np.cos(day_angle_start)
    day_angle_end = 2 * np.pi * (snow_season_end - 1) / 365.0
    T_at_end = 273.15 + T_offset - T_amplitude * np.cos(day_angle_end)
    tolerance = 3.0
    start_deviation = abs(T_at_start - 273.15)
    end_deviation = abs(T_at_end - 273.15)
    if start_deviation > tolerance or end_deviation > tolerance:
        pass


def _validate_phenology_snow_constraints(p: Dict[str, Any], rng: np.random.Generator):
    snow_season_end = p.get('snow_season_end')
    snow_season_start = p.get('snow_season_start')
    growth_day = p.get('growth_day')
    fall_day = p.get('fall_day')
    if any(param is None for param in [snow_season_end, snow_season_start, growth_day, fall_day]):
        return
    if growth_day <= snow_season_end:
        min_gap = 5.0
        p['growth_day'] = snow_season_end + min_gap + rng.uniform(0, 10)
    if fall_day >= snow_season_start:
        min_gap = 5.0
        p['fall_day'] = snow_season_start - min_gap - rng.uniform(0, 10)
    if fall_day <= growth_day:
        min_growing_season = 60.0
        new_fall_day = p['growth_day'] + min_growing_season + rng.uniform(0, 30)
        if new_fall_day >= snow_season_start:
            new_fall_day = snow_season_start - 5.0 - rng.uniform(0, 10)
        p['fall_day'] = new_fall_day


def get_stability_correction(z_ref: float, L: float) -> Tuple[float, float]:
    if abs(L) < 1e-6:
        L = -1e-6 if L < 0 else 1e-6
    zeta = z_ref / L
    if zeta >= 0:
        psi_m = -5.0 * zeta
        psi_h = -5.0 * zeta
    else:
        x = (1 - 16 * zeta) ** 0.25
        psi_m = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
        psi_h = 2 * np.log((1 + (1 - 16 * zeta) ** 0.5) / 2)
    return psi_m, psi_h


def h_aero(u: float, z_ref: float, z0: float, L: float, p: Dict) -> float:
    u = max(u, 0.1)
    _, psi_h = get_stability_correction(z_ref, L)
    ra_log_term = np.log(z_ref / z0) - psi_h
    if ra_log_term <= 0:
        ra_log_term = np.log(z_ref / z0)
    ra = ra_log_term**2 / (p['KAPPA']**2 * u)
    return p['RHO_AIR'] * p['CP_AIR'] / max(ra, 1.0)


def update_dynamic_parameters(p: Dict, day: int, hour: float, S: dict, L: float, rng: np.random.Generator):
    """
    Update parameters that vary at sub-daily timesteps.

    Generates daily stochastic temperature offsets and precipitation once per day
    and stores in p['daily_state'].
    """
    day_angle = 2 * np.pi * (day - 1) / 365.0

    daily_state = p.setdefault('daily_state', {})
    if daily_state.get('day') != day:
        temp_noise = rng.normal(0, p['T_daily_noise_std'])
        T_ls = 273.15 + p['T_annual_mean_offset'] - p['T_seasonal_amplitude'] * np.cos(day_angle) + temp_noise
        T_for_precip = T_ls - p['T_diurnal_amplitude'] * np.cos(2 * np.pi * (0 - p['T_hour_peak_diurnal']) / 24.0)

        rain_mm_day, snowfall_mm_day = 0.0, 0.0
        if T_for_precip > 274.15:
            if p['summer_day_start'] < day < p['summer_day_end']:
                if rng.random() < p['rain_summer_prob']:
                    temp_adjusted_precip = calculate_temperature_dependent_precipitation(
                        T_for_precip, p['rain_summer_mm_day'], 'summer', p
                    )
                    rain_mm_day = rng.exponential(temp_adjusted_precip)
            elif (p['shoulder_1_start'] < day < p['shoulder_1_end']) or (p['shoulder_2_start'] < day < p['shoulder_2_end']):
                if rng.random() < p['rain_shoulder_prob']:
                    temp_adjusted_precip = calculate_temperature_dependent_precipitation(
                        T_for_precip, p['rain_shoulder_mm_day'], 'shoulder', p
                    )
                    rain_mm_day = rng.exponential(temp_adjusted_precip)
        elif T_for_precip < 272.15 and (day > p['snow_season_start'] or day < p['snow_season_end']):
            if rng.random() < p['snow_winter_prob']:
                snowfall_mm_day = rng.exponential(p['winter_snow_mm_day'])

        daily_state.update({
            'day': day,
            'temp_noise': temp_noise,
            'rain_mm_day': rain_mm_day,
            'snow_mm_day': snowfall_mm_day,
        })

    temp_noise = daily_state['temp_noise']
    rain_mm_day = daily_state['rain_mm_day']
    snowfall_mm_day = daily_state['snow_mm_day']

    p["T_large_scale"] = 273.15 + p['T_annual_mean_offset'] - p['T_seasonal_amplitude'] * np.cos(day_angle) + temp_noise

    rain_adjusted_diurnal_amplitude = calculate_rain_adjusted_diurnal_amplitude(
        p['T_diurnal_amplitude'], rain_mm_day, p
    )
    p["T_atm"] = p["T_large_scale"] - rain_adjusted_diurnal_amplitude * np.cos(2 * np.pi * (hour - p['T_hour_peak_diurnal']) / 24.0)
    p['ea'] = p['mean_relative_humidity'] * esat_kPa(p['T_atm'])
    swc_frac = S['SWC_mm'] / p['SWC_max_mm']
    stress_range = 1.0 - p['soil_stress_threshold']
    p['soil_stress'] = np.clip((swc_frac - p['soil_stress_threshold']) / stress_range, 0.0, 1.0)

    p['rain_m_step'] = rain_mm_day / 1000.0 / p['STEPS_PER_DAY']
    p['snowfall_m_step'] = snowfall_mm_day / 1000.0 / p['STEPS_PER_DAY']

    decl = -23.45 * np.cos(2 * np.pi * (day + 10) / 365.0)
    cos_tz = (np.sin(np.deg2rad(p['latitude_deg'])) * np.sin(np.deg2rad(decl)) +
              np.cos(np.deg2rad(p['latitude_deg'])) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(15 * (hour - 12))))
    p["Q_solar"] = max(0.0, 1000.0 * cos_tz)

    leaf_on = 1 / (1 + np.exp(-p['growth_rate'] * (day - p['growth_day'])))
    leaf_off = 1 / (1 + np.exp(p['fall_rate'] * (day - p['fall_day'])))
    LAI_deciduous_actual = p["LAI_max_deciduous"] * leaf_on * leaf_off
    LAI_coniferous_actual = p['LAI_max_coniferous']
    p["LAI_actual"] = (LAI_coniferous_actual * p['coniferous_fraction']) + \
                      (LAI_deciduous_actual * (1.0 - p['coniferous_fraction']))
    total_woody_area = p['woody_area_index']
    p["LAI"] = p["LAI_actual"] + total_woody_area

    scaling_num = 1 - np.exp(-p['k_snow'] * p['LAI'])
    scaling_den = 1 - np.exp(-p['k_snow'] * (p['LAI_max'] + p['woody_area_index']))
    interception_scaling = scaling_num / scaling_den if scaling_den > 0 else 0.0
    p['rain_intercepted_m'] = p['rain_m_step'] * p['can_int_frac_rain'] * interception_scaling
    p['snow_intercepted_m'] = p['snowfall_m_step'] * p['can_int_frac_snow'] * interception_scaling
    p['evap_intercepted_rain_flux'] = (p['rain_intercepted_m'] * p['Lv'] * p['RHO_WATER']) / p['DT_SECONDS']

    p["K_can"] = np.exp(-p["k_ext"] * p["LAI"])
    p["alpha_can"] = p["alpha_can_base"] if p["LAI_actual"] > 0.1 else p["alpha_trunk"]
    p["A_can"] = p["A_can_max"] * (1 - np.exp(-p["k_ext"] * p["LAI"]))
    snow_frac = S['SWE'] / (S['SWE'] + p['SWE_SMOOTHING'])
    p["A_snow"] = (1.0 - p["A_trunk_plan"]) * snow_frac
    p["A_soil"] = 1.0 - p["A_trunk_plan"] - p["A_snow"]

    u = p['u_ref']
    p['h_can'] = max(h_aero(u, p['z_ref_h'], p['z0_can'], L, p), p['CANOPY_MIN_H'])
    h_soil_raw = h_aero(u, p['z_ref_soil'], p['z0_soil'], L, p)
    p['h_soil'] = max(h_soil_raw, 3.0) * (1 - snow_frac)
    p['h_snow'] = max(h_soil_raw, 3.0) * snow_frac * 0.5
    p["eps_atm"] = min(p['eps_atm_max'], p['eps_atm_coeff_a'] + p['eps_atm_coeff_b'] * np.tanh((p["T_atm"] - p['eps_atm_min_T']) / p['eps_atm_sensitivity']))
    return p

