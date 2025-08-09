from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from .config import get_model_config
from .constants import MIN_STEMS_HA, MAX_STEMS_HA
from .weather import (
    esat_kPa,
    delta_svp_kPa_per_K,
    calculate_temperature_dependent_precipitation,
    calculate_rain_adjusted_diurnal_amplitude,
    calculate_latitude_dependent_parameters,
    calculate_snow_season_dates,
    _validate_phenology_snow_constraints,
    update_dynamic_parameters,
    get_stability_correction,
    h_aero,
)
from .demography import calculate_natural_demography
from .utils import safe_update


def calculate_age_weighted_stand_attributes(age_distribution: dict, p: dict, 
                                           coniferous_fraction: float, stem_density: float) -> dict:
    age_dist = age_distribution
    total_canopy_area = 0.0
    total_lai = 0.0
    total_lue_weighted = 0.0
    canopy_factors = {
        'seedling': 0.1,
        'sapling': 0.3,
        'young': 0.7,
        'mature': 1.0,
        'old': 0.9,
    }
    lue_scaling_factors = {
        'seedling': 1.1,
        'sapling': 1.2,
        'young': 1.1,
        'mature': 0.9,
        'old': 0.7,
    }
    conifer_stems = sum(age_dist['conifer'].values())
    if conifer_stems > 0:
        for age_class, stems in age_dist['conifer'].items():
            factor = canopy_factors[age_class]
            lue_factor = lue_scaling_factors[age_class]
            total_canopy_area += stems * factor
            total_lai += stems * factor * p['LAI_max_conifer']
            total_lue_weighted += stems * lue_factor
    deciduous_stems = sum(age_dist['deciduous'].values())
    if deciduous_stems > 0:
        for age_class, stems in age_dist['deciduous'].items():
            factor = canopy_factors[age_class]
            lue_factor = lue_scaling_factors[age_class]
            total_canopy_area += stems * factor
            total_lai += stems * factor * p['LAI_max_deciduous']
            total_lue_weighted += stems * lue_factor
    total_stems = conifer_stems + deciduous_stems
    if total_stems > 0:
        avg_canopy_factor = total_canopy_area / total_stems
        stand_average_lai = total_lai / total_stems
        age_weighted_lue_factor = total_lue_weighted / total_stems
    else:
        avg_canopy_factor = 0.0
        stand_average_lai = 0.0
        age_weighted_lue_factor = 1.0
    return {
        'age_weighted_canopy_factor': avg_canopy_factor,
        'stand_average_lai': stand_average_lai,
        'age_weighted_lue_factor': age_weighted_lue_factor,
        'total_stems': total_stems,
    }


def get_baseline_parameters(
    p: Dict,
    coniferous_fraction: float,
    stem_density: float,
    rng: np.random.Generator,
    age_distribution: dict | None = None,
) -> Dict[str, Any]:
    _validate_phenology_snow_constraints(p, rng)
    p['coniferous_fraction'] = coniferous_fraction
    deciduous_fraction = 1.0 - coniferous_fraction
    u = p['u_ref']
    stem_density_per_m2 = stem_density / 10_000.0
    max_density_per_m2 = p['MAX_DENSITY_FOR_FULL_CANOPY'] / 10_000.0
    a_can_max_potential = 0.95
    age_params = None
    if age_distribution is not None:
        age_params = calculate_age_weighted_stand_attributes(age_distribution, p, coniferous_fraction, stem_density)
        age_weighted_canopy_factor = age_params['age_weighted_canopy_factor']
        density_canopy_factor = np.clip(stem_density_per_m2 / max_density_per_m2, 0.05, 1.0)
        combined_canopy_factor = density_canopy_factor * age_weighted_canopy_factor
        p['A_can_max'] = combined_canopy_factor * a_can_max_potential
    else:
        p['A_can_max'] = (
            np.clip(stem_density_per_m2 / max_density_per_m2, 0.05, 1.0) * a_can_max_potential
        )
    p['trunk_density_per_m2'] = stem_density_per_m2
    p['trunk_radius_m'] = np.sqrt(
        p['A_trunk_plan'] / (p['trunk_density_per_m2'] * np.pi + p['EPS'])
    )
    con_params = dict(
        alpha_can_base=p['alpha_can_base_conifer'], LAI_max=p['LAI_max_conifer'],
        k_ct=p['k_ct_base_con'], can_int_frac_rain=0.25, can_int_frac_snow=0.40
    )
    dec_params = dict(
        alpha_can_base=p['alpha_can_base_deciduous'], LAI_max=p['LAI_max_deciduous'],
        k_ct=p['k_ct_base_dec'], can_int_frac_rain=0.20, can_int_frac_snow=0.25
    )
    mixed_params = {}
    for key in con_params:
        mixed_params[key] = (con_params[key] * coniferous_fraction) + (dec_params[key] * deciduous_fraction)
    p.update(mixed_params)
    if age_params is not None:
        stand_average_lai = age_params['stand_average_lai']
        p['LAI_max'] = stand_average_lai
        base_lue = p['LUE_J_TO_G_C']
        p['LUE_J_TO_G_C'] = base_lue * age_params['age_weighted_lue_factor']
    else:
        p['LAI_max'] = (
            p['LAI_max_conifer'] * coniferous_fraction + p['LAI_max_deciduous'] * (1 - coniferous_fraction)
        )
    p['LAI_max_coniferous'] = p['LAI_max_conifer'] * (p['A_can_max'] / a_can_max_potential)
    p['LAI_max_deciduous'] = p['LAI_max_deciduous'] * (p['A_can_max'] / a_can_max_potential)
    p['h_trunk'] = p['h_trunk_const'] + p['h_trunk_wind_coeff'] * u
    p['k_ext'] = p['k_ext_factor']
    p['k_snow'] = p['k_ext'] * p['k_snow_factor']
    p['DT_SECONDS'] = p['TIME_STEP_MINUTES'] * 60
    p['STEPS_PER_DAY'] = int(24 * 60 / p['TIME_STEP_MINUTES'])
    p['J_PER_GC'] = 0.1 / p['LUE_J_TO_G_C']
    return p


def solve_canopy_energy_balance(T_guess: float, p: Dict, forcings: Dict, conduct: Dict, SWE_can: float) -> Tuple[float, Dict, float]:
    eps_can, A_can, h_can, k_ct, A_c2t, d_ct = p['eps_can'], p['A_can'], conduct['h_can'], p['k_ct'], p['A_c2t'], p['d_ct']
    PT_ALPHA, soil_stress = p['PT_ALPHA'], forcings['soil_stress']
    evap_intercepted_rain_flux = forcings['evap_intercepted_rain_flux']
    Q_abs_can, L_down_atm, L_up_grnd = forcings['Q_abs_can'], forcings['L_down_atm'], forcings['L_up_ground']
    T_trunk, T_air, ea = forcings['T_trunk'], forcings['T_air'], forcings['ea']

    def rnet(T):
        return Q_abs_can + A_can * (eps_can * (L_down_atm + L_up_grnd) - 2 * eps_can * p['SIGMA'] * T ** 4)

    def latent(T, Rn):
        if T <= 273.15 or Rn <= 0 or p['LAI_actual'] <= 0.1:
            return 0.0
        vpd = max(0.0, esat_kPa(T) - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        Delta = delta_svp_kPa_per_K(T)
        return PT_ALPHA * (Delta / (Delta + p['PSYCHROMETRIC_GAMMA'])) * Rn * vpd_stress * soil_stress

    def photosynthesis(T, Q_abs):
        if T <= 273.15 or Q_abs <= 0 or p['LAI_actual'] <= 0.1:
            return 0.0
        vpd = max(0.0, esat_kPa(T) - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        total_stress = vpd_stress * soil_stress
        par_abs = Q_abs * p['PAR_FRACTION']
        gpp = par_abs * p['LUE_J_TO_G_C'] * total_stress
        return gpp * p.get('gpp_scalar', 1.0)

    melt_energy_sink = 0.0
    if SWE_can > 0:
        T_freeze = 273.15
        Rn_at_freeze = rnet(T_freeze)
        H_at_freeze = h_can * (T_freeze - T_air)
        Cnd_at_freeze = k_ct * A_c2t / d_ct * (T_freeze - T_trunk)
        F_at_freeze = Rn_at_freeze - H_at_freeze - Cnd_at_freeze - evap_intercepted_rain_flux
        if F_at_freeze > 0:
            energy_to_melt_all = (SWE_can * p['Lf'] * p['RHO_WATER']) / p['DT_SECONDS']
            if F_at_freeze < energy_to_melt_all:
                flux_dict = {'Rnet_can': Rn_at_freeze, 'H_can': -H_at_freeze, 'LE_can': 0, 'G_photo_energy': 0, 'Cnd_can': -Cnd_at_freeze, 'Melt_flux_can': F_at_freeze, 'LE_int_rain': evap_intercepted_rain_flux}
                return T_freeze, flux_dict, 0.0
            else:
                melt_energy_sink = energy_to_melt_all

    T = np.clip(T_guess, p['T_MIN'] + 1.0, p['T_MAX'] - 1.0)
    gpp_g_m2_s = 0.0
    for _ in range(6):
        Rn = rnet(T)
        LE = latent(T, Rn)
        gpp_g_m2_s = photosynthesis(T, Q_abs_can)
        G_photo_energy = gpp_g_m2_s * p['J_PER_GC']
        H = h_can * (T - T_air)
        Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)
        F = Rn - H - LE - G_photo_energy - Cnd - melt_energy_sink - evap_intercepted_rain_flux
        if abs(F) < 1e-3:
            break
        dT = 0.1
        Rn_p, LE_p = rnet(T + dT), latent(T + dT, rnet(T + dT))
        gpp_p = photosynthesis(T + dT, Q_abs_can)
        G_p = gpp_p * p['J_PER_GC']
        H_p, Cnd_p = h_can * (T + dT - T_air), k_ct * A_c2t / d_ct * (T + dT - T_trunk)
        F_p = Rn_p - H_p - LE_p - G_p - Cnd_p - melt_energy_sink - evap_intercepted_rain_flux
        dF = (F_p - F) / dT
        if abs(dF) < 1e-4:
            dF = 1e-4
        T -= F / dF
        T = np.clip(T, p['T_MIN'] + 1.0, p['T_MAX'] - 1.0)
    Rn = rnet(T)
    LE = latent(T, Rn)
    H = h_can * (T - T_air)
    Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)
    gpp_g_m2_s = photosynthesis(T, Q_abs_can)
    G_photo_energy = gpp_g_m2_s * p['J_PER_GC']
    flux_dict = {'Rnet_can': Rn, 'H_can': -H, 'LE_can': -LE, 'G_photo_energy': -G_photo_energy, 'Cnd_can': -Cnd, 'Melt_flux_can': melt_energy_sink, 'LE_int_rain': evap_intercepted_rain_flux}
    return T, flux_dict, gpp_g_m2_s


def calculate_fluxes_and_melt(S: Dict, p: Dict) -> Tuple[Dict, float, float, float]:
    T_can_guess, T_trunk, T_snow = S['canopy'], S['trunk'], S['snow']
    T_soil_surf, T_soil_deep, T_air_model = S['soil_surf'], S['soil_deep'], S['atm_model']
    A_can, A_snow, A_soil = p['A_can'], p['A_snow'], p['A_soil']
    eps_can, eps_soil, eps_snow = p['eps_can'], p['eps_soil'], p['eps_snow']
    flux_report = {node: {} for node in S if 'SWE' not in node and 'SWC' not in node}
    lw = lambda T: p['SIGMA'] * T ** 4
    L_down_atm = p['eps_atm'] * lw(p['T_atm'])
    L_emit_soil, L_emit_snow = eps_soil * lw(T_soil_surf), eps_snow * lw(T_snow)
    L_up_ground = (A_soil * L_emit_soil) + (A_snow * L_emit_snow)
    Q_solar_on_canopy = p['Q_solar'] * p['A_can_max']
    Q_abs_can = Q_solar_on_canopy * (1 - p['K_can']) * (1 - p['alpha_can'])
    forcings = {'Q_abs_can': Q_abs_can, 'L_down_atm': L_down_atm, 'L_up_ground': L_up_ground, 'T_trunk': T_trunk, 'T_air': T_air_model, 'ea': p['ea'], 'soil_stress': p['soil_stress'], 'evap_intercepted_rain_flux': p['evap_intercepted_rain_flux']}
    T_can_step, can_flux, gpp_g_m2_s = solve_canopy_energy_balance(T_can_guess, p, forcings, {'h_can': p['h_can']}, S['SWE_can'])
    flux_report['canopy'] = {'Rnet': can_flux['Rnet_can'], 'H': can_flux['H_can'], 'LE_trans': can_flux['LE_can'], 'G_photo': can_flux.get('G_photo_energy', 0.0), 'Cnd_trunk': can_flux['Cnd_can'], 'Melt': -can_flux.get('Melt_flux_can', 0.0), 'LE_int_rain': -can_flux.get('LE_int_rain', 0.0)}
    Q_transmitted = Q_solar_on_canopy * p['K_can']
    Q_ground = p['Q_solar'] * (1 - p['A_can_max']) + Q_transmitted
    flux_report['soil_surf']['SW_in'] = A_soil * Q_ground * (1 - p['alpha_soil'])
    flux_report['snow']['SW_in'] = A_snow * Q_ground * (1 - p['alpha_snow'])
    L_emit_can = eps_can * lw(T_can_step) if A_can > 0 else 0.0
    gap_fraction = 1.0 - A_can
    L_down_transmitted_atm = (1 - eps_can) * L_down_atm
    LW_in_soil = A_soil * eps_soil * (gap_fraction*L_down_atm + A_can*(L_emit_can + L_down_transmitted_atm))
    LW_in_snow = A_snow * eps_snow * (gap_fraction*L_down_atm + A_can*(L_emit_can + L_down_transmitted_atm))
    flux_report['soil_surf']['LW_net'] = LW_in_soil - A_soil * L_emit_soil
    flux_report['snow']['LW_net'] = LW_in_snow - A_snow * L_emit_snow
    flux_report['snow']['Cnd_soil'] = 0.0
    flux_report['snow']['Cnd_trunk'] = 0.0
    flux_report['soil_surf']['Cnd_snow'] = 0.0
    flux_report['soil_surf']['Cnd_trunk'] = 0.0
    Rn_soil = flux_report['soil_surf']['SW_in'] + flux_report['soil_surf']['LW_net']
    LE_soil = 0.0
    if Rn_soil > 0 and A_soil > 0:
        Delta_soil = delta_svp_kPa_per_K(T_soil_surf)
        LE_soil = p['PT_ALPHA'] * (Delta_soil / (Delta_soil + p['PSYCHROMETRIC_GAMMA'])) * Rn_soil * p['soil_stress']
    flux_report['soil_surf']['LE_evap'] = -LE_soil
    k_soil = p['k_soil']
    flux_surf_deep = (k_soil / (0.5*(p['d_soil_surf']+p['d_soil_deep']))) * (T_soil_surf - T_soil_deep)
    flux_deep_bound = (k_soil / (0.5*p['d_soil_deep'])) * (T_soil_deep - p['T_deep_boundary'])
    flux_report['soil_surf']['Cnd_deep'] = -flux_surf_deep
    flux_report['soil_deep']['Cnd_surf'] = flux_surf_deep
    flux_report['soil_deep']['Cnd_boundary'] = -flux_deep_bound
    if p['A_snow'] > 0:
        snow_depth = (S['SWE'] * p['RHO_WATER']) / p['RHO_SNOW']
        R_soil = (0.5 * p['d_soil_surf']) / k_soil
        R_snow = (0.5 * snow_depth) / p['k_snow_pack']
        flux_soil_snow = (1/(R_soil+R_snow)) * p['A_snow'] * (T_soil_surf - T_snow) if (R_soil+R_snow) > 0 else 0.0
        flux_report['soil_surf']['Cnd_snow'] = -flux_soil_snow
        flux_report['snow']['Cnd_soil'] = flux_soil_snow
    H_trunk, H_soil, H_snow = p['h_trunk'] * (T_trunk - T_air_model), p['h_soil'] * (T_soil_surf - T_air_model), p['h_snow'] * (T_snow - T_air_model)
    flux_report['trunk']['H'], flux_report['soil_surf']['H'], flux_report['snow']['H'] = -H_trunk, -H_soil, -H_snow
    flux_report['atm_model']['H_can'], flux_report['atm_model']['H_trunk'], flux_report['atm_model']['H_soil'], flux_report['atm_model']['H_snow'] = flux_report['canopy']['H'], H_trunk, H_soil, H_snow
    LW_ground_to_atm = (gap_fraction * L_up_ground) + (A_can * (1 - eps_can) * L_up_ground)
    flux_report['atm_model']['LW_up'] = LW_ground_to_atm
    flux_report['atm_model']['Relax'] = (p['C_ATM']/p['tau_adv']) * (p['T_large_scale'] - T_air_model)
    flux_report['trunk']['Cnd_canopy'] = -flux_report['canopy']['Cnd_trunk']
    if S['SWE'] > 0:
        snow_depth = (S['SWE'] * p['RHO_WATER']) / p['RHO_SNOW']
        lat_area = p['trunk_density_per_m2'] * 2 * np.pi * p['trunk_radius_m'] * snow_depth
        flux_t_snow = (p['k_tsn']/p['d_tsn']) * lat_area * (T_trunk - T_snow)
        flux_report['trunk']['Cnd_ground'] = -flux_t_snow
        flux_report['snow']['Cnd_trunk'] = flux_t_snow
    else:
        flux_t_soil = (p['k_tso']/p['d_tso']) * p['A_trunk_plan'] * (T_trunk - T_soil_surf)
        flux_report['trunk']['Cnd_ground'] = -flux_t_soil
        flux_report['soil_surf']['Cnd_trunk'] = flux_t_soil
    d_SWE_melt_grd, d_SWE_melt_can = 0.0, 0.0
    if S['SWE_can'] > 0 and 'Melt' in flux_report['canopy']:
        d_SWE_melt_can = min((-flux_report['canopy']['Melt'] / (p['Lf'] * p['RHO_WATER'])) * p['DT_SECONDS'], S['SWE_can'])
    net_flux_snow = sum(flux_report.get('snow', {}).values())
    melt_energy_sink_grd = 0.0
    if S['SWE'] > 0 and T_snow >= 273.15 and net_flux_snow > 0:
        d_SWE_melt_grd = min((net_flux_snow / (p['Lf'] * p['RHO_WATER'])) * p['DT_SECONDS'], S['SWE'])
        melt_energy_sink_grd = net_flux_snow
    flux_report['snow']['Melt'] = -melt_energy_sink_grd
    flux_report['canopy']['T_new'] = T_can_step
    return flux_report, d_SWE_melt_grd, d_SWE_melt_can, gpp_g_m2_s


class ForestSimulator:
    def __init__(
        self,
        coniferous_fraction: float,
        stem_density: float,
        weather_seed: int,
        site_specific: bool = False,
        site_overrides: dict | None = None,
        deterministic_temp_noise: bool = False,
        remove_age_jitter: bool = False,
    ):
        self.rng = np.random.default_rng(weather_seed)
        self.site_specific = site_specific
        self.site_overrides = site_overrides or {}
        self.deterministic_temp_noise = deterministic_temp_noise
        self.remove_age_jitter = remove_age_jitter

        self.config = get_model_config()
        self.p = self._sample_parameters()
        # Force deterministic daily temperature noise if requested
        if self.deterministic_temp_noise:
            self.p['T_daily_noise_std'] = 0.0
        self.p = get_baseline_parameters(self.p, coniferous_fraction, stem_density, self.rng)
        self.S = {
            "canopy": 265.0, "trunk": 265.0, "snow": 268.0, "soil_surf": 270.0,
            "soil_deep": 270.0, "atm_model": 265.0, "SWE": 0.0, "SWE_can": 0.0,
            "SWC_mm": self.p['SWC_max_mm'] * 0.75,
        }
        self.L_stability = 1e6
        self.age_distribution = self._initialize_age_distribution(coniferous_fraction, stem_density)

    def _initialize_age_distribution(self, coniferous_fraction: float, stem_density: float) -> dict:
        age_classes = {
            'seedling': (0, 5),
            'sapling': (6, 15),
            'young': (16, 50),
            'mature': (51, 100),
            'old': (101, 200)
        }
        conifer_stems = int(stem_density * coniferous_fraction)
        deciduous_stems = int(stem_density * (1.0 - coniferous_fraction))
        age_dist = {
            'conifer': {age_class: 0 for age_class in age_classes.keys()},
            'deciduous': {age_class: 0 for age_class in age_classes.keys()},
            'age_classes': age_classes,
            'total_stems': stem_density
        }
        if conifer_stems > 0:
            age_weights = {
                'seedling': 0.05,
                'sapling': 0.15,
                'young': 0.40,
                'mature': 0.30,
                'old': 0.10
            }
            for age_class in age_classes.keys():
                base_fraction = age_weights[age_class]
                if self.remove_age_jitter:
                    adjusted_fraction = base_fraction
                else:
                    random_factor = self.rng.uniform(0.8, 1.2)
                    adjusted_fraction = base_fraction * random_factor
                age_dist['conifer'][age_class] = int(conifer_stems * adjusted_fraction)
            total_assigned = sum(age_dist['conifer'].values())
            if total_assigned > conifer_stems:
                scale_factor = conifer_stems / total_assigned
                for age_class in age_classes.keys():
                    age_dist['conifer'][age_class] = int(age_dist['conifer'][age_class] * scale_factor)
            elif total_assigned < conifer_stems:
                remaining = conifer_stems - total_assigned
                age_dist['conifer']['mature'] += remaining
        if deciduous_stems > 0:
            age_weights = {
                'seedling': 0.08,
                'sapling': 0.20,
                'young': 0.35,
                'mature': 0.25,
                'old': 0.12
            }
            for age_class in age_classes.keys():
                base_fraction = age_weights[age_class]
                if self.remove_age_jitter:
                    adjusted_fraction = base_fraction
                else:
                    random_factor = self.rng.uniform(0.8, 1.2)
                    adjusted_fraction = base_fraction * random_factor
                age_dist['deciduous'][age_class] = int(deciduous_stems * adjusted_fraction)
            total_assigned = sum(age_dist['deciduous'].values())
            if total_assigned > deciduous_stems:
                scale_factor = deciduous_stems / total_assigned
                for age_class in age_classes.keys():
                    age_dist['deciduous'][age_class] = int(age_dist['deciduous'][age_class] * scale_factor)
            elif total_assigned < deciduous_stems:
                remaining = deciduous_stems - total_assigned
                age_dist['deciduous']['mature'] += remaining
        return age_dist

    def update_age_distribution_after_management(self, density_change: float, management_conifer_fraction: float, thinning_oldest_first: bool = True) -> dict:
        age_dist = self.age_distribution.copy()
        if density_change > 0:
            new_conifer_stems = int(density_change * management_conifer_fraction)
            new_deciduous_stems = density_change - new_conifer_stems
            age_dist['conifer']['seedling'] += new_conifer_stems
            age_dist['deciduous']['seedling'] += new_deciduous_stems
        elif density_change < 0:
            stems_to_remove = abs(density_change)
            conifer_stems_to_remove = int(stems_to_remove * management_conifer_fraction)
            deciduous_stems_to_remove = stems_to_remove - conifer_stems_to_remove
            available_conifer_old = age_dist['conifer']['old']
            available_conifer_thinnable = available_conifer_old
            available_deciduous_old = age_dist['deciduous']['old']
            available_deciduous_thinnable = available_deciduous_old
            conifer_stems_to_remove = min(conifer_stems_to_remove, available_conifer_thinnable)
            deciduous_stems_to_remove = min(deciduous_stems_to_remove, available_deciduous_thinnable)
            age_classes_ordered = ['old']
            remaining_conifer = conifer_stems_to_remove
            for age_class in age_classes_ordered:
                if remaining_conifer <= 0:
                    break
                stems_in_class = age_dist['conifer'][age_class]
                stems_to_remove_from_class = min(remaining_conifer, stems_in_class)
                age_dist['conifer'][age_class] -= stems_to_remove_from_class
                remaining_conifer -= stems_to_remove_from_class
            remaining_deciduous = deciduous_stems_to_remove
            for age_class in age_classes_ordered:
                if remaining_deciduous <= 0:
                    break
                stems_in_class = age_dist['deciduous'][age_class]
                stems_to_remove_from_class = min(remaining_deciduous, stems_in_class)
                age_dist['deciduous'][age_class] -= stems_to_remove_from_class
                remaining_deciduous -= stems_to_remove_from_class
        age_dist['total_stems'] = sum(age_dist['conifer'].values()) + sum(age_dist['deciduous'].values())
        total_conifer_after = sum(age_dist['conifer'].values())
        total_stems_after = age_dist['total_stems']
        resulting_conifer_fraction = total_conifer_after / total_stems_after if total_stems_after > 0 else 0.5
        return {'age_distribution': age_dist, 'resulting_conifer_fraction': resulting_conifer_fraction}

    def age_trees_one_year(self) -> dict:
        age_dist = {
            'conifer': self.age_distribution['conifer'].copy(),
            'deciduous': self.age_distribution['deciduous'].copy(),
            'age_classes': self.age_distribution['age_classes'],
            'total_stems': self.age_distribution['total_stems']
        }
        age_classes = age_dist['age_classes']
        transition_rates = {
            'seedling': 1.0 / (age_classes['seedling'][1] - age_classes['seedling'][0] + 1),
            'sapling': 1.0 / (age_classes['sapling'][1] - age_classes['sapling'][0] + 1),
            'young': 1.0 / (age_classes['young'][1] - age_classes['young'][0] + 1),
            'mature': 1.0 / (age_classes['mature'][1] - age_classes['mature'][0] + 1),
            'old': 0.0
        }
        for species in ['conifer', 'deciduous']:
            transitions_out = {
                'seedling': int(age_dist[species]['seedling'] * transition_rates['seedling']),
                'sapling': int(age_dist[species]['sapling'] * transition_rates['sapling']),
                'young': int(age_dist[species]['young'] * transition_rates['young']),
                'mature': int(age_dist[species]['mature'] * transition_rates['mature']),
            }
            age_dist[species]['old'] += transitions_out['mature']
            age_dist[species]['mature'] = (age_dist[species]['mature'] - transitions_out['mature']) + transitions_out['young']
            age_dist[species]['young'] = (age_dist[species]['young'] - transitions_out['young']) + transitions_out['sapling']
            age_dist[species]['sapling'] = (age_dist[species]['sapling'] - transitions_out['sapling']) + transitions_out['seedling']
            age_dist[species]['seedling'] -= transitions_out['seedling']
        age_dist['total_stems'] = sum(age_dist['conifer'].values()) + sum(age_dist['deciduous'].values())
        self.age_distribution = age_dist
        return age_dist

    def _sample_parameters(self) -> Dict[str, Any]:
        p = self.config.copy()
        # Latitude: allow site override; else sample or midpoint under site_specific
        if 'latitude_deg' in self.site_overrides:
            latitude_deg = float(self.site_overrides['latitude_deg'])
        else:
            lat_range = self.config['latitude_deg_range']
            if self.site_specific:
                latitude_deg = 0.5 * (lat_range[0] + lat_range[1])
            else:
                latitude_deg = self.rng.uniform(low=lat_range[0], high=lat_range[1])
        p['latitude_deg'] = latitude_deg
        del p['latitude_deg_range']
        lat_dependent_params = calculate_latitude_dependent_parameters(latitude_deg, self.rng)
        p.update(lat_dependent_params)
        # In site-specific mode, remove stochastic jitters from latitude-dependent params
        if self.site_specific:
            # Only override if user didn't explicitly set them
            def not_overridden(k: str) -> bool:
                return k not in self.site_overrides
            if all(not_overridden(k) for k in [
                'T_annual_mean_offset', 'T_seasonal_amplitude', 'growth_day', 'fall_day'
            ]):
                base_temp_at_60N = -2.0
                temp_latitude_slope = -0.6
                base_T_annual_mean = base_temp_at_60N + temp_latitude_slope * (latitude_deg - 60.0)
                T_annual_mean_offset = base_T_annual_mean
                base_amplitude_at_60N = 22.0
                amplitude_latitude_slope = 0.3
                base_T_seasonal_amplitude = base_amplitude_at_60N + amplitude_latitude_slope * (latitude_deg - 60.0)
                T_seasonal_amplitude = base_T_seasonal_amplitude
                base_growth_day = 140
                growth_latitude_delay = 2.0
                growth_day = base_growth_day + growth_latitude_delay * max(0, latitude_deg - 60.0)
                temp_adjustment = max(0, -T_annual_mean_offset) * 2.0
                growth_day += temp_adjustment
                base_fall_day = 260
                fall_latitude_advance = 1.5
                fall_day = base_fall_day - fall_latitude_advance * max(0, latitude_deg - 60.0)
                temp_adjustment = max(0, -T_annual_mean_offset) * 1.5
                fall_day -= temp_adjustment
                min_growing_season = 60.0
                if fall_day - growth_day < min_growing_season:
                    fall_day = growth_day + min_growing_season
                fall_day = min(fall_day, 320)
                p['T_annual_mean_offset'] = T_annual_mean_offset
                p['T_seasonal_amplitude'] = T_seasonal_amplitude
                p['growth_day'] = growth_day
                p['fall_day'] = fall_day

        excluded_keys = [
            'latitude_deg_range', 'T_annual_mean_offset_range', 'T_seasonal_amplitude_range',
            'growth_day_range', 'fall_day_range', 'shoulder_1_start_range', 'shoulder_1_end_range',
            'summer_day_start_range', 'summer_day_end_range', 'shoulder_2_start_range', 'shoulder_2_end_range',
            'snow_season_start_range', 'snow_season_end_range', 'rain_diurnal_sensitivity_range',
            'rain_diurnal_threshold_range', 'max_diurnal_reduction_range', 'min_diurnal_amplitude_range',
        ]

        for key, value in list(self.config.items()):
            if key.endswith("_range") and key not in excluded_keys:
                base_key = key.replace("_range", "")
                if base_key in self.site_overrides:
                    sampled_value = float(self.site_overrides[base_key])
                elif self.site_specific:
                    sampled_value = 0.5 * (value[0] + value[1])
                else:
                    sampled_value = self.rng.uniform(low=value[0], high=value[1])
                p[base_key] = sampled_value
                del p[key]

        self._sample_seasonal_parameters_sequentially(p)

        # Final site-specific overrides: allow fixing any parameter explicitly
        if self.site_overrides:
            for k, v in self.site_overrides.items():
                p[k] = v
        # Re-validate phenology/snow consistency after overrides
        try:
            _validate_phenology_snow_constraints(p, self.rng)
        except Exception:
            pass
        return p

    def _sample_seasonal_parameters_sequentially(self, p: Dict[str, Any]):
        shoulder_1_start_range = p['shoulder_1_start_range']
        if self.site_specific:
            p['shoulder_1_start'] = 0.5 * (shoulder_1_start_range[0] + shoulder_1_start_range[1])
        else:
            p['shoulder_1_start'] = self.rng.uniform(low=shoulder_1_start_range[0], high=shoulder_1_start_range[1])
        del p['shoulder_1_start_range']

        shoulder_1_end_range = p['shoulder_1_end_range']
        min_shoulder_1_end = max(shoulder_1_end_range[0], p['shoulder_1_start'] + 10)
        max_shoulder_1_end = shoulder_1_end_range[1]
        if min_shoulder_1_end < max_shoulder_1_end:
            if self.site_specific:
                p['shoulder_1_end'] = 0.5 * (min_shoulder_1_end + max_shoulder_1_end)
            else:
                p['shoulder_1_end'] = self.rng.uniform(low=min_shoulder_1_end, high=max_shoulder_1_end)
        else:
            p['shoulder_1_end'] = min_shoulder_1_end
        del p['shoulder_1_end_range']

        summer_day_start_range = p['summer_day_start_range']
        min_summer_start = max(summer_day_start_range[0], p['shoulder_1_end'])
        max_summer_start = summer_day_start_range[1]
        if min_summer_start < max_summer_start:
            if self.site_specific:
                p['summer_day_start'] = 0.5 * (min_summer_start + max_summer_start)
            else:
                p['summer_day_start'] = self.rng.uniform(low=min_summer_start, high=max_summer_start)
        else:
            p['summer_day_start'] = min_summer_start
        del p['summer_day_start_range']

        summer_day_end_range = p['summer_day_end_range']
        min_summer_end = max(summer_day_end_range[0], p['summer_day_start'] + 30)
        max_summer_end = summer_day_end_range[1]
        if min_summer_end < max_summer_end:
            if self.site_specific:
                p['summer_day_end'] = 0.5 * (min_summer_end + max_summer_end)
            else:
                p['summer_day_end'] = self.rng.uniform(low=min_summer_end, high=max_summer_end)
        else:
            p['summer_day_end'] = min_summer_end
        del p['summer_day_end_range']

        shoulder_2_start_range = p['shoulder_2_start_range']
        min_shoulder_2_start = max(shoulder_2_start_range[0], p['summer_day_end'])
        max_shoulder_2_start = shoulder_2_start_range[1]
        if min_shoulder_2_start < max_shoulder_2_start:
            if self.site_specific:
                p['shoulder_2_start'] = 0.5 * (min_shoulder_2_start + max_shoulder_2_start)
            else:
                p['shoulder_2_start'] = self.rng.uniform(low=min_shoulder_2_start, high=max_shoulder_2_start)
        else:
            p['shoulder_2_start'] = min_shoulder_2_start
        del p['shoulder_2_start_range']

        shoulder_2_end_range = p['shoulder_2_end_range']
        min_shoulder_2_end = max(shoulder_2_end_range[0], p['shoulder_2_start'] + 10)
        max_shoulder_2_end = shoulder_2_end_range[1]
        if min_shoulder_2_end < max_shoulder_2_end:
            if self.site_specific:
                p['shoulder_2_end'] = 0.5 * (min_shoulder_2_end + max_shoulder_2_end)
            else:
                p['shoulder_2_end'] = self.rng.uniform(low=min_shoulder_2_end, high=max_shoulder_2_end)
        else:
            p['shoulder_2_end'] = min_shoulder_2_end
        del p['shoulder_2_end_range']

        # Snow season dates: in site mode, compute deterministically (no random variation)
        if self.site_specific:
            T_offset = p['T_annual_mean_offset']
            T_amplitude = p['T_seasonal_amplitude']
            cos_crossing = T_offset / T_amplitude if T_amplitude != 0 else 2.0
            if abs(cos_crossing) > 1.0:
                snow_season_end = 120
                snow_season_start = 280
            else:
                day_angle_crossing = np.arccos(cos_crossing)
                spring_day = 1 + (day_angle_crossing * 365.0) / (2 * np.pi)
                fall_day = 1 + ((2 * np.pi - day_angle_crossing) * 365.0) / (2 * np.pi)
                snow_season_end = int(np.clip(spring_day, 100, 150))
                snow_season_start = int(np.clip(fall_day, 260, 320))
                if snow_season_end > snow_season_start:
                    snow_season_end, snow_season_start = snow_season_start, snow_season_end
                    snow_season_end = int(np.clip(snow_season_end, 100, 150))
                    snow_season_start = int(np.clip(snow_season_start, 260, 320))
        else:
            snow_season_start, snow_season_end = calculate_snow_season_dates(p, self.rng)
        p['snow_season_start'] = snow_season_start
        p['snow_season_end'] = snow_season_end
        del p['snow_season_start_range']
        del p['snow_season_end_range']

    # The remainder of the class (run_annual_cycle, demography applications, etc.)
    # is kept nearly identical to the legacy implementation to preserve behavior.

    def run_annual_cycle(self, current_stem_density: float,
                         current_biomass_carbon_kg_m2: float, current_soil_carbon_kg_m2: float,
                         density_change: float = 0.0, management_conifer_fraction: float = 0.5,
                         warming_penalty_factor: float = 5.0) -> dict:
        total_gpp_kg_m2, total_autotrophic_resp_kg_m2, total_soil_resp_kg_m2 = 0.0, 0.0, 0.0
        total_litterfall_kg_m2 = 0.0
        positive_flux_sum = 0.0
        negative_flux_sum = 0.0
        initial_biomass_carbon_kg_m2 = current_biomass_carbon_kg_m2
        initial_soil_carbon_kg_m2 = current_soil_carbon_kg_m2
        new_stem_density = np.clip(current_stem_density + density_change, MIN_STEMS_HA, MAX_STEMS_HA)
        actual_density_change = new_stem_density - current_stem_density
        carbon_loss_thinning = 0.0
        hwp_carbon_stored = 0.0
        if actual_density_change < 0:
            total_carbon_removed = current_biomass_carbon_kg_m2 * abs(actual_density_change) / max(current_stem_density, 1.0)
            hwp_carbon_stored = total_carbon_removed * 0.95
            carbon_loss_thinning = total_carbon_removed * 0.05
            current_biomass_carbon_kg_m2 = max(0.0, current_biomass_carbon_kg_m2 - total_carbon_removed)
        current_conifer_fraction = self.p['coniferous_fraction']
        resulting_conifer_fraction = current_conifer_fraction
        if actual_density_change != 0:
            management_result = self.update_age_distribution_after_management(actual_density_change, management_conifer_fraction, thinning_oldest_first=True)
            self.age_distribution = management_result['age_distribution']
            resulting_conifer_fraction = management_result['resulting_conifer_fraction']
        self.p = get_baseline_parameters(self.p, resulting_conifer_fraction, new_stem_density, self.rng, self.age_distribution)
        dynamic_biomass_carbon_kg_m2 = current_biomass_carbon_kg_m2
        dynamic_soil_carbon_kg_m2 = current_soil_carbon_kg_m2
        drought_index = 0.0
        fire_losses = {}
        has_fire_occurred = False
        heat_caps = {
            "canopy": self.p['C_CANOPY_LEAF_OFF'], "trunk": self.p['C_TRUNK'], "snow": self.p['C_SNOW'],
            "atm_model": self.p['C_ATM'], "soil_surf": self.p['C_SOIL_TOTAL'] * 0.15, "soil_deep": self.p['C_SOIL_TOTAL'] * 0.85,
        }
        temp_nodes = [n for n in self.S if 'SWE' not in n and 'SWC' not in n]
        for day in range(1, 366):
            fall_start = self.p['fall_day'] - 10
            fall_end = self.p['fall_day'] + 20
            is_fall_season = (day >= fall_start and day <= fall_end)
            fall_days = fall_end - fall_start + 1
            if is_fall_season:
                daily_litterfall_rate = (self.p['LITTERFALL_FRACTION'] * self.p['LITTERFALL_SEASONALITY'] * dynamic_biomass_carbon_kg_m2 / fall_days)
            else:
                non_fall_days = 365 - fall_days
                daily_litterfall_rate = (self.p['LITTERFALL_FRACTION'] * (1.0 - self.p['LITTERFALL_SEASONALITY']) * dynamic_biomass_carbon_kg_m2 / non_fall_days)
            total_litterfall_kg_m2 += daily_litterfall_rate
            self.p['daily_state'] = {}
            potential_et_proxy = max(0, self.S['atm_model'] - 273.15) / 5.0
            drought_index += potential_et_proxy
            if self.p['daily_state'].get('rain_mm_day', 0) > 1.0:
                drought_index = max(0, drought_index - self.p['daily_state']['rain_mm_day'])
            is_summer = (self.p['summer_day_start'] < day < self.p['summer_day_end'])
            if is_summer and not has_fire_occurred:
                daily_fire_losses = self._calculate_fire_event(drought_index, self.S['atm_model'], dynamic_biomass_carbon_kg_m2)
                if daily_fire_losses.get("fire_mortality_fraction", 0.0) > 0:
                    fire_losses = daily_fire_losses
                    has_fire_occurred = True
            for t_step in range(self.p['STEPS_PER_DAY']):
                biomass_c = dynamic_biomass_carbon_kg_m2
                threshold = self.p['GPP_SCALING_BIOMASS_THRESHOLD_kg_m2']
                max_biomass = 15.0
                if biomass_c > threshold:
                    scale_range = max_biomass - threshold
                    biomass_over_threshold = biomass_c - threshold
                    gpp_scalar = 1.0 - 0.9 * (biomass_over_threshold / scale_range)
                    self.p['gpp_scalar'] = max(0.1, gpp_scalar)
                else:
                    self.p['gpp_scalar'] = 1.0
                hour = t_step * self.p['TIME_STEP_MINUTES'] / 60.0
                self.p = update_dynamic_parameters(self.p, day, hour, self.S, self.L_stability, self.rng)
                heat_caps['canopy'] = self.p['C_CANOPY_LEAF_ON'] if self.p['LAI_actual'] > 0.1 else self.p['C_CANOPY_LEAF_OFF']
                flux, dSWE_g, dSWE_c, gpp_g_m2_s = calculate_fluxes_and_melt(self.S, self.p)
                gpp_kg_m2_step = gpp_g_m2_s * 1e-3 * self.p['DT_SECONDS']
                total_gpp_kg_m2 += gpp_kg_m2_step
                r_auto_kg_m2_yr = (self.p['R_BASE_KG_M2_YR'] * (dynamic_biomass_carbon_kg_m2 / self.p['RESPIRATION_BIOMASS_SIZE_SCALAR_kg_m2']) * self.p['Q10'] ** ((self.S['soil_surf'] - self.p['T_REF_K']) / 10.0))
                auto_resp_step = r_auto_kg_m2_yr / (365 * self.p['STEPS_PER_DAY'])
                total_autotrophic_resp_kg_m2 += auto_resp_step
                r_soil_kg_m2_yr = (self.p['R_BASE_SOIL_KG_M2_YR'] * (dynamic_soil_carbon_kg_m2 / self.p['RESPIRATION_SOIL_SIZE_SCALAR_kg_m2']) * self.p['Q10'] ** ((self.S['soil_surf'] - self.p['T_REF_K']) / 10.0))
                soil_resp_step = r_soil_kg_m2_yr / (365 * self.p['STEPS_PER_DAY'])
                total_soil_resp_kg_m2 += soil_resp_step
                npp_step = gpp_kg_m2_step - auto_resp_step
                dynamic_biomass_carbon_kg_m2 += npp_step
                dynamic_soil_carbon_kg_m2 -= soil_resp_step
                if t_step == self.p['STEPS_PER_DAY'] - 1:
                    dynamic_biomass_carbon_kg_m2 -= daily_litterfall_rate
                    dynamic_soil_carbon_kg_m2 += daily_litterfall_rate
                dynamic_biomass_carbon_kg_m2 = max(0.0, dynamic_biomass_carbon_kg_m2)
                dynamic_soil_carbon_kg_m2 = max(0.0, dynamic_soil_carbon_kg_m2)
                self.S['SWE_can'] = max(0, self.S['SWE_can'] + self.p['snow_intercepted_m'] - dSWE_c)
                self.S['SWE'] = max(0, self.S['SWE'] + self.p['snowfall_m_step'] - self.p['snow_intercepted_m'] - dSWE_g)
                rain_throughfall = self.p['rain_m_step'] - self.p['rain_intercepted_m']
                water_in_mm = (rain_throughfall + dSWE_g + dSWE_c) * 1000.0
                LE_transpiration = -flux['canopy'].get('LE_trans', 0.0)
                LE_soil_evap = -flux['soil_surf'].get('LE_evap', 0.0)
                water_out_mm = ((LE_transpiration + LE_soil_evap) * self.p['DT_SECONDS']) / (self.p['Lv'] * self.p['RHO_WATER']) * 1000.0
                self.S['SWC_mm'] += water_in_mm - water_out_mm
                runoff_mm = max(0.0, self.S['SWC_mm'] - self.p['SWC_max_mm'])
                self.S['SWC_mm'] -= runoff_mm
                for node in temp_nodes:
                    net_flux = sum(flux.get(node, {}).values())
                    dT = (net_flux / heat_caps[node]) * self.p['DT_SECONDS'] if heat_caps.get(node, 0) > 0 else 0.0
                    self.S[node] = safe_update(self.S[node], dT, self.p) if node != 'canopy' else flux['canopy']['T_new']
                flux_deep_bound = flux['soil_deep'].get('Cnd_boundary', 0.0)
                thaw_energy_per_timestep = flux_deep_bound * self.p['DT_SECONDS']
                total_soil_heat_capacity = self.p['C_SOIL_TOTAL']
                thaw_temp_equivalent = thaw_energy_per_timestep / total_soil_heat_capacity
                timestep_duration_days = self.p['DT_SECONDS'] / 86400.0
                if thaw_temp_equivalent > 0:
                    positive_flux_sum += thaw_temp_equivalent * timestep_duration_days
                else:
                    negative_flux_sum += abs(thaw_temp_equivalent) * timestep_duration_days
                H_total = sum(flux['atm_model'].get(k, 0.0) for k in ['H_can', 'H_trunk', 'H_soil', 'H_snow'])
                if abs(H_total) > 1e-3:
                    u = self.p['u_ref']
                    psi_m, _ = get_stability_correction(self.p['z_ref_h'], self.L_stability)
                    u_star_log_term = np.log(self.p['z_ref_h'] / self.p['z0_can']) - psi_m
                    u_star = u * self.p['KAPPA'] / u_star_log_term if u_star_log_term > 0 else 0.1
                    L_den = self.p['KAPPA'] * self.p['G_ACCEL'] * H_total
                    self.L_stability = -self.p['RHO_AIR'] * self.p['CP_AIR'] * (u_star**3) * self.p['T_atm'] / L_den if abs(L_den) > 1e-9 else 1e6
                else:
                    self.L_stability = 1e6
        npp = total_gpp_kg_m2 - total_autotrophic_resp_kg_m2
        mortality_stems, recruitment_stems = calculate_natural_demography(
            current_density=new_stem_density,
            max_natural_density=self.p['MAX_NATURAL_DENSITY'],
            natural_mortality_rate=self.p['NATURAL_MORTALITY_RATE'],
            natural_recruitment_rate=self.p['NATURAL_RECRUITMENT_RATE'],
            density_dependent_mortality=self.p['DENSITY_DEPENDENT_MORTALITY'],
            density_mortality_threshold=self.p['DENSITY_MORTALITY_THRESHOLD'],
            environmental_stress=1.0,
            rng=self.rng,
        )
        winter_day_angle = 2 * np.pi * (1 - 1) / 365.0
        mean_winter_temp_c = self.p['T_annual_mean_offset'] - self.p['T_seasonal_amplitude'] * np.cos(winter_day_angle)
        insect_losses = self._calculate_insect_outbreak(mean_winter_temp_c, new_stem_density)
        self.age_distribution = self.apply_natural_demography_to_age_distribution(mortality_stems, recruitment_stems)
        insect_mortality_fraction = insect_losses['insect_mortality_fraction']
        self.age_distribution = self.apply_insect_mortality_to_age_distribution(insect_mortality_fraction)
        fire_mortality_fraction = fire_losses.get("fire_mortality_fraction", 0.0)
        if fire_mortality_fraction > 0:
            self.age_distribution = self.apply_fire_mortality_to_age_distribution(fire_mortality_fraction)
        grown_biomass_carbon_kg_m2 = dynamic_biomass_carbon_kg_m2
        insect_biomass_loss = 0.0
        if insect_mortality_fraction > 0:
            insect_biomass_loss = grown_biomass_carbon_kg_m2 * insect_mortality_fraction
            grown_biomass_carbon_kg_m2 -= insect_biomass_loss
            dynamic_soil_carbon_kg_m2 += insect_biomass_loss
        natural_mortality_biomass = 0.0
        if mortality_stems > 0 and new_stem_density > 0:
            natural_mortality_fraction = mortality_stems / new_stem_density
            natural_mortality_biomass = grown_biomass_carbon_kg_m2 * natural_mortality_fraction
            grown_biomass_carbon_kg_m2 -= natural_mortality_biomass
            dynamic_soil_carbon_kg_m2 += natural_mortality_biomass
        fire_biomass_loss = 0.0
        if fire_mortality_fraction > 0:
            fire_biomass_loss = grown_biomass_carbon_kg_m2 * fire_mortality_fraction
            grown_biomass_carbon_kg_m2 -= fire_biomass_loss
            fire_carbon_to_soil = fire_losses.get("fire_carbon_to_soil_kg_m2", 0.0)
            dynamic_soil_carbon_kg_m2 += fire_carbon_to_soil
        final_biomass_carbon_kg_m2 = max(0.0, grown_biomass_carbon_kg_m2)
        final_soil_carbon_kg_m2 = max(0.0, dynamic_soil_carbon_kg_m2)
        biomass_would_exceed_limit = final_biomass_carbon_kg_m2 > self.p['MAX_BIOMASS_CARBON_LIMIT_kg_m2']
        soil_would_exceed_limit = final_soil_carbon_kg_m2 > self.p['MAX_SOIL_CARBON_LIMIT_kg_m2']
        biomass_excess_before_clip = max(0, final_biomass_carbon_kg_m2 - self.p['MAX_BIOMASS_CARBON_LIMIT_kg_m2'])
        soil_excess_before_clip = max(0, final_soil_carbon_kg_m2 - self.p['MAX_SOIL_CARBON_LIMIT_kg_m2'])
        final_biomass_carbon_kg_m2 = min(final_biomass_carbon_kg_m2, self.p['MAX_BIOMASS_CARBON_LIMIT_kg_m2'])
        final_soil_carbon_kg_m2 = min(final_soil_carbon_kg_m2, self.p['MAX_SOIL_CARBON_LIMIT_kg_m2'])
        self.age_distribution = self.age_trees_one_year()
        final_stem_density_from_age_dist = self.age_distribution['total_stems']
        unclipped_stem_density = final_stem_density_from_age_dist
        final_stem_density = np.clip(unclipped_stem_density, MIN_STEMS_HA, MAX_STEMS_HA)
        asymmetric_thaw_reward = negative_flux_sum - (warming_penalty_factor * positive_flux_sum)
        total_thaw_degree_days = positive_flux_sum - negative_flux_sum
        total_initial_biomass = current_biomass_carbon_kg_m2 + npp
        total_mortality_losses = total_initial_biomass - final_biomass_carbon_kg_m2
        biomass_carbon_change = final_biomass_carbon_kg_m2 - initial_biomass_carbon_kg_m2
        soil_carbon_change = final_soil_carbon_kg_m2 - initial_soil_carbon_kg_m2
        return {
            "final_stem_density": final_stem_density,
            "unclipped_stem_density": unclipped_stem_density,
            "final_biomass_carbon_kg_m2": final_biomass_carbon_kg_m2,
            "final_soil_carbon_kg_m2": final_soil_carbon_kg_m2,
            "final_conifer_fraction": resulting_conifer_fraction,
            "thaw_degree_days": total_thaw_degree_days,
            "asymmetric_thaw_reward": asymmetric_thaw_reward,
            "positive_flux_sum": positive_flux_sum,
            "negative_flux_sum": negative_flux_sum,
            "warming_penalty_factor": warming_penalty_factor,
            "net_carbon_change": (final_biomass_carbon_kg_m2 + final_soil_carbon_kg_m2) - (initial_biomass_carbon_kg_m2 + initial_soil_carbon_kg_m2),
            "net_carbon_change_with_hwp": (final_biomass_carbon_kg_m2 + final_soil_carbon_kg_m2 + hwp_carbon_stored) - (initial_biomass_carbon_kg_m2 + initial_soil_carbon_kg_m2),
            "biomass_carbon_change": biomass_carbon_change,
            "soil_carbon_change": soil_carbon_change,
            "total_gpp_kg_m2": total_gpp_kg_m2,
            "total_litterfall_kg_m2": total_litterfall_kg_m2,
            "natural_mortality_stems": mortality_stems,
            "natural_recruitment_stems": recruitment_stems,
            "fire_mortality_fraction": fire_mortality_fraction,
            "insect_mortality_fraction": insect_losses['insect_mortality_fraction'],
            "final_drought_index": drought_index,
            "carbon_loss_thinning": carbon_loss_thinning,
            "hwp_carbon_stored": hwp_carbon_stored,
            "total_carbon_stock_with_hwp": final_biomass_carbon_kg_m2 + final_soil_carbon_kg_m2 + hwp_carbon_stored,
            "total_initial_biomass": total_initial_biomass,
            "total_mortality_losses": total_mortality_losses,
            "fire_biomass_loss": fire_biomass_loss,
            "insect_biomass_loss": insect_biomass_loss,
            "natural_mortality_biomass": natural_mortality_biomass,
            "total_autotrophic_resp_kg_m2": total_autotrophic_resp_kg_m2,
            "total_soil_resp_kg_m2": total_soil_resp_kg_m2,
            "biomass_would_exceed_limit": biomass_would_exceed_limit,
            "soil_would_exceed_limit": soil_would_exceed_limit,
            "biomass_excess_before_clip": biomass_excess_before_clip,
            "soil_excess_before_clip": soil_excess_before_clip,
        }

    def _calculate_fire_event(self, drought_index: float, T_air_k: float, current_biomass_kg_m2: float) -> dict:
        fire_occurs = False
        fire_losses = {"fire_mortality_fraction": 0.0, "fire_biomass_loss_kg_m2": 0.0, "fire_carbon_to_soil_kg_m2": 0.0}
        if drought_index > self.p['FIRE_DROUGHT_THRESHOLD'] and T_air_k > self.p['FIRE_TEMP_THRESHOLD_K']:
            prob_modifier = 1.0 + (self.p['coniferous_fraction'] * (self.p['FIRE_CONIFER_FLAMMABILITY_factor'] - 1.0))
            fire_prob = self.p['FIRE_BASE_PROB'] * prob_modifier
            if self.rng.random() < fire_prob:
                fire_occurs = True
        if fire_occurs:
            intensity = np.clip(current_biomass_kg_m2 / self.p['FIRE_INTENSITY_MAX_BIOMASS_kg_m2'], self.p['FIRE_INTENSITY_MIN_FACTOR'], self.p['FIRE_INTENSITY_MAX_FACTOR'])
            mortality_fraction = self.p['FIRE_MORTALITY_MAX_RATE'] * intensity * (0.5 + 0.5 * self.p['coniferous_fraction'])
            mortality_fraction = min(mortality_fraction, 0.2)
            biomass_loss = current_biomass_kg_m2 * mortality_fraction
            carbon_combusted = biomass_loss * self.p['FIRE_CARBON_COMBUSTED_frac']
            carbon_to_soil = biomass_loss * (1.0 - self.p['FIRE_CARBON_COMBUSTED_frac'])
            fire_losses.update({"fire_mortality_fraction": mortality_fraction, "fire_biomass_loss_kg_m2": biomass_loss, "fire_carbon_to_soil_kg_m2": carbon_to_soil})
        return fire_losses

    def apply_natural_demography_to_age_distribution(self, mortality_stems: float, recruitment_stems: float) -> dict:
        age_dist = {
            'conifer': self.age_distribution['conifer'].copy(),
            'deciduous': self.age_distribution['deciduous'].copy(),
            'age_classes': self.age_distribution['age_classes'],
            'total_stems': self.age_distribution['total_stems']
        }
        total_conifer_stems = sum(age_dist['conifer'].values())
        total_deciduous_stems = sum(age_dist['deciduous'].values())
        total_stems = total_conifer_stems + total_deciduous_stems
        if total_stems == 0:
            return age_dist
        conifer_mortality = mortality_stems * (total_conifer_stems / total_stems)
        deciduous_mortality = mortality_stems * (total_deciduous_stems / total_stems)
        conifer_recruitment = recruitment_stems * (total_conifer_stems / total_stems)
        deciduous_recruitment = recruitment_stems * (total_deciduous_stems / total_stems)
        for species in ['conifer', 'deciduous']:
            species_mortality = conifer_mortality if species == 'conifer' else deciduous_mortality
            species_stems = total_conifer_stems if species == 'conifer' else total_deciduous_stems
            if species_stems == 0:
                continue
            mortality_preferences = {
                'old': 1.0,
                'mature': 0.8,
                'seedling': 0.8,
                'sapling': 0.6,
                'young': 0.4,
            }
            total_preference_weighted = sum(
                age_dist[species][age_class] * mortality_preferences[age_class]
                for age_class in mortality_preferences.keys()
            )
            if total_preference_weighted == 0:
                continue
            stems_to_remove = int(species_mortality)
            preference_list = ['old', 'mature', 'seedling', 'sapling', 'young']
            for age_class in preference_list:
                if stems_to_remove <= 0:
                    break
                if age_dist[species][age_class] > 0:
                    max_removable = min(stems_to_remove, age_dist[species][age_class])
                    age_dist[species][age_class] -= max_removable
                    stems_to_remove -= max_removable
        if conifer_recruitment > 0:
            age_dist['conifer']['seedling'] += int(conifer_recruitment)
        if deciduous_recruitment > 0:
            age_dist['deciduous']['seedling'] += int(deciduous_recruitment)
        age_dist['total_stems'] = sum(age_dist['conifer'].values()) + sum(age_dist['deciduous'].values())
        return age_dist

    def apply_fire_mortality_to_age_distribution(self, fire_mortality_fraction: float) -> dict:
        if fire_mortality_fraction <= 0:
            return self.age_distribution
        age_dist = {
            'conifer': self.age_distribution['conifer'].copy(),
            'deciduous': self.age_distribution['deciduous'].copy(),
            'age_classes': self.age_distribution['age_classes'],
            'total_stems': self.age_distribution['total_stems']
        }
        fire_vulnerability = {'seedling': 1.2, 'sapling': 1.1, 'young': 1.0, 'mature': 0.9, 'old': 0.8}
        for species in ['conifer', 'deciduous']:
            species_stems = sum(age_dist[species].values())
            if species_stems == 0:
                continue
            total_fire_mortality = int(species_stems * fire_mortality_fraction)
            stems_to_remove = total_fire_mortality
            vulnerability_list = ['seedling', 'sapling', 'young', 'mature', 'old']
            for age_class in vulnerability_list:
                if stems_to_remove <= 0:
                    break
                if age_dist[species][age_class] > 0:
                    max_removable = min(stems_to_remove, age_dist[species][age_class])
                    age_dist[species][age_class] -= max_removable
                    stems_to_remove -= max_removable
        age_dist['total_stems'] = sum(age_dist['conifer'].values()) + sum(age_dist['deciduous'].values())
        return age_dist

    def apply_insect_mortality_to_age_distribution(self, insect_mortality_fraction: float) -> dict:
        if insect_mortality_fraction <= 0:
            return self.age_distribution
        age_dist = {
            'conifer': self.age_distribution['conifer'].copy(),
            'deciduous': self.age_distribution['deciduous'].copy(),
            'age_classes': self.age_distribution['age_classes'],
            'total_stems': self.age_distribution['total_stems']
        }
        insect_preference = {'seedling': 0.3, 'sapling': 0.5, 'young': 0.7, 'mature': 1.0, 'old': 1.2}
        for species in ['conifer', 'deciduous']:
            species_stems = sum(age_dist[species].values())
            if species_stems == 0:
                continue
            total_insect_mortality = int(species_stems * insect_mortality_fraction)
            stems_to_remove = total_insect_mortality
            total_preference_weighted = sum(
                age_dist[species][age_class] * insect_preference[age_class]
                for age_class in insect_preference.keys()
            )
            if total_preference_weighted == 0:
                continue
            preference_list = ['old', 'mature', 'young', 'sapling', 'seedling']
            for age_class in preference_list:
                if stems_to_remove <= 0:
                    break
                if age_dist[species][age_class] > 0:
                    max_removable = min(stems_to_remove, age_dist[species][age_class])
                    age_dist[species][age_class] -= max_removable
                    stems_to_remove -= max_removable
        age_dist['total_stems'] = sum(age_dist['conifer'].values()) + sum(age_dist['deciduous'].values())
        return age_dist

    def _calculate_insect_outbreak(self, mean_winter_temp_c: float, stem_density_ha: float) -> dict:
        outbreak_losses = {
            "insect_mortality_fraction": 0.0,
            "insect_carbon_to_soil_kg_m2": 0.0,
        }
        prob = 0.0
        if mean_winter_temp_c > self.p['INSECT_WARM_WINTER_THRESHOLD_C'] and stem_density_ha > self.p['INSECT_DENSITY_THRESHOLD_ha']:
            prob = self.p['INSECT_BASE_PROB']
        if self.rng.random() < prob:
            conifer_mortality_frac = self.p['INSECT_MORTALITY_RATE'] * self.p['INSECT_CONIFER_SUSCEPTIBILITY']
            deciduous_mortality_frac = self.p['INSECT_MORTALITY_RATE'] * (1.0 - self.p['INSECT_CONIFER_SUSCEPTIBILITY'])
            total_mortality_fraction = (self.p['coniferous_fraction'] * conifer_mortality_frac) + ((1.0 - self.p['coniferous_fraction']) * deciduous_mortality_frac)
            outbreak_losses["insect_mortality_fraction"] = total_mortality_fraction
        return outbreak_losses


