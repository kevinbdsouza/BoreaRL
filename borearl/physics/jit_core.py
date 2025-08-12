from __future__ import annotations

"""
Numba-accelerated kernels for the physics engine.

These functions avoid Python objects in hot loops by operating on scalars and
NumPy arrays only. When Numba is unavailable, they fall back to pure-Python
implementations via no-op decorators so the call sites remain valid.
"""

from typing import Tuple

import numpy as np

try:
    from numba import njit
    have_numba = True
except Exception:  # pragma: no cover - environments without numba
    have_numba = False

    def njit(signature_or_function=None, **kwargs):  # type: ignore
        def decorator(func):
            return func

        if callable(signature_or_function):
            return signature_or_function
        return decorator


@njit(cache=True, fastmath=True)
def esat_kPa_nb(T: float) -> float:
    return 0.6108 * np.exp(17.27 * (T - 273.15) / ((T - 273.15) + 237.3))


@njit(cache=True, fastmath=True)
def delta_svp_kPa_per_K_nb(T: float) -> float:
    es = esat_kPa_nb(T)
    return 4098.0 * es / ((T - 273.15) + 237.3) ** 2


@njit(cache=True, fastmath=True)
def solve_canopy_energy_balance_nb(
    T_guess: float,
    eps_can: float,
    A_can: float,
    h_can: float,
    k_ct: float,
    A_c2t: float,
    d_ct: float,
    PT_ALPHA: float,
    LAI_actual: float,
    LUE_J_TO_G_C: float,
    PAR_FRACTION: float,
    SIGMA: float,
    Lv: float,
    DT_SECONDS: float,
    soil_stress: float,
    Q_abs_can: float,
    L_down_atm: float,
    L_up_grnd: float,
    T_trunk: float,
    T_air: float,
    ea: float,
    SWE_can: float,
    Lf: float,
    RHO_WATER: float,
    max_iters: int,
    evap_intercepted_rain_flux: float,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Returns:
        T_canopy, Rnet_can, H_can, LE_can, G_photo_energy, Cnd_can, Melt_flux_can, LE_int_rain (all per-second flux units)
    """

    def rnet(T):
        return Q_abs_can + A_can * (eps_can * (L_down_atm + L_up_grnd) - 2.0 * eps_can * SIGMA * T ** 4)

    def latent(T, Rn):
        if T <= 273.15 or Rn <= 0.0 or LAI_actual <= 0.1:
            return 0.0
        vpd = max(0.0, esat_kPa_nb(T) - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        Delta = delta_svp_kPa_per_K_nb(T)
        return PT_ALPHA * (Delta / (Delta + 0.066)) * Rn * vpd_stress * soil_stress

    def photosynthesis(T, Q_abs):
        if T <= 273.15 or Q_abs <= 0.0 or LAI_actual <= 0.1:
            return 0.0
        vpd = max(0.0, esat_kPa_nb(T) - ea)
        vpd_stress = np.exp(-0.15 * vpd)
        total_stress = vpd_stress * soil_stress
        par_abs = Q_abs * PAR_FRACTION
        gpp = par_abs * LUE_J_TO_G_C * total_stress
        return gpp

    melt_energy_sink = 0.0

    if SWE_can > 0.0:
        T_freeze = 273.15
        Rn_freeze = rnet(T_freeze)
        H_freeze = h_can * (T_freeze - T_air)
        Cnd_freeze = k_ct * A_c2t / d_ct * (T_freeze - T_trunk)
        F_freeze = Rn_freeze - H_freeze - Cnd_freeze - evap_intercepted_rain_flux
        if F_freeze > 0.0:
            energy_to_melt_all = (SWE_can * Lf * RHO_WATER) / DT_SECONDS
            if F_freeze < energy_to_melt_all:
                # gpp_rate is zero in this branch
                return (
                    T_freeze,
                    Rn_freeze,
                    -H_freeze,
                    0.0,
                    0.0,
                    -Cnd_freeze,
                    F_freeze,
                    evap_intercepted_rain_flux,
                    0.0,
                )
            else:
                melt_energy_sink = energy_to_melt_all

    T = min(max(T_guess, 181.0), 329.0)
    gpp_g_m2_s = 0.0
    for _ in range(max_iters):
        Rn = rnet(T)
        LE = latent(T, Rn)
        gpp_g_m2_s = photosynthesis(T, Q_abs_can)
        G_photo_energy = gpp_g_m2_s * (0.1 / LUE_J_TO_G_C)
        H = h_can * (T - T_air)
        Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)
        F = Rn - H - LE - G_photo_energy - Cnd - melt_energy_sink - evap_intercepted_rain_flux
        if abs(F) < 1e-3:
            break
        dT = 0.1
        Rn_p = rnet(T + dT)
        LE_p = latent(T + dT, Rn_p)
        gpp_p = photosynthesis(T + dT, Q_abs_can)
        G_p = gpp_p * (0.1 / LUE_J_TO_G_C)
        H_p = h_can * (T + dT - T_air)
        Cnd_p = k_ct * A_c2t / d_ct * (T + dT - T_trunk)
        F_p = Rn_p - H_p - LE_p - G_p - Cnd_p - melt_energy_sink - evap_intercepted_rain_flux
        dF = (F_p - F) / dT
        if abs(dF) < 1e-4:
            dF = 1e-4
        T -= F / dF
        if T < 181.0:
            T = 181.0
        elif T > 329.0:
            T = 329.0

    Rn = rnet(T)
    LE = latent(T, Rn)
    H = h_can * (T - T_air)
    Cnd = k_ct * A_c2t / d_ct * (T - T_trunk)
    G_photo_energy = gpp_g_m2_s * (0.1 / LUE_J_TO_G_C)
    # Return negative convention for flux components to match Python solver
    # Also return gpp rate in g m^-2 s^-1 for downstream accumulation
    gpp_rate = gpp_g_m2_s
    return T, Rn, -H, -LE, -G_photo_energy, -Cnd, melt_energy_sink, evap_intercepted_rain_flux, gpp_rate



