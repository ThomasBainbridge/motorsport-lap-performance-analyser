from __future__ import annotations

from dataclasses import replace

import numpy as np

from .track_model import TrackGeometry
from .vehicle_model import VehicleParams, G


def _lateral_accel(v_ms: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return v_ms ** 2 * np.abs(kappa)


def _longitudinal_accel(v_ms: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    return v_ms * np.gradient(v_ms, distance_m)


def calibrate_from_lap(
    track: TrackGeometry,
    speed_ms: np.ndarray,
    throttle_pct: np.ndarray,
    brake: np.ndarray,
    *,
    initial: VehicleParams | None = None,
) -> VehicleParams:
    """Fit mu_lat, mu_long, ClA, CdA, P_max from a single lap's telemetry.

    All input arrays must be aligned to `track.distance`. Fitted values are
    sanity-bounded; failed fits fall back to `initial`.
    """
    if initial is None:
        initial = VehicleParams()

    a_lat = _lateral_accel(speed_ms, track.curvature)
    a_lon = _longitudinal_accel(speed_ms, track.distance)

    # ---- mu_lat from low-speed peak cornering (aero contribution negligible) ----
    low_speed = speed_ms < (150.0 / 3.6)
    cornering = a_lat > 0.5 * G
    low_corner = low_speed & cornering
    if np.any(low_corner):
        mu_lat = float(np.quantile(a_lat[low_corner] / G, 0.97))
    else:
        mu_lat = initial.mu_lat

    # ---- ClA from a_lat_peak(v) = mu_lat * g * (1 + 0.5*rho*ClA*v^2/(m*g)) ----
    # Linear fit of y = a_lat/g vs x = v^2 on the upper envelope of fast corners.
    high_corner = (speed_ms > (140.0 / 3.6)) & cornering
    cla = initial.cla_m2
    if np.count_nonzero(high_corner) > 20:
        y_all = a_lat[high_corner] / G
        x_all = speed_ms[high_corner] ** 2
        bins = np.linspace(x_all.min(), x_all.max(), 8)
        ids = np.digitize(x_all, bins)
        xs, ys = [], []
        for b in np.unique(ids):
            sel = ids == b
            if np.count_nonzero(sel) < 3:
                continue
            thr = np.quantile(y_all[sel], 0.9)
            keep = sel & (y_all >= thr)
            xs.append(x_all[keep])
            ys.append(y_all[keep])
        if xs:
            x_fit = np.concatenate(xs)
            y_fit = np.concatenate(ys)
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            if slope > 0 and intercept > 0:
                mu_lat = max(mu_lat, float(intercept))
                cla_fit = float(slope * initial.mass_kg * G * 2.0
                                / (mu_lat * initial.rho_air_kgm3))
                if 0.5 < cla_fit < 6.0:
                    cla = cla_fit

    # ---- mu_long from low-speed straight-line braking (aero term small) ----
    straight = a_lat < 0.5 * G
    braking = (brake > 0.5) & (a_lon < -0.3 * G)
    low_brake = straight & braking & (speed_ms < 150.0 / 3.6)
    if np.any(low_brake):
        mu_long = float(np.quantile(-a_lon[low_brake] / G, 0.97))
    else:
        mu_long = initial.mu_long
    mu_long = max(mu_long, mu_lat)  # longitudinal grip usually meets-or-exceeds lateral

    # ---- P_max and CdA from full-throttle straight-line acceleration ----
    # m*a + F_roll = eta*P/v - 0.5*rho*CdA*v^2
    # Linear regression in (eta*P, CdA) with predictors [1/v, -0.5*rho*v^2].
    p_max = initial.p_max_w
    cda = initial.cda_m2
    full_throttle_straight = (throttle_pct > 95.0) & (brake < 0.5) & (a_lat < 0.3 * G) & (a_lon > 0.1 * G)
    if np.count_nonzero(full_throttle_straight) > 20:
        v = speed_ms[full_throttle_straight]
        a = a_lon[full_throttle_straight]
        n_load = initial.mass_kg * G + 0.5 * initial.rho_air_kgm3 * cla * v ** 2
        f_roll = initial.crr * n_load
        y = initial.mass_kg * a + f_roll
        X = np.column_stack([1.0 / v, -0.5 * initial.rho_air_kgm3 * v ** 2])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            p_eff = float(coef[0])
            cda_fit = float(coef[1])
            p_fit = p_eff / initial.drivetrain_efficiency
            if 0.5 < cda_fit < 2.5 and 3.0e5 < p_fit < 1.2e6:
                p_max = p_fit
                cda = cda_fit
        except np.linalg.LinAlgError:
            pass

    return replace(
        initial,
        mu_lat=mu_lat,
        mu_long=mu_long,
        cla_m2=cla,
        cda_m2=cda,
        p_max_w=p_max,
    )
