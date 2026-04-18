from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track_model import TrackGeometry


G = 9.80665
RHO_AIR_DEFAULT = 1.225


@dataclass
class VehicleParams:
    """Quasi-steady-state vehicle parameters. All SI units."""
    mass_kg: float = 800.0                  # F1 2025 minimum including driver
    mu_lat: float = 1.70                    # peak lateral grip coefficient
    mu_long: float = 1.80                   # peak longitudinal grip coefficient
    cda_m2: float = 1.10                    # drag area
    cla_m2: float = 3.50                    # downforce area, positive = down
    p_max_w: float = 760_000.0              # peak total power at crank (W)
    drivetrain_efficiency: float = 0.94
    crr: float = 0.012                      # rolling resistance
    rho_air_kgm3: float = RHO_AIR_DEFAULT


@dataclass
class EnvelopeSolution:
    distance: np.ndarray
    curvature: np.ndarray
    v_corner: np.ndarray          # m/s, lateral-grip-limited
    v_forward: np.ndarray         # m/s, forward acceleration pass (periodic)
    v_backward: np.ndarray        # m/s, backward braking pass (periodic)
    v_model: np.ndarray           # m/s, min of the three
    a_lat_model: np.ndarray       # m/s^2, always >= 0
    a_lon_model: np.ndarray       # m/s^2, signed
    lap_time_s: float
    params: VehicleParams
    periodic_residual_ms: float   # |v_model[0] - v_model[-1]|, should be small


def max_corner_speed(kappa: np.ndarray, params: VehicleParams, *, v_cap_ms: float = 400.0 / 3.6) -> np.ndarray:
    """Lateral-grip-limited speed including aero downforce."""
    kappa_abs = np.abs(np.asarray(kappa, dtype=float))
    aero_term = params.mu_lat * 0.5 * params.rho_air_kgm3 * params.cla_m2
    denom = params.mass_kg * kappa_abs - aero_term
    v2 = np.full_like(kappa_abs, v_cap_ms ** 2)
    safe = denom > 1e-6
    v2[safe] = params.mu_lat * params.mass_kg * G / denom[safe]
    return np.sqrt(np.clip(v2, 0.0, v_cap_ms ** 2))


def _available_long_accel(v_ms: float, kappa: float, params: VehicleParams, mode: str) -> float:
    """Signed a_lon the car can produce at this speed while using some lateral grip."""
    v_safe = max(v_ms, 1.0)
    n_load = params.mass_kg * G + 0.5 * params.rho_air_kgm3 * params.cla_m2 * v_safe ** 2
    f_drag = 0.5 * params.rho_air_kgm3 * params.cda_m2 * v_safe ** 2
    f_roll = params.crr * n_load

    a_lat = v_safe * v_safe * abs(kappa)
    a_lat_max = params.mu_lat * n_load / params.mass_kg
    used_lat_frac = min(a_lat / max(a_lat_max, 1e-6), 1.0)
    ellipse_long_frac = np.sqrt(max(1.0 - used_lat_frac ** 2, 0.0))

    f_long_grip = params.mu_long * n_load * ellipse_long_frac

    if mode == "drive":
        f_drive_power = params.drivetrain_efficiency * params.p_max_w / v_safe
        f_drive = min(f_drive_power, f_long_grip)
        return (f_drive - f_drag - f_roll) / params.mass_kg

    if mode == "brake":
        return -(f_long_grip + f_drag + f_roll) / params.mass_kg

    raise ValueError(f"mode must be 'drive' or 'brake', got {mode!r}")


def forward_pass(
    distance: np.ndarray,
    kappa: np.ndarray,
    v_corner: np.ndarray,
    params: VehicleParams,
    v_start_ms: float | None = None,
) -> np.ndarray:
    """Integrate v forward under drive+grip limits, capped at v_corner everywhere."""
    n = len(distance)
    v = np.empty(n, dtype=float)
    v[0] = min(v_start_ms, v_corner[0]) if v_start_ms is not None else v_corner[0]
    for i in range(n - 1):
        ds = distance[i + 1] - distance[i]
        a = _available_long_accel(v[i], kappa[i], params, mode="drive")
        v_next_sq = max(v[i] ** 2 + 2.0 * a * ds, 0.0)
        v[i + 1] = min(np.sqrt(v_next_sq), v_corner[i + 1])
    return v


def backward_pass(
    distance: np.ndarray,
    kappa: np.ndarray,
    v_corner: np.ndarray,
    params: VehicleParams,
    v_end_ms: float | None = None,
) -> np.ndarray:
    """Integrate v backward under brake+grip limits, capped at v_corner everywhere."""
    n = len(distance)
    v = np.empty(n, dtype=float)
    v[-1] = min(v_end_ms, v_corner[-1]) if v_end_ms is not None else v_corner[-1]
    for i in range(n - 1, 0, -1):
        ds = distance[i] - distance[i - 1]
        a = _available_long_accel(v[i], kappa[i], params, mode="brake")  # a < 0
        v_prev_sq = max(v[i] ** 2 - 2.0 * a * ds, 0.0)
        v[i - 1] = min(np.sqrt(v_prev_sq), v_corner[i - 1])
    return v


def _periodic_forward_pass(
    distance: np.ndarray,
    kappa: np.ndarray,
    v_corner: np.ndarray,
    params: VehicleParams,
    *,
    max_iters: int = 6,
    tol_ms: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Forward pass iterated to a periodic fixed point: v[0] == v[-1].

    Returns (v_fwd, residual_ms). The iteration is guaranteed to reach
    a fixed point within a few passes because v_corner clipping forces
    the solution through the slowest corner regardless of boundary.
    """
    v_start = float(v_corner[0])
    v_fwd = forward_pass(distance, kappa, v_corner, params, v_start_ms=v_start)
    residual = abs(v_fwd[-1] - v_fwd[0])
    for _ in range(max_iters - 1):
        if residual < tol_ms:
            break
        v_start = float(v_fwd[-1])
        v_fwd = forward_pass(distance, kappa, v_corner, params, v_start_ms=v_start)
        residual = abs(v_fwd[-1] - v_fwd[0])
    return v_fwd, residual


def _periodic_backward_pass(
    distance: np.ndarray,
    kappa: np.ndarray,
    v_corner: np.ndarray,
    params: VehicleParams,
    *,
    max_iters: int = 6,
    tol_ms: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Backward pass iterated to a periodic fixed point: v[0] == v[-1]."""
    v_end = float(v_corner[-1])
    v_bwd = backward_pass(distance, kappa, v_corner, params, v_end_ms=v_end)
    residual = abs(v_bwd[0] - v_bwd[-1])
    for _ in range(max_iters - 1):
        if residual < tol_ms:
            break
        v_end = float(v_bwd[0])
        v_bwd = backward_pass(distance, kappa, v_corner, params, v_end_ms=v_end)
        residual = abs(v_bwd[0] - v_bwd[-1])
    return v_bwd, residual


def solve_envelope(
    track: TrackGeometry,
    params: VehicleParams,
    *,
    max_iters: int = 6,
    tol_ms: float = 0.05,
) -> EnvelopeSolution:
    """Full QSS envelope on a closed lap.

    The forward and backward passes are iterated to a periodic fixed point
    so that v_model(0) ~= v_model(L). Driver boundary conditions are
    intentionally not accepted -- on a closed loop the start/finish speed
    is determined by the track and car, not by whatever noisy first sample
    the driver's telemetry happens to contain.
    """
    v_corner = max_corner_speed(track.curvature, params)

    v_fwd, fwd_res = _periodic_forward_pass(
        track.distance, track.curvature, v_corner, params,
        max_iters=max_iters, tol_ms=tol_ms,
    )
    v_bwd, bwd_res = _periodic_backward_pass(
        track.distance, track.curvature, v_corner, params,
        max_iters=max_iters, tol_ms=tol_ms,
    )

    v_model = np.minimum.reduce([v_corner, v_fwd, v_bwd])

    a_lat = v_model ** 2 * np.abs(track.curvature)
    a_lon = v_model * np.gradient(v_model, track.distance)

    ds = np.gradient(track.distance)
    lap_time = float(np.sum(ds / np.clip(v_model, 1.0, None)))

    periodic_residual = float(abs(v_model[0] - v_model[-1]))

    return EnvelopeSolution(
        distance=track.distance,
        curvature=track.curvature,
        v_corner=v_corner,
        v_forward=v_fwd,
        v_backward=v_bwd,
        v_model=v_model,
        a_lat_model=a_lat,
        a_lon_model=a_lon,
        lap_time_s=lap_time,
        params=params,
        periodic_residual_ms=periodic_residual,
    )
