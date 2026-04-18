from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .track_model import TrackGeometry
from .vehicle_model import EnvelopeSolution, VehicleParams, G


@dataclass
class EnvelopeComparison:
    distance: np.ndarray
    v_driver_ms: np.ndarray
    v_model_ms: np.ndarray
    speed_gap_ms: np.ndarray          # v_model - v_driver, positive = headroom
    grip_utilisation: np.ndarray      # sqrt((a_lat/a_lat_max)^2 + (a_lon/a_lon_max)^2)
    lap_time_driver_s: float
    lap_time_model_s: float
    unused_time_s: float              # driver - model
    per_point_ds: np.ndarray


def _smoothed_derivative(y: np.ndarray, x: np.ndarray, half_window: int = 5) -> np.ndarray:
    """Centred finite difference over a wider stencil than np.gradient.

    Uses points at i - half_window and i + half_window rather than adjacent
    neighbours. This damps derivative noise substantially without broadening
    peaks in the underlying y -- the value at i is unchanged; we just use
    a wider denominator.
    """
    n = len(y)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n - 1, i + half_window)
        if hi <= lo:
            continue
        out[i] = (y[hi] - y[lo]) / max(x[hi] - x[lo], 1e-9)
    return out


def _integrate_time(distance: np.ndarray, speed_ms: np.ndarray) -> float:
    """Trapezoidal integration of dt = ds/v, robust to mid-lap pit-lane speeds."""
    v = np.clip(speed_ms, 1.0, None)
    # dt_i = ds_i / v_avg_i, using trapezoidal rule between samples
    ds = np.diff(distance)
    v_avg = 0.5 * (v[:-1] + v[1:])
    return float(np.sum(ds / v_avg))


def compare_lap_to_envelope(
    track: TrackGeometry,
    solution: EnvelopeSolution,
    speed_driver_ms: np.ndarray,
    params: VehicleParams,
) -> EnvelopeComparison:
    """Compare driver lap to its QSS envelope on the SAME distance grid.

    Driver speed, track curvature and envelope solution must all share
    track.distance as their distance axis. This is now guaranteed by the
    rewritten pipeline (native FastF1 grid used throughout), so no further
    alignment is done here.
    """
    if len(speed_driver_ms) != len(track.distance):
        raise ValueError(
            f"speed_driver_ms ({len(speed_driver_ms)}) must match track.distance ({len(track.distance)})"
        )

    d = track.distance
    kappa = track.curvature

    a_lat = speed_driver_ms ** 2 * np.abs(kappa)
    # Wider stencil (~10 samples ~= 10-20 m) damps derivative noise on a_lon
    a_lon = speed_driver_ms * _smoothed_derivative(speed_driver_ms, d, half_window=5)

    n_load = params.mass_kg * G + 0.5 * params.rho_air_kgm3 * params.cla_m2 * speed_driver_ms ** 2
    a_lat_limit = params.mu_lat * n_load / params.mass_kg
    a_lon_limit = params.mu_long * n_load / params.mass_kg
    util = np.sqrt(
        (a_lat / np.clip(a_lat_limit, 1e-3, None)) ** 2
        + (a_lon / np.clip(a_lon_limit, 1e-3, None)) ** 2
    )

    t_driver = _integrate_time(d, speed_driver_ms)
    t_model = _integrate_time(d, solution.v_model)

    # Per-point ds for downstream segment integration. Use symmetric spacings.
    ds = np.zeros_like(d)
    ds[1:-1] = 0.5 * (d[2:] - d[:-2])
    ds[0] = d[1] - d[0]
    ds[-1] = d[-1] - d[-2]

    return EnvelopeComparison(
        distance=d,
        v_driver_ms=speed_driver_ms,
        v_model_ms=solution.v_model,
        speed_gap_ms=solution.v_model - speed_driver_ms,
        grip_utilisation=util,
        lap_time_driver_s=t_driver,
        lap_time_model_s=t_model,
        unused_time_s=t_driver - t_model,
        per_point_ds=ds,
    )


def summarise_by_segment(comparison: EnvelopeComparison, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate envelope results over detected corner segments."""
    rows = []
    d = comparison.distance
    for seg in segments_df.itertuples(index=False):
        s_start = float(seg.StartDistance)
        s_end = float(seg.EndDistance)
        mask = (d >= s_start) & (d <= s_end)
        if not np.any(mask):
            continue
        d_seg = d[mask]
        if len(d_seg) < 2:
            continue
        v_drv_seg = comparison.v_driver_ms[mask]
        v_mdl_seg = comparison.v_model_ms[mask]
        t_driver = _integrate_time(d_seg, v_drv_seg)
        t_model = _integrate_time(d_seg, v_mdl_seg)
        rows.append({
            "SegmentId": int(seg.SegmentId),
            "SegmentLabel": getattr(seg, "SegmentLabel", f"S{int(seg.SegmentId)}"),
            "driver_time_s": t_driver,
            "model_time_s": t_model,
            "unused_time_s": t_driver - t_model,
            "mean_util": float(np.mean(comparison.grip_utilisation[mask])),
            "peak_util": float(np.max(comparison.grip_utilisation[mask])),
            "mean_speed_gap_ms": float(np.mean(comparison.speed_gap_ms[mask])),
        })
    return pd.DataFrame(rows)
