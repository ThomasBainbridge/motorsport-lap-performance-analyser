from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackGeometry:
    """Track geometry on the native distance grid from FastF1.

    Critically, we do NOT resample to a uniform grid. The same distance array
    carries x, y, curvature, and is shared with driver telemetry channels
    (speed, throttle, brake). This means alignment between driver data and
    track features is exact -- not interpolation-dependent.
    """
    distance: np.ndarray       # metres, monotone non-decreasing, possibly non-uniform
    x: np.ndarray              # metres
    y: np.ndarray              # metres
    heading: np.ndarray        # radians
    curvature: np.ndarray      # 1/metres, signed, same grid as distance


def _menger_signed_curvature(
    distance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    chord_m: float = 8.0,
) -> np.ndarray:
    """Signed curvature by fitting a circle through three chord-length-spaced points.

    At each index i, take points at s_i - chord_m, s_i, and s_i + chord_m
    (using nearest-available indices), fit a circle, and return 1/R signed
    by the cross product. This is a direct geometric measure of curvature,
    not a differentiation of noisy position data, and preserves apex peaks
    that Savitzky-Golay smoothing would wipe out.

    Points near the start/end of the array where a full chord doesn't fit
    are returned as zero (endpoints are almost always on straights).
    """
    n = len(distance)
    kappa = np.zeros(n, dtype=float)
    for i in range(n):
        s_back = distance[i] - chord_m
        s_fwd = distance[i] + chord_m
        j_back = int(np.searchsorted(distance, s_back, side="right") - 1)
        j_fwd = int(np.searchsorted(distance, s_fwd, side="left"))
        if j_back < 0 or j_fwd >= n or j_back == i or j_fwd == i:
            continue
        x1, y1 = x[j_back], y[j_back]
        x2, y2 = x[i], y[i]
        x3, y3 = x[j_fwd], y[j_fwd]
        d12 = np.hypot(x2 - x1, y2 - y1)
        d13 = np.hypot(x3 - x1, y3 - y1)
        d23 = np.hypot(x3 - x2, y3 - y2)
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        denom = d12 * d13 * d23
        if denom < 1e-9:
            continue
        kappa[i] = 2.0 * cross / denom
    return kappa


def _robust_median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Median filter that suppresses isolated outliers without broadening real peaks."""
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = len(values)
    out = values.copy()
    for i in range(half, n - half):
        out[i] = np.median(values[i - half : i + half + 1])
    return out


def build_track_from_position(
    distance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chord_m: float = 8.0,
    median_window: int = 5,
) -> TrackGeometry:
    """Build a TrackGeometry directly on the input distance grid.

    No uniform-grid resampling happens here. The returned .distance is the
    same array that was passed in. This eliminates the alignment errors
    that arose from interpolating driver speed onto a separate track grid.
    """
    distance = np.asarray(distance, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(distance) < 20:
        raise ValueError("Need at least 20 samples to build track geometry.")

    kappa_raw = _menger_signed_curvature(distance, x, y, chord_m=chord_m)
    kappa = _robust_median_filter(kappa_raw, window=median_window)

    # Heading from a coarse finite difference, purely for diagnostic use
    dx = np.gradient(x, distance)
    dy = np.gradient(y, distance)
    heading = np.arctan2(dy, dx)

    return TrackGeometry(
        distance=distance,
        x=x,
        y=y,
        heading=heading,
        curvature=kappa,
    )


# Backwards-compatibility shim -- old call sites imported estimate_curvature
# and resample_to_track. Preserve those names but route them through the
# new robust implementation.

def estimate_curvature(
    distance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    step_m: float = 2.0,          # kept for API compatibility, no longer used internally
    smoothing_window_m: float = 30.0,  # kept for API compatibility, no longer used internally
    polyorder: int = 3,           # kept for API compatibility, no longer used internally
) -> TrackGeometry:
    """Deprecated signature; now routes to build_track_from_position.

    The step_m / smoothing_window_m / polyorder arguments are ignored. Kept
    for compatibility so older callers do not crash.
    """
    del step_m, smoothing_window_m, polyorder
    return build_track_from_position(distance, x, y)


def resample_to_track(track: TrackGeometry, source_distance: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate a scalar channel onto the track's distance grid.

    With the rebuilt pipeline, track.distance is the native FastF1 grid, so
    this is usually a no-op when called with source_distance == track.distance.
    Kept for callers that still want to line up externally-sampled channels.
    """
    return np.interp(track.distance, source_distance, values)
