from __future__ import annotations

import numpy as np
import pandas as pd

from .telemetry import make_monotonic_time_seconds


CONTINUOUS_CHANNELS = ("Speed", "Throttle", "RPM")
DISCRETE_CHANNELS = ("Brake", "nGear", "DRS")


def _step_interpolate(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(x_old, x_new, side="right") - 1
    idx = np.clip(idx, 0, len(x_old) - 1)
    return y_old[idx]


def _interp_continuous(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x_old, y_old)


def align_telemetry_pair(
    reference_tel: pd.DataFrame,
    comparison_tel: pd.DataFrame,
    *,
    distance_step_m: float = 5.0,
) -> pd.DataFrame:
    """Align two laps onto a common distance basis."""
    if distance_step_m <= 0:
        raise ValueError("distance_step_m must be > 0")

    ref_x = reference_tel["Distance"].to_numpy(dtype=float)
    cmp_x = comparison_tel["Distance"].to_numpy(dtype=float)

    start_distance = max(ref_x.min(), cmp_x.min())
    end_distance = min(ref_x.max(), cmp_x.max())
    if end_distance <= start_distance:
        raise ValueError("Aligned overlap between laps is empty.")

    common_distance = np.arange(start_distance, end_distance + distance_step_m, distance_step_m)

    aligned = {"Distance": common_distance}

    ref_time = make_monotonic_time_seconds(reference_tel)
    cmp_time = make_monotonic_time_seconds(comparison_tel)

    for prefix, tel, x_old, time_old in (
        ("ref", reference_tel, ref_x, ref_time),
        ("cmp", comparison_tel, cmp_x, cmp_time),
    ):
        aligned[f"{prefix}_TimeSeconds"] = _interp_continuous(common_distance, x_old, time_old)

        for channel in CONTINUOUS_CHANNELS:
            if channel in tel.columns:
                y_old = tel[channel].to_numpy(dtype=float)
                aligned[f"{prefix}_{channel}"] = _interp_continuous(common_distance, x_old, y_old)

        for channel in DISCRETE_CHANNELS:
            if channel in tel.columns:
                y_old = tel[channel].to_numpy(dtype=float)
                aligned[f"{prefix}_{channel}"] = _step_interpolate(common_distance, x_old, y_old)

    aligned_df = pd.DataFrame(aligned)
    aligned_df["DeltaSeconds"] = (
        aligned_df["cmp_TimeSeconds"] - aligned_df["ref_TimeSeconds"]
    )

    for prefix in ("ref", "cmp"):
        speed_col = f"{prefix}_Speed"
        if speed_col in aligned_df.columns:
            aligned_df[f"{prefix}_SpeedGradient"] = np.gradient(
                aligned_df[speed_col].to_numpy(dtype=float),
                aligned_df["Distance"].to_numpy(dtype=float),
            )

    return aligned_df
