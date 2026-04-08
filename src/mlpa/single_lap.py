from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _extract_interval_features
from .telemetry import lap_to_car_telemetry, make_monotonic_time_seconds


SINGLE_LAP_CLUSTER_ALIAS_MAP = {
    "cmp_entry_speed_kph": "lap_entry_speed_kph",
    "cmp_min_speed_kph": "lap_min_speed_kph",
    "cmp_exit_speed_kph": "lap_exit_speed_kph",
    "cmp_mean_throttle_pct": "lap_mean_throttle_pct",
    "cmp_brake_fraction": "lap_brake_fraction",
    "cmp_segment_length_m": "lap_segment_length_m",
    "cmp_apex_to_exit_gain_kph": "lap_apex_to_exit_gain_kph",
}


def build_single_lap_analysis_df(lap) -> pd.DataFrame:
    telemetry = lap_to_car_telemetry(lap)
    distance = telemetry["Distance"].to_numpy(dtype=float)

    data: dict[str, np.ndarray] = {
        "Distance": distance,
        "ref_TimeSeconds": make_monotonic_time_seconds(telemetry),
    }

    for channel in ("Speed", "Throttle", "RPM", "Brake", "nGear", "DRS"):
        if channel in telemetry.columns:
            data[f"ref_{channel}"] = telemetry[channel].to_numpy(dtype=float)

    analysis_df = pd.DataFrame(data)
    if "ref_Speed" in analysis_df.columns and len(analysis_df) > 1:
        analysis_df["ref_SpeedGradient"] = np.gradient(
            analysis_df["ref_Speed"].to_numpy(dtype=float),
            analysis_df["Distance"].to_numpy(dtype=float),
        )
    else:
        analysis_df["ref_SpeedGradient"] = 0.0

    return analysis_df


def compute_single_lap_segment_features(
    analysis_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    *,
    throttle_pickup_threshold: float = 90.0,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for row in segments_df.itertuples(index=False):
        start_idx = int(row.StartIdx)
        end_idx = int(row.EndIdx)

        lap_feats = _extract_interval_features(
            analysis_df,
            prefix="ref",
            start_idx=start_idx,
            end_idx=end_idx,
            throttle_pickup_threshold=throttle_pickup_threshold,
        )

        combined = {
            "SegmentId": int(row.SegmentId),
            "SegmentLabel": getattr(row, "SegmentLabel", f"S{int(row.SegmentId)}"),
            "StartDistance": float(row.StartDistance),
            "EndDistance": float(row.EndDistance),
            "BrakeStartDistance": float(row.BrakeStartDistance),
            "ApexDistance": float(row.ApexDistance),
            "ThrottlePickupDistance": float(row.ThrottlePickupDistance),
        }

        for key, value in lap_feats.items():
            combined[key.replace("ref_", "lap_")] = value

        for alias_name, source_name in SINGLE_LAP_CLUSTER_ALIAS_MAP.items():
            combined[alias_name] = combined[source_name]

        rows.append(combined)

    return pd.DataFrame(rows)


def single_lap_overall_summary(analysis_df: pd.DataFrame, segments_df: pd.DataFrame) -> dict[str, float]:
    speed = analysis_df["ref_Speed"].to_numpy(dtype=float)
    throttle = analysis_df.get("ref_Throttle", pd.Series(np.full(len(analysis_df), 100.0))).to_numpy(dtype=float)
    brake = analysis_df.get("ref_Brake", pd.Series(np.zeros(len(analysis_df)))).to_numpy(dtype=float)
    time_s = analysis_df["ref_TimeSeconds"].to_numpy(dtype=float)

    heavy_braking_events = int(segments_df["SegmentId"].nunique()) if not segments_df.empty else 0

    return {
        "lap_time_s": float(time_s[-1] - time_s[0]) if len(time_s) else np.nan,
        "top_speed_kph": float(np.max(speed)) if len(speed) else np.nan,
        "mean_speed_kph": float(np.mean(speed)) if len(speed) else np.nan,
        "full_throttle_fraction": float(np.mean(throttle >= 90.0)) if len(throttle) else np.nan,
        "brake_fraction": float(np.mean(brake > 0.5)) if len(brake) else np.nan,
        "n_segments": float(len(segments_df)),
        "n_heavy_braking_events": float(heavy_braking_events),
    }
