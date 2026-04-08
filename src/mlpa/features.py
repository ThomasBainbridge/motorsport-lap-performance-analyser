from __future__ import annotations

import numpy as np
import pandas as pd


CHANNEL_DEFAULTS = {
    "Throttle": 100.0,
    "Brake": 0.0,
}


def _robust_entry_speed(speed: np.ndarray, brake: np.ndarray) -> float:
    """Estimate corner entry speed more robustly than a single boundary sample."""
    brake_mask = brake > 0.5
    if np.any(brake_mask):
        brake_start_local = int(np.argmax(brake_mask))
        window_start = max(0, brake_start_local - 3)
        window_end = brake_start_local + 1
        return float(np.mean(speed[window_start:window_end]))

    n = len(speed)
    if n <= 1:
        return float(speed[0])
    window_end = max(1, int(np.ceil(0.15 * n)))
    return float(np.mean(speed[:window_end]))



def _extract_interval_features(
    aligned_df: pd.DataFrame,
    *,
    prefix: str,
    start_idx: int,
    end_idx: int,
    throttle_pickup_threshold: float = 90.0,
) -> dict[str, float]:
    segment = aligned_df.iloc[start_idx : end_idx + 1].copy()
    distance = segment["Distance"].to_numpy(dtype=float)
    speed = segment[f"{prefix}_Speed"].to_numpy(dtype=float)
    throttle = segment.get(
        f"{prefix}_Throttle", pd.Series(np.full(len(segment), CHANNEL_DEFAULTS["Throttle"]))
    ).to_numpy(dtype=float)
    brake = segment.get(
        f"{prefix}_Brake", pd.Series(np.zeros(len(segment)) + CHANNEL_DEFAULTS["Brake"])
    ).to_numpy(dtype=float)
    time_s = segment[f"{prefix}_TimeSeconds"].to_numpy(dtype=float)

    local_brake_mask = brake > 0.5
    if np.any(local_brake_mask):
        brake_start_local = int(np.argmax(local_brake_mask))
        brake_end_local = len(local_brake_mask) - 1 - int(np.argmax(local_brake_mask[::-1]))
    else:
        brake_start_local = 0
        brake_end_local = 0

    apex_local = int(np.argmin(speed))
    post_apex_throttle = throttle[apex_local:]
    pickup_candidates = np.flatnonzero(post_apex_throttle >= throttle_pickup_threshold)
    throttle_pickup_local = int(apex_local + pickup_candidates[0]) if pickup_candidates.size else len(speed) - 1

    entry_speed = _robust_entry_speed(speed, brake)
    min_speed = float(np.min(speed))
    exit_speed = float(speed[-1])
    mean_speed = float(np.mean(speed))
    straightline_gain = float(exit_speed - min_speed)
    braking_drop = float(entry_speed - min_speed)

    return {
        f"{prefix}_segment_time_s": float(time_s[-1] - time_s[0]),
        f"{prefix}_segment_length_m": float(distance[-1] - distance[0]),
        f"{prefix}_entry_speed_kph": float(entry_speed),
        f"{prefix}_min_speed_kph": min_speed,
        f"{prefix}_exit_speed_kph": exit_speed,
        f"{prefix}_max_speed_kph": float(np.max(speed)),
        f"{prefix}_mean_speed_kph": mean_speed,
        f"{prefix}_entry_to_apex_drop_kph": braking_drop,
        f"{prefix}_apex_to_exit_gain_kph": straightline_gain,
        f"{prefix}_brake_start_distance_m": float(distance[brake_start_local]),
        f"{prefix}_brake_end_distance_m": float(distance[brake_end_local]),
        f"{prefix}_apex_distance_m": float(distance[apex_local]),
        f"{prefix}_throttle_pickup_distance_m": float(distance[throttle_pickup_local]),
        f"{prefix}_full_throttle_fraction": float(np.mean(throttle >= throttle_pickup_threshold)),
        f"{prefix}_mean_throttle_pct": float(np.mean(throttle)),
        f"{prefix}_brake_fraction": float(np.mean(brake > 0.5)),
    }



def compute_segment_features(
    aligned_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    *,
    throttle_pickup_threshold: float = 90.0,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for row in segments_df.itertuples(index=False):
        start_idx = int(row.StartIdx)
        end_idx = int(row.EndIdx)

        ref_feats = _extract_interval_features(
            aligned_df,
            prefix="ref",
            start_idx=start_idx,
            end_idx=end_idx,
            throttle_pickup_threshold=throttle_pickup_threshold,
        )
        cmp_feats = _extract_interval_features(
            aligned_df,
            prefix="cmp",
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
        combined.update(ref_feats)
        combined.update(cmp_feats)

        combined["time_loss_s"] = combined["cmp_segment_time_s"] - combined["ref_segment_time_s"]
        combined["entry_speed_delta_kph"] = combined["cmp_entry_speed_kph"] - combined["ref_entry_speed_kph"]
        combined["min_speed_delta_kph"] = combined["cmp_min_speed_kph"] - combined["ref_min_speed_kph"]
        combined["exit_speed_delta_kph"] = combined["cmp_exit_speed_kph"] - combined["ref_exit_speed_kph"]
        combined["mean_speed_delta_kph"] = combined["cmp_mean_speed_kph"] - combined["ref_mean_speed_kph"]
        combined["brake_start_delta_m"] = combined["cmp_brake_start_distance_m"] - combined["ref_brake_start_distance_m"]
        combined["brake_end_delta_m"] = combined["cmp_brake_end_distance_m"] - combined["ref_brake_end_distance_m"]
        combined["apex_delta_m"] = combined["cmp_apex_distance_m"] - combined["ref_apex_distance_m"]
        combined["throttle_pickup_delta_m"] = (
            combined["cmp_throttle_pickup_distance_m"] - combined["ref_throttle_pickup_distance_m"]
        )
        combined["mean_throttle_delta_pct"] = combined["cmp_mean_throttle_pct"] - combined["ref_mean_throttle_pct"]
        combined["full_throttle_fraction_delta"] = (
            combined["cmp_full_throttle_fraction"] - combined["ref_full_throttle_fraction"]
        )
        combined["brake_fraction_delta"] = combined["cmp_brake_fraction"] - combined["ref_brake_fraction"]
        combined["entry_to_apex_drop_delta_kph"] = (
            combined["cmp_entry_to_apex_drop_kph"] - combined["ref_entry_to_apex_drop_kph"]
        )
        combined["apex_to_exit_gain_delta_kph"] = (
            combined["cmp_apex_to_exit_gain_kph"] - combined["ref_apex_to_exit_gain_kph"]
        )
        rows.append(combined)

    return pd.DataFrame(rows)
