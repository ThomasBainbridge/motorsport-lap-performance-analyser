from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import contiguous_true_regions, first_valid_index_where, nearest_index


def detect_reference_segments(
    aligned_df: pd.DataFrame,
    *,
    min_braking_zone_length_m: float = 35.0,
    apex_search_lookahead_m: float = 140.0,
    throttle_pickup_threshold: float = 90.0,
    low_throttle_threshold: float = 15.0,
    exit_search_window_m: float = 180.0,
) -> pd.DataFrame:
    """Detect braking/apex/exit segments from the reference lap."""
    distance = aligned_df["Distance"].to_numpy(dtype=float)
    brake = aligned_df.get("ref_Brake", pd.Series(np.zeros(len(aligned_df)))).to_numpy(dtype=float)
    throttle = aligned_df.get("ref_Throttle", pd.Series(np.full(len(aligned_df), 100.0))).to_numpy(dtype=float)
    speed_grad = aligned_df.get("ref_SpeedGradient", pd.Series(np.zeros(len(aligned_df)))).to_numpy(dtype=float)

    brake_like = (brake > 0.5) | ((throttle <= low_throttle_threshold) & (speed_grad < -0.25))
    regions = contiguous_true_regions(brake_like)

    records: list[dict] = []
    step_guess = np.median(np.diff(distance)) if len(distance) > 1 else 5.0

    for seg_id, (start_idx, end_idx) in enumerate(regions, start=1):
        zone_length = distance[end_idx] - distance[start_idx]
        if zone_length < min_braking_zone_length_m:
            continue

        apex_search_end_distance = min(
            distance[-1], distance[end_idx] + apex_search_lookahead_m
        )
        apex_search_end_idx = nearest_index(distance, apex_search_end_distance)
        apex_slice = slice(start_idx, max(apex_search_end_idx, end_idx) + 1)
        apex_local_idx = int(np.argmin(aligned_df.loc[apex_slice, "ref_Speed"].to_numpy()))
        apex_idx = start_idx + apex_local_idx

        exit_search_end_distance = min(distance[-1], distance[apex_idx] + exit_search_window_m)
        exit_search_end_idx = nearest_index(distance, exit_search_end_distance)
        post_apex_throttle = throttle[apex_idx: exit_search_end_idx + 1]
        post_apex_speed_grad = speed_grad[apex_idx: exit_search_end_idx + 1]
        pickup_rel = first_valid_index_where(
            (post_apex_throttle >= throttle_pickup_threshold) & (post_apex_speed_grad > 0.1)
        )
        if pickup_rel is None:
            throttle_pickup_idx = exit_search_end_idx
        else:
            throttle_pickup_idx = apex_idx + pickup_rel

        records.append(
            {
                "SegmentId": len(records) + 1,
                "StartIdx": int(start_idx),
                "EndIdx": int(min(exit_search_end_idx, len(distance) - 1)),
                "BrakeStartIdx": int(start_idx),
                "BrakeEndIdx": int(end_idx),
                "ApexIdx": int(apex_idx),
                "ThrottlePickupIdx": int(throttle_pickup_idx),
                "StartDistance": float(distance[start_idx]),
                "EndDistance": float(distance[min(exit_search_end_idx, len(distance) - 1)]),
                "BrakeStartDistance": float(distance[start_idx]),
                "BrakeEndDistance": float(distance[end_idx]),
                "ApexDistance": float(distance[apex_idx]),
                "ThrottlePickupDistance": float(distance[throttle_pickup_idx]),
                "ApproxLength": float(distance[min(exit_search_end_idx, len(distance) - 1)] - distance[start_idx]),
                "DistanceStep": float(step_guess),
            }
        )

    return pd.DataFrame(records)
