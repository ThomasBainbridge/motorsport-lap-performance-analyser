from __future__ import annotations

import numpy as np
import pandas as pd


def _segment_descriptor(row) -> str:
    label = getattr(row, "SegmentLabel", None)
    if isinstance(label, str) and label.strip():
        return label
    return f"S{int(row.SegmentId)}"



def rank_segments(segment_features: pd.DataFrame) -> pd.DataFrame:
    if segment_features.empty:
        return segment_features.copy()

    ranked = segment_features.copy()
    if "SegmentLabel" not in ranked.columns:
        ranked["SegmentLabel"] = ranked["SegmentId"].map(lambda x: f"S{int(x)}")
    ranked["abs_time_loss_s"] = ranked["time_loss_s"].abs()
    ranked = ranked.sort_values(["abs_time_loss_s", "SegmentId"], ascending=[False, True]).reset_index(drop=True)

    narratives: list[str] = []
    for row in ranked.itertuples(index=False):
        drivers: list[str] = []

        if row.brake_start_delta_m < -5.0:
            drivers.append("earlier braking")
        elif row.brake_start_delta_m > 5.0:
            drivers.append("later braking")

        if row.min_speed_delta_kph < -3.0:
            drivers.append("lower minimum speed")
        elif row.min_speed_delta_kph > 3.0:
            drivers.append("higher minimum speed")

        if row.throttle_pickup_delta_m > 10.0:
            drivers.append("later throttle pickup")
        elif row.throttle_pickup_delta_m < -10.0:
            drivers.append("earlier throttle pickup")

        if row.exit_speed_delta_kph < -3.0:
            drivers.append("weaker exit speed")
        elif row.exit_speed_delta_kph > 3.0:
            drivers.append("stronger exit speed")

        if not drivers:
            drivers.append("small distributed effects")

        sign = "lost" if row.time_loss_s >= 0 else "gained"
        label = _segment_descriptor(row)
        narratives.append(
            f"{label}: {sign} {abs(row.time_loss_s):.3f} s, driven mainly by "
            + ", ".join(drivers)
            + "."
        )

    ranked["Narrative"] = narratives
    return ranked



def overall_summary(aligned_df: pd.DataFrame) -> dict[str, float]:
    total_delta = float(aligned_df["DeltaSeconds"].iloc[-1] - aligned_df["DeltaSeconds"].iloc[0])
    max_loss = float(np.max(aligned_df["DeltaSeconds"]))
    min_gain = float(np.min(aligned_df["DeltaSeconds"]))
    return {
        "total_delta_s": total_delta,
        "max_cumulative_loss_s": max_loss,
        "max_cumulative_gain_s": min_gain,
    }
