from __future__ import annotations

import numpy as np
import pandas as pd

from mlpa.single_lap import compute_single_lap_segment_features, single_lap_overall_summary


def _make_analysis_df() -> pd.DataFrame:
    distance = np.arange(0.0, 60.0, 10.0)
    speed = np.array([250.0, 220.0, 140.0, 110.0, 170.0, 230.0])
    return pd.DataFrame(
        {
            "Distance": distance,
            "ref_TimeSeconds": np.array([0.0, 0.2, 0.5, 0.9, 1.4, 2.0]),
            "ref_Speed": speed,
            "ref_Throttle": np.array([100.0, 20.0, 0.0, 30.0, 85.0, 100.0]),
            "ref_Brake": np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            "ref_SpeedGradient": np.gradient(speed, distance),
        }
    )


def test_single_lap_segment_features_basic_columns() -> None:
    analysis_df = _make_analysis_df()
    segments_df = pd.DataFrame([
        {
            "SegmentId": 1,
            "SegmentLabel": "Test Corner",
            "StartIdx": 0,
            "EndIdx": 5,
            "StartDistance": 0.0,
            "EndDistance": 50.0,
            "BrakeStartDistance": 10.0,
            "ApexDistance": 30.0,
            "ThrottlePickupDistance": 40.0,
        }
    ])
    features = compute_single_lap_segment_features(analysis_df, segments_df, throttle_pickup_threshold=90.0)
    assert "lap_entry_speed_kph" in features.columns
    assert "lap_min_speed_kph" in features.columns
    assert "lap_exit_speed_kph" in features.columns
    assert "cmp_min_speed_kph" in features.columns


def test_single_lap_overall_summary_counts_segments() -> None:
    analysis_df = _make_analysis_df()
    segments_df = pd.DataFrame([{"SegmentId": 1}, {"SegmentId": 2}])
    summary = single_lap_overall_summary(analysis_df, segments_df)
    assert summary["n_segments"] == 2.0
    assert summary["n_heavy_braking_events"] == 2.0
