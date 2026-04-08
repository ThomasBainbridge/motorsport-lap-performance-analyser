import pandas as pd
import numpy as np

from mlpa.alignment import align_telemetry_pair


def build_tel(offset_time: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Distance": [0.0, 50.0, 100.0, 150.0],
            "Time": pd.to_timedelta([0.0 + offset_time, 1.0 + offset_time, 2.0 + offset_time, 3.0 + offset_time], unit="s"),
            "Speed": [100.0, 120.0, 110.0, 130.0],
            "Throttle": [100.0, 50.0, 60.0, 100.0],
            "Brake": [0, 1, 0, 0],
            "nGear": [7, 5, 6, 8],
            "RPM": [10000, 9000, 9500, 11000],
            "DRS": [0, 0, 0, 2],
        }
    )


def test_align_telemetry_pair_basic():
    ref = build_tel(0.0)
    cmp = build_tel(0.2)

    aligned = align_telemetry_pair(ref, cmp, distance_step_m=50.0)

    assert "DeltaSeconds" in aligned.columns
    assert np.isclose(aligned["DeltaSeconds"].iloc[0], 0.2)
    assert len(aligned) >= 4
