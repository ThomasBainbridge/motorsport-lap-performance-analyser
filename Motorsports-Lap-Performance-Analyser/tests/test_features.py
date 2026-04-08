import numpy as np
import pandas as pd

from mlpa.features import compute_segment_features


def test_compute_segment_features_outputs_delta():
    df = pd.DataFrame(
        {
            "Distance": np.arange(0.0, 100.0, 10.0),
            "ref_TimeSeconds": np.linspace(0.0, 1.0, 10),
            "cmp_TimeSeconds": np.linspace(0.0, 1.1, 10),
            "ref_Speed": np.linspace(200.0, 150.0, 10),
            "cmp_Speed": np.linspace(198.0, 145.0, 10),
            "ref_Throttle": np.linspace(100.0, 20.0, 10),
            "cmp_Throttle": np.linspace(100.0, 30.0, 10),
            "ref_Brake": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            "cmp_Brake": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    segments = pd.DataFrame(
        {
            "SegmentId": [1],
            "StartIdx": [0],
            "EndIdx": [9],
            "StartDistance": [0.0],
            "EndDistance": [90.0],
            "BrakeStartDistance": [20.0],
            "ApexDistance": [90.0],
            "ThrottlePickupDistance": [90.0],
        }
    )

    out = compute_segment_features(df, segments)
    assert "time_loss_s" in out.columns
    assert len(out) == 1
