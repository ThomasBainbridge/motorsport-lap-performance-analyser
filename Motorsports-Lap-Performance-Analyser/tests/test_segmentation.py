import numpy as np
import pandas as pd

from mlpa.segmentation import detect_reference_segments


def test_detect_reference_segments_finds_zone():
    distance = np.arange(0.0, 500.0, 5.0)
    speed = np.full_like(distance, 250.0, dtype=float)
    speed[20:30] = np.linspace(250, 120, 10)
    speed[30:45] = np.linspace(120, 200, 15)

    throttle = np.full_like(distance, 100.0, dtype=float)
    throttle[20:35] = 5.0
    brake = np.zeros_like(distance, dtype=float)
    brake[20:28] = 1.0

    df = pd.DataFrame(
        {
            "Distance": distance,
            "ref_Speed": speed,
            "ref_Throttle": throttle,
            "ref_Brake": brake,
        }
    )
    df["ref_SpeedGradient"] = np.gradient(df["ref_Speed"], df["Distance"])

    segments = detect_reference_segments(df)
    assert len(segments) >= 1
