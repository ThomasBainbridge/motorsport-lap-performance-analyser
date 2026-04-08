import pandas as pd

from mlpa.filtering import filter_candidate_laps


def test_filter_candidate_laps_quality_flags():
    laps = pd.DataFrame(
        {
            "LapTime": pd.to_timedelta([90, 91, 92], unit="s"),
            "IsAccurate": [True, False, True],
            "Deleted": [False, False, True],
            "FastF1Generated": [False, False, False],
            "TrackStatus": ["1", "1", "1"],
        }
    )

    filtered = filter_candidate_laps(
        laps,
        require_accuracy=True,
        exclude_deleted=True,
        exclude_generated=True,
        quicklaps=False,
        green_flag_only=True,
    )
    assert len(filtered) == 1
