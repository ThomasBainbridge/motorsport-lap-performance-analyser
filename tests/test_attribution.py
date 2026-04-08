import pandas as pd

from mlpa.attribution import compute_segment_contributions, rank_segments



def test_rank_segments_uses_gain_consistent_language():
    df = pd.DataFrame(
        {
            "SegmentId": [1],
            "SegmentLabel": ["Roggia"],
            "time_loss_s": [-0.08],
            "brake_start_delta_m": [12.0],
            "entry_speed_delta_kph": [4.0],
            "brake_fraction_delta": [-0.05],
            "min_speed_delta_kph": [6.0],
            "apex_delta_m": [-2.0],
            "throttle_pickup_delta_m": [-20.0],
            "exit_speed_delta_kph": [8.0],
            "mean_throttle_delta_pct": [5.0],
            "apex_to_exit_gain_delta_kph": [6.0],
        }
    )
    ranked = rank_segments(df)
    narrative = ranked.loc[0, "Narrative"]
    assert "gained 0.080 s" in narrative
    assert "earlier throttle pickup" in narrative or "stronger exit speed" in narrative



def test_compute_segment_contributions_contains_phase_rows():
    df = pd.DataFrame(
        {
            "SegmentId": [1],
            "SegmentLabel": ["Ascari"],
            "time_loss_s": [0.05],
            "brake_start_delta_m": [-8.0],
            "entry_speed_delta_kph": [-3.0],
            "brake_fraction_delta": [0.04],
            "min_speed_delta_kph": [-5.0],
            "apex_delta_m": [7.0],
            "throttle_pickup_delta_m": [18.0],
            "exit_speed_delta_kph": [-4.0],
            "mean_throttle_delta_pct": [-5.0],
            "apex_to_exit_gain_delta_kph": [-7.0],
        }
    )
    contrib = compute_segment_contributions(df)
    assert set(contrib["Phase"]) == {"Braking", "Minimum-speed", "Traction/exit"}
