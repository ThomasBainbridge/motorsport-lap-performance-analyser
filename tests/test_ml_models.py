import numpy as np
import pandas as pd

from mlpa.ml_models import run_style_clustering, train_time_loss_regressor



def build_training_df(n_rows: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = pd.DataFrame(
        {
            "cmp_entry_speed_kph": rng.normal(220, 12, n_rows),
            "cmp_min_speed_kph": rng.normal(140, 18, n_rows),
            "cmp_exit_speed_kph": rng.normal(185, 15, n_rows),
            "cmp_brake_fraction": rng.uniform(0.1, 0.5, n_rows),
            "cmp_mean_throttle_pct": rng.uniform(35, 80, n_rows),
            "cmp_segment_length_m": rng.uniform(120, 260, n_rows),
            "cmp_apex_to_exit_gain_kph": rng.normal(45, 10, n_rows),
            "entry_speed_delta_kph": rng.normal(0, 4, n_rows),
            "min_speed_delta_kph": rng.normal(0, 5, n_rows),
            "exit_speed_delta_kph": rng.normal(0, 5, n_rows),
            "mean_speed_delta_kph": rng.normal(0, 4, n_rows),
            "brake_start_delta_m": rng.normal(0, 10, n_rows),
            "brake_end_delta_m": rng.normal(0, 8, n_rows),
            "apex_delta_m": rng.normal(0, 6, n_rows),
            "throttle_pickup_delta_m": rng.normal(0, 12, n_rows),
            "mean_throttle_delta_pct": rng.normal(0, 6, n_rows),
            "full_throttle_fraction_delta": rng.normal(0, 0.08, n_rows),
            "brake_fraction_delta": rng.normal(0, 0.08, n_rows),
            "entry_to_apex_drop_delta_kph": rng.normal(0, 4, n_rows),
            "apex_to_exit_gain_delta_kph": rng.normal(0, 5, n_rows),
        }
    )
    target = (
        0.004 * base["brake_start_delta_m"]
        - 0.003 * base["min_speed_delta_kph"]
        + 0.005 * base["throttle_pickup_delta_m"]
        - 0.004 * base["exit_speed_delta_kph"]
        - 0.003 * base["mean_throttle_delta_pct"]
        + rng.normal(0, 0.01, n_rows)
    )
    base["time_loss_s"] = target
    return base



def test_run_style_clustering_outputs_profiles():
    df = build_training_df()
    clustered, centers, profiles = run_style_clustering(df, n_clusters=3, random_state=42)
    assert "StyleCluster" in clustered.columns
    assert not centers.empty
    assert not profiles.empty



def test_train_time_loss_regressor_returns_metrics_and_predictions():
    df = build_training_df()
    result = train_time_loss_regressor(df, random_state=42, cv_folds=4)
    assert result.metrics["selected_model"] in {"ridge", "random_forest", "gradient_boosting"}
    assert not result.feature_importance.empty
    assert not result.predictions.empty
