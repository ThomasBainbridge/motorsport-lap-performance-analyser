from __future__ import annotations

import argparse

import pandas as pd

from .alignment import align_telemetry_pair
from .attribution import compute_segment_contributions, overall_summary, rank_segments
from .features import compute_segment_features
from .filtering import select_driver_lap, select_training_laps
from .io import load_config, load_session_from_config, prepare_output_dirs
from .ml_models import run_style_clustering, train_time_loss_regressor
from .plotting import (
    plot_brake_overlay,
    plot_cluster_map,
    plot_cluster_profiles,
    plot_delta_trace,
    plot_feature_importance,
    plot_regression_parity,
    plot_segment_contributions,
    plot_selected_segment_losses,
    plot_single_lap_brake_trace,
    plot_single_lap_segment_metrics,
    plot_single_lap_speed_trace,
    plot_single_lap_throttle_trace,
    plot_speed_overlay,
    plot_throttle_overlay,
)
from .reporting import write_single_lap_summary_markdown, write_summary_markdown
from .segmentation import detect_reference_segments
from .single_lap import build_single_lap_analysis_df, compute_single_lap_segment_features, single_lap_overall_summary
from .telemetry import lap_to_car_telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motorsport Lap Performance Analyzer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def get_analysis_mode(config: dict) -> str:
    mode = str(config.get("analysis", {}).get("mode", "compare_laps")).strip().lower()
    if mode in {"compare_laps", "compare", "comparison"}:
        return "compare_laps"
    if mode in {"single_lap", "single", "singlelap"}:
        return "single_lap"
    raise ValueError(f"Unsupported analysis.mode: {mode}")


def apply_segment_labels(segments_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    labelled = segments_df.copy()
    labels = config.get("segmentation", {}).get("segment_labels", {}) or {}
    if not labels:
        labelled["SegmentLabel"] = labelled["SegmentId"].map(lambda x: f"S{int(x)}")
        return labelled

    def _resolve(seg_id: int) -> str:
        return str(labels.get(seg_id, labels.get(str(seg_id), f"S{int(seg_id)}")))

    labelled["SegmentLabel"] = labelled["SegmentId"].map(_resolve)
    return labelled


def build_training_segment_dataset(session, config: dict, reference_driver: str, reference_tel: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    ml_cfg = config["ml"]
    selection_cfg = config["lap_selection"]
    alignment_cfg = config["alignment"]
    training_drivers = ml_cfg.get("training_drivers", [reference_driver])
    top_n = int(ml_cfg.get("top_n_laps_per_driver", 8))
    rows = []
    for driver in training_drivers:
        training_laps = select_training_laps(session, driver, selection_cfg, top_n=top_n)
        for _, lap in training_laps.iterrows():
            try:
                candidate_tel = lap_to_car_telemetry(lap)
                aligned = align_telemetry_pair(reference_tel, candidate_tel, distance_step_m=float(alignment_cfg.get("distance_step_m", 5.0)))
                segment_features = compute_segment_features(aligned, segments_df, throttle_pickup_threshold=float(config["segmentation"].get("throttle_pickup_threshold", 90.0)))
                segment_features["Driver"] = lap["Driver"]
                segment_features["LapNumber"] = int(lap["LapNumber"])
                segment_features["LapTime"] = lap["LapTime"]
                segment_features["IsReferenceDriver"] = lap["Driver"] == reference_driver
                rows.append(segment_features)
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _empty_compare_ml_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    clustered_df = pd.DataFrame()
    cluster_centers_df = pd.DataFrame(columns=["Cluster", "Archetype", "Feature", "CenterValue"])
    cluster_profiles_df = pd.DataFrame(columns=["Cluster", "Archetype"])
    regression_metrics_df = pd.DataFrame([{"n_rows": 0.0, "selected_model": "disabled", "cv_r2_mean": pd.NA, "cv_r2_std": pd.NA, "cv_mae_mean": pd.NA, "cv_mae_std": pd.NA, "test_r2": pd.NA, "test_mae_s": pd.NA}])
    regression_feature_importance_df = pd.DataFrame(columns=["Feature", "Importance"])
    regression_predictions_df = pd.DataFrame(columns=["actual_time_loss_s", "predicted_time_loss_s", "subset"])
    regression_metrics = regression_metrics_df.iloc[0].to_dict()
    return clustered_df, cluster_centers_df, cluster_profiles_df, regression_feature_importance_df, regression_predictions_df, regression_metrics


def _detect_and_label_segments(analysis_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    segments_df = detect_reference_segments(
        analysis_df,
        min_braking_zone_length_m=float(config["segmentation"].get("min_braking_zone_length_m", 35.0)),
        apex_search_lookahead_m=float(config["segmentation"].get("apex_search_lookahead_m", 140.0)),
        throttle_pickup_threshold=float(config["segmentation"].get("throttle_pickup_threshold", 90.0)),
        low_throttle_threshold=float(config["segmentation"].get("low_throttle_threshold", 15.0)),
        exit_search_window_m=float(config["segmentation"].get("exit_search_window_m", 180.0)),
    )
    if segments_df.empty:
        raise RuntimeError("No braking/corner segments were detected. Try another session or relax segmentation thresholds.")
    return apply_segment_labels(segments_df, config)


def run_compare_laps(config: dict) -> None:
    output_dirs = prepare_output_dirs(config["paths"]["outputs_dir"])
    session = load_session_from_config(config)
    reference_driver = config["drivers"]["reference"]
    comparison_driver = config["drivers"]["comparison"]
    selection_cfg = config["lap_selection"]
    reference_lap = select_driver_lap(session, driver=reference_driver, selection_cfg=selection_cfg)
    comparison_lap = select_driver_lap(session, driver=comparison_driver, selection_cfg=selection_cfg)
    reference_tel = lap_to_car_telemetry(reference_lap)
    comparison_tel = lap_to_car_telemetry(comparison_lap)
    aligned_df = align_telemetry_pair(reference_tel, comparison_tel, distance_step_m=float(config["alignment"].get("distance_step_m", 5.0)))
    segments_df = _detect_and_label_segments(aligned_df, config)
    segment_features = compute_segment_features(aligned_df, segments_df, throttle_pickup_threshold=float(config["segmentation"].get("throttle_pickup_threshold", 90.0)))
    segment_ranking = rank_segments(segment_features)
    contributions_df = compute_segment_contributions(segment_features)
    overall_metrics = overall_summary(aligned_df)
    training_segment_features = pd.DataFrame()
    ml_enabled = bool(config.get("ml", {}).get("enabled", True))
    if ml_enabled:
        training_segment_features = build_training_segment_dataset(session=session, config=config, reference_driver=reference_driver, reference_tel=reference_tel, segments_df=segments_df)
        clustered_df, cluster_centers_df, cluster_profiles_df = run_style_clustering(training_segment_features, n_clusters=int(config["ml"].get("n_clusters", 4)), random_state=int(config["ml"].get("random_state", 42)))
        regression_result = train_time_loss_regressor(training_segment_features, random_state=int(config["ml"].get("random_state", 42)), cv_folds=int(config["ml"].get("cv_folds", 5)), test_size=float(config["ml"].get("test_size", 0.30)), candidate_names=list(config["ml"].get("regression_models", ["ridge", "random_forest", "gradient_boosting"])))
        regression_feature_importance_df = regression_result.feature_importance
        regression_predictions_df = regression_result.predictions
        regression_metrics = regression_result.metrics
    else:
        clustered_df, cluster_centers_df, cluster_profiles_df, regression_feature_importance_df, regression_predictions_df, regression_metrics = _empty_compare_ml_outputs()
    regression_metrics_df = pd.DataFrame([regression_metrics])
    aligned_df.to_csv(output_dirs["tables"] / "aligned_trace.csv", index=False)
    segments_df.to_csv(output_dirs["tables"] / "segments.csv", index=False)
    segment_features.to_csv(output_dirs["tables"] / "segment_features.csv", index=False)
    segment_ranking.to_csv(output_dirs["tables"] / "segment_ranking.csv", index=False)
    contributions_df.to_csv(output_dirs["tables"] / "segment_contributions.csv", index=False)
    training_segment_features.to_csv(output_dirs["tables"] / "training_segment_features.csv", index=False)
    clustered_df.to_csv(output_dirs["tables"] / "clustered_segment_features.csv", index=False)
    cluster_centers_df.to_csv(output_dirs["tables"] / "cluster_centers.csv", index=False)
    cluster_profiles_df.to_csv(output_dirs["tables"] / "cluster_profiles.csv", index=False)
    regression_feature_importance_df.to_csv(output_dirs["tables"] / "regression_feature_importance.csv", index=False)
    regression_metrics_df.to_csv(output_dirs["tables"] / "regression_metrics.csv", index=False)
    regression_predictions_df.to_csv(output_dirs["tables"] / "regression_predictions.csv", index=False)
    plot_speed_overlay(aligned_df, segments_df, output_dirs["figures"] / "speed_overlay.png")
    plot_throttle_overlay(aligned_df, segments_df, output_dirs["figures"] / "throttle_overlay.png")
    plot_brake_overlay(aligned_df, segments_df, output_dirs["figures"] / "brake_overlay.png")
    plot_delta_trace(aligned_df, segments_df, output_dirs["figures"] / "delta_trace.png")
    plot_selected_segment_losses(segment_ranking, output_dirs["figures"] / "segment_losses.png")
    plot_segment_contributions(contributions_df, output_dirs["figures"] / "segment_contributions.png")
    plot_cluster_map(clustered_df, output_dirs["figures"] / "cluster_map.png")
    plot_cluster_profiles(cluster_profiles_df, output_dirs["figures"] / "cluster_profiles.png")
    plot_feature_importance(regression_feature_importance_df, output_dirs["figures"] / "regression_feature_importance.png")
    plot_regression_parity(regression_predictions_df, output_dirs["figures"] / "regression_parity.png")
    write_summary_markdown(output_dirs["reports"] / "summary.md", config=config, reference_lap=reference_lap, comparison_lap=comparison_lap, overall_metrics=overall_metrics, segment_ranking_df=segment_ranking, contributions_df=contributions_df, regression_metrics=regression_metrics, feature_importance_df=regression_feature_importance_df, cluster_profiles_df=cluster_profiles_df)
    print("MLPA Version 2.1 compare-laps run complete.")
    print(f"Tables:   {output_dirs['tables']}")
    print(f"Figures:  {output_dirs['figures']}")
    print(f"Reports:  {output_dirs['reports']}")


def run_single_lap(config: dict) -> None:
    output_dirs = prepare_output_dirs(config["paths"]["outputs_dir"])
    session = load_session_from_config(config)
    selection_cfg = config["lap_selection"]
    driver_code = config.get("drivers", {}).get("single") or config.get("drivers", {}).get("reference")
    if not driver_code:
        raise ValueError("Single-lap mode requires drivers.single or drivers.reference in the config.")
    lap = select_driver_lap(session, driver=driver_code, selection_cfg=selection_cfg)
    analysis_df = build_single_lap_analysis_df(lap)
    segments_df = _detect_and_label_segments(analysis_df, config)
    segment_features = compute_single_lap_segment_features(analysis_df, segments_df, throttle_pickup_threshold=float(config["segmentation"].get("throttle_pickup_threshold", 90.0)))
    overall_metrics = single_lap_overall_summary(analysis_df, segments_df)
    ml_enabled = bool(config.get("ml", {}).get("enabled", True))
    if ml_enabled:
        clustered_df, cluster_centers_df, cluster_profiles_df = run_style_clustering(segment_features.copy(), n_clusters=int(config["ml"].get("n_clusters", 4)), random_state=int(config["ml"].get("random_state", 42)))
    else:
        clustered_df = segment_features.copy()
        clustered_df["StyleCluster"] = -1
        clustered_df["Archetype"] = "Unassigned"
        clustered_df["PC1"] = pd.NA
        clustered_df["PC2"] = pd.NA
        cluster_centers_df = pd.DataFrame(columns=["Cluster", "Archetype", "Feature", "CenterValue"])
        cluster_profiles_df = pd.DataFrame(columns=["Cluster", "Archetype"])
    analysis_df.to_csv(output_dirs["tables"] / "single_lap_trace.csv", index=False)
    segments_df.to_csv(output_dirs["tables"] / "segments.csv", index=False)
    segment_features.to_csv(output_dirs["tables"] / "single_lap_segment_features.csv", index=False)
    clustered_df.to_csv(output_dirs["tables"] / "clustered_segment_features.csv", index=False)
    cluster_centers_df.to_csv(output_dirs["tables"] / "cluster_centers.csv", index=False)
    cluster_profiles_df.to_csv(output_dirs["tables"] / "cluster_profiles.csv", index=False)
    plot_single_lap_speed_trace(analysis_df, segments_df, output_dirs["figures"] / "speed_trace.png")
    plot_single_lap_throttle_trace(analysis_df, segments_df, output_dirs["figures"] / "throttle_trace.png")
    plot_single_lap_brake_trace(analysis_df, segments_df, output_dirs["figures"] / "brake_trace.png")
    plot_single_lap_segment_metrics(segment_features, output_dirs["figures"] / "segment_metrics.png")
    plot_cluster_map(clustered_df, output_dirs["figures"] / "cluster_map.png")
    plot_cluster_profiles(cluster_profiles_df, output_dirs["figures"] / "cluster_profiles.png")
    write_single_lap_summary_markdown(output_dirs["reports"] / "summary.md", config=config, lap=lap, overall_metrics=overall_metrics, segment_features_df=segment_features, cluster_profiles_df=cluster_profiles_df)
    print("MLPA Version 2.1 single-lap run complete.")
    print(f"Tables:   {output_dirs['tables']}")
    print(f"Figures:  {output_dirs['figures']}")
    print(f"Reports:  {output_dirs['reports']}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    mode = get_analysis_mode(config)
    if mode == "compare_laps":
        run_compare_laps(config)
        return
    if mode == "single_lap":
        run_single_lap(config)
        return
    raise RuntimeError(f"Unhandled analysis mode: {mode}")


if __name__ == "__main__":
    main()
