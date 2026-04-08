from __future__ import annotations

import argparse

import pandas as pd

from .alignment import align_telemetry_pair
from .attribution import overall_summary, rank_segments
from .features import compute_segment_features
from .filtering import select_driver_lap, select_training_laps
from .io import load_config, load_session_from_config, prepare_output_dirs
from .ml_models import run_style_clustering, train_time_loss_regressor
from .plotting import (
    plot_brake_overlay,
    plot_cluster_map,
    plot_delta_trace,
    plot_feature_importance,
    plot_selected_segment_losses,
    plot_speed_overlay,
    plot_throttle_overlay,
)
from .reporting import write_summary_markdown
from .segmentation import detect_reference_segments
from .telemetry import lap_to_car_telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motorsport Lap Performance Analyzer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


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


def build_training_segment_dataset(
    session,
    config: dict,
    reference_driver: str,
    comparison_driver: str,
    reference_tel: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    ml_cfg = config["ml"]
    selection_cfg = config["lap_selection"]
    alignment_cfg = config["alignment"]
    training_drivers = ml_cfg.get("training_drivers", [reference_driver, comparison_driver])
    top_n = int(ml_cfg.get("top_n_laps_per_driver", 8))

    rows = []
    for driver in training_drivers:
        training_laps = select_training_laps(session, driver, selection_cfg, top_n=top_n)
        for _, lap in training_laps.iterrows():
            try:
                candidate_tel = lap_to_car_telemetry(lap)
                aligned = align_telemetry_pair(
                    reference_tel,
                    candidate_tel,
                    distance_step_m=float(alignment_cfg.get("distance_step_m", 5.0)),
                )
                segment_features = compute_segment_features(
                    aligned,
                    segments_df,
                    throttle_pickup_threshold=float(
                        config["segmentation"].get("throttle_pickup_threshold", 90.0)
                    ),
                )
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dirs = prepare_output_dirs(config["paths"]["outputs_dir"])

    session = load_session_from_config(config)

    reference_driver = config["drivers"]["reference"]
    comparison_driver = config["drivers"]["comparison"]
    selection_cfg = config["lap_selection"]

    reference_lap = select_driver_lap(session, driver=reference_driver, selection_cfg=selection_cfg)
    comparison_lap = select_driver_lap(session, driver=comparison_driver, selection_cfg=selection_cfg)

    reference_tel = lap_to_car_telemetry(reference_lap)
    comparison_tel = lap_to_car_telemetry(comparison_lap)

    aligned_df = align_telemetry_pair(
        reference_tel,
        comparison_tel,
        distance_step_m=float(config["alignment"].get("distance_step_m", 5.0)),
    )

    segments_df = detect_reference_segments(
        aligned_df,
        min_braking_zone_length_m=float(config["segmentation"].get("min_braking_zone_length_m", 35.0)),
        apex_search_lookahead_m=float(config["segmentation"].get("apex_search_lookahead_m", 140.0)),
        throttle_pickup_threshold=float(config["segmentation"].get("throttle_pickup_threshold", 90.0)),
        low_throttle_threshold=float(config["segmentation"].get("low_throttle_threshold", 15.0)),
        exit_search_window_m=float(config["segmentation"].get("exit_search_window_m", 180.0)),
    )
    if segments_df.empty:
        raise RuntimeError(
            "No braking/corner segments were detected. Try another session or relax segmentation thresholds."
        )
    segments_df = apply_segment_labels(segments_df, config)

    segment_features = compute_segment_features(
        aligned_df,
        segments_df,
        throttle_pickup_threshold=float(config["segmentation"].get("throttle_pickup_threshold", 90.0)),
    )
    segment_ranking = rank_segments(segment_features)
    overall_metrics = overall_summary(aligned_df)

    training_segment_features = build_training_segment_dataset(
        session=session,
        config=config,
        reference_driver=reference_driver,
        comparison_driver=comparison_driver,
        reference_tel=reference_tel,
        segments_df=segments_df,
    )

    clustered_df, cluster_centers_df = run_style_clustering(
        training_segment_features,
        n_clusters=int(config["ml"].get("n_clusters", 3)),
        random_state=int(config["ml"].get("random_state", 42)),
    )

    regression_result = train_time_loss_regressor(
        training_segment_features,
        random_state=int(config["ml"].get("random_state", 42)),
    )

    aligned_df.to_csv(output_dirs["tables"] / "aligned_trace.csv", index=False)
    segments_df.to_csv(output_dirs["tables"] / "segments.csv", index=False)
    segment_features.to_csv(output_dirs["tables"] / "segment_features.csv", index=False)
    segment_ranking.to_csv(output_dirs["tables"] / "segment_ranking.csv", index=False)
    training_segment_features.to_csv(output_dirs["tables"] / "training_segment_features.csv", index=False)
    clustered_df.to_csv(output_dirs["tables"] / "clustered_segment_features.csv", index=False)
    cluster_centers_df.to_csv(output_dirs["tables"] / "cluster_centers.csv", index=False)
    regression_result.feature_importance.to_csv(
        output_dirs["tables"] / "regression_feature_importance.csv",
        index=False,
    )

    plot_speed_overlay(aligned_df, segments_df, output_dirs["figures"] / "speed_overlay.png")
    plot_throttle_overlay(aligned_df, segments_df, output_dirs["figures"] / "throttle_overlay.png")
    plot_brake_overlay(aligned_df, segments_df, output_dirs["figures"] / "brake_overlay.png")
    plot_delta_trace(aligned_df, segments_df, output_dirs["figures"] / "delta_trace.png")
    plot_selected_segment_losses(segment_ranking, output_dirs["figures"] / "segment_losses.png")
    plot_cluster_map(clustered_df, output_dirs["figures"] / "cluster_map.png")
    plot_feature_importance(
        regression_result.feature_importance,
        output_dirs["figures"] / "regression_feature_importance.png",
    )

    write_summary_markdown(
        output_dirs["reports"] / "summary.md",
        config=config,
        reference_lap=reference_lap,
        comparison_lap=comparison_lap,
        overall_metrics=overall_metrics,
        segment_ranking_df=segment_ranking,
        regression_metrics=regression_result.metrics,
    )

    print("MLPA Version 1.1 run complete.")
    print(f"Tables:   {output_dirs['tables']}")
    print(f"Figures:  {output_dirs['figures']}")
    print(f"Reports:  {output_dirs['reports']}")


if __name__ == "__main__":
    main()
