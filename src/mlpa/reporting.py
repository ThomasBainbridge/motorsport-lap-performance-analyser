from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import format_seconds, markdown_table_from_dataframe


def write_summary_markdown(
    output_path: str | Path,
    *,
    config: dict[str, Any],
    reference_lap,
    comparison_lap,
    overall_metrics: dict[str, float],
    segment_ranking_df: pd.DataFrame,
    contributions_df: pd.DataFrame,
    regression_metrics: dict[str, float | str],
    feature_importance_df: pd.DataFrame,
    cluster_profiles_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    selected_sum = float(segment_ranking_df["time_loss_s"].sum()) if not segment_ranking_df.empty else float("nan")
    lines.append("# Motorsport Lap Performance Analyzer — comparison summary")
    lines.append("")
    lines.append("## Session")
    lines.append("")
    lines.append(f"- Year: {config['session']['year']}")
    lines.append(f"- Grand Prix: {config['session']['grand_prix']}")
    lines.append(f"- Session: {config['session']['session']}")
    lines.append("")
    lines.append("## Compared laps")
    lines.append("")
    lines.append(f"- Reference: {reference_lap['Driver']} lap {int(reference_lap['LapNumber'])} ({reference_lap['LapTime']})")
    lines.append(f"- Comparison: {comparison_lap['Driver']} lap {int(comparison_lap['LapNumber'])} ({comparison_lap['LapTime']})")
    lines.append("")
    lines.append("## Overall comparison")
    lines.append("")
    lines.append(f"- Full-lap comparison minus reference delta: {format_seconds(overall_metrics['total_delta_s'])} s")
    lines.append(f"- Maximum cumulative loss on aligned trace: {format_seconds(overall_metrics['max_cumulative_loss_s'])} s")
    lines.append(f"- Maximum cumulative gain on aligned trace: {format_seconds(overall_metrics['max_cumulative_gain_s'])} s")
    lines.append(f"- Sum of selected-segment losses: {format_seconds(selected_sum)} s")
    lines.append("")
    lines.append("## Selected-segment ranking")
    lines.append("")
    display_cols = [c for c in ["SegmentLabel", "time_loss_s", "DominantPhase", "Narrative"] if c in segment_ranking_df.columns]
    lines.append(markdown_table_from_dataframe(segment_ranking_df[display_cols], max_rows=12))
    lines.append("")
    lines.append("## Phase contribution summary")
    lines.append("")
    if contributions_df.empty:
        lines.append("_No contribution summary available._")
    else:
        contrib_pivot = contributions_df.pivot(index="SegmentLabel", columns="Phase", values="ContributionScore").reset_index().fillna(0.0)
        lines.append(markdown_table_from_dataframe(contrib_pivot, max_rows=12))
    lines.append("")
    lines.append("## Regression model summary")
    lines.append("")
    lines.append(f"- Rows used: {regression_metrics.get('n_rows', float('nan')):.0f}")
    lines.append(f"- Selected model: {regression_metrics.get('selected_model', 'not available')}")
    cv_r2_mean = regression_metrics.get("cv_r2_mean")
    cv_r2_std = regression_metrics.get("cv_r2_std")
    cv_mae_mean = regression_metrics.get("cv_mae_mean")
    cv_mae_std = regression_metrics.get("cv_mae_std")
    test_r2 = regression_metrics.get("test_r2")
    test_mae = regression_metrics.get("test_mae_s")
    lines.append(f"- CV R²: {cv_r2_mean:.3f} ± {cv_r2_std:.3f}" if pd.notna(cv_r2_mean) and pd.notna(cv_r2_std) else "- CV R²: not available")
    lines.append(f"- CV MAE (s): {cv_mae_mean:.4f} ± {cv_mae_std:.4f}" if pd.notna(cv_mae_mean) and pd.notna(cv_mae_std) else "- CV MAE (s): not available")
    lines.append(f"- Holdout R²: {test_r2:.3f}" if pd.notna(test_r2) else "- Holdout R²: not available")
    lines.append(f"- Holdout MAE (s): {test_mae:.4f}" if pd.notna(test_mae) else "- Holdout MAE (s): not available")
    lines.append("")
    lines.append("### Top model features")
    lines.append("")
    if feature_importance_df.empty:
        lines.append("_No feature-importance results available._")
    else:
        lines.append(markdown_table_from_dataframe(feature_importance_df.head(8), max_rows=8))
    lines.append("")
    lines.append("## Cluster archetype summary")
    lines.append("")
    if cluster_profiles_df.empty:
        lines.append("_No clustering results available._")
    else:
        lines.append(markdown_table_from_dataframe(cluster_profiles_df, max_rows=8))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The full-lap cumulative delta is a distance-aligned whole-lap trace. The selected-segment ranking covers automatically detected braking/apex/exit regions, so the sum of selected-segment losses does not have to equal the full-lap delta.")
    lines.append("- Corner archetype clustering is an unsupervised grouping of segment behaviour in feature space.")
    lines.append("- The regression layer is designed for explainable local attribution within this pipeline, not as a broadly validated predictive model.")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_single_lap_summary_markdown(
    output_path: str | Path,
    *,
    config: dict[str, Any],
    lap,
    overall_metrics: dict[str, float],
    segment_features_df: pd.DataFrame,
    cluster_profiles_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Motorsport Lap Performance Analyzer — single-lap summary")
    lines.append("")
    lines.append("## Session")
    lines.append("")
    lines.append(f"- Year: {config['session']['year']}")
    lines.append(f"- Grand Prix: {config['session']['grand_prix']}")
    lines.append(f"- Session: {config['session']['session']}")
    lines.append("")
    lines.append("## Analysed lap")
    lines.append("")
    lines.append(f"- Driver: {lap['Driver']}")
    lines.append(f"- Lap number: {int(lap['LapNumber'])}")
    lines.append(f"- Lap time: {lap['LapTime']}")
    lines.append("")
    lines.append("## Overall lap metrics")
    lines.append("")
    lines.append(f"- Lap time (telemetry trace): {format_seconds(overall_metrics['lap_time_s'])} s")
    lines.append(f"- Top speed: {overall_metrics['top_speed_kph']:.1f} km/h")
    lines.append(f"- Mean speed: {overall_metrics['mean_speed_kph']:.1f} km/h")
    lines.append(f"- Full-throttle fraction: {overall_metrics['full_throttle_fraction']:.3f}")
    lines.append(f"- Brake fraction: {overall_metrics['brake_fraction']:.3f}")
    lines.append(f"- Detected selected segments: {int(overall_metrics['n_segments'])}")
    lines.append(f"- Heavy-braking events: {int(overall_metrics['n_heavy_braking_events'])}")
    lines.append("")
    lines.append("## Segment summary")
    lines.append("")
    display_cols = [c for c in ["SegmentLabel", "lap_entry_speed_kph", "lap_min_speed_kph", "lap_exit_speed_kph", "lap_brake_fraction", "lap_full_throttle_fraction"] if c in segment_features_df.columns]
    lines.append(markdown_table_from_dataframe(segment_features_df[display_cols], max_rows=20))
    lines.append("")
    lines.append("## Cluster archetype summary")
    lines.append("")
    if cluster_profiles_df.empty:
        lines.append("_No clustering results available._")
    else:
        lines.append(markdown_table_from_dataframe(cluster_profiles_df, max_rows=8))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Single-lap mode describes the structure and control application of one lap. It does not assign absolute time loss without a reference lap.")
    lines.append("- Corner archetypes are unsupervised groupings of segment behaviour in feature space.")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
