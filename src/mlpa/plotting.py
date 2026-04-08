from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PHASE_ORDER = ["Braking", "Minimum-speed", "Traction/exit"]


def _save(fig, output_path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _segment_label(row) -> str:
    label = getattr(row, "SegmentLabel", None)
    if isinstance(label, str) and label.strip():
        return label
    return f"S{int(row.SegmentId)}"


def plot_speed_overlay(aligned_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aligned_df["Distance"], aligned_df["ref_Speed"], label="Reference")
    ax.plot(aligned_df["Distance"], aligned_df["cmp_Speed"], label="Comparison")
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.8)
        y_top = ax.get_ylim()[1]
        ax.text(row.ApexDistance, y_top * 0.96, _segment_label(row), ha="center", va="top")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed overlay with selected segment markers")
    ax.legend()
    _save(fig, output_path)


def plot_throttle_overlay(aligned_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(aligned_df["Distance"], aligned_df["ref_Throttle"], label="Reference throttle")
    ax.plot(aligned_df["Distance"], aligned_df["cmp_Throttle"], label="Comparison throttle")
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Throttle (%)")
    ax.set_title("Throttle overlay")
    ax.legend()
    _save(fig, output_path)


def plot_brake_overlay(aligned_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(aligned_df["Distance"], aligned_df["ref_Brake"] * 100.0, label="Reference brake")
    ax.plot(aligned_df["Distance"], aligned_df["cmp_Brake"] * 100.0, label="Comparison brake")
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Brake signal (0 or 100)")
    ax.set_title("Brake overlay")
    ax.legend()
    _save(fig, output_path)


def plot_delta_trace(aligned_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aligned_df["Distance"], aligned_df["DeltaSeconds"])
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Comparison - reference time (s)")
    ax.set_title("Full-lap cumulative delta")
    _save(fig, output_path)


def plot_selected_segment_losses(segment_ranking_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = segment_ranking_df.get("SegmentLabel", segment_ranking_df["SegmentId"].astype(str))
    ax.bar(labels.astype(str), segment_ranking_df["time_loss_s"])
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Selected segment")
    ax.set_ylabel("Time loss (s)")
    ax.set_title("Selected-segment time loss")
    _save(fig, output_path)


def plot_segment_contributions(contributions_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if contributions_df.empty:
        ax.text(0.5, 0.5, "No contribution data available", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return
    pivot = contributions_df.pivot(index="SegmentLabel", columns="Phase", values="ContributionScore").fillna(0.0)
    labels = pivot.index.astype(str).tolist()
    x = range(len(labels))
    width = 0.25
    for idx, phase in enumerate(PHASE_ORDER):
        if phase not in pivot.columns:
            continue
        offsets = [val + (idx - 1) * width for val in x]
        ax.bar(offsets, pivot[phase].to_numpy(dtype=float), width=width, label=phase)
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Selected segment")
    ax.set_ylabel("Contribution score")
    ax.set_title("Segment contribution breakdown")
    ax.legend()
    _save(fig, output_path)


def plot_cluster_map(clustered_df: pd.DataFrame, output_path: str | Path) -> None:
    usable = clustered_df[clustered_df["StyleCluster"] >= 0].copy() if "StyleCluster" in clustered_df.columns else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 6))
    if usable.empty or "PC1" not in usable.columns or "PC2" not in usable.columns:
        ax.text(0.5, 0.5, "Not enough data for clustering", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return
    for (_, archetype), group in usable.groupby(["StyleCluster", "Archetype"]):
        ax.scatter(group["PC1"], group["PC2"], label=archetype)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Corner archetype clustering")
    ax.legend()
    _save(fig, output_path)


def plot_cluster_profiles(profile_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if profile_df.empty:
        ax.text(0.5, 0.5, "Not enough data for cluster profiles", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return
    x = range(len(profile_df))
    ax.plot(x, profile_df["cmp_entry_speed_kph"], marker="o", label="Entry speed")
    ax.plot(x, profile_df["cmp_min_speed_kph"], marker="o", label="Minimum speed")
    ax.plot(x, profile_df["cmp_exit_speed_kph"], marker="o", label="Exit speed")
    ax.plot(x, profile_df["cmp_mean_throttle_pct"], marker="o", label="Mean throttle")
    ax.set_xticks(list(x))
    ax.set_xticklabels(profile_df["Archetype"], rotation=15, ha="right")
    ax.set_ylabel("Profile value")
    ax.set_title("Cluster archetype profiles")
    ax.legend()
    _save(fig, output_path)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if importance_df.empty:
        ax.text(0.5, 0.5, "Not enough data for regression", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return
    ordered = importance_df.iloc[::-1]
    ax.barh(ordered["Feature"], ordered["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Segment time-loss model feature importance")
    _save(fig, output_path)


def plot_regression_parity(predictions_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if predictions_df.empty:
        ax.text(0.5, 0.5, "Not enough data for parity plot", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return
    for subset, group in predictions_df.groupby("subset"):
        ax.scatter(group["actual_time_loss_s"], group["predicted_time_loss_s"], label=subset.title())
    min_val = min(predictions_df["actual_time_loss_s"].min(), predictions_df["predicted_time_loss_s"].min())
    max_val = max(predictions_df["actual_time_loss_s"].max(), predictions_df["predicted_time_loss_s"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Actual time loss (s)")
    ax.set_ylabel("Predicted time loss (s)")
    ax.set_title("Regression parity plot")
    ax.legend()
    _save(fig, output_path)


def plot_single_lap_speed_trace(analysis_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(analysis_df["Distance"], analysis_df["ref_Speed"], label="Lap speed")
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.8)
        y_top = ax.get_ylim()[1]
        ax.text(row.ApexDistance, y_top * 0.96, _segment_label(row), ha="center", va="top")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Single-lap speed trace with selected segment markers")
    ax.legend()
    _save(fig, output_path)


def plot_single_lap_throttle_trace(analysis_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(analysis_df["Distance"], analysis_df["ref_Throttle"], label="Throttle")
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Throttle (%)")
    ax.set_title("Single-lap throttle trace")
    ax.legend()
    _save(fig, output_path)


def plot_single_lap_brake_trace(analysis_df: pd.DataFrame, segments_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(analysis_df["Distance"], analysis_df["ref_Brake"] * 100.0, label="Brake")
    for row in segments_df.itertuples(index=False):
        ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Brake signal (0 or 100)")
    ax.set_title("Single-lap brake trace")
    ax.legend()
    _save(fig, output_path)


def plot_single_lap_segment_metrics(segment_features_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = segment_features_df.get("SegmentLabel", segment_features_df["SegmentId"].astype(str)).astype(str)
    ax.plot(labels, segment_features_df["lap_entry_speed_kph"], marker="o", label="Entry speed")
    ax.plot(labels, segment_features_df["lap_min_speed_kph"], marker="o", label="Minimum speed")
    ax.plot(labels, segment_features_df["lap_exit_speed_kph"], marker="o", label="Exit speed")
    ax.set_xlabel("Selected segment")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Single-lap segment speed profile")
    ax.legend()
    _save(fig, output_path)
