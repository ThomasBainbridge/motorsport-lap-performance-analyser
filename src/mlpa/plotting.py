from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_cluster_map(clustered_df: pd.DataFrame, output_path: str | Path) -> None:
    usable = clustered_df[clustered_df["StyleCluster"] >= 0].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    if usable.empty:
        ax.text(0.5, 0.5, "Not enough data for clustering", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return

    for cluster_id, group in usable.groupby("StyleCluster"):
        ax.scatter(group["cmp_min_speed_kph"], group["cmp_exit_speed_kph"], label=f"Archetype {cluster_id}")

    ax.set_xlabel("Minimum speed (km/h)")
    ax.set_ylabel("Exit speed (km/h)")
    ax.set_title("Corner archetype clustering")
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
