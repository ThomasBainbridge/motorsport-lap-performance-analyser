from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from .envelope import EnvelopeComparison
from .track_model import TrackGeometry
from .vehicle_model import EnvelopeSolution, VehicleParams, G


def _save(fig, output_path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _segment_label(row) -> str:
    label = getattr(row, "SegmentLabel", None)
    if isinstance(label, str) and label.strip():
        return label
    return f"S{int(row.SegmentId)}"


def plot_envelope_overlay(
    comparison: EnvelopeComparison,
    solution: EnvelopeSolution,
    segments_df: pd.DataFrame | None,
    output_path: str | Path,
    *,
    driver_label: str = "Driver",
) -> None:
    """Driver speed vs model envelope vs corner-limited speed."""
    fig, ax = plt.subplots(figsize=(12, 5))
    d = comparison.distance
    ax.plot(d, comparison.v_driver_ms * 3.6, label=f"{driver_label} (measured)", linewidth=1.4)
    ax.plot(d, solution.v_model * 3.6, label="QSS envelope v_model", linewidth=1.4)
    ax.plot(d, solution.v_corner * 3.6, label="Corner-limit v_corner", linestyle="--", alpha=0.6)
    if segments_df is not None and not segments_df.empty:
        y_top = ax.get_ylim()[1]
        for row in segments_df.itertuples(index=False):
            ax.axvline(row.BrakeStartDistance, linestyle="--", linewidth=0.7, alpha=0.4)
            ax.text(row.ApexDistance, y_top * 0.97, _segment_label(row), ha="center", va="top", fontsize=8)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Driver vs. QSS envelope -- unused {comparison.unused_time_s:.3f} s")
    ax.legend(loc="lower right")
    _save(fig, output_path)


def plot_grip_utilisation_map(
    track: TrackGeometry,
    comparison: EnvelopeComparison,
    output_path: str | Path,
    *,
    cmap: str = "viridis",
    vmax: float = 1.1,
) -> None:
    """Plot the track X, Y outline with colour = driver grip utilisation (0..~1)."""
    pts = np.column_stack([track.x, track.y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    util = np.clip(comparison.grip_utilisation, 0.0, vmax)
    util_seg = 0.5 * (util[:-1] + util[1:])

    fig, ax = plt.subplots(figsize=(8, 8))
    lc = LineCollection(segs, cmap=cmap, array=util_seg, linewidth=3)
    lc.set_clim(0.0, vmax)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Driver grip utilisation along the track")
    fig.colorbar(lc, ax=ax, label="Utilisation (1.0 = on the ellipse)")
    _save(fig, output_path)


def plot_segment_unused_time(segment_summary: pd.DataFrame, output_path: str | Path) -> None:
    """Bar chart of unused time per segment (positive = driver slower than envelope)."""
    if segment_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No segments available", ha="center", va="center")
        _save(fig, output_path)
        return
    df = segment_summary.sort_values("unused_time_s", ascending=False).copy()
    labels = df["SegmentLabel"].tolist()
    values = df["unused_time_s"].to_numpy()
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(labels) + 1)))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Unused time vs. QSS envelope (s)")
    ax.set_title("Per-segment headroom (positive = driver slower than theoretical)")
    ax.invert_yaxis()
    _save(fig, output_path)


def plot_gg_diagram(
    track: TrackGeometry,
    speed_ms: np.ndarray,
    params: VehicleParams,
    output_path: str | Path,
    *,
    driver_label: str = "Driver",
) -> None:
    """g-g scatter of driver accelerations with the v-dependent friction ellipse overlaid."""
    a_lon = speed_ms * np.gradient(speed_ms, track.distance)
    # Signed lateral acceleration using curvature sign separates left/right corners on the plot.
    a_lat_signed = speed_ms ** 2 * track.curvature

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(a_lat_signed / G, a_lon / G, c=speed_ms * 3.6, cmap="plasma", s=6, alpha=0.75)
    fig.colorbar(sc, ax=ax, label="Speed (km/h)")

    speeds_ref_kph = [80, 160, 240, 320]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    for v_kph in speeds_ref_kph:
        v = v_kph / 3.6
        n_load = params.mass_kg * G + 0.5 * params.rho_air_kgm3 * params.cla_m2 * v ** 2
        a_lat_max = params.mu_lat * n_load / params.mass_kg / G
        a_lon_max = params.mu_long * n_load / params.mass_kg / G
        ax.plot(a_lat_max * np.cos(theta), a_lon_max * np.sin(theta),
                linestyle="--", linewidth=0.9, alpha=0.6, label=f"{v_kph} km/h")

    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral acceleration (g)")
    ax.set_ylabel("Longitudinal acceleration (g)")
    ax.set_title(f"g-g diagram -- {driver_label} with friction ellipse at reference speeds")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, output_path)
