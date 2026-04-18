from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import calibrate_from_lap
from .envelope import EnvelopeComparison, compare_lap_to_envelope, summarise_by_segment
from .envelope_plotting import (
    plot_envelope_overlay,
    plot_gg_diagram,
    plot_grip_utilisation_map,
    plot_segment_unused_time,
)
from .telemetry import lap_to_merged_telemetry
from .track_model import TrackGeometry, build_track_from_position
from .vehicle_model import EnvelopeSolution, VehicleParams, solve_envelope


def _vehicle_params_from_config(env_cfg: dict, fitted: VehicleParams) -> VehicleParams:
    """Allow the config to pin any VehicleParams field, overriding the calibrated value."""
    overrides = {}
    for field_name in ("mass_kg", "mu_lat", "mu_long", "cda_m2", "cla_m2",
                       "p_max_w", "drivetrain_efficiency", "crr", "rho_air_kgm3"):
        if field_name in env_cfg:
            overrides[field_name] = float(env_cfg[field_name])
    return replace(fitted, **overrides) if overrides else fitted


def _prepare_lap_data(lap, env_cfg: dict) -> tuple[TrackGeometry, dict[str, np.ndarray]]:
    """Extract merged telemetry and build a TrackGeometry on FastF1's native distance grid.

    All channels (speed, throttle, brake, X, Y, curvature) end up on the SAME
    distance axis, eliminating any interpolation-induced misalignment between
    driver data and track features.
    """
    tel = lap_to_merged_telemetry(lap)
    if not {"X", "Y", "Distance", "Speed"}.issubset(tel.columns):
        raise RuntimeError("Merged telemetry missing required columns X/Y/Distance/Speed.")

    # FastF1's X/Y are delivered in 1/10 of a metre; Distance is in metres.
    # Detect scale robustly by comparing path lengths.
    x_raw = tel["X"].to_numpy(dtype=float)
    y_raw = tel["Y"].to_numpy(dtype=float)
    distance_m = tel["Distance"].to_numpy(dtype=float)

    raw_path = float(np.sum(np.hypot(np.diff(x_raw), np.diff(y_raw))))
    dist_path = float(distance_m[-1] - distance_m[0])
    if dist_path <= 0 or raw_path <= 0:
        raise RuntimeError("Degenerate telemetry -- cannot build track.")
    scale_to_m = dist_path / raw_path
    if not (1e-4 < scale_to_m < 10.0):
        raise RuntimeError(f"X/Y scale out of range: {scale_to_m:.3e}")

    x_m = x_raw * scale_to_m
    y_m = y_raw * scale_to_m

    chord_m = float(env_cfg.get("curvature_chord_m", 8.0))
    median_window = int(env_cfg.get("curvature_median_window", 5))

    track = build_track_from_position(
        distance=distance_m, x=x_m, y=y_m,
        chord_m=chord_m, median_window=median_window,
    )

    channels = {
        "speed_ms": tel["Speed"].to_numpy(dtype=float) / 3.6,
        "throttle_pct": tel["Throttle"].to_numpy(dtype=float) if "Throttle" in tel.columns
                        else np.full(len(distance_m), 100.0),
        "brake": tel["Brake"].to_numpy(dtype=float) if "Brake" in tel.columns
                 else np.zeros(len(distance_m)),
    }
    return track, channels


def _driver_speed_on_reference_track(
    reference_track: TrackGeometry, comparison_lap, env_cfg: dict
) -> np.ndarray:
    """Sample another driver's speed onto the reference track's distance axis."""
    other_track, other_channels = _prepare_lap_data(comparison_lap, env_cfg)
    return np.interp(reference_track.distance, other_track.distance, other_channels["speed_ms"])


def _calibrate(track: TrackGeometry, channels: dict[str, np.ndarray], env_cfg: dict) -> VehicleParams:
    base = VehicleParams()
    if env_cfg.get("calibrate_from_reference", True):
        fitted = calibrate_from_lap(
            track=track,
            speed_ms=channels["speed_ms"],
            throttle_pct=channels["throttle_pct"],
            brake=channels["brake"],
            initial=base,
        )
    else:
        fitted = base
    return _vehicle_params_from_config(env_cfg, fitted)


def _write_vehicle_params(params: VehicleParams, path: Path) -> None:
    pd.DataFrame([asdict(params)]).to_csv(path, index=False)


def _write_envelope_solution(solution: EnvelopeSolution, path: Path) -> None:
    pd.DataFrame({
        "distance_m": solution.distance,
        "curvature_1pm": solution.curvature,
        "v_corner_ms": solution.v_corner,
        "v_forward_ms": solution.v_forward,
        "v_backward_ms": solution.v_backward,
        "v_model_ms": solution.v_model,
        "a_lat_model_ms2": solution.a_lat_model,
        "a_lon_model_ms2": solution.a_lon_model,
    }).to_csv(path, index=False)


def _write_comparison(comparison: EnvelopeComparison, path: Path) -> None:
    pd.DataFrame({
        "distance_m": comparison.distance,
        "v_driver_ms": comparison.v_driver_ms,
        "v_model_ms": comparison.v_model_ms,
        "speed_gap_ms": comparison.speed_gap_ms,
        "grip_utilisation": comparison.grip_utilisation,
    }).to_csv(path, index=False)


def run_envelope_stage_compare(
    config: dict,
    reference_lap,
    comparison_lap,
    segments_df: pd.DataFrame,
    output_dirs: dict,
) -> None:
    env_cfg = config.get("envelope", {}) or {}
    if not env_cfg.get("enabled", False):
        return

    tables = Path(output_dirs["tables"])
    figures = Path(output_dirs["figures"])
    ref_driver = config["drivers"]["reference"]
    cmp_driver = config["drivers"]["comparison"]

    ref_track, ref_channels = _prepare_lap_data(reference_lap, env_cfg)
    params = _calibrate(ref_track, ref_channels, env_cfg)
    _write_vehicle_params(params, tables / "vehicle_params.csv")

    solution = solve_envelope(ref_track, params)
    _write_envelope_solution(solution, tables / "envelope_solution.csv")

    # Comparison driver on reference track grid
    cmp_speed = _driver_speed_on_reference_track(ref_track, comparison_lap, env_cfg)

    cmp_ref = compare_lap_to_envelope(ref_track, solution, ref_channels["speed_ms"], params)
    _write_comparison(cmp_ref, tables / f"envelope_comparison_{ref_driver}.csv")
    ref_seg = summarise_by_segment(cmp_ref, segments_df)
    ref_seg.to_csv(tables / f"envelope_segment_summary_{ref_driver}.csv", index=False)

    cmp_cmp = compare_lap_to_envelope(ref_track, solution, cmp_speed, params)
    _write_comparison(cmp_cmp, tables / f"envelope_comparison_{cmp_driver}.csv")
    cmp_seg = summarise_by_segment(cmp_cmp, segments_df)
    cmp_seg.to_csv(tables / f"envelope_segment_summary_{cmp_driver}.csv", index=False)

    plot_envelope_overlay(cmp_ref, solution, segments_df,
                          figures / f"envelope_overlay_{ref_driver}.png", driver_label=ref_driver)
    plot_envelope_overlay(cmp_cmp, solution, segments_df,
                          figures / f"envelope_overlay_{cmp_driver}.png", driver_label=cmp_driver)
    plot_grip_utilisation_map(ref_track, cmp_ref,
                              figures / f"grip_utilisation_map_{ref_driver}.png")
    plot_grip_utilisation_map(ref_track, cmp_cmp,
                              figures / f"grip_utilisation_map_{cmp_driver}.png")
    plot_segment_unused_time(ref_seg, figures / f"segment_unused_time_{ref_driver}.png")
    plot_segment_unused_time(cmp_seg, figures / f"segment_unused_time_{cmp_driver}.png")
    plot_gg_diagram(ref_track, ref_channels["speed_ms"], params,
                    figures / f"gg_diagram_{ref_driver}.png", driver_label=ref_driver)
    plot_gg_diagram(ref_track, cmp_speed, params,
                    figures / f"gg_diagram_{cmp_driver}.png", driver_label=cmp_driver)

    # One-line console summary so you can see at a glance whether things are sane
    print(
        f"[envelope] {ref_driver}: driver={cmp_ref.lap_time_driver_s:.2f}s "
        f"model={cmp_ref.lap_time_model_s:.2f}s unused={cmp_ref.unused_time_s:+.3f}s "
        f"peak_util={cmp_ref.grip_utilisation.max():.2f} | "
        f"{cmp_driver}: driver={cmp_cmp.lap_time_driver_s:.2f}s "
        f"unused={cmp_cmp.unused_time_s:+.3f}s peak_util={cmp_cmp.grip_utilisation.max():.2f} | "
        f"periodic_residual={solution.periodic_residual_ms:.3f}m/s"
    )


def run_envelope_stage_single(
    config: dict,
    lap,
    segments_df: pd.DataFrame,
    output_dirs: dict,
) -> None:
    env_cfg = config.get("envelope", {}) or {}
    if not env_cfg.get("enabled", False):
        return

    tables = Path(output_dirs["tables"])
    figures = Path(output_dirs["figures"])
    driver_code = config.get("drivers", {}).get("single") or config.get("drivers", {}).get("reference")

    track, channels = _prepare_lap_data(lap, env_cfg)
    params = _calibrate(track, channels, env_cfg)
    _write_vehicle_params(params, tables / "vehicle_params.csv")

    solution = solve_envelope(track, params)
    _write_envelope_solution(solution, tables / "envelope_solution.csv")

    comparison = compare_lap_to_envelope(track, solution, channels["speed_ms"], params)
    _write_comparison(comparison, tables / f"envelope_comparison_{driver_code}.csv")
    seg = summarise_by_segment(comparison, segments_df)
    seg.to_csv(tables / f"envelope_segment_summary_{driver_code}.csv", index=False)

    plot_envelope_overlay(comparison, solution, segments_df,
                          figures / f"envelope_overlay_{driver_code}.png", driver_label=driver_code)
    plot_grip_utilisation_map(track, comparison,
                              figures / f"grip_utilisation_map_{driver_code}.png")
    plot_segment_unused_time(seg, figures / f"segment_unused_time_{driver_code}.png")
    plot_gg_diagram(track, channels["speed_ms"], params,
                    figures / f"gg_diagram_{driver_code}.png", driver_label=driver_code)

    print(
        f"[envelope] {driver_code}: driver={comparison.lap_time_driver_s:.2f}s "
        f"model={comparison.lap_time_model_s:.2f}s unused={comparison.unused_time_s:+.3f}s "
        f"peak_util={comparison.grip_utilisation.max():.2f} "
        f"periodic_residual={solution.periodic_residual_ms:.3f}m/s"
    )
