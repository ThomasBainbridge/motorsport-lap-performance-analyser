# Motorsport Lap Performance Analyser (MLPA)

A modular Python project that combines FastF1 telemetry analysis with a first-principles vehicle dynamics model to analyse laps, compare drivers, and quantify how close each driver got to the physical limit of the car.

## Overview

MLPA runs two complementary analyses on every lap:

1. **Data-driven comparison** — aligns two laps on distance, segments into braking/apex/exit phases, attributes time gain/loss per corner, and fits ML models to segment-level features.
2. **Physics-based envelope** — reconstructs the track from position data, builds a quasi-steady-state (QSS) lap-time model of the car, and compares the driver's actual speed trace against the theoretical best-possible lap given the car's grip, downforce, drag, and power.

The comparison stage answers *"who was faster where, and in what way?"* The envelope stage answers *"how much time was left on the table relative to what the car could physically do?"* Together they give both a relative (driver vs driver) and an absolute (driver vs physics) picture of a lap.

Two analysis modes are supported:

- **compare_laps** — both stages run; two drivers' laps are analysed against each other and against the envelope.
- **single_lap** — both stages run on one lap; the comparison stage describes the lap's structure, the envelope stage quantifies its headroom.

## Features

### Comparison stage

- FastF1 session loading and caching
- Accurate-lap filtering
- Two-lap telemetry alignment on a common distance basis
- Automatic braking-zone and corner segmentation
- Selected-segment time-loss attribution with phase-level drivers (braking, minimum-speed, traction/exit)
- Telemetry-derived segment feature extraction
- Corner archetype clustering with PCA map coordinates
- Regression model selection (Ridge, Random Forest, Gradient Boosting) with cross-validated metrics and held-out parity analysis
- Figure, table, and markdown-summary generation

### Envelope stage

- Track reconstruction from FastF1 X/Y position data using Menger curvature (preserves apex peaks that smoothing-based methods lose)
- Quasi-steady-state vehicle model with friction ellipse, speed-dependent aerodynamic downforce, drag and power limits
- Three-pass lap solver: cornering limit, forward acceleration pass, backward braking pass
- Periodic closed-loop boundary condition: `v(0) = v(L)` on any closed circuit
- Per-point and per-segment "unused time" — how much the driver trailed the theoretical envelope
- Grip utilisation on the friction ellipse (driver as a fraction of theoretical limit)
- Four new visualisations: envelope overlay, g-g diagram with speed-dependent ellipses, grip-utilisation track map, per-segment headroom bar chart

## Example result — 2025 Italian Grand Prix qualifying

Pinned 2025 F1 parameters (800 kg, μ_lat = 1.70, ClA = 3.2, CdA = 0.95, P = 820 kW):

| Driver | Real lap time | Envelope lap time | Unused time | Envelope utilisation |
|--------|--------------:|------------------:|------------:|---------------------:|
| VER (pole) | 78.75 s | 78.04 s | **+0.71 s** | **99.1%** |
| LEC | 78.80 s | 78.04 s | **+0.77 s** | **99.0%** |

Closed-loop periodic residual: 0.000 m/s. Real lap times recovered to 10 ms.

## Project structure

    motorsport-lap-performance-analyser/
    ├─ README.md
    ├─ pyproject.toml
    ├─ requirements.txt
    ├─ .gitignore
    ├─ configs/
    │  ├─ default.yaml
    │  ├─ monza_2025_q_ver_lec.yaml
    │  ├─ monza_2025_q_ver_single.yaml
    │  └─ silverstone_2025_q_nor_pia.yaml
    ├─ src/
    │  └─ mlpa/
    │     ├─ __init__.py
    │     ├─ main.py
    │     ├─ io.py
    │     ├─ filtering.py
    │     ├─ telemetry.py
    │     ├─ alignment.py
    │     ├─ segmentation.py
    │     ├─ features.py
    │     ├─ attribution.py
    │     ├─ ml_models.py
    │     ├─ plotting.py
    │     ├─ reporting.py
    │     ├─ single_lap.py
    │     ├─ utils.py
    │     ├─ track_model.py         # envelope: Menger curvature, track geometry
    │     ├─ vehicle_model.py       # envelope: QSS solver, VehicleParams
    │     ├─ calibration.py         # envelope: parameter fitting from telemetry
    │     ├─ envelope.py            # envelope: driver vs envelope comparison
    │     ├─ envelope_plotting.py   # envelope: 4 new plots
    │     └─ envelope_pipeline.py   # envelope: orchestration
    ├─ tests/
    │  ├─ test_alignment.py
    │  ├─ test_attribution.py
    │  ├─ test_features.py
    │  ├─ test_filtering.py
    │  ├─ test_ml_models.py
    │  ├─ test_segmentation.py
    │  ├─ test_single_lap.py
    │  └─ test_vehicle_model.py     # envelope: 10 physics and closed-loop tests
    ├─ outputs/
    │  ├─ compare_laps/
    │  └─ single_lap/
    └─ cache/

## Installation

    python -m venv .venv

    # Windows PowerShell
    .\.venv\Scripts\Activate.ps1

    # macOS/Linux
    source .venv/bin/activate

    python -m pip install -r requirements.txt
    python -m pip install -e .

## Run

Compare two laps (runs both comparison and envelope stages):

    python -m mlpa.main --config configs/monza_2025_q_ver_lec.yaml

Analyse a single lap:

    python -m mlpa.main --config configs/monza_2025_q_ver_single.yaml

Or after editable install:

    mlpa --config configs/monza_2025_q_ver_lec.yaml

At the end of each envelope run a one-line summary prints to the terminal showing real vs model lap time, unused time, peak utilisation, and the periodic residual. If the residual is below 0.5 m/s the closed-loop solver has converged.

## Configuration

YAML configs in `configs/` define the analysis mode, session, drivers, lap selection, output locations, ML settings, and envelope parameters. The envelope stage is gated by an `envelope.enabled` flag and is a no-op if absent, so existing configs remain backwards-compatible.

Minimal envelope block:

    envelope:
      enabled: true
      calibrate_from_reference: false
      mass_kg: 800
      mu_lat: 1.70
      mu_long: 1.85
      cda_m2: 0.95
      cla_m2: 3.20
      p_max_w: 820000
      curvature_chord_m: 12.0
      curvature_median_window: 7

With `calibrate_from_reference: true` the pipeline attempts to fit μ, ClA, CdA, P_max from the reference lap; see the Limitations section below.

## Typical outputs

### Compare-laps tables

- `aligned_trace.csv`
- `segments.csv`
- `segment_features.csv`
- `segment_ranking.csv`
- `segment_contributions.csv`
- `training_segment_features.csv`
- `clustered_segment_features.csv`
- `cluster_centers.csv`
- `cluster_profiles.csv`
- `regression_feature_importance.csv`
- `regression_metrics.csv`
- `regression_predictions.csv`
- `vehicle_params.csv` (envelope)
- `envelope_solution.csv` (envelope)
- `envelope_comparison_<DRIVER>.csv` (envelope, one per driver)
- `envelope_segment_summary_<DRIVER>.csv` (envelope, one per driver)

### Compare-laps figures

- `speed_overlay.png`
- `throttle_overlay.png`
- `brake_overlay.png`
- `delta_trace.png`
- `segment_losses.png`
- `segment_contributions.png`
- `cluster_map.png`
- `cluster_profiles.png`
- `regression_feature_importance.png`
- `regression_parity.png`
- `envelope_overlay_<DRIVER>.png` (envelope)
- `gg_diagram_<DRIVER>.png` (envelope)
- `grip_utilisation_map_<DRIVER>.png` (envelope)
- `segment_unused_time_<DRIVER>.png` (envelope)

### Single-lap tables

- `single_lap_trace.csv`
- `segments.csv`
- `single_lap_segment_features.csv`
- `clustered_segment_features.csv`
- `cluster_centers.csv`
- `cluster_profiles.csv`
- Envelope tables as above (one per driver)

### Single-lap figures

- `speed_trace.png`
- `throttle_trace.png`
- `brake_trace.png`
- `segment_metrics.png`
- `cluster_map.png`
- `cluster_profiles.png`
- Envelope figures as above (one per driver)

### Report

- `summary.md`

## Method notes

### Curvature estimation

Track curvature `κ(s)` is estimated by fitting a circle through three points spaced by a chord length (default 8–12 m) along the driven line, following the Menger-curvature formula. Earlier iterations used Savitzky-Golay smoothing of x(s), y(s) followed by double differentiation; this wiped out apex peaks at tight chicanes and was replaced. A short median filter removes isolated outliers without broadening real peaks.

### QSS lap solver

At each track point, corner-limited speed `v_corner` is derived by balancing centripetal demand against available lateral grip including aerodynamic downforce:

    m · v² · |κ| = μ_lat · (m · g + ½ · ρ · ClA · v²)

A forward pass integrates `v · dv/ds = a_lon` under drive-plus-grip limits (limited by power, by the friction ellipse, and by drag/rolling). A backward pass does the same under braking limits. The solution is `v_model(s) = min(v_corner, v_forward, v_backward)` at every point.

For a closed circuit the forward and backward passes are iterated to a periodic fixed point so that `v_model(0) = v_model(L)`, which must hold physically. This was an early source of wrong results before it was made periodic.

### Grip utilisation

The driver's per-point combined grip usage is:

    util = √((a_lat / a_lat_max)² + (a_lon / a_lon_max)²)

where `a_lat_max` and `a_lon_max` are the current speed's friction ellipse half-axes. Values near 1.0 mean the driver is on the ellipse; below ~0.9 means headroom available; sustained above 1.0 is either real driver performance exceeding the assumed parameters or numerical noise in the derivatives.

## Limitations

1. **Vehicle parameters are currently pinned, not fitted.** The calibration layer (`calibration.py`) can fit μ_lat, μ_long, ClA, CdA, P_max from a single lap but the fit is under-determined without DRS-state awareness and multi-lap pooling. The default recommended configuration pins reasonable 2025-regulation values per session rather than fitting them. Proper per-session calibration is the identified next piece of work.

2. **Peak utilisation has known artefacts at the fastest corners.** On the example Monza run, peak utilisation spikes to ~1.7 at Ascari and Parabolica. This is not a bug in the solver — it is a symptom of single-set parameters not simultaneously fitting both lap-time integral and instantaneous high-speed-corner grip. Mean utilisation is unaffected.

3. **No tyre load sensitivity.** μ is constant with normal load. Real F1 tyres lose some grip as N grows, which matters most under high downforce at top-speed corners.

4. **Constant-capability car assumption.** DRS opening/closing, battery deployment state, and fuel load are not modelled. Best used on qualifying laps where these are approximately steady.

## Running tests

    pytest

The envelope stage in particular has a 10-test suite (`tests/test_vehicle_model.py`) including:

- `test_chicane_apex_curvature_is_resolved` — verifies the curvature estimator recovers a 30 m chicane apex within 30% of ground truth
- `test_solve_envelope_is_periodic_on_closed_loop` — verifies the closed-loop boundary condition converges
- `test_envelope_comparison_at_98pct_is_plausible` — verifies a synthetic driver at 98% of envelope produces ~2% unused time and peak utilisation below 1.15

Both physics bugs encountered during development (non-periodic solver, apex-curvature wipeout) have dedicated tests that would catch them on a future regression.

## Usage notes

- Compare laps from the same session for the most meaningful results.
- Qualifying sessions are usually the easiest starting point for clean lap-to-lap comparisons and for envelope analysis; races have more DRS and fuel-load variation.
- Single-lap mode describes a lap on its own; the comparison stage cannot assign absolute time loss without a reference lap, but the envelope stage can.
- Cached FastF1 data is stored in the configured cache directory for faster repeated runs.
- Output files are written to analysis-specific subfolders under `outputs/`.