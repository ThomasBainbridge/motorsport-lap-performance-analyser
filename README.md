# Motorsport Lap Performance Analyser (MLPA)

A modular Python project that uses FastF1 timing and telemetry data to analyse laps, compare two drivers, and apply machine learning to telemetry-derived performance features.

## Overview

MLPA supports two analysis modes:

- **compare_laps**: aligns two laps from the same session and quantifies where time is gained or lost.
- **single_lap**: analyses one lap on its own and summarises its structure, control application, and segment characteristics.

The project combines telemetry alignment, automatic segment detection, feature extraction, comparative analysis, unsupervised clustering, regression model selection, and report generation in a single Python workflow.

## Features

- FastF1 session loading and caching
- accurate-lap filtering
- two-lap telemetry alignment over a common distance basis
- single-lap telemetry analysis mode
- automatic braking-zone and corner segmentation
- selected-segment time-loss attribution with phase-level drivers
- telemetry-derived segment feature extraction
- corner archetype clustering with PCA map coordinates
- regression model selection across multiple candidates
- cross-validated regression metrics and holdout parity analysis
- figure, table, and summary report generation
- optional track-specific segment labels through the config file

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
    │     └─ utils.py
    ├─ tests/
    │  ├─ test_alignment.py
    │  ├─ test_attribution.py
    │  ├─ test_features.py
    │  ├─ test_filtering.py
    │  ├─ test_ml_models.py
    │  ├─ test_segmentation.py
    │  └─ test_single_lap.py
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

Compare two laps:

    python -m mlpa.main --config configs/monza_2025_q_ver_lec.yaml

Analyse a single lap:

    python -m mlpa.main --config configs/monza_2025_q_ver_single.yaml

or after editable install:

    mlpa --config configs/monza_2025_q_ver_lec.yaml

## Configuration

The project is driven by YAML config files stored in `configs/`. A config defines the analysis mode, session, drivers, lap-selection settings, output locations, clustering settings, regression settings, and optional segment labels.

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

### Single-lap tables

- `single_lap_trace.csv`
- `segments.csv`
- `single_lap_segment_features.csv`
- `clustered_segment_features.csv`
- `cluster_centers.csv`
- `cluster_profiles.csv`

### Single-lap figures

- `speed_trace.png`
- `throttle_trace.png`
- `brake_trace.png`
- `segment_metrics.png`
- `cluster_map.png`
- `cluster_profiles.png`

### Report

- `summary.md`

## Usage notes

- Compare laps from the same session for the most meaningful results.
- Qualifying sessions are usually the easiest starting point for clean lap-to-lap comparisons.
- Single-lap mode describes a lap on its own; it does not assign absolute time loss without a reference lap.
- Cached FastF1 data is stored in the configured cache directory for faster repeated runs.
- Output files are written to analysis-specific subfolders under `outputs/`.

## Running tests

    pytest
