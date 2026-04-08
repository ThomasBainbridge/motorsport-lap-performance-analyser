# Motorsport Lap Performance Analyser (MLPA)

A modular Python project that uses FastF1 timing and telemetry data to compare laps, decompose lap-time differences into selected braking/apex/exit segments, and apply machine learning to telemetry-derived performance features.

## Overview

MLPA is designed to analyse two laps from the same session and quantify where time is gained or lost around the circuit. It combines telemetry alignment, automatic segment detection, feature extraction, comparative analysis, unsupervised clustering, and regression-based feature importance in a single Python workflow.

## Features

- FastF1 session loading and caching
- accurate-lap filtering
- two-lap telemetry alignment over a common distance basis
- automatic braking-zone and corner segmentation
- selected-segment time-loss attribution
- telemetry-derived segment feature extraction
- corner archetype clustering
- regression-based segment time-loss feature importance
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
    │  └─ monza_2025_q_ver_lec.yaml
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
    │     └─ utils.py
    ├─ tests/
    │  ├─ test_alignment.py
    │  ├─ test_features.py
    │  ├─ test_filtering.py
    │  └─ test_segmentation.py
    ├─ outputs/
    │  ├─ figures/
    │  ├─ tables/
    │  └─ reports/
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

    python -m mlpa.main --config configs/monza_2025_q_ver_lec.yaml

or after editable install:

    mlpa --config configs/monza_2025_q_ver_lec.yaml

## Configuration

The project is driven by YAML config files stored in `configs/`. A config defines the session, drivers, lap-selection settings, output locations, clustering settings, regression settings, and optional segment labels.

Example:

    session:
      year: 2025
      grand_prix: "Monza"
      session: "Q"

    comparison:
      reference_driver: "VER"
      comparison_driver: "LEC"

    segmentation:
      segment_labels:
        1: "First chicane"
        2: "Roggia"
        3: "Lesmo 1"
        4: "Lesmo 2"
        5: "Ascari"
        6: "Parabolica"

If no labels are provided, the tool falls back to `S1`, `S2`, `S3`, and so on.

## Typical outputs

### Tables

- `outputs/tables/aligned_trace.csv`
- `outputs/tables/segments.csv`
- `outputs/tables/segment_features.csv`
- `outputs/tables/segment_ranking.csv`
- `outputs/tables/training_segment_features.csv`
- `outputs/tables/clustered_segment_features.csv`
- `outputs/tables/cluster_centers.csv`
- `outputs/tables/regression_feature_importance.csv`

### Figures

- `outputs/figures/speed_overlay.png`
- `outputs/figures/throttle_overlay.png`
- `outputs/figures/brake_overlay.png`
- `outputs/figures/delta_trace.png`
- `outputs/figures/segment_losses.png`
- `outputs/figures/cluster_map.png`
- `outputs/figures/regression_feature_importance.png`

### Report

- `outputs/reports/summary.md`

## Main outputs explained

### Speed overlay with selected segment markers

Compares the two laps on a common distance basis and marks the detected segments used for local comparison.

### Throttle overlay

Shows throttle application differences along the lap distance.

### Brake overlay

Shows braking application differences along the lap distance.

### Full-lap cumulative delta

Shows how the comparison lap gains or loses time relative to the reference lap across the full aligned lap.

### Selected-segment time loss

Ranks the automatically detected segments by local time gain or loss.

### Corner archetype clustering

Groups segments in feature space using telemetry-derived corner and exit behaviour.

### Segment time-loss model feature importance

Ranks the engineered features most associated with local segment time loss.

## Usage notes

- Compare laps from the same session for the most meaningful results.
- Qualifying sessions are usually the easiest starting point for clean lap-to-lap comparisons.
- Cached FastF1 data is stored in the configured cache directory for faster repeated runs.
- Output files are regenerated each time the analysis is run.

## Running tests

    pytest

## Dependencies

Main dependencies include:

- FastF1
- NumPy
- pandas
- Matplotlib
- scikit-learn
- SciPy
- PyYAML
- pytest

