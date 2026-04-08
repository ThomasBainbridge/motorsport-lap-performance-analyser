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
    regression_metrics: dict[str, float],
) -> None:
    lines: list[str] = []

    selected_sum = float(segment_ranking_df["time_loss_s"].sum()) if not segment_ranking_df.empty else float("nan")

    lines.append("# Motorsport Lap Performance Analyzer — Version 1.1 summary")
    lines.append("")
    lines.append("## Session")
    lines.append("")
    lines.append(f"- Year: {config['session']['year']}")
    lines.append(f"- Grand Prix: {config['session']['grand_prix']}")
    lines.append(f"- Session: {config['session']['session']}")
    lines.append("")
    lines.append("## Compared laps")
    lines.append("")
    lines.append(
        f"- Reference: {reference_lap['Driver']} lap {int(reference_lap['LapNumber'])} ({reference_lap['LapTime']})"
    )
    lines.append(
        f"- Comparison: {comparison_lap['Driver']} lap {int(comparison_lap['LapNumber'])} ({comparison_lap['LapTime']})"
    )
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
    display_cols = [c for c in ["SegmentLabel", "time_loss_s", "Narrative"] if c in segment_ranking_df.columns]
    if not display_cols:
        display_cols = ["SegmentId", "time_loss_s", "Narrative"]
    lines.append(markdown_table_from_dataframe(segment_ranking_df[display_cols], max_rows=10))
    lines.append("")
    lines.append("## Regression metrics")
    lines.append("")
    lines.append(f"- Rows used: {regression_metrics.get('n_rows', float('nan')):.0f}")
    r2 = regression_metrics.get("r2")
    mae = regression_metrics.get("mae_s")
    lines.append(f"- R²: {r2:.3f}" if pd.notna(r2) else "- R²: not available")
    lines.append(f"- MAE (s): {mae:.4f}" if pd.notna(mae) else "- MAE (s): not available")
    lines.append("")
    lines.append("## Interpretation caveats")
    lines.append("")
    lines.append(
        "- The full-lap cumulative delta is a distance-aligned whole-lap trace. "
        "The selected-segment ranking covers only automatically detected braking/apex/exit regions, "
        "so the sum of selected-segment losses does not have to equal the full-lap delta."
    )
    lines.append(
        "- Corner archetype clustering is an unsupervised grouping of segment behaviour in feature space. "
        "It should not be described as direct driver-style identification in Version 1."
    )
    lines.append(
        "- The regression model is an explainable local attribution layer built on engineered features. "
        "It is intended for interpretation within this pipeline, not as a broadly validated predictive model."
    )

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
