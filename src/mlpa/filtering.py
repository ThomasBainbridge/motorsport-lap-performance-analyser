from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import coerce_bool_series, standardize_trackstatus_to_green


def filter_candidate_laps(
    laps: pd.DataFrame,
    *,
    require_accuracy: bool = True,
    exclude_deleted: bool = True,
    exclude_generated: bool = True,
    quicklaps: bool = True,
    green_flag_only: bool = True,
) -> pd.DataFrame:
    """Apply quality-control filters to a FastF1 Laps dataframe."""
    if laps is None or len(laps) == 0:
        return laps

    filtered = laps.copy()
    filtered = filtered[filtered["LapTime"].notna()]

    if quicklaps and hasattr(filtered, "pick_quicklaps"):
        try:
            filtered = filtered.pick_quicklaps()
        except Exception:
            pass

    if require_accuracy and "IsAccurate" in filtered.columns:
        filtered = filtered[coerce_bool_series(filtered["IsAccurate"], default=False)]

    if exclude_deleted and "Deleted" in filtered.columns:
        filtered = filtered[~coerce_bool_series(filtered["Deleted"], default=False)]

    if exclude_generated and "FastF1Generated" in filtered.columns:
        filtered = filtered[~coerce_bool_series(filtered["FastF1Generated"], default=False)]

    if green_flag_only and "TrackStatus" in filtered.columns:
        filtered = filtered[filtered["TrackStatus"].apply(standardize_trackstatus_to_green)]

    return filtered.sort_values("LapTime").reset_index(drop=True)


def select_driver_lap(
    session,
    *,
    driver: str,
    selection_cfg: dict[str, Any],
):
    laps = session.laps.pick_drivers(driver)
    candidates = filter_candidate_laps(
        laps,
        require_accuracy=selection_cfg.get("require_accuracy", True),
        exclude_deleted=selection_cfg.get("exclude_deleted", True),
        exclude_generated=selection_cfg.get("exclude_generated", True),
        quicklaps=selection_cfg.get("quicklaps", True),
        green_flag_only=selection_cfg.get("green_flag_only", True),
    )

    if len(candidates) == 0:
        raise ValueError(f"No usable laps found for driver '{driver}' after filtering.")

    mode = str(selection_cfg.get("mode", "fastest")).lower()
    only_by_time = bool(selection_cfg.get("only_by_time", False))

    if mode == "fastest":
        lap = candidates.pick_fastest(only_by_time=only_by_time)
        if lap is None:
            raise ValueError(f"Unable to select fastest lap for driver '{driver}'.")
        return lap

    if mode == "lap_number":
        lap_number = selection_cfg.get("lap_number")
        if lap_number is None:
            raise ValueError("lap_selection.mode='lap_number' requires 'lap_number'.")
        selected = candidates[candidates["LapNumber"] == float(lap_number)]
        if len(selected) == 0:
            raise ValueError(f"Lap {lap_number} not found for driver '{driver}'.")
        return selected.iloc[0]

    raise ValueError(f"Unsupported lap selection mode: {mode}")


def select_training_laps(session, driver: str, selection_cfg: dict[str, Any], top_n: int) -> pd.DataFrame:
    laps = session.laps.pick_drivers(driver)
    candidates = filter_candidate_laps(
        laps,
        require_accuracy=selection_cfg.get("require_accuracy", True),
        exclude_deleted=selection_cfg.get("exclude_deleted", True),
        exclude_generated=selection_cfg.get("exclude_generated", True),
        quicklaps=selection_cfg.get("quicklaps", True),
        green_flag_only=selection_cfg.get("green_flag_only", True),
    )
    if len(candidates) == 0:
        return candidates
    return candidates.sort_values("LapTime").head(int(top_n)).reset_index(drop=True)
