from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def timedelta_to_seconds(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return pd.to_timedelta(value).total_seconds()


def series_timedelta_to_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds()


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index ranges for contiguous True regions."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    regions: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            regions.append((start, idx - 1))
            start = None
    if start is not None:
        regions.append((start, mask.size - 1))
    return regions


def safe_event_name(session) -> str:
    try:
        return str(session.event["EventName"])
    except Exception:
        return "Unknown Event"


def safe_session_name(session) -> str:
    for attr in ("name", "session_info"):
        value = getattr(session, attr, None)
        if isinstance(value, str):
            return value
    try:
        return str(session.name)
    except Exception:
        return "Unknown Session"


def coerce_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    return series.fillna(default).astype(bool)


def first_valid_index_where(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def last_valid_index_where(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[-1])


def nearest_index(distance: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(distance - target)))


def markdown_table_from_dataframe(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows available._"

    preview = df.head(max_rows).copy()
    preview = preview.replace({np.nan: ""})
    preview = preview.astype(str)

    headers = list(preview.columns)
    separator = ["---"] * len(headers)
    rows = preview.values.tolist()

    def _row(values: list[str]) -> str:
        escaped = [str(v).replace("|", r"\|") for v in values]
        return "| " + " | ".join(escaped) + " |"

    lines = [_row(headers), _row(separator)]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def standardize_trackstatus_to_green(status_value) -> bool:
    if status_value is None or (isinstance(status_value, float) and np.isnan(status_value)):
        return False
    return str(status_value) == "1"


def format_seconds(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"
