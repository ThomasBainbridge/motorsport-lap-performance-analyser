from __future__ import annotations

import numpy as np
import pandas as pd


def lap_to_car_telemetry(lap) -> pd.DataFrame:
    """Extract single-lap car telemetry and add cumulative distance."""
    telemetry = lap.get_car_data().copy()
    telemetry = telemetry.add_distance()
    telemetry = telemetry.sort_values("Distance").drop_duplicates(subset="Distance", keep="first")
    telemetry = telemetry.reset_index(drop=True)

    if "Throttle" in telemetry.columns:
        throttle = telemetry["Throttle"].astype(float).replace(104.0, np.nan)
        telemetry["Throttle"] = throttle.ffill().bfill().clip(0.0, 100.0)

    if "Brake" in telemetry.columns:
        telemetry["Brake"] = telemetry["Brake"].fillna(False).astype(int)

    for col in ("Speed", "RPM", "nGear", "DRS", "Distance"):
        if col in telemetry.columns:
            telemetry[col] = pd.to_numeric(telemetry[col], errors="coerce")

    telemetry = telemetry.dropna(subset=["Distance", "Speed", "Time"])
    telemetry = telemetry[telemetry["Distance"].diff().fillna(0) >= 0].reset_index(drop=True)
    return telemetry


def make_monotonic_time_seconds(telemetry: pd.DataFrame) -> np.ndarray:
    time_seconds = pd.to_timedelta(telemetry["Time"]).dt.total_seconds().to_numpy(dtype=float)
    if time_seconds.size == 0:
        return time_seconds
    time_seconds = np.maximum.accumulate(time_seconds)
    return time_seconds


def lap_to_merged_telemetry(lap) -> pd.DataFrame:
    """Extract lap telemetry including X, Y position, on a common distance basis.

    FastF1's `get_telemetry()` merges car channels with position data and
    resamples; `get_car_data()` used elsewhere in this package does not
    carry X/Y and is not suitable for vehicle-model analysis.
    """
    tel = lap.get_telemetry().copy()
    tel = tel.sort_values("Distance").drop_duplicates(subset="Distance", keep="first")
    tel = tel.reset_index(drop=True)

    if "Throttle" in tel.columns:
        throttle = tel["Throttle"].astype(float).replace(104.0, np.nan)
        tel["Throttle"] = throttle.ffill().bfill().clip(0.0, 100.0)
    if "Brake" in tel.columns:
        tel["Brake"] = tel["Brake"].fillna(False).astype(int)

    for col in ("Speed", "RPM", "nGear", "DRS", "Distance", "X", "Y", "Z"):
        if col in tel.columns:
            tel[col] = pd.to_numeric(tel[col], errors="coerce")

    tel = tel.dropna(subset=["Distance", "Speed", "X", "Y"])
    tel = tel[tel["Distance"].diff().fillna(0) >= 0].reset_index(drop=True)
    return tel
