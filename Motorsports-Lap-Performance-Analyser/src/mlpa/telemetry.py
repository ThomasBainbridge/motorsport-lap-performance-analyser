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
