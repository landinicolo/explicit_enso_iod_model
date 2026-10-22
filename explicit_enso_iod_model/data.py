"""Read / write the three monthly input series as a plain CSV (no xarray needed).

File format (header required, one row per month)::

    time,E,I,h
    1958-01-16,0.31,-0.05,1.8
    ...

``E`` Nino3.4, ``I`` DMI, ``h`` equatorial-Pacific thermocline-depth anomaly (all monthly
anomalies, detrended).  ``time`` is any date parsable by pandas, one per month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_indices_csv(path) -> dict:
    """Return ``dict(time, E, I, h)`` like ``synthetic.make_synthetic_obs``."""
    df = pd.read_csv(path, parse_dates=["time"])
    missing = {"time", "E", "I", "h"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}; expected time,E,I,h")
    df = df.sort_values("time")
    if df[["E", "I", "h"]].isna().any().any():
        raise ValueError(f"{path}: NaNs in E/I/h; the model needs gap-free monthly series")
    gaps = np.diff(df["time"].values.astype("datetime64[M]").astype(int))
    if (gaps != 1).any():
        raise ValueError(f"{path}: the time axis is not strictly monthly")
    return dict(time=pd.DatetimeIndex(df["time"]), E=df["E"].to_numpy(float),
                I=df["I"].to_numpy(float), h=df["h"].to_numpy(float))


def save_indices_csv(path, time, E, I, h) -> None:
    pd.DataFrame({"time": pd.DatetimeIndex(time).strftime("%Y-%m-%d"),
                  "E": np.asarray(E, float), "I": np.asarray(I, float),
                  "h": np.asarray(h, float)}).to_csv(path, index=False, float_format="%.6g")
