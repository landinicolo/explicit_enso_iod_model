"""Synthetic ENSO/IOD/thermocline data, so examples and tests run without observations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import simulate_free_evolution


def make_synthetic_obs(n_years: int = 66, seed: int = 0, start="1958-01-16") -> dict:
    """Generate (E, I, h) by running the seasonal model itself with noise.

    The result is only meant to look vaguely like Nino3.4 / DMI / thermocline
    anomalies (E std ~0.8, I std ~0.4, ~3-4 yr oscillation); it is *not* real data.
    Returns a dict with ``time`` (mid-month DatetimeIndex), ``E``, ``I``, ``h``.
    """
    n = n_years * 12
    # [L0, L1c, L1s, L2c, L2s]
    params = dict(
        alpha_E=[0.045, 0.010, 0.012, 0.0, 0.0],
        a=[0.010, 0.002, 0.0, 0.0, 0.0],
        gamma_E=0.05,
        beta_E=0.09,
        alpha_I=[0.20, 0.03, -0.02, 0.0, 0.0],
        c=[0.06, 0.01, 0.0, 0.0, 0.0],
        gamma_I=0.20,
        F1=0.11, F2=0.18, eps_h=0.037,
        sigma_E=0.22, sigma_I=0.22, sigma_h=0.30,
    )
    x = simulate_free_evolution(np.zeros(5), n + 120, 1.0, params,
                                process_noise_scale=1.0, seed=seed)[120:]   # drop spin-up
    month_start = pd.Timestamp(start).to_period("M").to_timestamp()
    time = pd.date_range(month_start, periods=n, freq="MS") + pd.Timedelta(days=15)
    return dict(time=time, E=x[:, 0] + x[:, 1], I=x[:, 2] + x[:, 3], h=12.0 * x[:, 4])
