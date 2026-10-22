"""High-level workflow object tying the pieces together."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import diagnostics as dg
from .core import STATE_NAMES, simulate_free_evolution
from .ensembles import (Hindcasts, calibrate_process_noise_scale, run_freerun_ensemble,
                        run_hindcasts)
from .estimation import (KalmanConfig, attach_recharge_terms, decompose, fit_seasonal_direct,
                         summarize_direct_fit)


class ExplicitENSOIODModel:
    """Fit the seasonal model to monthly E, I, h series and run experiments with it.

    Parameters
    ----------
    E, I, h : monthly anomaly series (array-like or xarray ``DataArray``) of the ENSO
        index, IOD index and thermocline depth, equal length, detrended / anomalies.
    time : monthly datetimes (one per step).  Defaults to ``E.time`` if present, else
        a synthetic monthly axis starting in January.
    dt : step length in months.

    ``decompose()`` (Kalman split only) sets ``baseline``, ``kalman_params`` and ``x_kalman``;
    ``fit()`` additionally sets the rest.  After ``fit()`` the following attributes exist:

    ``baseline``      constant coefficients (``fit_baseline``)
    ``kalman_params`` filter parameters actually used
    ``x_kalman``      (T, 5) filtered latent states ``[E_I, E_D, I_I, I_D, h]``
    ``params``        fitted seasonal parameter dict (feeds ``simulate_free_evolution``)
    ``diagnostics``   per-equation regression diagnostics
    ``noise_scale``   process-noise multiplier calibrated to the observed variance
    """

    def __init__(self, E, I, h, time=None, dt: float = 1.0):
        self.E = np.asarray(E, dtype=float)
        self.I = np.asarray(I, dtype=float)
        self.h = np.asarray(h, dtype=float)
        if not (len(self.E) == len(self.I) == len(self.h)):
            raise ValueError("E, I and h must have the same length")
        if time is None:
            time = getattr(E, "time", None)
            time = None if time is None else np.asarray(time)
        if time is None:
            time = pd.date_range("2000-01-15", periods=len(self.E), freq="MS")
        self.time = pd.DatetimeIndex(time)
        self.dt = dt

        self.baseline = self.kalman_params = self.x_kalman = None
        self.params = self.diagnostics = self.noise_scale = None

    # -- bookkeeping -------------------------------------------------------
    @property
    def months(self) -> np.ndarray:
        return self.time.month.values

    @property
    def start_month(self) -> int:
        return int(self.months[0])

    def _require_fit(self):
        if self.params is None:
            raise RuntimeError("call .fit() first")

    # -- fitting -----------------------------------------------------------
    def decompose(self, kalman: KalmanConfig | None = None):
        """Steps 1-2 only: baseline coefficients and Kalman split into E_I, E_D, I_I, I_D.

        Sets ``baseline``, ``kalman_params`` and ``x_kalman``.  Use this alone if you only
        want the four-term decomposition; see ``decomposition()`` for a DataFrame.
        """
        self.x_kalman, self.baseline, self.kalman_params = decompose(
            self.E, self.I, self.h, self.dt, kalman)
        return self

    def decomposition(self) -> pd.DataFrame:
        """DataFrame (indexed by time) with E, I, their four latent parts and standardised h.

        The first row is the filter initialisation and is returned as NaN (in the copy only:
        ``x_kalman`` itself is never modified).
        """
        if self.x_kalman is None:
            raise RuntimeError("call .decompose() or .fit() first")
        df = pd.DataFrame(self.x_kalman.copy(), index=self.time, columns=list(STATE_NAMES[:4]) + ["h_norm"])
        df.insert(0, "I", self.I)
        df.insert(0, "E", self.E)
        df.iloc[0, 2:] = np.nan
        return df

    def fit(self, kalman: KalmanConfig | None = None, calibrate_noise: bool = True):
        """Decomposition -> seasonal regression -> noise scale."""
        self.decompose(kalman)
        params, self.diagnostics = fit_seasonal_direct(self.x_kalman, self.dt, self.start_month)
        kp = self.kalman_params
        self.params = attach_recharge_terms(params, kp["F1"], kp["F2"], kp["eps_h"])
        if calibrate_noise:
            self.noise_scale = calibrate_process_noise_scale(
                self.E, self.I, self.x_kalman[0], self.params, self.dt,
                start_month=self.start_month)
        return self

    def fit_summary(self) -> dict:
        self._require_fit()
        return summarize_direct_fit(self.params, self.diagnostics)

    # -- experiments -------------------------------------------------------
    def simulate(self, n_steps: int | None = None, seed: int = 7, noise: bool = True,
                 x0=None) -> np.ndarray:
        """One free run (default: record length), from the first Kalman state unless ``x0`` is given."""
        self._require_fit()
        x0 = self.x_kalman[0] if x0 is None else x0
        scale = (self.noise_scale or 0.0) if noise else 0.0
        return simulate_free_evolution(x0, n_steps or len(self.E), self.dt, self.params,
                                       process_noise_scale=scale, seed=seed,
                                       start_month=self.start_month)

    def freerun_ensemble(self, n_members: int = 100, x0_jitter_scale: float = 0.05,
                         seed: int = 13, n_steps: int | None = None, x0=None) -> np.ndarray:
        """Ensemble of free runs, shape ``(n_members, n_steps, 5)``."""
        self._require_fit()
        x0 = self.x_kalman[0] if x0 is None else x0
        return run_freerun_ensemble(
            self.params, x0, n_steps or len(self.E), self.dt, n_members=n_members,
            process_noise_scale=self.noise_scale or 0.0, x0_jitter_scale=x0_jitter_scale,
            seed=seed, start_month=self.start_month)

    def hindcast(self, max_lead: int = 24, init_stride: int = 1, n_members: int = 20,
                 x0_jitter_scale: float = 0.05, seed: int = 123) -> Hindcasts:
        """In-sample lagged-initialisation hindcasts (see ``ensembles.run_hindcasts``)."""
        self._require_fit()
        return run_hindcasts(
            self.x_kalman, self.E, self.I, self.params, self.months, self.dt,
            max_lead=max_lead, init_stride=init_stride, n_members=n_members,
            process_noise_scale=self.noise_scale or 0.0, x0_jitter_scale=x0_jitter_scale,
            seed=seed)

    # -- convenience diagnostics ------------------------------------------
    def obs(self, var: str) -> np.ndarray:
        return self.E if var.upper() == "E" else self.I

    def observed_peak_months(self, var: str = "E", min_abs_peak: float = 0.01) -> list:
        return dg.peak_months_per_year(self.obs(var), self.time.year.values, self.months,
                                       min_abs_peak)
