"""Parameter estimation.

Pipeline (each step feeds the next):

1. ``fit_baseline``        - constant AR(1)-type coefficients from the observed E, I, h
2. ``KalmanConfig`` + ``kalman_filter_abs``
                           - split E and I into independent/dependent latent parts
3. ``fit_seasonal_direct`` - regress the latent-state tendencies on the seasonal basis
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import SEASONAL_PARAM_NAMES, seasonal_basis, seasonal_phase


# --------------------------------------------------------------------------
# 1. Baseline (constant) coefficients
# --------------------------------------------------------------------------
def decay_rate_from_lag1(x, dt: float = 1.0):
    """AR(1) damping rate from the lag-1 autocorrelation: ``alpha = -ln(rho1)/dt``.

    Returns ``(rho1, alpha)``; ``alpha`` is NaN if ``rho1 <= 0``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    rho1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    alpha = -np.log(rho1) / dt if rho1 > 0 else np.nan
    return rho1, alpha


def fit_enso_iod_core(E, I, dt: float = 1.0) -> dict:
    """Constant damping and coupling between E and I.

    ``alpha_E``/``alpha_I`` come from the lag-1 autocorrelations; the couplings
    are least-squares fits of ``dE/dt + alpha_E E`` on ``I`` and of
    ``dI/dt + alpha_I I`` on ``E``.

    (Old notebook names: ``a, b, c, d`` = ``alpha_E, coupling_I_to_E,
    coupling_E_to_I, alpha_I``.)
    """
    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)
    mask = np.isfinite(E) & np.isfinite(I)
    E = E[mask] - np.mean(E[mask])
    I = I[mask] - np.mean(I[mask])

    _, alpha_E = decay_rate_from_lag1(E, dt)
    _, alpha_I = decay_rate_from_lag1(I, dt)

    dE = np.diff(E) / dt
    dI = np.diff(I) / dt

    coupling_I_to_E = np.linalg.lstsq(I[:-1, None], dE + alpha_E * E[:-1], rcond=None)[0][0]
    coupling_E_to_I = np.linalg.lstsq(E[1:, None], dI + alpha_I * I[:-1], rcond=None)[0][0]
    return dict(alpha_E=alpha_E, coupling_I_to_E=coupling_I_to_E,
                coupling_E_to_I=coupling_E_to_I, alpha_I=alpha_I)


def fit_recharge_oscillator(E, h, alpha_E: float, eps_h: float, dt: float = 1.0):
    """Thermocline couplings ``(F1, F2)``: h -> E and E -> h, by least squares."""
    E = np.asarray(E, dtype=float)
    h = np.asarray(h, dtype=float)
    mask = np.isfinite(E) & np.isfinite(h)
    E = E[mask] - np.mean(E[mask])
    h = h[mask] - np.mean(h[mask])

    dE = np.diff(E) / dt
    dh = np.diff(h) / dt

    F1 = np.linalg.lstsq(h[:-1, None], dE + alpha_E * E[:-1], rcond=None)[0][0]
    F2 = np.linalg.lstsq(E[:-1, None], -(dh + eps_h * h[:-1]), rcond=None)[0][0]
    return F1, F2


def fit_baseline(E, I, h, dt: float = 1.0) -> dict:
    """All constant coefficients, plus the standardised thermocline series.

    ``h`` is demeaned and divided by its std before fitting, so ``F1``/``F2``/
    ``eps_h`` refer to the standardised thermocline (returned as ``h_norm``).
    """
    h_raw = np.asarray(h, dtype=float)
    h_raw = h_raw - np.nanmean(h_raw)
    h_std = float(np.nanstd(h_raw))
    h_norm = h_raw / h_std

    out = fit_enso_iod_core(E, I, dt)
    rho_h, eps_h = decay_rate_from_lag1(h_norm, dt)
    F1, F2 = fit_recharge_oscillator(E, h_norm, out["alpha_E"], eps_h, dt)

    out.update(F1=F1, F2=F2, eps_h=eps_h, rho_h=rho_h, h_std=h_std,
               osc_strength=F1 * F2, eigen_timescale=1.0 / np.sqrt(F1 * F2),
               h_norm=h_norm)
    return out


# --------------------------------------------------------------------------
# 2. Extended Kalman filter: E -> (E_I, E_D), I -> (I_I, I_D)
# --------------------------------------------------------------------------
@dataclass
class KalmanConfig:
    """Hand-tuned choices that turn baseline coefficients into filter parameters.

    The defaults are the values used in the original notebook.  They are
    *tuning knobs*, not fitted quantities:

    beta_E, beta_I        nonlinear coupling strengths (``E_I |I|`` and ``E I_I`` terms)
    sigma_E/sigma_I/sigma_h   process-noise std of E_I, I_I, h
    coupling_I_to_E_scale multiplies the fitted I->E coupling (0 switches it off)
    coupling_E_to_I_scale multiplies the fitted E->I coupling
    round_decimals        parameters are rounded before filtering (as in the notebook); None = no rounding
    """
    beta_E: float = 0.09
    beta_I: float = 0.0
    sigma_E: float = 1.0
    sigma_I: float = 1.0
    sigma_h: float = 1.0
    coupling_I_to_E_scale: float = 0.0
    coupling_E_to_I_scale: float = 3.0
    round_decimals: int | None = 4

    def build_params(self, baseline: dict) -> dict:
        p = dict(
            alpha_E=baseline["alpha_E"], alpha_I=baseline["alpha_I"],
            gamma_E=baseline["alpha_E"], gamma_I=baseline["alpha_I"],
            a=baseline["coupling_I_to_E"] * self.coupling_I_to_E_scale,
            c=baseline["coupling_E_to_I"] * self.coupling_E_to_I_scale,
            beta_E=self.beta_E, beta_I=self.beta_I,
            sigma_E=self.sigma_E, sigma_I=self.sigma_I, sigma_h=self.sigma_h,
            F1=baseline["F1"], F2=baseline["F2"], eps_h=baseline["eps_h"],
        )
        p = {k: float(v) for k, v in p.items()}
        if self.round_decimals is not None:
            p = {k: round(v, self.round_decimals) for k, v in p.items()}
        return p


def kalman_filter_abs(E, I, h, dt: float, p: dict, obs_noise: float = 0.01) -> np.ndarray:
    """Extended Kalman filter for ``[E_I, E_D, I_I, I_D, h]`` observed through E, I, h.

    ``p`` is a dict as built by ``KalmanConfig.build_params``.  The nonlinearity
    is ``beta_E * E_I * |I|`` in the E_D equation.  Observations are the sums
    ``E = E_I + E_D``, ``I = I_I + I_D`` and ``h`` itself.

    Returns the filtered states, shape ``(T, 5)``.  Row 0 is the filter's
    initialisation (zeros), not an estimate.
    """
    alpha_E, alpha_I = p["alpha_E"], p["alpha_I"]
    gamma_E, gamma_I = p["gamma_E"], p["gamma_I"]
    a, c = p["a"], p["c"]
    beta_E, beta_I = p["beta_E"], p["beta_I"]
    F1, F2, eps_h = p["F1"], p["F2"], p["eps_h"]

    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)
    h = np.asarray(h, dtype=float)
    T = len(E)
    n = 5

    H = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    eps_q = 1e-8  # tiny process noise on the "dependent" states
    Q = np.diag([p["sigma_E"] ** 2, eps_q, p["sigma_I"] ** 2, eps_q, p["sigma_h"] ** 2])
    R = np.eye(3) * max(obs_noise, 1e-6)

    x_est = np.zeros((T, n))
    P = np.eye(n)
    y = np.vstack([E, I, h]).T

    for t in range(1, T):
        E_I, E_D, I_I, I_D, h_ = x_est[t - 1]
        I_sum = I_I + I_D

        dE_I = -alpha_E * E_I + F1 * h_
        dE_D = a * I_sum - gamma_E * E_D - beta_E * E_I * np.abs(I_sum)
        dI_I = -alpha_I * I_I
        dI_D = c * (E_I + E_D) - gamma_I * I_D + beta_I * (E_I + E_D) * I_I
        dh = -F2 * E_I - eps_h * h_

        x_pred = np.array([
            E_I + dt * dE_I,
            E_D + dt * dE_D,
            I_I + dt * dI_I,
            I_D + dt * dI_D,
            h_ + dt * dh,
        ])

        # Jacobian of the discrete map
        F = np.array([
            [1 - alpha_E * dt, 0, 0, 0, dt * F1],
            [-beta_E * dt * np.abs(I_sum), 1 - gamma_E * dt,
             dt * (a - beta_E * E_I * np.sign(I_sum)),
             dt * (a - beta_E * E_I * np.sign(I_sum)), 0],
            [0, 0, 1 - alpha_I * dt, 0, 0],
            [dt * (c + beta_I * I_I), dt * (c + beta_I * I_I),
             dt * beta_I * (E_I + E_D), 1 - gamma_I * dt, 0],
            [-dt * F2, 0, 0, 0, 1 - dt * eps_h],
        ])

        P_pred = F @ P @ F.T + Q
        if not np.isfinite(P_pred).all():
            P = np.eye(n)
            P_pred = F @ P @ F.T + Q

        S = H @ P_pred @ H.T + R
        S += np.eye(S.shape[0]) * 1e-8
        try:
            K = (P_pred @ H.T) @ np.linalg.solve(S, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            K = P_pred @ H.T @ np.linalg.pinv(S)

        x_est[t] = x_pred + K @ (y[t] - H @ x_pred)
        P = (np.eye(n) - K @ H) @ P_pred

    return x_est


def decompose(E, I, h, dt: float = 1.0, config: KalmanConfig | None = None):
    """Steps 1-2 only: split E and I into their four latent parts.

    Returns ``(x, baseline, kalman_params)`` where ``x`` has shape ``(T, 5)`` with columns
    ``[E_I, E_D, I_I, I_D, h_norm]`` (``h_norm`` is the standardised thermocline).
    Row 0 is the filter initialisation (zeros), not an estimate.
    """
    baseline = fit_baseline(E, I, h, dt)
    kalman_params = (config or KalmanConfig()).build_params(baseline)
    x = kalman_filter_abs(E, I, baseline["h_norm"], dt, kalman_params)
    return x, baseline, kalman_params


# --------------------------------------------------------------------------
# 3. Direct seasonal regression on the Kalman states
# --------------------------------------------------------------------------
def _safe_rows(*arrays, eps: float = 1e-10):
    mask = np.ones_like(np.asarray(arrays[0], dtype=float), dtype=bool)
    for array in arrays:
        values = np.asarray(array, dtype=float)
        mask &= np.isfinite(values) & (np.abs(values) > eps)
    return mask


def _least_squares(design, target, rcond=None) -> dict:
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    coef, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=rcond)
    resid = target - design @ coef
    return {"coef": coef, "residual": resid, "rss": float(np.sum(resid ** 2)),
            "rank": int(rank), "singular_values": singular_values}


def fit_seasonal_direct(x_kalman, dt: float = 1.0, start_month: int = 1, rcond=None):
    """Fit all seasonal coefficients by linear least squares, treating the Kalman
    states as truth.

    Regressed tendencies (each coefficient is expanded on the 5-term seasonal basis)::

        dE_I/dt = -alpha_E(t) E_I
        dE_D/dt =  a(t) I_tot - gamma_E(t) E_D - beta_E(t) E_I |I_tot|
        dI_I/dt = -alpha_I(t) I_I
        dI_D/dt =  c(t) E_tot - gamma_I(t) I_D

    Note the thermocline term ``F1 h`` is *not* in the ``E_I`` regression, so the
    fitted ``alpha_E`` implicitly absorbs it.

    Returns
    -------
    params : dict name -> 5-vector, plus ``sigma_E`` / ``sigma_I`` (std of the
        ``alpha`` regression residuals times ``dt``).
    diagnostics : dict of per-equation fit info (``coef``, ``residual``, ``rss``, ...).
    """
    x = np.asarray(x_kalman, dtype=float)
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("x_kalman must have shape (n_steps, 5)")
    if x.shape[0] < 2:
        raise ValueError("x_kalman must contain at least 2 time points")

    E_I, E_D, I_I, I_D = (x[:-1, k] for k in range(4))
    dE_I, dE_D, dI_I, dI_D = (np.diff(x[:, k]) / dt for k in range(4))

    B = seasonal_basis(seasonal_phase(x.shape[0] - 1, dt, start_month))
    params, diag = {}, {}

    mask = _safe_rows(E_I)
    fit = _least_squares(B[mask] * (-E_I[mask, None]), dE_I[mask], rcond)
    params["alpha_E"], diag["alpha_E"] = fit["coef"], fit

    mask = _safe_rows(I_I)
    fit = _least_squares(B[mask] * (-I_I[mask, None]), dI_I[mask], rcond)
    params["alpha_I"], diag["alpha_I"] = fit["coef"], fit

    I_tot, E_tot = I_I + I_D, E_I + E_D

    design = np.column_stack([
        B * I_tot[:, None],
        B * (-E_D[:, None]),
        B * (-(E_I * np.abs(I_tot))[:, None]),
    ])
    mask = np.all(np.isfinite(design), axis=1) & np.isfinite(dE_D)
    fit = _least_squares(design[mask], dE_D[mask], rcond)
    params["a"], params["gamma_E"], params["beta_E"] = (fit["coef"][i:i + 5] for i in (0, 5, 10))
    diag["E_D_equation"] = fit

    design = np.column_stack([B * E_tot[:, None], B * (-I_D[:, None])])
    mask = np.all(np.isfinite(design), axis=1) & np.isfinite(dI_D)
    fit = _least_squares(design[mask], dI_D[mask], rcond)
    params["c"], params["gamma_I"] = fit["coef"][0:5], fit["coef"][5:10]
    diag["I_D_equation"] = fit

    params["sigma_E"] = float(np.std(diag["alpha_E"]["residual"] * dt, ddof=1))
    params["sigma_I"] = float(np.std(diag["alpha_I"]["residual"] * dt, ddof=1))
    return params, diag


def attach_recharge_terms(params: dict, F1: float, F2: float, eps_h: float,
                          sigma_h: float | None = None) -> dict:
    """Add the thermocline terms the regression does not estimate.

    ``sigma_h`` defaults to ``sigma_E`` (as in the notebook).  Returns a new dict.
    """
    out = dict(params)
    out.setdefault("sigma_h", params["sigma_E"] if sigma_h is None else sigma_h)
    out.setdefault("F1", F1)
    out.setdefault("F2", F2)
    out.setdefault("eps_h", eps_h)
    return out


def summarize_direct_fit(params: dict, diagnostics: dict) -> dict:
    """One-row summary: annual-mean coefficients, noise levels and regression RSS."""
    row = {f"{name}_L0": float(params[name][0]) for name in SEASONAL_PARAM_NAMES}
    row.update(sigma_E=params["sigma_E"], sigma_I=params["sigma_I"],
               rss_alpha_E=diagnostics["alpha_E"]["rss"],
               rss_alpha_I=diagnostics["alpha_I"]["rss"],
               rss_E_D=diagnostics["E_D_equation"]["rss"],
               rss_I_D=diagnostics["I_D_equation"]["rss"])
    return row
