"""Numerical diagnostics (no plotting): spectra, peak months, amplitudes, skill matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .ensembles import Hindcasts

APR_CYCLE = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def state_total(x, var: str = "E") -> np.ndarray:
    """E = E_I + E_D or I = I_I + I_D from states on the last axis (``(..., 5)``)."""
    x = np.asarray(x, dtype=float)
    if var.upper() == "E":
        return x[..., 0] + x[..., 1]
    if var.upper() == "I":
        return x[..., 2] + x[..., 3]
    raise ValueError("var must be 'E' or 'I'")


def lag1_autocorr(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.std(x) == 0:
        return np.nan
    x = x - x.mean()
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fit_metrics(E_obs, I_obs, x_model, x_truth=None) -> dict:
    """RMSE/correlation of the model E, I totals vs observations (+ latent RMSE vs ``x_truth``)."""
    E_obs, I_obs = np.asarray(E_obs, float), np.asarray(I_obs, float)
    E_m, I_m = state_total(x_model, "E"), state_total(x_model, "I")
    out = {"E_rmse": _rmse(E_obs, E_m), "I_rmse": _rmse(I_obs, I_m),
           "E_corr": float(np.corrcoef(E_obs, E_m)[0, 1]),
           "I_corr": float(np.corrcoef(I_obs, I_m)[0, 1])}
    if x_truth is not None:
        for k, name in enumerate(("E_I", "E_D", "I_I", "I_D")):
            out[f"{name}_rmse"] = _rmse(np.asarray(x_truth)[:, k], np.asarray(x_model)[:, k])
    return out


# --------------------------------------------------------------------------
# Ensemble summary statistics
# --------------------------------------------------------------------------
def ensemble_mean_std(ens, names=("E_I", "E_D", "I_I", "I_D", "h")) -> dict:
    """Std over time, averaged over members, for each state and for the E/I totals."""
    ens = np.asarray(ens, dtype=float)             # (members, T, 5)
    out = {n: float(np.nanmean(np.nanstd(ens[..., k], axis=1))) for k, n in enumerate(names)}
    out["E_total"] = float(np.nanmean(np.nanstd(state_total(ens, "E"), axis=1)))
    out["I_total"] = float(np.nanmean(np.nanstd(state_total(ens, "I"), axis=1)))
    return out


def std_autocorr_table(obs, ens, var: str = "E", label: str = "model") -> pd.DataFrame:
    """Observed vs ensemble std and lag-1 autocorrelation (one-row DataFrame)."""
    obs = np.asarray(obs, dtype=float)
    obs = obs[np.isfinite(obs)]
    series = state_total(ens, var)                 # (members, T)
    stds = np.nanstd(series, axis=1)
    acfs = np.array([lag1_autocorr(s) for s in series])
    return pd.DataFrame([{
        "label": label, "var": var.upper(),
        "obs_std": float(np.nanstd(obs)), "model_std_mean": float(np.nanmean(stds)),
        "model_std_std": float(np.nanstd(stds)),
        "obs_acf1": lag1_autocorr(obs), "model_acf1_mean": float(np.nanmean(acfs)),
        "model_acf1_std": float(np.nanstd(acfs)),
    }])


def power_spectrum(y, dt: float = 1.0):
    """Periodogram ``(freqs, power)`` without the zero frequency."""
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    freqs = np.fft.rfftfreq(len(y), d=dt)
    power = np.abs(np.fft.rfft(y)) ** 2 / len(y)
    return freqs[1:], power[1:]


def spectrum_compare(obs, ens, var: str = "E", dt: float = 1.0) -> dict:
    """Observed periodogram vs ensemble-mean periodogram, on a period axis (months)."""
    f_obs, p_obs = power_spectrum(obs, dt)
    series = state_total(ens, var)
    f_mod, _ = power_spectrum(series[0], dt)
    good = [s for s in series if np.isfinite(s).all()]
    p_mod = (np.mean([power_spectrum(s, dt)[1] for s in good], axis=0)
             if good else np.full_like(f_mod, np.nan))
    return {"period_obs": 1 / f_obs, "power_obs": p_obs,
            "period_model": 1 / f_mod, "power_model": p_mod}


# --------------------------------------------------------------------------
# Health checks: run these first when a model curve is missing from a plot
# --------------------------------------------------------------------------
def ensemble_health(ens, explode: float = 20.0) -> dict:
    """Quick sanity report for an ensemble ``(members, T, 5)``.

    ``n_nonfinite``: members containing NaN/inf (their spectra and peaks are unusable).
    ``n_exploded``: finite members whose |E| or |I| ever exceeds ``explode`` (far beyond ENSO/IOD).
    ``n_flat``: members whose E total never leaves ~0 (noise scale of 0 / failed calibration).
    """
    ens = np.asarray(ens, dtype=float)
    finite = np.isfinite(ens).all(axis=(1, 2))
    E, I = state_total(ens, "E"), state_total(ens, "I")
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(np.abs(np.concatenate([E, I], axis=1)), axis=1)
    return {"n_members": int(ens.shape[0]), "n_nonfinite": int((~finite).sum()),
            "n_exploded": int((finite & (peak > explode)).sum()),
            "n_flat": int((finite & (np.nanmax(np.abs(E), axis=1) < 1e-3)).sum()),
            "max_abs_E_or_I": float(np.nanmax(peak)) if np.isfinite(peak).any() else float("nan")}


def parameter_health(params: dict) -> dict:
    """Over the 12 months: smallest value of each damping coefficient, and 100-year growth.

    A negative damping (``alpha_E``, ``alpha_I``, ``gamma_E``, ``gamma_I``) in some month is
    allowed only if the annual mean is positive, but large negative values can make the
    noise-free model grow.  ``growth_100yr`` is |x| after 100 years of a deterministic run
    from a unit state divided by its start (>> 1 means unstable, ~0 means decaying).
    """
    from .core import seasonal_value, simulate_free_evolution
    t = np.arange(12) / 12.0
    out = {f"min_{k}": float(np.min(seasonal_value(params[k], t)))
           for k in ("alpha_E", "alpha_I", "gamma_E", "gamma_I")}
    x = simulate_free_evolution(np.array([1.0, 0, 0.5, 0, 0.5]), 1200, 1.0, params)
    out["growth_100yr"] = float(np.linalg.norm(x[-1]) / np.linalg.norm(x[0]))
    out["finite"] = bool(np.isfinite(x).all())
    return out


# --------------------------------------------------------------------------
# Peak month / seasonal amplitude
# --------------------------------------------------------------------------
def peak_months_per_year(series, years, months, min_abs_peak: float = 0.5) -> list:
    """Month (0-11) of the largest |value| in each calendar year with a big enough peak.

    One sample per year makes an observed record comparable to an ensemble with
    one peak per member.
    """
    series = np.asarray(series, dtype=float)
    years, months = np.asarray(years), np.asarray(months)
    out = []
    for yr in np.unique(years):
        m = years == yr
        vals = np.abs(series[m])
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        idx = int(np.argmax(np.where(finite, vals, -np.inf)))
        if vals[idx] >= min_abs_peak:
            out.append(int(months[m][idx]) - 1)
    return out


def peak_months_ensemble(ens, var: str = "E", min_abs_peak: float = 0.5,
                         start_month: int = 1) -> list:
    """Month (0-11) of the global |peak| of each member's whole run.

    Members with no finite values, or whose peak is below ``min_abs_peak``, are skipped.
    """
    out = []
    for s in state_total(ens, var):
        a = np.where(np.isfinite(s), np.abs(s), -np.inf)
        if not np.isfinite(a.max()):
            continue
        idx = int(np.argmax(a))
        if a[idx] >= min_abs_peak:
            out.append((start_month - 1 + idx) % 12)
    return out


def month_locking_score(months, target_month: int = 11) -> float:
    """Fraction of peaks falling in ``target_month`` (0-based; 11 = December)."""
    months = np.asarray(months, dtype=int)
    return float("nan") if months.size == 0 else float(np.mean(months == target_month))


def monthly_abs_amplitude(values, months) -> np.ndarray:
    """Mean |value| for each calendar month (length-12 array, Jan..Dec)."""
    values, months = np.asarray(values, dtype=float), np.asarray(months, dtype=int)
    return np.array([np.nanmean(np.abs(values[months == m])) if np.any(months == m) else np.nan
                     for m in range(1, 13)])


def ensemble_monthly_abs_amplitude(ens, months, var: str = "E"):
    """Ensemble mean and std (across members) of ``monthly_abs_amplitude``."""
    curves = np.array([monthly_abs_amplitude(s, months) for s in state_total(ens, var)])
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)


# --------------------------------------------------------------------------
# Skill seasonality (initial month x target month)
# --------------------------------------------------------------------------
def effective_sample_size_bretherton(pred, true) -> float:
    """Effective sample size of a correlation between autocorrelated series (Bretherton et al. 1999)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[ok], true[ok]
    n = len(pred)
    if n < 3:
        return np.nan
    r1, r2 = lag1_autocorr(pred), lag1_autocorr(true)
    if not (np.isfinite(r1) and np.isfinite(r2)) or 1 + r1 * r2 <= 0:
        return np.nan
    return float(np.clip(n * (1 - r1 * r2) / (1 + r1 * r2), 1.0, n))


def seasonality_skill_matrix(hc: Hindcasts, var: str = "E", *, max_lead: int | None = None,
                             alpha: float = 0.05, min_n: int = 5):
    """Correlation skill by (initial month, target month) with significance.

    Rows are initial months ordered Apr..Mar (``APR_CYCLE``); column ``c`` of row ``r``
    is the target ``r + lead - 1`` months after April.  Significance uses a two-sided
    t-test on the correlation with the Bretherton effective sample size.

    Returns ``skill`` (12, 11+max_lead) and boolean ``signif`` of the same shape.
    """
    max_lead = hc.max_lead if max_lead is None else max_lead
    k = hc.var_index(var)
    skill = np.full((12, 11 + max_lead), np.nan)
    signif = np.zeros_like(skill, dtype=bool)

    for init_month in range(1, 13):
        row = (init_month - 4) % 12
        sel = hc.init_month == init_month
        for lead in range(1, max_lead + 1):
            p, t = hc.pred[sel, lead, k], hc.truth[sel, lead, k]
            ok = np.isfinite(p) & np.isfinite(t)
            p, t = p[ok], t[ok]
            if len(p) < min_n:
                continue
            r = np.corrcoef(p, t)[0, 1]
            if not np.isfinite(r):
                continue
            col = row + lead - 1
            skill[row, col] = r
            if abs(r) >= 1:
                continue
            n_eff = effective_sample_size_bretherton(p, t)
            if not np.isfinite(n_eff) or n_eff <= 2:
                continue
            tstat = r * np.sqrt((n_eff - 2) / (1 - r ** 2))
            signif[row, col] = 2 * stats.t.sf(abs(tstat), df=n_eff - 2) < alpha
    return skill, signif


def seasonality_xticks(max_lead: int = 24, every: int = 3):
    """Tick positions/labels (``Mon$^{year}$``) for the heat-map's target-month axis."""
    cols, labels = [], []
    for col in range(11 + max_lead):
        if col % every == 0:
            i = col + 1
            cols.append(col)
            labels.append(f"{APR_CYCLE[i % 12]}$^{{{(i + 3) // 12}}}$")
    return cols, labels
