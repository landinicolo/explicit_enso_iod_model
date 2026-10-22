"""Stochastic ensembles: free runs, noise calibration and lead-time hindcasts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import simulate_free_evolution


def _perturb(x0: np.ndarray, jitter_scale: float, rng) -> np.ndarray:
    """Relative jitter on the initial state (scale floored at 1 so near-zero states move too)."""
    if jitter_scale <= 0:
        return x0
    jitter = rng.normal(scale=jitter_scale, size=x0.shape)
    return x0 + jitter * np.maximum(np.abs(x0), 1.0)


# --------------------------------------------------------------------------
# Free-running ensembles
# --------------------------------------------------------------------------
def run_freerun_ensemble(
    params: dict, x0, n_steps: int, dt: float = 1.0, *, n_members: int = 100,
    process_noise_scale: float = 0.05, x0_jitter_scale: float = 0.05,
    seed: int = 0, start_month: int = 1,
) -> np.ndarray:
    """``n_members`` independent noisy runs from (jittered) ``x0``.

    Returns an array of shape ``(n_members, n_steps, 5)``.
    """
    x0 = np.asarray(x0, dtype=float)
    master = np.random.default_rng(seed)
    members = []
    for _ in range(n_members):
        member_seed = int(master.integers(0, 2 ** 32 - 1))
        x0_m = _perturb(x0, x0_jitter_scale, np.random.default_rng(member_seed))
        members.append(simulate_free_evolution(
            x0_m, n_steps, dt, params, process_noise_scale=process_noise_scale,
            seed=member_seed, start_month=start_month))
    return np.asarray(members)


def calibrate_process_noise_scale(
    E_obs, I_obs, x0, params: dict, dt: float = 1.0, *,
    probe_members: int = 20, seed: int = 0, start_month: int = 1,
) -> float:
    """Noise multiplier that makes the ensemble's mean std of (E, I) match the observed one.

    Runs a unit-noise probe ensemble and rescales by (observed std) / (probe std).
    The model is linear in the noise only approximately, so treat the result as a
    first-order calibration.
    """
    E_obs = np.asarray(E_obs, dtype=float)
    I_obs = np.asarray(I_obs, dtype=float)
    target = float(np.nanmean([np.nanstd(E_obs), np.nanstd(I_obs)]))
    if not np.isfinite(target) or target <= 0:
        return 1e-6

    probe = run_freerun_ensemble(
        params, x0, len(E_obs), dt, n_members=probe_members,
        process_noise_scale=1.0, x0_jitter_scale=0.0, seed=seed, start_month=start_month)
    stds = [np.nanmean([np.nanstd(m[:, 0] + m[:, 1]), np.nanstd(m[:, 2] + m[:, 3])])
            for m in probe]
    probe_scale = float(np.nanmean(stds))
    if not np.isfinite(probe_scale) or probe_scale <= 0:
        return 1e-6
    return float(max(target / probe_scale, 1e-6))


# --------------------------------------------------------------------------
# Hindcasts
# --------------------------------------------------------------------------
@dataclass
class Hindcasts:
    """Ensemble-mean hindcasts of E and I from many start dates.

    Attributes
    ----------
    t0 : (n_init,) record index of each start date.
    init_month : (n_init,) calendar month (1-12) of each start date.
    pred, truth : (n_init, max_lead+1, 2) forecast / verifying values; the last axis is
        ``[E, I]``.  ``pred[:, 0]`` is the initial state's own E/I totals.
    """
    t0: np.ndarray
    init_month: np.ndarray
    pred: np.ndarray
    truth: np.ndarray
    var_names: tuple = ("E", "I")

    @property
    def max_lead(self) -> int:
        return self.pred.shape[1] - 1

    def var_index(self, var: str) -> int:
        return self.var_names.index(var.upper())


def run_hindcasts(
    x_init, E_obs, I_obs, params: dict, months, dt: float = 1.0, *,
    max_lead: int = 24, init_stride: int = 1, n_members: int = 20,
    process_noise_scale: float = 0.0, x0_jitter_scale: float = 0.0, seed: int = 123,
) -> Hindcasts:
    """Re-initialise the model from the filtered state at many start months and integrate forward.

    One pass produces both E and I, for every lead 0..``max_lead``, so the same
    object feeds ``skill_by_lead`` and the seasonality heat-maps.  Verification
    uses ``E_obs[t0 + lead]``, hence start dates stop ``max_lead`` before the record end.

    Parameters
    ----------
    x_init : (T, 5) latent states to initialise from (the Kalman estimates).
    months : (T,) calendar month (1-12) of each record step.
    """
    x_init = np.asarray(x_init, dtype=float)
    obs = np.column_stack([np.asarray(E_obs, dtype=float), np.asarray(I_obs, dtype=float)])
    months = np.asarray(months, dtype=int)
    T = len(obs)

    t0s = np.arange(0, T - max_lead, init_stride)
    pred = np.zeros((len(t0s), max_lead + 1, 2))
    master = np.random.default_rng(seed)

    for row, t0 in enumerate(t0s):
        x0 = x_init[t0]
        acc = np.zeros((max_lead + 1, 2))
        for _ in range(n_members):
            member_seed = int(master.integers(0, 2 ** 32 - 1))
            x0_m = _perturb(x0, x0_jitter_scale, np.random.default_rng(member_seed))
            traj = simulate_free_evolution(
                x0_m, max_lead + 1, dt, params, process_noise_scale=process_noise_scale,
                seed=member_seed, start_month=int(months[t0]))
            acc[:, 0] += traj[:, 0] + traj[:, 1]
            acc[:, 1] += traj[:, 2] + traj[:, 3]
        pred[row] = acc / n_members
        pred[row, 0] = (x0[0] + x0[1], x0[2] + x0[3])   # lead 0 = initial state itself

    leads = np.arange(max_lead + 1)
    truth = obs[t0s[:, None] + leads[None, :]]
    return Hindcasts(t0=t0s, init_month=months[t0s], pred=pred, truth=truth)


def persistence_hindcasts(hc: Hindcasts) -> Hindcasts:
    """Baseline with the same start dates: the value at ``t0`` is held fixed."""
    pred = np.repeat(hc.truth[:, :1, :], hc.pred.shape[1], axis=1)
    return Hindcasts(hc.t0, hc.init_month, pred, hc.truth, hc.var_names)


def skill_by_lead(hc: Hindcasts, var: str = "E", stride: int = 1):
    """Correlation and RMSE against lead time (index = lead, in steps).

    ``stride`` keeps only start dates with ``t0 % stride == 0``.
    """
    k = hc.var_index(var)
    keep = hc.t0 % stride == 0
    corr = np.full(hc.max_lead + 1, np.nan)
    rmse = np.full(hc.max_lead + 1, np.nan)
    for lead in range(hc.max_lead + 1):
        p, t = hc.pred[keep, lead, k], hc.truth[keep, lead, k]
        ok = np.isfinite(p) & np.isfinite(t)
        p, t = p[ok], t[ok]
        if len(p) < 3:
            continue
        corr[lead] = np.corrcoef(p, t)[0, 1]
        rmse[lead] = np.sqrt(np.mean((p - t) ** 2))
    return corr, rmse
