"""Model definition: seasonal parameters and the free-running simulator.

The model has five latent states, ``x = [E_I, E_D, I_I, I_D, h]``:

* ``E_I`` / ``E_D``: independent / IOD-driven parts of the ENSO index E (Nino3.4)
* ``I_I`` / ``I_D``: independent / ENSO-driven parts of the IOD index I (DMI)
* ``h``: equatorial-Pacific thermocline depth (recharge-oscillator memory)

and the observable indices are the *sums* ``E = E_I + E_D`` and ``I = I_I + I_D``.

Every time-dependent coefficient is a low-order Fourier series in the time of
year (``t`` in years)::

    L(t) = L0 + L1c cos(wt) + L1s sin(wt) + L2c cos(2wt) + L2s sin(2wt),  w = 2*pi

A coefficient is stored either as a scalar (constant) or as a length-5 array
``[L0, L1c, L1s, L2c, L2s]``.  A full parameter set is a plain ``dict`` (see
``SEASONAL_PARAM_NAMES`` and ``simulate_free_evolution``).
"""
from __future__ import annotations

import numpy as np

# Coefficients that are fitted with a seasonal cycle (order matters: it is the
# order used when flattening parameters into a vector).
SEASONAL_PARAM_NAMES = (
    "alpha_E",   # damping of E_I
    "a",         # IOD -> ENSO coupling (drives E_D)
    "gamma_E",   # damping of E_D
    "beta_E",    # nonlinear E_I * |I| term in the E_D equation
    "alpha_I",   # damping of I_I
    "c",         # ENSO -> IOD coupling (drives I_D)
    "gamma_I",   # damping of I_D
)
STATE_NAMES = ("E_I", "E_D", "I_I", "I_D", "h")
OMEGA = 2.0 * np.pi


# --------------------------------------------------------------------------
# Seasonal cycle helpers
# --------------------------------------------------------------------------
def seasonal_basis(t, omega: float = OMEGA) -> np.ndarray:
    """Design matrix ``(len(t), 5)`` with columns ``[1, cos wt, sin wt, cos 2wt, sin 2wt]``.

    ``t`` is in years, so one monthly step is ``1/12``.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.column_stack([
        np.ones_like(t),
        np.cos(omega * t),
        np.sin(omega * t),
        np.cos(2.0 * omega * t),
        np.sin(2.0 * omega * t),
    ])


def seasonal_value(theta, t, omega: float = OMEGA):
    """Evaluate a seasonal coefficient at time(s) ``t`` (years).

    ``theta`` is a scalar (constant coefficient) or ``[L0, L1c, L1s, L2c, L2s]``.
    Returns a float for scalar ``t`` and an array for array ``t``.
    """
    theta = np.asarray(theta, dtype=float)
    scalar_t = np.ndim(t) == 0
    if theta.ndim == 0:
        out = np.full(np.shape(t), float(theta))
    else:
        out = seasonal_basis(t, omega) @ theta
    return float(np.ravel(out)[0]) if scalar_t else out


def seasonal_phase(n_steps: int, dt: float = 1.0, start_month: int = 1) -> np.ndarray:
    """Time of year (in years) at each step, for a record starting in ``start_month`` (1-12)."""
    phase0 = (int(start_month) - 1) / 12.0
    return phase0 + np.arange(n_steps, dtype=float) * (dt / 12.0)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
def simulate_free_evolution(
    x0,
    n_steps: int,
    dt: float,
    params: dict,
    *,
    process_noise_scale: float = 0.0,
    seed: int = 0,
    start_month: int = 1,
) -> np.ndarray:
    """Forward-Euler integration of the seasonal model.

    Parameters
    ----------
    x0 : (5,) initial state ``[E_I, E_D, I_I, I_D, h]``.
    n_steps : number of states returned (including ``x0``).
    dt : time step in months.
    params : dict with the seasonal coefficients (``SEASONAL_PARAM_NAMES``), plus
        ``F1``, ``F2`` (thermocline coupling), ``eps_h`` (thermocline damping) and
        the noise amplitudes ``sigma_E``, ``sigma_I``, ``sigma_h``.
    process_noise_scale : multiplies all noise amplitudes; 0 gives a deterministic run.
        Noise is added to ``E_I``, ``I_I`` and ``h`` only.
    seed : seed of the noise generator.
    start_month : calendar month (1-12) of ``x0``; sets the seasonal phase.

    Returns
    -------
    (n_steps, 5) array of states.
    """
    x = np.zeros((n_steps, 5), dtype=float)
    x[0] = np.asarray(x0, dtype=float)
    if not np.isfinite(x[0]).all():
        raise ValueError("x0 contains NaN/inf, so every state of the run would be NaN")
    if n_steps < 2:
        return x

    n_trans = n_steps - 1
    t = seasonal_phase(n_trans, dt, start_month)
    series = {
        name: np.broadcast_to(seasonal_value(params[name], t), (n_trans,)).tolist()
        for name in (*SEASONAL_PARAM_NAMES, "F1", "F2")
    }
    eps_h = float(params["eps_h"])

    # One (n_trans, 3) block of N(0,1) draws -> columns are the E, I, h noises.
    # Drawing in a block gives the same stream as drawing step by step.
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_trans, 3))
    amp = process_noise_scale * np.sqrt(dt) * np.array(
        [params["sigma_E"], params["sigma_I"], params["sigma_h"]], dtype=float
    )
    noise = (z * amp).tolist()

    alpha_E, a, gamma_E = series["alpha_E"], series["a"], series["gamma_E"]
    beta_E, alpha_I = series["beta_E"], series["alpha_I"]
    c, gamma_I = series["c"], series["gamma_I"]
    F1, F2 = series["F1"], series["F2"]

    E_I, E_D, I_I, I_D, h = x[0].tolist()
    for i in range(n_trans):
        nE, nI, nh = noise[i]
        I_sum = I_I + I_D

        dE_I = -alpha_E[i] * E_I + F1[i] * h
        dE_D = a[i] * I_sum - gamma_E[i] * E_D - beta_E[i] * E_I * abs(I_sum)
        dI_I = -alpha_I[i] * I_I
        dI_D = c[i] * (E_I + E_D) - gamma_I[i] * I_D
        dh = -F2[i] * E_I - eps_h * h

        E_I, E_D, I_I, I_D, h = (
            E_I + dt * dE_I + nE,
            E_D + dt * dE_D,
            I_I + dt * dI_I + nI,
            I_D + dt * dI_D,
            h + dt * dh + nh,
        )
        x[i + 1] = (E_I, E_D, I_I, I_D, h)
    return x
