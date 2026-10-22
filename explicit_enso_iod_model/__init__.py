"""Seasonal recharge-oscillator model of coupled ENSO / IOD variability.

Typical use::

    from explicit_enso_iod_model import ExplicitENSOIODModel
    model = ExplicitENSOIODModel(E, I, h, time=time).fit()
    ens = model.freerun_ensemble(n_members=100)

The building blocks (``core``, ``estimation``, ``ensembles``, ``diagnostics``,
``plotting``) are plain functions that can be used without the class.
"""
from .core import (SEASONAL_PARAM_NAMES, STATE_NAMES, seasonal_basis, seasonal_phase,
                   seasonal_value, simulate_free_evolution)
from .estimation import (KalmanConfig, attach_recharge_terms, decompose, fit_baseline,
                         fit_seasonal_direct, kalman_filter_abs, summarize_direct_fit)
from .ensembles import (Hindcasts, calibrate_process_noise_scale, persistence_hindcasts,
                        run_freerun_ensemble, run_hindcasts, skill_by_lead)
from .model import ExplicitENSOIODModel
from .data import load_indices_csv, save_indices_csv
from .plotting import kalman_stds_from_segments
from . import diagnostics, plotting, synthetic

__all__ = [n for n in dir() if not n.startswith("_")]
