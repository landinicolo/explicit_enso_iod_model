"""Gridded-data helpers used by the example notebook (need ``xarray``).

Only what the workflow uses is kept: monthly anomalies, polynomial detrending and
regression maps.  ``xarray`` is imported lazily so the core package does not need it.
"""
from __future__ import annotations

import numpy as np


def detrend_dim(da, dim: str, deg: int = 1):
    """Remove a polynomial trend of degree ``deg`` along ``dim`` (xarray DataArray)."""
    import xarray as xr
    p = da.polyfit(dim=dim, deg=deg)
    return da - xr.polyval(da[dim], p.polyfit_coefficients)


def anom(xarr):
    """Monthly-climatology anomalies (trend NOT removed). Returns ``(anomalies, climatology)``."""
    clm = xarr.groupby("time.month").mean("time")
    return xarr.groupby("time.month") - clm, clm


def npcovariance(fld_tyx, idx_t):
    """Covariance map between a field ``(T, Y, X)`` and a time series ``(T,)``."""
    fld = np.asarray(fld_tyx, dtype=float)
    fld = fld - np.nanmean(fld, 0)
    idx = np.asarray(idx_t, dtype=float)
    idx = idx - np.nanmean(idx)
    return np.einsum("kij,k->ij", np.nan_to_num(fld), idx) / len(idx)


def npregress(fld_tyx, idx_t, per_std: bool = True):
    """Regression map of a field on an index.

    ``per_std=True``  -> field units per 1 standard deviation of the index (cov / std).
    ``per_std=False`` -> field units per unit of the index (cov / var).
    """
    idx = np.asarray(idx_t, dtype=float)
    cov = npcovariance(fld_tyx, idx)
    return cov / np.nanstd(idx) if per_std else cov / np.nanvar(idx)


def regression_maps(field, indices: dict):
    """Regress an xarray field ``(time, lat, lon)`` on several indices, per 1 sigma.

    Parameters
    ----------
    field : xarray DataArray, anomalies, dims ``(time, lat, lon)``.
    indices : name -> 1-D series with the same length as ``field.time``
        (e.g. ``{"E_I": x[:, 0], "E_D": x[:, 1], "I_I": x[:, 2], "I_D": x[:, 3]}``).

    Returns
    -------
    reg_maps_std : name -> DataArray (lat, lon), units of ``field`` per 1 sigma of the index.
    sigmas : name -> standard deviation of each index.
    """
    reg_maps, sigmas = {}, {}
    template = field.isel(time=0, drop=True)
    for name, series in indices.items():
        series = np.asarray(series, dtype=float)
        reg_maps[name] = template.copy(data=npregress(field.values, series))
        sigmas[name] = float(np.nanstd(series))
    return reg_maps, sigmas
