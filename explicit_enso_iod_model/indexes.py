"""Climate indices from gridded anomalies (xarray DataArrays with ``lat``/``lon`` coordinates).

Each function returns a 1-D (time) DataArray.  Optional ``pp="d"`` / ``"z"`` / ``"dz"``
detrends linearly and/or standardises the result.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _to_180(xarr):
    """Longitudes 0-360 -> -180..180 (sorted), if needed."""
    if xarr.lon.max() > 180:
        xarr = xarr.assign_coords(lon=(((xarr.lon + 180) % 360) - 180)).sortby("lon")
    return xarr


def _box_mean(xarr, lat_min, lat_max, lon_min, lon_max):
    """Area-unweighted mean over a lat/lon box (lat ascending or descending)."""
    box = xarr.where(
        (xarr.lat >= lat_min) & (xarr.lat <= lat_max) &
        (xarr.lon >= lon_min) & (xarr.lon <= lon_max), drop=True)
    return box.mean(dim=("lat", "lon"), skipna=True)


def _postprocess(index, pp):
    if pp:
        if "d" in pp:
            index.values = signal.detrend(index.values)
        if "z" in pp:
            index.values = index.values / np.nanstd(index.values)
    return index


def N34(xarr, lat_halfwidth: float = 6.0, pp=None):
    """Nino3.4 box: ``lat_halfwidth`` deg S-N (default 6, as in the original analysis;
    the conventional box is 5S-5N), 170W-120W."""
    xarr = _to_180(xarr)
    return _postprocess(_box_mean(xarr, -lat_halfwidth, lat_halfwidth, -170, -120), pp)


def DMI_NOAA(xarr, pp=None):
    """Dipole Mode Index, NOAA definition: West (50-70E, 10S-10N) minus East (90-110E, 10S-0)."""
    xarr = _to_180(xarr)
    west = _box_mean(xarr, -10, 10, 50, 70)
    east = _box_mean(xarr, -10, 0, 90, 110)
    return _postprocess(west - east, pp)


def EqPac(xarr, pp=None):
    """Equatorial-Pacific mean (120E-80W, 5S-5N); used for the thermocline depth anomaly."""
    lon360 = (xarr.lon + 360) % 360
    box = xarr.where((xarr.lat >= -5) & (xarr.lat <= 5) &
                     (lon360 >= 120) & (lon360 <= 280), drop=True)
    return _postprocess(box.mean(dim=("lat", "lon"), skipna=True), pp)
