"""Tests for the CSV loader, regression helpers and the extra figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import explicit_enso_iod_model as es
from explicit_enso_iod_model import climate_base as cb


def _skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        print("  skipped:", reason)


def test_csv_roundtrip(tmp_path=None):
    import tempfile, os
    d = es.synthetic.make_synthetic_obs(n_years=5, seed=3)
    path = os.path.join(tempfile.mkdtemp(), "idx.csv")
    es.save_indices_csv(path, d["time"], d["E"], d["I"], d["h"])
    r = es.load_indices_csv(path)
    assert len(r["E"]) == 60
    assert np.allclose(r["E"], d["E"], atol=1e-5) and np.allclose(r["h"], d["h"], atol=1e-4)
    assert (r["time"].to_period("M") == d["time"].to_period("M")).all()


def test_csv_rejects_gaps_and_missing_columns():
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "bad.csv")
    pd.DataFrame({"time": ["2000-01-15", "2000-03-15"], "E": [0, 1], "I": [0, 1], "h": [0, 1]}).to_csv(path, index=False)
    try:
        es.load_indices_csv(path)
    except ValueError as e:
        assert "monthly" in str(e)
    else:
        raise AssertionError("gap not detected")
    pd.DataFrame({"time": ["2000-01-15"], "E": [0]}).to_csv(path, index=False)
    try:
        es.load_indices_csv(path)
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("missing column not detected")


def test_npregress_recovers_slope_per_sigma():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000) * 2.0
    fld = np.zeros((2000, 2, 3))
    fld[:, 0, 0] = 0.5 * x                   # slope 0.5 per unit -> 1.0 per sigma(=2)
    fld[:, 1, 2] = -1.5 * x + rng.normal(size=2000) * 0.1
    m = cb.npregress(fld, x)
    assert np.isclose(m[0, 0], 1.0, atol=0.03) and np.isclose(m[1, 2], -3.0, atol=0.05)
    assert np.isclose(cb.npregress(fld, x, per_std=False)[0, 0], 0.5, atol=0.01)


def test_std_phase_space_runs():
    d = es.synthetic.make_synthetic_obs(n_years=30, seed=0)
    segs = []
    for s in range(3):
        dd = es.synthetic.make_synthetic_obs(n_years=30, seed=10 + s)
        segs.append(es.ExplicitENSOIODModel(dd["E"], dd["I"], dd["h"], time=dd["time"]).decompose().x_kalman)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.uniform(0.2, 1.0, 3) for c in
                       ("E_ctrl", "I_ctrl", "E_noenso", "I_noenso", "E_noiod", "I_noiod")})
    fig, ax, summ = es.plotting.plot_std_phase_space(df, d["E"], d["I"], x_est_segments=segs)
    assert len(summ["kalman_stds"]["E_I"]) == 3 and np.isfinite(summ["std_E_obs"])
    plt.close(fig)


def test_regression_map_plot():
    try:
        import cartopy  # noqa: F401
        import xarray as xr
    except ImportError:
        return _skip("cartopy/xarray not installed")
    rng = np.random.default_rng(0)
    lat, lon = np.arange(-30, 31, 5.0), np.arange(30, 291, 5.0)
    fld = xr.DataArray(rng.normal(size=(120, lat.size, lon.size)), dims=("time", "lat", "lon"),
                       coords=dict(time=pd.date_range("2000-01-01", periods=120, freq="MS"), lat=lat, lon=lon))
    idxs = {k: rng.normal(size=120) for k in ("E_I", "E_D", "I_I", "I_D")}
    maps, sig = cb.regression_maps(fld, idxs)
    fig, axes = es.plotting.plot_regression_maps(maps, sig)
    assert axes.shape == (2, 2)
    plt.close(fig)


def test_decomposition_does_not_modify_x_kalman():
    """On pandas < 3 (no copy-on-write) the DataFrame used to alias x_kalman, so calling
    decomposition() put NaN into x_kalman[0] and every free run started from NaN."""
    d = es.synthetic.make_synthetic_obs(n_years=10, seed=0)
    m = es.ExplicitENSOIODModel(d["E"], d["I"], d["h"], time=d["time"]).fit()
    before = m.x_kalman.copy()
    df = m.decomposition()
    assert np.isnan(df.iloc[0, 2:]).all()
    assert np.array_equal(m.x_kalman, before) and np.isfinite(m.x_kalman).all()
    assert np.isfinite(m.freerun_ensemble(n_members=2, n_steps=24)).all()
    try:
        es.simulate_free_evolution(np.full(5, np.nan), 10, 1.0, m.params)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN x0 not rejected")


def test_health_checks_and_nan_robustness():
    from explicit_enso_iod_model import diagnostics as dg
    d = es.synthetic.make_synthetic_obs(n_years=30, seed=0)
    m = es.ExplicitENSOIODModel(d["E"], d["I"], d["h"], time=d["time"]).fit()
    ens = m.freerun_ensemble(n_members=6, n_steps=240)
    h = dg.ensemble_health(ens)
    assert h["n_nonfinite"] == 0 and h["n_exploded"] == 0 and h["n_flat"] == 0
    assert dg.parameter_health(m.params)["finite"]
    bad = ens.copy()
    bad[0, 100:] = np.nan                      # one member goes non-finite
    assert dg.ensemble_health(bad)["n_nonfinite"] == 1
    assert len(dg.peak_months_ensemble(bad, "E", 0.01)) == 6    # still counted, month unshifted
    spec = dg.spectrum_compare(d["E"][:240], bad, "E")
    assert np.isfinite(spec["power_model"]).all()               # NaN member excluded
    allbad = np.full_like(ens, np.nan)
    assert dg.peak_months_ensemble(allbad, "E", 0.01) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
