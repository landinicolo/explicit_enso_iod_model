"""Smoke and correctness tests.  Run with ``pytest`` (or ``python tests/test_basic.py``)."""
import numpy as np

import explicit_enso_iod_model as es
from explicit_enso_iod_model import diagnostics as dg

TRUE = dict(
    alpha_E=[0.045, 0.010, 0.012, 0.0, 0.0], a=[0.010, 0.002, 0.0, 0.0, 0.0], gamma_E=0.05,
    beta_E=0.09, alpha_I=[0.20, 0.03, -0.02, 0.0, 0.0], c=[0.06, 0.01, 0.003, 0.0, 0.0],
    gamma_I=[0.20, 0.0, 0.01, 0.0, 0.0], F1=0.11, F2=0.18, eps_h=0.037,
    sigma_E=0.22, sigma_I=0.22, sigma_h=0.3,
)


def test_seasonal_value_scalar_and_vector():
    assert es.seasonal_value(0.3, 0.7) == 0.3
    theta = [1.0, 0.5, 0.0, 0.0, 0.0]
    assert np.isclose(es.seasonal_value(theta, 0.0), 1.5)
    t = np.linspace(0, 1, 7)
    assert es.seasonal_value(theta, t).shape == (7,)


def test_simulation_is_deterministic_and_seed_controlled():
    x0 = np.array([1.0, 0, 0.2, 0, 0.5])
    a = es.simulate_free_evolution(x0, 100, 1.0, TRUE, process_noise_scale=1.0, seed=1)
    b = es.simulate_free_evolution(x0, 100, 1.0, TRUE, process_noise_scale=1.0, seed=1)
    c = es.simulate_free_evolution(x0, 100, 1.0, TRUE, process_noise_scale=1.0, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    z = es.simulate_free_evolution(x0, 100, 1.0, TRUE, process_noise_scale=0.0, seed=1)
    z2 = es.simulate_free_evolution(x0, 100, 1.0, TRUE, process_noise_scale=0.0, seed=9)
    assert np.array_equal(z, z2)


def _as_vec5(v):
    return np.asarray(v, float) if np.ndim(v) else np.array([v, 0, 0, 0, 0], float)


def test_direct_fit_recovers_noise_free_equations():
    """The E_D and I_D tendencies contain no noise, so the regression is exact."""
    x = es.simulate_free_evolution(np.zeros(5), 3000, 1.0, TRUE, process_noise_scale=1.0, seed=4,
                                   start_month=3)
    p, _ = es.fit_seasonal_direct(x, 1.0, start_month=3)
    for name in ("a", "gamma_E", "beta_E", "c", "gamma_I"):
        assert np.allclose(p[name], _as_vec5(TRUE[name]), atol=1e-6), name


def test_full_workflow_on_synthetic_data():
    d = es.synthetic.make_synthetic_obs(n_years=40, seed=1)
    m = es.ExplicitENSOIODModel(d["E"], d["I"], d["h"], time=d["time"]).fit()
    assert m.x_kalman.shape == (480, 5) and np.isfinite(m.x_kalman).all()
    assert m.noise_scale > 0
    ens = m.freerun_ensemble(n_members=5, n_steps=200)
    assert ens.shape == (5, 200, 5)
    hc = m.hindcast(max_lead=6, n_members=2)
    assert hc.pred.shape == (480 - 6, 7, 2)
    corr, rmse = es.skill_by_lead(hc, "E")
    assert np.isclose(corr[0], 1.0, atol=0.05)            # lead 0 ~ identity
    pers = es.persistence_hindcasts(hc)
    c_p, _ = es.skill_by_lead(pers, "E")
    assert np.isclose(c_p[0], 1.0)


def test_decompose_only_matches_fit_and_sums_to_obs():
    d = es.synthetic.make_synthetic_obs(n_years=20, seed=5)
    m = es.ExplicitENSOIODModel(d["E"], d["I"], d["h"], time=d["time"]).decompose()
    assert m.params is None and m.x_kalman.shape == (240, 5)
    df = m.decomposition()
    assert list(df.columns) == ["E", "I", "E_I", "E_D", "I_I", "I_D", "h_norm"]
    assert df.iloc[0, 2:].isna().all()
    ok = df.iloc[1:]
    assert np.corrcoef(ok["E"], ok["E_I"] + ok["E_D"])[0, 1] > 0.99
    assert np.corrcoef(ok["I"], ok["I_I"] + ok["I_D"])[0, 1] > 0.99
    x, _, _ = es.decompose(d["E"], d["I"], d["h"])
    assert np.array_equal(x, m.x_kalman)
    assert np.array_equal(es.ExplicitENSOIODModel(d["E"], d["I"], d["h"], time=d["time"]).fit().x_kalman, x)


def test_hindcast_start_dates_stop_before_record_end():
    d = es.synthetic.make_synthetic_obs(n_years=10, seed=2)
    m = es.ExplicitENSOIODModel(d["E"], d["I"], d["h"], time=d["time"]).fit()
    hc = m.hindcast(max_lead=12, n_members=1)
    assert hc.t0.max() + 12 < len(d["E"])
    assert np.isfinite(hc.truth).all()


def test_bretherton_sample_size_bounds():
    rng = np.random.default_rng(0)
    white = rng.normal(size=500)
    n_eff = dg.effective_sample_size_bretherton(white, rng.normal(size=500))
    assert 400 < n_eff <= 500
    red = np.cumsum(rng.normal(size=500))
    assert dg.effective_sample_size_bretherton(red, red) < 100


def test_peak_months_per_year():
    months = np.tile(np.arange(1, 13), 3)
    years = np.repeat([2000, 2001, 2002], 12)
    s = np.zeros(36)
    s[[11, 12 + 5, 24 + 11]] = [2.0, -3.0, 1.0]            # Dec, Jun, Dec
    assert dg.peak_months_per_year(s, years, months, 0.5) == [11, 5, 11]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
