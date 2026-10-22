# explicit_enso_iod_model

**Separate ENSO and the Indian Ocean Dipole into four terms (independent and mutually driven parts) with a Kalman filter. Optionally continue to a seasonal model for free runs and hindcasts (Python).**

Given monthly anomalies of the ENSO index **E** (Niño3.4), the IOD index **I** (DMI) and equatorial-Pacific thermocline depth **h**, the package splits each index into a part that evolves on its own and a part that is driven by the other mode:

```
E = E_I + E_D        E_I : ENSO-independent        E_D : ENSO driven by the IOD
I = I_I + I_D        I_I : IOD-independent         I_D : IOD driven by ENSO
```

The decomposition (steps 1–2 below) is the core of the package and can be used on its own. The seasonal fit, ensembles and hindcasts (step 3 onward) are optional extensions that reuse the four latent series.

---

## 1. Quick start: the four-term decomposition

Requires Python ≥ 3.9 with numpy, pandas, scipy and matplotlib.

```bash
pip install -e .          # from the repository root
```

```python
import explicit_enso_iod_model as es

# E, I, h: monthly anomaly series of equal length; time: monthly datetimes
model = es.ExplicitENSOIODModel(E, I, h, time=time).decompose()

df = model.decomposition()     # columns: E, I, E_I, E_D, I_I, I_D, h_norm
```

or, without the class:

```python
x, baseline, kalman_params = es.decompose(E, I, h)   # x: (T, 5) = [E_I, E_D, I_I, I_D, h_norm]
```

* `E_I + E_D` reproduces `E`, and `I_I + I_D` reproduces `I`, up to the filter's observation noise (correlation ≈ 0.99+).
* The first row is the filter's initialisation, not an estimate. `decomposition()` returns it as NaN.
* `h` is standardised (zero mean, unit std) before use and is returned as `h_norm`.
* Plot: `es.plotting.plot_kalman_decomposition(df.index, E, I, model.x_kalman)`.

## 2. How the decomposition works

**Step 1: baseline coefficients (constant, from the observations).**
Damping rates `alpha_E`, `alpha_I` from the lag-1 autocorrelation (`alpha = -ln(rho1)/dt`); the I→E and E→I couplings and the thermocline couplings `F1`, `F2` by one-parameter least squares on the tendencies. (`fit_baseline`)

**Step 2: extended Kalman filter.**
The filter runs a five-state model, `[E_I, E_D, I_I, I_D, h]`, with a forward-Euler step of 1 month and constant coefficients:

```
dE_I/dt = -alpha_E E_I + F1 h                    (+ process noise)
dE_D/dt =  a I - gamma_E E_D - beta_E E_I |I|    (no process noise)
dI_I/dt = -alpha_I I_I                           (+ process noise)
dI_D/dt =  c E - gamma_I I_D                     (no process noise)
dh/dt   = -F2 E_I - eps_h h                      (+ process noise)
```

and is observed through the three sums `E = E_I + E_D`, `I = I_I + I_D`, `h = h`, with small observation noise.

How to read the result:

* `E_D` and `I_D` are the *model-predicted* responses to the other mode: they follow the driven equations and receive no process noise.
* `E_I` and `I_I` absorb whatever the driven parts cannot explain, because the filter's process noise acts on them (and on `h`).
* Hence the split depends on the coupling terms and their strengths. It is a model-based attribution, not a unique or purely statistical separation.

**Tuning knobs (`KalmanConfig`).**
A few filter settings are choices, not estimates. The defaults are those of the original analysis:

| field | default | meaning |
|---|---|---|
| `beta_E` | 0.09 | strength of the nonlinear term `E_I · abs(I)` (the absolute value makes it independent of the IOD sign) |
| `coupling_I_to_E_scale` | 0 | multiplier on the fitted linear I→E coupling `a` (0 = linear coupling off, so `E_D` is driven only by the nonlinear term) |
| `coupling_E_to_I_scale` | 3 | multiplier on the fitted linear E→I coupling `c` |
| `sigma_E`, `sigma_I`, `sigma_h` | 1 | process-noise std of `E_I`, `I_I`, `h` |
| `beta_I` | 0 | optional nonlinear E·I_I term in the `I_D` equation |
| `round_decimals` | 4 | filter parameters are rounded before use (`None` = off) |

```python
cfg = es.KalmanConfig(beta_E=0.045, coupling_I_to_E_scale=1.0, coupling_E_to_I_scale=2.0)
model = es.ExplicitENSOIODModel(E, I, h, time=time).decompose(cfg)
```

Varying these and checking how `E_D` and `I_D` respond is the recommended way to judge how robust the separation is for your data.

## 3. Optional: seasonal model, free runs and hindcasts

`model.fit()` continues from the decomposition:

3. **Seasonal coefficients.** Treating the filtered states as truth, each tendency is regressed on a seasonal basis (ordinary least squares). Every coefficient of `alpha_E, a, gamma_E, beta_E, alpha_I, c, gamma_I` becomes a low-order Fourier series in the time of year (`t` in years):

   ```
   L(t) = L0 + L1c cos(2 pi t) + L1s sin(2 pi t) + L2c cos(4 pi t) + L2s sin(4 pi t)
   ```

   `F1`, `F2`, `eps_h` stay constant. (`fit_seasonal_direct`)
4. **Noise calibration.** A multiplier on the process noise (`noise_scale`) is chosen so that free-run ensembles reproduce the observed mean std of E and I.

Then:

```python
model.fit()
ens = model.freerun_ensemble(n_members=100)        # (members, time, 5), any length
hc  = model.hindcast(max_lead=24)                  # lagged-initialisation hindcasts
corr, rmse = es.skill_by_lead(hc, "E")
```

Diagnostics provided: power spectra, per-year peak-month histograms and December locking score, mean absolute amplitude by calendar month, ensemble std and lag-1 autocorrelation vs. observations, skill vs. lead time against persistence, and skill by initial month × target month with Bretherton effective-sample-size significance.

## 4. Maps

The example notebook (section 8) also draws **regression maps** of SST on the standardised latent parts `E_I, E_D, I_I, I_D`:
`climate_base.regression_maps(field, {...})` then `plotting.plot_regression_maps(reg_maps, sigmas)`. It needs the gridded SST (`DATA_SOURCE = "oras"`), `xarray` and `cartopy` (`pip install -e ".[maps]"`); otherwise the cell is skipped.

`plotting.plot_std_phase_space` (standard deviations of Nino3.4 and DMI in control / noENSO / noIOD experiments, next to the observations and the Kalman-decomposed parts) is included as a function, but no example or model output is distributed with it.

## 5. Repository layout

| path | contents |
|---|---|
| `explicit_enso_iod_model/estimation.py` | **decomposition**: `fit_baseline`, `KalmanConfig`, `kalman_filter_abs`, `decompose`; plus the seasonal regression |
| `explicit_enso_iod_model/model.py` | `ExplicitENSOIODModel`: `decompose()`, `decomposition()`, `fit()`, `freerun_ensemble()`, `hindcast()` |
| `explicit_enso_iod_model/core.py` | seasonal basis and `simulate_free_evolution` |
| `explicit_enso_iod_model/ensembles.py` | free-run ensembles, noise calibration, hindcasts, skill |
| `explicit_enso_iod_model/diagnostics.py` | spectra, peak months, amplitudes, significance, skill matrix |
| `explicit_enso_iod_model/plotting.py` | one function per figure (returns `fig, ax`), including maps and the STD phase-space plot |
| `explicit_enso_iod_model/data.py` | read/write the `time,E,I,h` CSV |
| `explicit_enso_iod_model/indexes.py`, `climate_base.py` | Nino3.4 / DMI / thermocline boxes, anomalies, detrending, regression maps (need `xarray`) |
| `explicit_enso_iod_model/synthetic.py` | synthetic demo data |
| `notebooks/explicit_enso_iod_model_example.ipynb` | end-to-end example (saved figures from ORAS5; synthetic fallback when no data are present) |
| `scripts/make_indices_csv.py` | build the index CSV from your own ORAS file |
| `data/` | where input files go (see `data/README.md`) |
| `tests/` | unit and workflow tests |

## 6. Data

**No third-party data are included.** Everything runs out of the box on synthetic data. To reproduce the analysis you need the monthly series below; `data/README.md` explains how to obtain and place them.

The observations used in the paper were built from the ORAS5 reanalysis (1958–2023, monthly, low resolution): Niño3.4 and DMI computed from SST anomalies, and equatorial-Pacific thermocline depth from the depth of the 20 °C isotherm. All fields are anomalies from the monthly climatology with a quadratic trend removed.

In the notebook, `DATA_SOURCE = "auto"` (default) picks the first source that exists (oras, then csv) and otherwise falls back to synthetic data. You can force one:

* `"synthetic"`: no data needed;
* `"csv"`: a `time,E,I,h` file at `CSV_PATH`;
* `"oras"`: your own ORAS file at `ORAS_PATH` (variables `sosstsst`, `so20chgt`); also enables the regression maps.

Any other monthly E, I and h anomaly series can be used the same way.

## 7. Limitations and caveats

* **The split is model-dependent** (see section 2). It depends on the tuned `KalmanConfig` values, and in particular with the default linear I→E coupling switched off, `E_D` reflects only the nonlinear response.
* **Start state.** The first row of the filter output is its zero initialisation. Free runs and the first hindcast start from it, so initial transients are possible.
* **Seasonal fit (optional part).** Each seasonal coefficient has five parameters estimated from a short record, with no uncertainty ranges provided. The `E_I` regression omits the `F1 h` term that the simulator includes, so the fitted `alpha_E` may partly absorb the thermocline feedback. `sigma_h` is set equal to `sigma_E`, while the filter uses `sigma_h = 1` for the standardised `h`.
* **Hindcasts are in-sample**: fitted and verified on the same record, so skill is optimistic. `h` is not verified.
* Forward-Euler integration with a 1-month step is meant for seasonal-to-interannual statistics, not sub-monthly dynamics.
* **Nino3.4 box.** `indexes.N34` averages 6°S–6°N (as in the original analysis), not the conventional 5°S–5°N; change `lat_halfwidth` to compare.

## 8. Tests

```bash
pip install -e ".[test]"
pytest
```

(The map test is skipped unless `cartopy` and `xarray` are installed.)

## 9. Citing

If you use this code, please cite the Zenodo record (see `CITATION.cff`):

> Landi, N. (2026). *explicit_enso_iod_model* (version 0.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23558633

(Replace the year, version and DOI after publishing, and add the associated paper if there is one.)

## 10. License

MIT, see `LICENSE`.
# explicit_enso_iod_model
