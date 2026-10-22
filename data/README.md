# Input data

No third-party data are redistributed with this package. The model needs three monthly anomaly series,
`E` (Nino3.4), `I` (DMI) and `h` (equatorial-Pacific thermocline depth), all detrended.

## Option A: CSV (simplest)

Create `data/indices_oras5_1958_2023.csv` with a header and one row per month:

```
time,E,I,h
1958-01-16,0.31,-0.05,1.8
...
```

then set `DATA_SOURCE = "csv"` in the notebook (or call `explicit_enso_iod_model.load_indices_csv`).
Gaps are not allowed; the time axis must be strictly monthly.

## Option B: build the CSV from ORAS5

1. Get the ORAS5 monthly fields *sea surface temperature* and *depth of the 20 °C isotherm* from the
   Copernicus Climate Data Store (https://cds.climate.copernicus.eu), regrid/merge them into one NetCDF file
   with variables `sosstsst` and `so20chgt`, coordinates `lat`, `lon`, `time_counter`.
2. Run (needs `pip install -e ".[notebook]"`)

   ```bash
   python scripts/make_indices_csv.py path/to/ORAS_LowRes.nc data/indices_oras5_1958_2023.csv
   ```

   Anomalies from the monthly climatology, quadratic detrend, 1958-01 to 2023-12.

Or set `DATA_SOURCE = "oras"` and `ORAS_PATH` in the notebook to use the NetCDF directly (also needed for the maps).
