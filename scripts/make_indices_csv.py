"""Build data/indices_oras5_1958_2023.csv from your own ORAS file (needs xarray + netCDF4).

Run from anywhere, e.g. from the project root:

    python scripts/make_indices_csv.py path/to/ORAS_LowRes.nc data/indices_oras5_1958_2023.csv

The file must contain monthly fields ``sosstsst`` (SST) and ``so20chgt`` (depth of the 20 C
isotherm) on ``lat``/``lon`` with a ``time_counter`` axis.  Anomalies are taken from the
monthly climatology and a quadratic trend is removed, as in the paper.
"""
import os
import sys

# Work without `pip install -e .`: put the project root (parent of scripts/) on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(nc_path, out_csv, start="1958-01", end="2023-12"):
    import xarray as xr

    from explicit_enso_iod_model import climate_base as cb, indexes as idx
    from explicit_enso_iod_model.data import save_indices_csv

    ds = xr.open_dataset(nc_path)
    ds = ds.assign_coords(time=("time_counter", ds.time_counter.data))
    ds = ds.swap_dims({"time_counter": "time"}).drop_vars("time_counter")
    ano, _ = cb.anom(ds)
    sst = cb.detrend_dim(ano.sosstsst.sel(lat=slice(30, -30)), "time", deg=2).sel(time=slice(start, end))
    th = cb.detrend_dim(ano.so20chgt.sel(lat=slice(30, -30)), "time", deg=2).sel(time=slice(start, end))
    E, I, h = idx.N34(sst), idx.DMI_NOAA(sst), idx.EqPac(th)
    save_indices_csv(out_csv, E.time.values, E.values, I.values, h.values)
    print(f"wrote {out_csv}: {len(E)} months")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/make_indices_csv.py <ORAS_LowRes.nc> <output.csv>\n"
                 "example: python scripts/make_indices_csv.py ../DATA/ORAS_LowRes.nc data/indices_oras5_1958_2023.csv")
    main(*sys.argv[1:3])
