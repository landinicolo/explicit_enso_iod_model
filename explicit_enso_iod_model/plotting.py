"""Plot helpers.  Each returns ``(fig, ax_or_axes)`` and never calls ``plt.show()``."""
from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .core import SEASONAL_PARAM_NAMES, seasonal_value, simulate_free_evolution
from .diagnostics import APR_CYCLE, MONTH_ABBR, seasonality_xticks, state_total

COLORS = {"E": "#BE067C", "I": "#008A69"}
ENSO_PARAMS = ("alpha_E", "a", "gamma_E", "beta_E")     # coloured like E, the rest like I


def _grid(ax):
    ax.grid(True, alpha=0.3)


def plot_kalman_decomposition(time, E_obs, I_obs, x_kalman, figsize=(16, 4)):
    """Observed E / I against their Kalman latent parts (independent, dependent, sum)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, var, obs, (i_ind, i_dep) in zip(axes, "EI", (E_obs, I_obs), ((0, 1), (2, 3))):
        c = COLORS[var]
        tot = x_kalman[:, i_ind] + x_kalman[:, i_dep]
        corr = np.corrcoef(obs, tot)[0, 1]
        ax.plot(time, x_kalman[:, i_ind], "--", color=c, lw=1.2, alpha=.8, label=f"{var}_I est")
        ax.plot(time, x_kalman[:, i_dep], ":", color=c, lw=1.2, alpha=.8, label=f"{var}_D est")
        ax.plot(time, tot, color=c, lw=1.2, alpha=.8, label=f"{var}_I + {var}_D")
        ax.plot(time, obs, color="k", lw=1.0, alpha=.6, label=f"{var} observed")
        ax.set_title(f"{var} decomposition (corr={corr:.3f})", fontweight="bold")
        ax.legend(fontsize=9)
        _grid(ax)
    fig.tight_layout()
    return fig, axes


def plot_seasonal_parameters(params, *, figsize=(12, 14), title="Seasonal parameter curves"):
    """Annual cycle of every seasonal coefficient (12 monthly values)."""
    t = np.arange(12) / 12.0
    fig, axes = plt.subplots(len(SEASONAL_PARAM_NAMES), 1, figsize=figsize, sharex=True)
    for ax, name in zip(axes, SEASONAL_PARAM_NAMES):
        ax.plot(range(12), seasonal_value(params[name], t), lw=1.6,
                color=COLORS["E"] if name in ENSO_PARAMS else COLORS["I"], marker="o", ms=3)
        ax.set_ylabel(name)
        _grid(ax)
    axes[-1].set_xticks(range(12))
    axes[-1].set_xticklabels(MONTH_ABBR)
    axes[-1].set_xlabel("Month within seasonal cycle")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_latent_overlay(time, x_truth, x_model, figsize=(16, 4)):
    """Kalman ('truth') vs model latent states E_I, E_D, I_I, I_D."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for k, (ax, name) in enumerate(zip(axes, ("E_I", "E_D", "I_I", "I_D"))):
        ax.plot(time, x_truth[:, k], "k", lw=1.1, label="Kalman")
        ax.plot(time, x_model[:, k], lw=1.1, color=COLORS["E" if k < 2 else "I"], label="model")
        ax.set_title(name)
        loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        _grid(ax)
    axes[0].legend()
    fig.tight_layout()
    return fig, axes


def plot_spectrum_compare(spec: dict, var="E", figsize=(10, 5)):
    """Output of ``spectrum_compare``: observed vs ensemble-mean power spectrum."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(spec["period_obs"], spec["power_obs"], "k", label="observed")
    if not np.isfinite(spec["power_model"]).any():
        warnings.warn(f"{var} spectrum: no finite model power; run diagnostics.ensemble_health(ens).")
    ax.plot(spec["period_model"], spec["power_model"], color=COLORS[var], label="freerun model")
    ax.axvspan(24, 84, color="C2", alpha=.12, label="2-7 yr band")
    ax.axvline(12, color="gray", ls="--", lw=1, label="annual")
    ax.set(xlim=(4, 120), xlabel="Period (months)", ylabel="Power", title=f"{var} spectrum")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    return fig, ax


def plot_peak_month_histogram(obs_months, model_months, var="E", figsize=(10, 5)):
    """Histogram of peak months (0-11): observed (one per year) vs ensemble."""
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(-0.5, 12.5, 1.0)
    ax.hist(obs_months, bins=bins, alpha=.6, color="k", density=True, label="observed")
    if len(model_months) == 0:
        warnings.warn(f"{var} peak months: the ensemble gave no peaks (all members non-finite or "
                      "below min_abs_peak); run diagnostics.ensemble_health(ens).")
    ax.hist(model_months, bins=bins, alpha=.55, color=COLORS[var], density=True, label="freerun model")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_ABBR)
    ax.set(xlabel="Peak month", ylabel="Density", title=f"{var} peak month distribution")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    return fig, ax


def plot_monthly_abs_amplitude(obs_curve, ens_mean, ens_std, var="E", figsize=(10, 5)):
    """Mean |value| by calendar month: observed vs ensemble mean +/- std."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(12)
    ax.plot(x, obs_curve, "k-o", lw=2, label="observed")
    if not np.isfinite(ens_mean).any():
        warnings.warn(f"{var} amplitude: model curve is all NaN; run diagnostics.ensemble_health(ens).")
    ax.plot(x, ens_mean, "-o", color=COLORS[var], lw=1.8, label="model mean")
    ax.fill_between(x, ens_mean - ens_std, ens_mean + ens_std, color=COLORS[var], alpha=.15)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_ABBR)
    ax.set(ylabel="Mean |value|", title=f"Mean absolute {var} amplitude by month")
    ax.legend(loc="upper right")
    _grid(ax)
    fig.tight_layout()
    return fig, ax


def plot_skill_vs_lead(skills: dict, figsize=(14, 5)):
    """``skills``: label -> (corr, rmse) arrays indexed by lead, e.g. from ``skill_by_lead``.

    Labels containing 'persistence' are drawn dashed; labels ending in 'E'/'I' pick the colour.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for label, (corr, rmse) in skills.items():
        style = dict(color=COLORS[label.strip()[-1]], marker="o", ms=4)
        if "persistence" in label:
            style.update(ls="--", alpha=.5)
        axes[0].plot(np.arange(len(corr)), corr, label=label, **style)
        axes[1].plot(np.arange(len(rmse)), rmse, label=label, **style)
    axes[0].axhline(0, color="grey", lw=.8)
    axes[0].set(xlabel="Lead time (months)", ylabel="Correlation with obs",
                title="In-sample skill: correlation")
    axes[1].set(xlabel="Lead time (months)", ylabel="RMSE", title="In-sample skill: RMSE")
    for ax in axes:
        ax.legend(fontsize=9)
        _grid(ax)
    fig.tight_layout()
    return fig, axes


def plot_example_hindcasts(x_init, obs, params, months, t0s, var="E", max_lead=16, dt=1.0,
                           figsize=(12, 5)):
    """Deterministic forecast trajectories from a few start months over the observed series."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(obs, "k", lw=1.2, label="observed", zorder=5)
    for t0 in t0s:
        traj = simulate_free_evolution(x_init[t0], max_lead + 1, dt, params,
                                       start_month=int(months[t0]))
        ax.plot(np.arange(t0, t0 + max_lead + 1), state_total(traj, var), color=COLORS[var],
                lw=1.6, marker="o", ms=3)
    ax.set(xlabel="Time index (months)", ylabel=var,
           title=f"Example {max_lead}-month hindcast trajectories ({var})")
    ax.legend(["observed"], fontsize=8)
    _grid(ax)
    fig.tight_layout()
    return fig, ax


def plot_freerun_timeline(time_obs, obs, full_time, freerun, var="E", figsize=(14, 4)):
    """Observed record with one long free run (extending past the record) over it."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_obs, obs, "k", lw=1.1, label="observed", zorder=5)
    ax.plot(full_time, freerun, color=COLORS[var], lw=1.1, alpha=.85, label="freerun")
    ax.axvspan(time_obs[-1], full_time[-1], color="grey", alpha=.07, zorder=0)
    ax.axvline(time_obs[-1], color="grey", lw=.9, ls="--", alpha=.7)
    ax.set(ylabel=var, title=f"{var}: observed vs. freerun example")
    ax.legend(fontsize=9, loc="upper left")
    _grid(ax)
    fig.tight_layout()
    return fig, ax


def plot_seasonality_skill(skill, signif, title, max_lead=24, vmin=0.3, vmax=1.0, figsize=(11, 5)):
    """Heat-map of correlation skill by initial month (y) and target month (x); hatched = not significant."""
    cmap = plt.cm.Reds.copy()
    cmap.set_bad("lightgrey")
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(skill), origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    for r in range(skill.shape[0]):
        for c in range(skill.shape[1]):
            if np.isfinite(skill[r, c]) and not signif[r, c]:
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                           hatch="//", edgecolor="black", linewidth=0))
    fs = 12
    ax.set_yticks(range(12))
    ax.set_yticklabels(APR_CYCLE, fontsize=fs)
    ax.set_ylabel("Initial time", fontsize=fs)
    cols, labels = seasonality_xticks(max_lead)
    ax.set_xticks(cols)
    ax.set_xticklabels(labels, fontsize=fs)
    ax.set_xlabel("Target month", fontsize=fs)
    ax.set_title(title, fontsize=fs)
    fig.colorbar(im, ax=ax).set_label("Correlation skill", fontsize=fs)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Regression maps (need cartopy)
# --------------------------------------------------------------------------
def plot_regression_maps(reg_maps, sigmas, *, extent=(30, 290, -30, 30), ncols=2,
                         figsize=(15, 8), cmap="RdBu_r", n_levels=31,
                         cbar_label="Regression coefficient (°C per 1σ index)"):
    """Regression maps (e.g. SST on E_I, E_D, I_I, I_D) on a Pacific-centred PlateCarree map.

    Parameters
    ----------
    reg_maps : name -> 2-D DataArray (lat, lon), regression per 1 sigma of the index;
        ``climate_base.regression_maps`` builds it.
    sigmas : name -> std of the index, written in each panel title.
    extent : (lon_min, lon_max, lat_min, lat_max) in degrees east (0-360).

    Returns ``(fig, axes)``.  Requires ``cartopy``.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker

    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180)
    names = list(reg_maps)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": proj},
                             constrained_layout=True, squeeze=False)

    vmax = max(float(np.nanmax(np.abs(np.asarray(m)))) for m in reg_maps.values())
    levels = np.linspace(-vmax, vmax, n_levels)
    lon_min, lon_max, lat_min, lat_max = extent
    lon_lines = np.arange(np.ceil(lon_min / 30) * 30, lon_max + 1, 30)
    lat_lines = np.arange(-30, 31, 10)
    lon_fmt = cticker.LongitudeFormatter(zero_direction_label=True, dateline_direction_label=True)

    cf = None
    for ax, name in zip(axes.flat, names):
        m = reg_maps[name]
        cf = ax.contourf(m.lon, m.lat, m, levels=levels, cmap=cmap, extend="both",
                         transform=data_crs)
        ax.contour(m.lon, m.lat, m, levels=levels, colors="k", linewidths=0.2, alpha=0.3,
                   transform=data_crs)
        ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=10)
        ax.coastlines(resolution="50m", linewidth=0.8)
        ax.set_extent(list(extent), crs=data_crs)

        # Gridlines and tick labels are drawn by hand: with a central longitude of 180,
        # cartopy's automatic labels are unreliable.
        for lon in lon_lines:
            ax.plot([lon, lon], [lat_min, lat_max], transform=data_crs, color="gray",
                    lw=0.4, alpha=0.5, ls="--", zorder=5)
            ax.text(lon, lat_min - 2, lon_fmt(lon), transform=data_crs, ha="center",
                    va="top", fontsize=9, clip_on=False)
        for lat in lat_lines:
            ax.plot([lon_min, lon_max], [lat, lat], transform=data_crs, color="gray",
                    lw=0.4, alpha=0.5, ls="--", zorder=5)
            if lat != lat_min and lat != lat_max:
                ax.text(lon_min - 2, lat, f"{abs(lat)}°{'S' if lat < 0 else 'N' if lat > 0 else ''}",
                        transform=data_crs, ha="right", va="center", fontsize=9, clip_on=False)

        ax.set_title(rf"$\bf{{{name}}}$" "\n" rf"std={sigmas[name]:.2f}", fontsize=12)

    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    cbar = fig.colorbar(cf, ax=axes, orientation="horizontal", shrink=0.9, pad=0.06)
    cbar.set_label(cbar_label, fontsize=12)
    return fig, axes


# --------------------------------------------------------------------------
# Variance (standard-deviation) phase space: CTRL / noENSO / noIOD + latent parts
# --------------------------------------------------------------------------
DEFAULT_REGIMES = {
    "CTRL":   ("E_ctrl",   "I_ctrl",   "black"),
    "noENSO": ("E_noenso", "I_noenso", "red"),
    "noIOD":  ("E_noiod",  "I_noiod",  "#0d3bff"),
}


def kalman_stds_from_segments(x_est_segments) -> dict:
    """Std of each latent part (and of the E, I totals) for every segment.

    ``x_est_segments``: list of ``(T, 5)`` arrays ``[E_I, E_D, I_I, I_D, h]``, e.g. one
    ``model.x_kalman`` per segment of a long control run.
    """
    return {
        "E_I":   [np.nanstd(x[:, 0]) for x in x_est_segments],
        "E_D":   [np.nanstd(x[:, 1]) for x in x_est_segments],
        "I_I":   [np.nanstd(x[:, 2]) for x in x_est_segments],
        "I_D":   [np.nanstd(x[:, 3]) for x in x_est_segments],
        "E_sum": [np.nanstd(x[:, 0] + x[:, 1]) for x in x_est_segments],
        "I_sum": [np.nanstd(x[:, 2] + x[:, 3]) for x in x_est_segments],
    }


def plot_std_phase_space(df_stds, E_obs, I_obs, kalman_stds=None, x_est_segments=None,
                         regimes=None, fontsz=16, ax=None, label_offsets=None):
    """STD phase space (x: IOD std, y: Nino3.4 std).

    Dots: CTRL / noENSO / noIOD experiments (one dot per segment/member, columns of
    ``df_stds`` named in ``regimes``).  Star: observations.  Crosses: Kalman-decomposed
    parts of the control segments, ``I_I`` on the x axis, ``E_I`` on the y axis and the
    totals ``(I, E)`` in the interior.

    Parameters
    ----------
    df_stds : DataFrame with one column of stds per regime variable
        (default columns: E_ctrl, I_ctrl, E_noenso, I_noenso, E_noiod, I_noiod).
    kalman_stds : output of ``kalman_stds_from_segments`` (or give ``x_est_segments``).
    label_offsets : optional name -> (dx, dy) in data units added to a label position,
        e.g. ``{"noENSO": (0, 0.05)}``.

    Returns ``(fig, ax, summary)``.
    """
    regimes = DEFAULT_REGIMES if regimes is None else regimes
    if kalman_stds is None:
        if x_est_segments is None:
            raise ValueError("Provide either kalman_stds or x_est_segments.")
        kalman_stds = kalman_stds_from_segments(x_est_segments)
    label_offsets = {"noENSO": (0.0, 0.05), "(I, E)": (0.0, -0.1), "I_I": (0.05, 0.0)} \
        if label_offsets is None else label_offsets

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    def label(text, x, y, color, offset):
        dx, dy = label_offsets.get(text, (0.0, 0.0))
        ax.annotate(text, (x + dx, y + dy), xytext=offset, textcoords="offset points",
                    color=color, fontsize=fontsz, weight="bold")

    for name, (e_col, i_col, color) in regimes.items():
        xs, ys = df_stds[i_col].values, df_stds[e_col].values
        ax.scatter(xs, ys, s=120, color=color, marker="o", edgecolor="k", linewidth=1,
                   alpha=0.85, zorder=3)
        label(name, float(np.nanmean(xs)), float(np.nanmean(ys)), color, (8, 8))

    std_E_obs, std_I_obs = float(np.nanstd(E_obs)), float(np.nanstd(I_obs))
    ax.scatter(std_I_obs, std_E_obs, s=180, marker="*", color="black", edgecolor="white",
               linewidth=0.9, zorder=7)
    label("OBS", std_I_obs, std_E_obs, "black", (8, -12))

    cross = dict(s=220, marker="x", linewidth=3.0, alpha=1.0, zorder=2)
    ax.scatter(kalman_stds["I_I"], np.zeros(len(kalman_stds["I_I"])), color="green", **cross)
    label("I_I", float(np.nanmean(kalman_stds["I_I"])), 0.0, "green", (8, -15))
    ax.scatter(np.zeros(len(kalman_stds["E_I"])), kalman_stds["E_I"], color="purple", **cross)
    label("E_I", 0.0, float(np.nanmean(kalman_stds["E_I"])), "purple", (-15, 8))
    ax.scatter(kalman_stds["I_sum"], kalman_stds["E_sum"], color="magenta", **cross)
    label("(I, E)", float(np.nanmean(kalman_stds["I_sum"])),
          float(np.nanmean(kalman_stds["E_sum"])), "magenta", (8, 6))

    ax.set_xlabel("STD IOD [°C]", fontsize=fontsz + 2, weight="bold")
    ax.set_ylabel("STD N34 [°C]", fontsize=fontsz + 2, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=fontsz)
    ax.grid(True, alpha=0.35, linestyle="--")

    all_vals = np.concatenate(
        [np.asarray(df_stds[c].values, dtype=float) for c in df_stds.columns]
        + [np.asarray(v, dtype=float) for v in kalman_stds.values()]
        + [np.asarray([std_I_obs, std_E_obs], dtype=float)])
    mmin, mmax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = max(0.05, 0.1 * (mmax - mmin))
    ax.set_xlim(-pad, mmax + pad)
    ax.set_ylim(-pad, mmax + pad)
    ax.axvline(0, color="gray", ls="--", lw=1.5, alpha=0.6)
    ax.axhline(0, color="gray", ls="--", lw=1.5, alpha=0.6)
    fig.tight_layout()

    return fig, ax, {"std_E_obs": std_E_obs, "std_I_obs": std_I_obs, "kalman_stds": kalman_stds}
