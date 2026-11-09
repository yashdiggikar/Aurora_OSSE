#!/usr/bin/env python3
"""
Spatial bias (mean signed error, forecast - truth) for the Aurora Correlated
OSSE Perturbation experiments (T_PERTURBED_CORR, UV_PERTURBED_CORR).

Computed over the POST-INJECTION period only (step >= 120, day 30-60), for
both the control forecast and the experiment forecast — see
FCN/compute_spatial_bias_corr.py's docstring for the full rationale (this is
the same analysis, same variable choices, applied to Aurora's grid/naming).

Output per experiment (into PERTURBATION_EXPERIMENTS/Aurora/<experiment>/):
    bias_control_<VAR>.npy, bias_experiment_<VAR>.npy   (lat,lon) mean signed error
    spatial_bias_<experiment>.csv                        summary table (global + 3 lat bands)
    spatial_bias_map_<experiment>_<VAR>.png              experiment-minus-control bias difference map

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_spatial_bias_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
CC_DIR    = BASE / "CC_aurora"
DATA_DIR  = BASE / "AURORA_DATA"
ERA5_DIR  = DATA_DIR / "rmse_data"
CTRL_FILE = BASE / "CC_aurora" / "CONTROL_MATCHED_60D" / "output" / "aurora_control_matched_60d.nc"
# (previously auroraoutput_1year_1stfeb23.nc -- see
# compute_global_rmse_corr.py for why this was replaced 2026-07-28.)
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "Aurora"

EXPERIMENTS = [
    ("T_PERTURBED_CORR",  CC_DIR / "T_PERTURBED_CORR"  / "output" / "aurora_t_perturbed_corr.nc",  "temp"),
    ("UV_PERTURBED_CORR", CC_DIR / "UV_PERTURBED_CORR" / "output" / "aurora_uv_perturbed_corr.nc", "uv"),
]

BASE_TIME   = datetime(2023, 2, 1, 12, 0)
LEVELS = [300, 500, 1000]  # hPa
UPPER_VARS = {"T": "t", "U": "u", "V": "v", "Z": "z", "Q": "q"}  # label -> ERA5 shortName (== Aurora's own var prefix)
SFC_VARS = [("T2M", "t2m", "2t"), ("U10", "u10", "10u"), ("V10", "v10", "10v")]  # label, era5 key, Aurora var name
INJECT_STEP = 120

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

AURORA_LATS = np.arange(90.0, -90.0, -0.25)
AURORA_LONS = np.arange(0.0, 360.0, 0.25)

POLAR_MASK  = np.abs(AURORA_LATS) > 60
MIDLAT_MASK = (np.abs(AURORA_LATS) <= 60) & (np.abs(AURORA_LATS) >= 30)
TROP_MASK   = np.abs(AURORA_LATS) < 30


def make_lead_times(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        return ds.variables["lead_time"][:].data.copy()


def read_base_time(nc_file):
    # Read from the file itself -- some CC_aurora files had a +6h base_time
    # mislabeling bug, since patched in place; this avoids assuming BASE_TIME
    # applies uniformly to every file.
    with nc4.Dataset(str(nc_file)) as ds:
        return datetime.utcfromtimestamp(int(ds.variables["base_time"][...]))


def make_valid_times(lead_h_arr, base=BASE_TIME):
    return [base + timedelta(hours=int(h)) for h in lead_h_arr]


def load_era5_var(shortname, level_hpa, grib_files):
    dsets = []
    for f in grib_files:
        if not f.exists():
            continue
        dsets.append(xr.open_dataset(str(f), engine="cfgrib",
                     filter_by_keys={"shortName": shortname, "typeOfLevel": "isobaricInhPa",
                                      "level": int(level_hpa)}, indexpath=None))
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times, data = combined.time.values, combined[shortname].values.astype(np.float32)
    for ds in dsets: ds.close()
    return times, data


def load_era5_sfc(shortname, grib_files):
    dsets = []
    for f in grib_files:
        if not f.exists():
            continue
        dsets.append(xr.open_dataset(str(f), engine="cfgrib", indexpath=None))
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times, data = combined.time.values, combined[shortname].values.astype(np.float32)
    for ds in dsets: ds.close()
    return times, data


def era5_idx(era5_times, valid_time_dt, tol_hours=3):
    vt = np.datetime64(valid_time_dt)
    deltas = np.abs(era5_times - vt)
    idx = int(np.argmin(deltas))
    return idx if deltas[idx] <= np.timedelta64(tol_hours, "h") else None


def load_var(ds, varname, n_steps):
    return ds.variables[varname][:n_steps].data.astype(np.float32)



# Cosine-latitude weights -- an unweighted np.nanmean over an equirectangular
# (lat,lon) grid overweights the poles (same row count as the equator despite
# covering far less actual area). Fixed 2026-07-28; see
# PERTURBATION_EXPERIMENTS/*/spatial_bias_*.csv regeneration note.
_W_ROW = np.clip(np.cos(np.deg2rad(AURORA_LATS)), 0, None)


def _wmean(bias_map, row_mask=None):
    m = bias_map if row_mask is None else bias_map[row_mask, :]
    w = np.broadcast_to((_W_ROW if row_mask is None else _W_ROW[row_mask])[:, None], m.shape)
    finite = np.isfinite(m)
    if not np.any(finite):
        return float("nan")
    return float(np.sum(m[finite] * w[finite]) / np.sum(w[finite]))


def region_stats(bias_map):
    return {
        "global_mean_bias": _wmean(bias_map),
        "global_mean_abs_bias": _wmean(np.abs(bias_map)),
        "polar_mean_bias": _wmean(bias_map, POLAR_MASK),
        "midlat_mean_bias": _wmean(bias_map, MIDLAT_MASK),
        "tropics_mean_bias": _wmean(bias_map, TROP_MASK),
    }


print(f"Loading ERA5 truth: {list(UPPER_VARS)} x {LEVELS}hPa + surface {[l for l,_,_ in SFC_VARS]} ...")
ERA5_UPPER = {}
for label, sn in UPPER_VARS.items():
    for lev in LEVELS:
        try:
            ERA5_UPPER[(label, lev)] = load_era5_var(sn, lev, ERA5_PRES)
        except Exception as e:
            print(f"  {label}{lev}: FAILED to load ({e!r}), skipping this combination.")
            ERA5_UPPER[(label, lev)] = (None, None)
ERA5_SFC = {label: load_era5_sfc(era5_key, ERA5_SURF) for label, era5_key, _ in SFC_VARS}

VAR_PAIRS = [(f"{label}{lev}", f"{UPPER_VARS[label]}{lev}", *ERA5_UPPER[(label, lev)])
             for label in UPPER_VARS for lev in LEVELS]
VAR_PAIRS += [(label, aur_var, *ERA5_SFC[label]) for label, _, aur_var in SFC_VARS]

print("Loading Aurora control lead_times ...")
# Opened once and kept open for the whole script -- repeatedly open/closing
# the same multi-GB file per-variable was found to segfault on this NFS
# filesystem in the FCN analysis scripts; same fix applied here proactively.
ctrl_ds = nc4.Dataset(str(CTRL_FILE))
ctrl_lead_h = make_lead_times(CTRL_FILE)
ctrl_valid  = make_valid_times(ctrl_lead_h, base=read_base_time(CTRL_FILE))
ctrl_valid_np = np.array([np.datetime64(v) for v in ctrl_valid])

summary_rows = []

for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    lead_h = make_lead_times(nc_file)
    valid_times = make_valid_times(lead_h, base=read_base_time(nc_file))
    exp_ds = nc4.Dataset(str(nc_file))
    n_steps = len(lead_h)
    post_steps = [s for s in range(n_steps) if s >= INJECT_STEP]

    # Every variable/level is evaluated for every experiment (cross-variable
    # coupling included), not just the directly-inserted family.
    var_pairs = VAR_PAIRS
    last_step = n_steps - 1  # day-60-only snapshot, alongside the day-30-60 mean below

    for label, varname, era5_times, era5_data in var_pairs:
        if era5_times is None:
            print(f"  [{exp_name}] {label}: unavailable, skipping.")
            continue
        try:
            fc_data = load_var(exp_ds, varname, n_steps)
        except (KeyError, IndexError) as e:
            print(f"  [{exp_name}] {label}: variable '{varname}' missing ({e}), skipping.")
            continue
        ctrl_data_full = load_var(ctrl_ds, varname, len(ctrl_lead_h))

        exp_diffs, ctrl_diffs = [], []
        for s in post_steps:
            vt = valid_times[s]
            ei = era5_idx(era5_times, vt)
            if ei is None:
                continue
            truth_720 = era5_data[ei][:-1]   # 721 -> 720, drop south-pole row
            exp_diffs.append(fc_data[s] - truth_720)
            ci = era5_idx(ctrl_valid_np, np.datetime64(vt))
            if ci is not None:
                ctrl_diffs.append(ctrl_data_full[ci] - truth_720)

        if not exp_diffs:
            print(f"  [{exp_name}] {label}: no matched timesteps, skipping.")
            continue

        exp_bias = np.nanmean(np.stack(exp_diffs), axis=0)
        ctrl_bias = np.nanmean(np.stack(ctrl_diffs), axis=0) if ctrl_diffs else np.full_like(exp_bias, np.nan)

        np.save(out_dir / f"bias_experiment_{label}.npy", exp_bias)
        np.save(out_dir / f"bias_control_{label}.npy", ctrl_bias)

        # Day-60-only snapshot (single step), alongside the day-30-60 mean above.
        ei60 = era5_idx(era5_times, valid_times[last_step])
        exp_bias_day60 = np.full_like(exp_bias, np.nan)
        ctrl_bias_day60 = np.full_like(exp_bias, np.nan)
        if ei60 is not None:
            truth60_720 = era5_data[ei60][:-1]
            exp_bias_day60 = fc_data[last_step] - truth60_720
            ci60 = era5_idx(ctrl_valid_np, np.datetime64(valid_times[last_step]))
            if ci60 is not None:
                ctrl_bias_day60 = ctrl_data_full[ci60] - truth60_720
        np.save(out_dir / f"bias_experiment_{label}_day60.npy", exp_bias_day60)
        np.save(out_dir / f"bias_control_{label}_day60.npy", ctrl_bias_day60)

        exp_stats  = region_stats(exp_bias)
        ctrl_stats = region_stats(ctrl_bias)
        exp_stats_day60  = region_stats(exp_bias_day60)
        ctrl_stats_day60 = region_stats(ctrl_bias_day60)
        row = {"experiment": exp_name, "variable": label}
        row.update({f"control_{k}": v for k, v in ctrl_stats.items()})
        row.update({f"experiment_{k}": v for k, v in exp_stats.items()})
        row.update({f"control_day60_{k}": v for k, v in ctrl_stats_day60.items()})
        row.update({f"experiment_day60_{k}": v for k, v in exp_stats_day60.items()})
        summary_rows.append(row)
        print(f"  [{exp_name}] {label}: control global_mean_bias={ctrl_stats['global_mean_bias']:+.4f}  "
              f"experiment global_mean_bias={exp_stats['global_mean_bias']:+.4f}  "
              f"(polar: ctrl={ctrl_stats['polar_mean_bias']:+.4f} exp={exp_stats['polar_mean_bias']:+.4f})")

        # Write the CSV after every variable (not just once at the end of the
        # experiment) so the numeric results survive even if the plotting
        # step below crashes.
        df = pd.DataFrame([r for r in summary_rows if r["experiment"] == exp_name])
        out_csv = out_dir / f"spatial_bias_{exp_name}.csv"
        df.to_csv(out_csv, index=False)

        diff_map = exp_bias - ctrl_bias
        vmax = np.nanpercentile(np.abs(diff_map), 98)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(AURORA_LONS, AURORA_LATS, diff_map, transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{exp_name}  {label}  bias(experiment) - bias(control)\n"
                      f"post-injection mean (day 30-60)", fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label=label)
        fig.savefig(out_dir / f"spatial_bias_map_{exp_name}_{label}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        diff_map_day60 = exp_bias_day60 - ctrl_bias_day60
        vmax60 = np.nanpercentile(np.abs(diff_map_day60), 98)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(AURORA_LONS, AURORA_LATS, diff_map_day60, transform=ccrs.PlateCarree(),
                            cmap="RdBu_r", vmin=-vmax60, vmax=vmax60, shading="auto")
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{exp_name}  {label}  bias(experiment) - bias(control)\n"
                      f"day 60 only", fontsize=10)
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label=label)
        fig.savefig(out_dir / f"spatial_bias_map_{exp_name}_{label}_day60.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {out_dir / f'spatial_bias_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
