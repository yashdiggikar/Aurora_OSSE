#!/usr/bin/env python3
"""
Global (cosine-lat-weighted) RMSE vs ERA5 for the Aurora Correlated OSSE
Perturbation experiments (T_PERTURBED_CORR, UV_PERTURBED_CORR).

Reuses the exact same area-weighting/time-matching methodology as
CC_aurora/analyze_osse.py (global_rmse(), era5_idx(), cos_weights_1d())
applied to the two new global-insertion experiments, comparing against the
SAME control forecast (auroraoutput_1year_1stfeb23.nc) used everywhere else
in this project for Aurora.

Full variable/level grid (per user request): T, U, V, Z, Q (specific
humidity) at 300/500/1000 hPa, plus surface 2t/10u/10v -- computed for BOTH
experiments regardless of which family was directly inserted, so this also
reports cross-variable coupling.

Output: one CSV per experiment, columns:
    lead_hour, valid_time, control_rmse_<VAR>, experiment_rmse_<VAR>, ...
written to this directory (PERTURBATION_EXPERIMENTS/Aurora/<experiment>/).

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_global_rmse_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
CC_DIR    = BASE / "CC_aurora"
DATA_DIR  = BASE / "AURORA_DATA"
ERA5_DIR  = DATA_DIR / "rmse_data"
CTRL_FILE = BASE / "CC_aurora" / "CONTROL_MATCHED_60D" / "output" / "aurora_control_matched_60d.nc"
# (previously auroraoutput_1year_1stfeb23.nc -- a separately-built, 12-hourly
# cadence, 1-year rollout not forked from the same IC as the two experiments
# below, found to cause a real day 1-30 control/experiment RMSE mismatch
# before insertion even starts. Replaced 2026-07-28 with a purpose-built,
# IC- and cadence-matched free-running control; see
# CC_aurora/CONTROL_MATCHED_60D/inf.py and
# PERTURBATION_EXPERIMENTS/Aurora/IMPLEMENTATION.md Section 5.)
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "Aurora"

EXPERIMENTS = [
    ("T_PERTURBED_CORR",  CC_DIR / "T_PERTURBED_CORR"  / "output" / "aurora_t_perturbed_corr.nc",  "temp"),
    ("UV_PERTURBED_CORR", CC_DIR / "UV_PERTURBED_CORR" / "output" / "aurora_uv_perturbed_corr.nc", "uv"),
]

BASE_TIME  = datetime(2023, 2, 1, 12, 0)   # all CC_aurora experiments share this IC
LEVELS = [300, 500, 1000]  # hPa
# label -> ERA5 shortName; Aurora's own saved variable name is exactly f"{shortname}{level}"
UPPER_VARS = {"T": "t", "U": "u", "V": "v", "Z": "z", "Q": "q"}
# label, ERA5 lookup key, Aurora's own saved surface-variable name
SFC_VARS = [("T2M", "t2m", "2t"), ("U10", "u10", "10u"), ("V10", "v10", "10v")]

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

AURORA_LATS  = np.arange(90.0, -90.0, -0.25)   # 720 pts, no south-pole row
COS_W_AURORA = np.cos(np.deg2rad(AURORA_LATS))[:, np.newaxis]


def global_rmse(fc, truth, w2d):
    w_total = float(w2d.sum()) * fc.shape[1]
    return float(np.sqrt(np.sum(w2d * (fc - truth) ** 2) / w_total))


def make_lead_times(nc_file):
    with nc4.Dataset(str(nc_file)) as ds:
        return ds.variables["lead_time"][:].data.copy()


def read_base_time(nc_file):
    # Read base_time from the file itself rather than assuming BASE_TIME --
    # some CC_aurora experiment files had a +6h mislabeling bug (base_time was
    # captured from the first forecast's own valid time instead of the true
    # pre-rollout reference time); those files were patched in place, so this
    # always reflects the correct value per-file rather than a shared guess.
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
    return ds.variables[varname][:n_steps].data.astype(np.float32)   # (n_steps, 720, 1440)


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

    # Every variable/level is evaluated for every experiment (cross-variable
    # coupling included), not just the directly-inserted family.
    var_pairs = VAR_PAIRS

    result = {"lead_hour": lead_h.tolist(), "valid_time": [str(v) for v in valid_times]}
    for label, varname, era5_times, era5_data in var_pairs:
        if era5_times is None:
            print(f"  [{exp_name}] {label}: ERA5 data unavailable, skipping.")
            continue
        try:
            fc_data = load_var(exp_ds, varname, n_steps)   # (n_steps, 720, 1440)
        except (KeyError, IndexError) as e:
            print(f"  [{exp_name}] {label}: variable '{varname}' not in file ({e}), skipping.")
            continue
        ctrl_data_full = load_var(ctrl_ds, varname, len(ctrl_lead_h))

        exp_rmse  = np.full(n_steps, np.nan, dtype=np.float64)
        ctrl_rmse = np.full(n_steps, np.nan, dtype=np.float64)
        for s, vt in enumerate(valid_times):
            ei = era5_idx(era5_times, vt)
            if ei is None:
                continue
            truth_720 = era5_data[ei][:-1]   # drop south-pole row: 721 -> 720
            exp_rmse[s] = global_rmse(fc_data[s], truth_720, COS_W_AURORA)
            ci = era5_idx(ctrl_valid_np, np.datetime64(vt))
            if ci is not None:
                ctrl_rmse[s] = global_rmse(ctrl_data_full[ci], truth_720, COS_W_AURORA)
        result[f"control_rmse_{label}"] = ctrl_rmse
        result[f"experiment_rmse_{label}"] = exp_rmse
        print(f"  [{exp_name}] {label}: mean control_rmse={np.nanmean(ctrl_rmse):.4f}  "
              f"mean experiment_rmse={np.nanmean(exp_rmse):.4f}")

        out_csv = out_dir / f"global_rmse_{exp_name}.csv"
        pd.DataFrame(result).to_csv(out_csv, index=False)

    print(f"  Saved: {out_dir / f'global_rmse_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
