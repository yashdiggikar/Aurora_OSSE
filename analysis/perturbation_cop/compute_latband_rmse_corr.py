#!/usr/bin/env python3
"""
Single-latitude-slice RMSE vs ERA5 for the Aurora Correlated OSSE Perturbation
experiments (T_PERTURBED_CORR, UV_PERTURBED_CORR).

Four individual latitude rows, as requested (NOT band averages): 2N, 14N,
40N, 60N. Each is the RMSE at that one nearest grid row only -- cosine-
latitude weighting is unnecessary for a single fixed latitude (see FCN's
compute_latband_rmse_corr.py docstring for why).

Output: one CSV per experiment, columns:
    lead_hour, valid_time,
    control_rmse_<VAR>_2N, experiment_rmse_<VAR>_2N,
    control_rmse_<VAR>_14N, experiment_rmse_<VAR>_14N,
    control_rmse_<VAR>_40N, experiment_rmse_<VAR>_40N,
    control_rmse_<VAR>_60N, experiment_rmse_<VAR>_60N, ...
written to PERTURBATION_EXPERIMENTS/Aurora/<experiment>/.

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_latband_rmse_corr.py
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
# (previously auroraoutput_1year_1stfeb23.nc -- see compute_global_rmse_corr.py
# for why this was replaced 2026-07-28.)
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "Aurora"

EXPERIMENTS = [
    ("T_PERTURBED_CORR",  CC_DIR / "T_PERTURBED_CORR"  / "output" / "aurora_t_perturbed_corr.nc",  "temp"),
    ("UV_PERTURBED_CORR", CC_DIR / "UV_PERTURBED_CORR" / "output" / "aurora_uv_perturbed_corr.nc", "uv"),
]

BASE_TIME  = datetime(2023, 2, 1, 12, 0)
LEVELS = [300, 500, 1000]  # hPa
UPPER_VARS = {"T": "t", "U": "u", "V": "v", "Z": "z", "Q": "q"}  # label -> ERA5 shortName (== Aurora's own var prefix)
SFC_VARS = [("T2M", "t2m", "2t"), ("U10", "u10", "10u"), ("V10", "v10", "10v")]  # label, era5 key, Aurora var name

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

AURORA_LATS  = np.arange(90.0, -90.0, -0.25)   # 720 pts, no south-pole row
COS_W_AURORA = np.cos(np.deg2rad(AURORA_LATS))[:, np.newaxis]

LAT_SLICES = {"2N": 2.0, "14N": 14.0, "40N": 40.0, "60N": 60.0}
LAT_ROWS = {name: int(np.argmin(np.abs(AURORA_LATS - lat))) for name, lat in LAT_SLICES.items()}
for name, row in LAT_ROWS.items():
    print(f"Slice {name}: nearest row {row} (lat={AURORA_LATS[row]:.3f})")


def slice_rmse(fc, truth, row):
    return float(np.sqrt(np.mean((fc[row, :] - truth[row, :]) ** 2)))


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
            print(f"  [{exp_name}] {label}: unavailable, skipping.")
            continue
        try:
            fc_data = load_var(exp_ds, varname, n_steps)
        except (KeyError, IndexError) as e:
            print(f"  [{exp_name}] {label}: variable '{varname}' missing ({e}), skipping.")
            continue
        ctrl_data_full = load_var(ctrl_ds, varname, len(ctrl_lead_h))

        for slice_name, row in LAT_ROWS.items():
            exp_rmse  = np.full(n_steps, np.nan, dtype=np.float64)
            ctrl_rmse = np.full(n_steps, np.nan, dtype=np.float64)
            for s, vt in enumerate(valid_times):
                ei = era5_idx(era5_times, vt)
                if ei is None:
                    continue
                truth_720 = era5_data[ei][:-1]   # 721 -> 720
                exp_rmse[s] = slice_rmse(fc_data[s], truth_720, row)
                ci = era5_idx(ctrl_valid_np, np.datetime64(vt))
                if ci is not None:
                    ctrl_rmse[s] = slice_rmse(ctrl_data_full[ci], truth_720, row)
            result[f"control_rmse_{label}_{slice_name}"] = ctrl_rmse
            result[f"experiment_rmse_{label}_{slice_name}"] = exp_rmse
            print(f"  [{exp_name}] {label} [{slice_name}]: mean control={np.nanmean(ctrl_rmse):.4f}  "
                  f"mean experiment={np.nanmean(exp_rmse):.4f}")

        out_csv = out_dir / f"latband_rmse_{exp_name}.csv"
        pd.DataFrame(result).to_csv(out_csv, index=False)

    print(f"  Saved: {out_dir / f'latband_rmse_{exp_name}.csv'}")
    exp_ds.close()

ctrl_ds.close()
print("\nDone.")
