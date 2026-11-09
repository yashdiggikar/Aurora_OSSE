#!/usr/bin/env python3
"""
Global NRMSE for Aurora — 4 runs vs ERA5 truth
Grid: 720×1440 (0.25°, 90°N to -89.75°; South Pole excluded).
IC: 2023-02-01 12:00 UTC  |  120 steps × 12 hr = 60 days

Control run: first 120 steps of auroraoutput_1year_1stfeb23.nc
  (auroraoutput_60day_12hr.nc is a truncated write of the same run)

Normalization: cosine-latitude weighted global mean of ERA5 temporal std
  σ_global(var) = cos-lat-weighted mean of std_t( ERA5(lat,lon,t) ) over 120 steps
  NRMSE < 1.0 = skillful (error < natural variability)

Outputs (all in nrmse/values/):
  Aurora_{var}_global_rmse.csv   — step, lead_day, control, t_ins, uv_ins, z_ins
  Aurora_{var}_global_nrmse.csv  — same, divided by σ_global(var)
  Aurora_era5_std_normalization.csv — per-variable normalization factors
"""

import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files")
ERA5DIR   = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
OUTDIR    = BASE / "nrmse"
VALDIR    = OUTDIR / "values"
LOGFILE   = OUTDIR / "logs" / "compute_aurora_nrmse_log.txt"

LOGFILE.parent.mkdir(parents=True, exist_ok=True)
VALDIR.mkdir(parents=True, exist_ok=True)

log = open(LOGFILE, "w", buffering=1)
def L(msg): print(msg); log.write(msg + "\n")

# ─── Aurora run files ─────────────────────────────────────────────────────────
# Control = 1-year file (same IC, same format); read first 120 steps
CTRL_FILE  = Path("/umbc/rs/pi_halem/users/yashd1/auroraoutput_1year_1stfeb23.nc")
TINS_FILE  = BASE / "auroraoutput_temp_inserted.nc"
UVINS_FILE = BASE / "auroraoutput_uv_inserted.nc"
ZINS_FILE  = BASE / "auroraoutput_geopotential_inserted.nc"

AUR_FILES = {
    "control":  (CTRL_FILE,  slice(0, 120)),
    "t_ins":    (TINS_FILE,  slice(None)),
    "uv_ins":   (UVINS_FILE, slice(None)),
    "z_ins":    (ZINS_FILE,  slice(None)),
}
RUNS = ["control", "t_ins", "uv_ins", "z_ins"]

# ─── ERA5 files (Feb–Apr 2023) ────────────────────────────────────────────────
ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

# ─── Time axis ────────────────────────────────────────────────────────────────
# Aurora IC = 2023-02-01 12:00 UTC; lead_time stored as hours
AURORA_IC = datetime(2023, 2, 1, 12, 0)
N_STEPS   = 120
aur_times = [AURORA_IC + timedelta(hours=12 * k) for k in range(N_STEPS)]
lead_days = np.array([0.5 * k for k in range(N_STEPS)])

# ─── Grid (Aurora = 720×1440, no South Pole) ─────────────────────────────────
NLAT, NLON = 720, 1440
lats_1d    = np.arange(90.0, -90.0, -0.25)          # 720 points, 90 → -89.75
cos_w_1d   = np.cos(np.deg2rad(lats_1d))             # (720,)
cos_w_2d   = cos_w_1d[:, None] * np.ones((1, NLON))  # (720, 1440)
total_w    = cos_w_2d.sum()

# ─── ERA5 loading ─────────────────────────────────────────────────────────────
def _load_era5_month(fpath, shortname, type_of_level, level):
    kw = {"typeOfLevel": type_of_level, "shortName": shortname}
    if level is not None:
        kw["level"] = int(level)
    ds = xr.open_dataset(str(fpath), engine="cfgrib",
                         filter_by_keys=kw,
                         backend_kwargs={"indexpath": ""})
    drop = [c for c in ds.coords if ds.coords[c].dims == () or
            c in ("step", "number", "isobaricInhPa", "surface",
                  "valid_time", "heightAboveGround")]
    ds = ds.drop_vars(drop, errors="ignore")
    varname = list(ds.data_vars)[0]
    return ds[varname]


def load_era5_array(shortname, type_of_level, level, era5_files):
    """Load ERA5 → float32 (N_STEPS, 720, 1440), NaN where not found.
    ERA5 has 721 lat rows (includes South Pole); we drop the last row to
    match Aurora's 720-row grid (90 → -89.75)."""
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in era5_files:
        if not fpath.exists():
            L(f"    MISSING: {fpath.name}"); continue
        L(f"    Loading {fpath.name}  var={shortname}")
        try:
            da = _load_era5_month(fpath, shortname, type_of_level, level)
        except Exception as e:
            L(f"      SKIP ({e})"); continue
        era5_ts  = pd.to_datetime(da.time.values)
        lat_name = "latitude" if "latitude" in da.dims else "lat"
        lat_vals = da[lat_name].values
        if lat_vals[0] < lat_vals[-1]:          # ascending → flip to descending
            da = da.isel({lat_name: slice(None, None, -1)})
        arr = da.values.astype(np.float32)      # (T_era5, 721, 1440)
        if arr.shape[1] == 721:                 # drop South Pole row
            arr = arr[:, :720, :]
        da.close(); gc.collect()
        for k, tgt in enumerate(aur_times):
            if not np.all(np.isnan(result[k])):
                continue
            deltas = np.abs(era5_ts - pd.Timestamp(tgt))
            idx    = deltas.argmin()
            if deltas[idx].total_seconds() <= 3600:
                result[k] = arr[idx]
        del arr; gc.collect()
    matched = int(np.sum(~np.all(np.isnan(result), axis=(1, 2))))
    L(f"    ERA5 {shortname} (lev={level}): matched {matched}/{N_STEPS} steps")
    return result


def load_aurora_var(run_name, var_name):
    """Load (N_STEPS, 720, 1440) float32 array for a named variable."""
    fpath, sl = AUR_FILES[run_name]
    with h5py.File(str(fpath), "r") as f:
        arr = f[var_name][sl, :, :]
    return arr.astype(np.float32)

# ─── Metric helpers ───────────────────────────────────────────────────────────
def global_rmse(aur_arr, era5_arr):
    sq   = (aur_arr - era5_arr) ** 2
    rmse = np.sqrt((sq * cos_w_2d).sum(axis=(1, 2)) / total_w)
    return rmse.astype(np.float64)


def era5_global_std(era5_arr):
    valid_mask = ~np.all(np.isnan(era5_arr), axis=0)
    std_field  = np.nanstd(era5_arr, axis=0)
    w          = cos_w_2d * valid_mask
    wsum       = w.sum()
    return float((std_field * w).sum() / wsum) if wsum > 0 else np.nan

# ─── Variable definitions ─────────────────────────────────────────────────────
# (key: (label, unit, era5_short, era5_level, era5_type, era5_files))
# aurora_var_name = key (e.g. "t500"); for winds use None → derive
VARIABLES = {
    "t500":    ("T500",    "K",      "t",   500, "isobaricInhPa", ERA5_PRES),
    "u500":    ("U500",    "m s⁻¹",  "u",   500, "isobaricInhPa", ERA5_PRES),
    "z500":    ("Z500",    "m²s⁻²",  "z",   500, "isobaricInhPa", ERA5_PRES),
    "wind500": ("Wind500", "m s⁻¹",  None,  None, None,            None),
    "wind10":  ("Wind10",  "m s⁻¹",  None,  None, None,            None),
}

# ─── Main loop ────────────────────────────────────────────────────────────────
L("=" * 70)
L("AURORA GLOBAL RMSE + NRMSE  |  IC=2023-02-01 12Z  |  120 steps × 0.5 day")
L("Grid: 720×1440 (0.25°, South Pole excluded)")
L("Control: auroraoutput_1year_1stfeb23.nc  (steps 0–119)")
L("=" * 70)

norm_rows = []

for var_key, (lbl, unit, era5_short, era5_level, era5_type, era5_files) in VARIABLES.items():
    L(f"\n{'─'*60}")
    L(f"Variable: {var_key}  ({lbl}  [{unit}])")

    # ── Load ERA5 ────────────────────────────────────────────────────────────
    if var_key == "wind500":
        L("  Loading ERA5 U500 + V500 for Wind500 …")
        e_u = load_era5_array("u", "isobaricInhPa", 500, ERA5_PRES)
        e_v = load_era5_array("v", "isobaricInhPa", 500, ERA5_PRES)
        era5_arr = np.sqrt(e_u**2 + e_v**2)
        del e_u, e_v; gc.collect()
    elif var_key == "wind10":
        L("  Loading ERA5 10U + 10V for Wind10 …")
        e_u = load_era5_array("10u", "surface", None, ERA5_SURF)
        e_v = load_era5_array("10v", "surface", None, ERA5_SURF)
        era5_arr = np.sqrt(e_u**2 + e_v**2)
        del e_u, e_v; gc.collect()
    else:
        L(f"  Loading ERA5 {era5_short} …")
        era5_arr = load_era5_array(era5_short, era5_type, era5_level, era5_files)

    # ── Normalization factor ──────────────────────────────────────────────────
    sigma = era5_global_std(era5_arr)
    L(f"  ERA5 global temporal std (σ) = {sigma:.6f}  [{unit}]")
    norm_rows.append({"variable": var_key, "label": lbl, "unit": unit,
                      "era5_global_std": sigma})

    # ── RMSE for all runs ────────────────────────────────────────────────────
    rmse_cols  = {"step": np.arange(N_STEPS), "lead_day": lead_days}
    nrmse_cols = {"step": np.arange(N_STEPS), "lead_day": lead_days}

    for run in RUNS:
        L(f"  Run: {run}")
        if var_key == "wind500":
            a_u     = load_aurora_var(run, "u500")
            a_v     = load_aurora_var(run, "v500")
            aur_arr = np.sqrt(a_u**2 + a_v**2)
            del a_u, a_v; gc.collect()
        elif var_key == "wind10":
            a_u     = load_aurora_var(run, "10u")
            a_v     = load_aurora_var(run, "10v")
            aur_arr = np.sqrt(a_u**2 + a_v**2)
            del a_u, a_v; gc.collect()
        else:
            aur_arr = load_aurora_var(run, var_key)

        rmse_ts = global_rmse(aur_arr, era5_arr)
        rmse_cols[run]  = rmse_ts
        nrmse_cols[run] = rmse_ts / sigma if sigma > 0 else np.full_like(rmse_ts, np.nan)

        mature_mean  = rmse_ts[lead_days >= 30].mean()
        nrmse_mature = mature_mean / sigma if sigma > 0 else np.nan
        L(f"    mature RMSE={mature_mean:.4f}  NRMSE={nrmse_mature:.4f}")
        del aur_arr; gc.collect()

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(rmse_cols).to_csv(
        VALDIR / f"Aurora_{var_key}_global_rmse.csv", index=False)
    pd.DataFrame(nrmse_cols).to_csv(
        VALDIR / f"Aurora_{var_key}_global_nrmse.csv", index=False)
    L(f"  Saved: Aurora_{var_key}_global_rmse.csv + nrmse.csv")
    del era5_arr; gc.collect()

# ─── Normalization factors CSV ────────────────────────────────────────────────
df_norm = pd.DataFrame(norm_rows)
df_norm.to_csv(VALDIR / "Aurora_era5_std_normalization.csv", index=False)
L(f"\nNormalization factors saved: Aurora_era5_std_normalization.csv")
L(df_norm.to_string(index=False))

L("\n" + "=" * 70)
L("DONE — all Aurora global RMSE and NRMSE CSVs written.")
L("=" * 70)
log.close()
