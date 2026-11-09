#!/usr/bin/env python3
"""
Add Q NRMSE for Aurora — specific humidity at 500 hPa (variable q500).
ERA5 q at 500 hPa already exists in era5_pressure_2023_0?.grib.
Writes Aurora_q500_global_nrmse.csv and Aurora_q500_global_rmse.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import gc, os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path("/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files")
ERA5DIR = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
VALDIR  = BASE / "nrmse/values"
LOGFILE = BASE / "nrmse/logs/compute_q_nrmse_aurora.txt"
LOGFILE.parent.mkdir(parents=True, exist_ok=True)

log = open(LOGFILE, "w", buffering=1)
def L(msg): print(msg); log.write(msg + "\n")

# ── Aurora run files ───────────────────────────────────────────────────────────
AUR_FILES = {
    "control":  (Path("/umbc/rs/pi_halem/users/yashd1/auroraoutput_1year_1stfeb23.nc"), slice(0, 120)),
    "t_ins":    (BASE / "auroraoutput_temp_inserted.nc",          slice(None)),
    "uv_ins":   (BASE / "auroraoutput_uv_inserted.nc",            slice(None)),
    "z_ins":    (BASE / "auroraoutput_geopotential_inserted.nc",  slice(None)),
}
RUNS     = ["control", "t_ins", "uv_ins", "z_ins"]
ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

# ── Time / grid ───────────────────────────────────────────────────────────────
AURORA_IC = datetime(2023, 2, 1, 12, 0)
N_STEPS   = 120
aur_times = [AURORA_IC + timedelta(hours=12 * k) for k in range(N_STEPS)]
lead_days = np.array([0.5 * k for k in range(N_STEPS)])

NLAT, NLON = 720, 1440
lats_1d  = np.linspace(89.875, -89.875, NLAT)
cos_w_2d = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, NLON))
total_w  = cos_w_2d.sum()

# ── ERA5 loader ───────────────────────────────────────────────────────────────
def load_era5_q500(era5_files):
    """Return ERA5 specific humidity at 500 hPa: (N_STEPS, 720, 1440)."""
    result = np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
    for fpath in era5_files:
        if not fpath.exists():
            continue
        L(f"    Loading {fpath.name}  var=q  lev=500")
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 filter_by_keys={"shortName": "q",
                                                 "typeOfLevel": "isobaricInhPa",
                                                 "level": 500},
                                 backend_kwargs={"indexpath": ""})
            ds = ds.drop_vars(["step", "isobaricInhPa", "valid_time"], errors="ignore")
            da = ds["q"]
        except Exception as e:
            L(f"      SKIP: {e}"); continue
        era5_ts = pd.to_datetime(da.time.values)
        if da.latitude.values[0] < da.latitude.values[-1]:
            da = da.isel(latitude=slice(None, None, -1))
        # Crop south-pole row so 721→720
        arr = da.values.astype(np.float32)
        if arr.shape[-2] == 721:
            arr = arr[:, 1:, :]
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
    L(f"    ERA5 q500: matched {matched}/{N_STEPS} steps")
    return result

# ── Metrics ───────────────────────────────────────────────────────────────────
def global_rmse(pred, truth):
    sq = (pred - truth) ** 2
    return np.sqrt((sq * cos_w_2d).sum(axis=(1, 2)) / total_w).astype(np.float64)

def era5_global_std(arr):
    std_field = np.nanstd(arr, axis=0)
    w         = cos_w_2d * (~np.all(np.isnan(arr), axis=0))
    wsum      = w.sum()
    return float((std_field * w).sum() / wsum) if wsum > 0 else np.nan

# ── Main ──────────────────────────────────────────────────────────────────────
L("=" * 70)
L("Aurora Q500 NRMSE  |  IC=2023-02-01 12Z")
L("=" * 70)

L("Loading ERA5 q500 ...")
era5_q = load_era5_q500(ERA5_PRES)
sigma  = era5_global_std(era5_q)
L(f"  ERA5 q500 global std σ = {sigma:.2e} kg/kg")

rmse_cols  = {"step": np.arange(N_STEPS), "lead_day": lead_days}
nrmse_cols = {"step": np.arange(N_STEPS), "lead_day": lead_days}

for run in RUNS:
    L(f"  Run: {run}")
    fpath, sl = AUR_FILES[run]
    with h5py.File(str(fpath), "r") as f:
        aur_q = f["q500"][sl, :, :].astype(np.float32)
    rmse_ts = global_rmse(aur_q, era5_q)
    rmse_cols[run]  = rmse_ts
    nrmse_cols[run] = rmse_ts / sigma if sigma > 0 else np.full_like(rmse_ts, np.nan)
    mature = rmse_ts[lead_days >= 30].mean()
    L(f"    mature RMSE={mature:.2e}  NRMSE={mature/sigma:.4f}")
    del aur_q; gc.collect()

del era5_q; gc.collect()

pd.DataFrame(rmse_cols ).to_csv(VALDIR / "Aurora_q500_global_rmse.csv",  index=False)
pd.DataFrame(nrmse_cols).to_csv(VALDIR / "Aurora_q500_global_nrmse.csv", index=False)
L("Saved: Aurora_q500_global_nrmse.csv + Aurora_q500_global_rmse.csv")
L("DONE")
log.close()
