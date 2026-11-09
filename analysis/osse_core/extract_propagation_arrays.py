#!/usr/bin/env python3
"""
Lightweight extraction of the propagation-map |error| arrays for the 4
land/ocean insertion experiments (land_temp, ocean_temp, land_uv, ocean_uv)
— NOT the full analyze_osse.py pipeline. Loads only the 3 day-snapshot
timesteps needed (First-Injection / Day45 / Day60) from each experiment's
raw forecast .nc plus the matching ERA5 T500/U500 truth snapshots, computes
the per-grid |error| array exactly as analyze_osse.py does, and saves each
as a .npy so propagation maps can be replotted later without touching the
raw 10GB .nc files again.

Output: CC_aurora/{T_LAND,T_OCEAN,UV_LAND,UV_OCEAN}/propagation_error_{exp}.npy
        shape (3, 720, 1440), plus propagation_meta_{exp}.json (vmax, day labels).
"""
import warnings; warnings.filterwarnings("ignore")
import gc, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import netCDF4 as nc4

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
CC_DIR    = BASE / "CC_aurora"
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

N_STEPS = 240
DAY45_IDX, DAY60_IDX = 179, 239
DAY45_H, DAY60_H = 1080, 1440
INJ_START_IDX, INJ_START_H = 121, 732
PLOT_LEVEL = 500

EXPERIMENTS = [
    ("land_temp",  CC_DIR / "T_LAND/output/aurora_t_land.nc",   CC_DIR / "T_LAND",  "temp"),
    ("ocean_temp", CC_DIR / "T_OCEAN/output/aurora_t_ocean.nc", CC_DIR / "T_OCEAN", "temp"),
    ("land_uv",    CC_DIR / "UV_LAND/output/aurora_uv_land.nc", CC_DIR / "UV_LAND", "uv"),
    ("ocean_uv",   CC_DIR / "UV_OCEAN/output/aurora_uv_ocean.nc", CC_DIR / "UV_OCEAN", "uv"),
]


def read_base_time(nc_file):
    # These files had a +6h base_time mislabeling bug (base_time was captured
    # from the first forecast's own valid time instead of the true
    # pre-rollout reference time); patched in place, so read per-file rather
    # than hardcoding datetime(2023, 2, 1, 12, 0).
    with nc4.Dataset(str(nc_file)) as ds:
        return datetime.utcfromtimestamp(int(ds.variables["base_time"][...]))


# All 4 experiments share the same corrected reference time.
BASE_TIME = read_base_time(EXPERIMENTS[0][1])

DAY_CONFIGS = [
    ("First Injection (Day 30+12h)", INJ_START_IDX, BASE_TIME + timedelta(hours=INJ_START_H)),
    ("Day 45", DAY45_IDX, BASE_TIME + timedelta(hours=DAY45_H)),
    ("Day 60", DAY60_IDX, BASE_TIME + timedelta(hours=DAY60_H)),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_era5_var(shortname, level_hpa, grib_files):
    log(f"Loading ERA5 {shortname} @ {level_hpa} hPa ...")
    dsets = []
    for f in grib_files:
        if not f.exists():
            continue
        ds = xr.open_dataset(str(f), engine="cfgrib",
                              filter_by_keys={"shortName": shortname,
                                              "typeOfLevel": "isobaricInhPa",
                                              "level": int(level_hpa)},
                              indexpath=None)
        dsets.append(ds)
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data = combined[shortname].values.astype(np.float32)
    for ds in dsets:
        ds.close()
    del combined
    log(f"  -> shape {data.shape}")
    return times, data


def era5_idx(era5_times, valid_time_dt):
    vt = np.datetime64(valid_time_dt)
    deltas = np.abs(era5_times - vt)
    idx = int(np.argmin(deltas))
    if deltas[idx] > np.timedelta64(3, "h"):
        return None
    return idx


log("=== Loading ERA5 T500 truth snapshots ===")
era5_tm, era5_t500m = load_era5_var("t", PLOT_LEVEL, ERA5_PRES)
t500_snaps = np.stack([era5_t500m[era5_idx(era5_tm, vt), :-1, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
del era5_t500m; gc.collect()

log("=== Loading ERA5 U500 truth snapshots ===")
era5_um, era5_u500m = load_era5_var("u", PLOT_LEVEL, ERA5_PRES)
u500_snaps = np.stack([era5_u500m[era5_idx(era5_um, vt), :-1, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
del era5_u500m; gc.collect()

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    log(f"Extracting propagation array: {exp_name} ...")
    var_fc = "t500" if ins_type == "temp" else "u500"
    var_label = "T @ 500 hPa (K)" if ins_type == "temp" else "U @ 500 hPa (m/s)"
    truth_snaps = t500_snaps if ins_type == "temp" else u500_snaps

    with nc4.Dataset(str(nc_file)) as ds:
        exp_slices = np.stack([
            ds.variables[var_fc][idx].data.astype(np.float32) for _, idx, _ in DAY_CONFIGS
        ])

    errors = np.abs(exp_slices - truth_snaps)
    vmax = max(float(np.nanpercentile(errors, 97)), 0.1)
    vmax = round(vmax, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"propagation_error_{exp_name}.npy", errors)
    meta = {
        "exp_name": exp_name, "ins_type": ins_type, "var_label": var_label,
        "vmax": vmax, "day_labels": [d for d, _, _ in DAY_CONFIGS],
        "shape": list(errors.shape),
    }
    with open(out_dir / f"propagation_meta_{exp_name}.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  saved -> {out_dir}/propagation_error_{exp_name}.npy  (vmax={vmax})")
    del exp_slices, errors

log("Done - all 4 Aurora propagation arrays extracted.")
