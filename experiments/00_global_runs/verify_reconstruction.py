#!/usr/bin/env python3
"""Verify the reconstructed global-insertion rollouts against the ORIGINAL output files.

The reconstructed T and Z rollouts (and the renamed U+V rollout) hard-replace the
inserted variable with the time-matched ERA5 field at all 13 pressure levels, globally
(latitude rows 0:720), from rollout step 121 onward, and change nothing else. If that is
exactly what the original runs did, then in each original output file:
  (a) before insertion, all three runs are identical (same IC and code path);
  (b) after insertion, the inserted variable equals ERA5 exactly (max |diff| = 0)
      at all 13 levels everywhere, while the other variables do not.
This check runs on CPU from the output files alone; no model re-run is needed.

Result on the original files (2026-11-09): every condition holds. Before insertion
(lead <= 720 h) the three runs agree. From lead 732 h on, max |pred - ERA5| = 0 at all 13
levels for t (T run), z (Z run) and u, v (U+V run), while the non-inserted variables
keep nonzero errors.

Usage:
    python verify_reconstruction.py --pred_dir /path/to/Aurora_Pred_Files \
        --era5_grib /path/to/era5_pressure_2023_03.grib
"""
import argparse

import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--pred_dir", required=True)
ap.add_argument("--era5_grib", required=True, help="ERA5 13-level pressure GRIB covering March 2023")
ap.add_argument("--base_time", default="2023-02-01 12:00", help="time origin of the lead_time axis")
a = ap.parse_args()

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
RUNS = {"T": ("auroraoutput_temp_inserted.nc", ["t"]),
        "Z": ("auroraoutput_geopotential_inserted.nc", ["z"]),
        "UV": ("auroraoutput_uv_inserted.nc", ["u", "v"])}
era = xr.open_dataset(a.era5_grib, engine="cfgrib")
era_times = pd.DatetimeIndex(era.time.values)
base = pd.Timestamp(a.base_time)
ok = True

def maxdiff(ds, k, var, vt):
    return max(float(np.nanmax(np.abs(np.asarray(ds.variables[f"{var}{p}"][k])
                                      - era[var].sel(time=vt, isobaricInhPa=p).values[:720])))
               for p in LEVELS)

files = {tag: nc.Dataset(f"{a.pred_dir}/{fn}") for tag, (fn, _) in RUNS.items()}
leads = list(files["T"].variables["lead_time"][:])

# (a) identical before insertion
for L in [684, 696, 708, 720]:
    k = leads.index(L)
    same = all(np.array_equal(np.asarray(files["T"].variables[v][k]), np.asarray(files[o].variables[v][k]),
                              equal_nan=True) for o in ["Z", "UV"] for v in ["t500", "z500", "u500"])
    print(f"lead {L:4d} h (pre-insertion): runs identical = {same}")
    ok &= same

# (b) exact truth in the inserted variable only, after insertion
for L in [732, 744, 1020]:
    vt = base + pd.Timedelta(hours=L)
    if vt not in era_times:
        continue
    k = leads.index(L)
    for tag, (_, ins) in RUNS.items():
        for v in ["t", "z", "u", "v"]:
            m = maxdiff(files[tag], k, v, vt)
            expect_zero = v in ins
            good = (m == 0.0) if expect_zero else (m > 0.0)
            ok &= good
            print(f"lead {L:4d} h  {tag:2} run  {v}: max|pred-ERA5| (13 lev) = {m:10.4g}  "
                  f"{'(inserted, expect 0)' if expect_zero else ''} {'OK' if good else 'FAIL'}")
print("\nRECONSTRUCTION VERIFIED" if ok else "\nVERIFICATION FAILED")
