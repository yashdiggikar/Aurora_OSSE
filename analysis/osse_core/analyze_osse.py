#!/usr/bin/env python3
"""
Aurora CC OSSE Analysis
=======================
Computes for 6 insertion experiments (T_LAND, UV_LAND, T_OCEAN, UV_OCEAN, T_COASTAL, UV_COASTAL):
  1. Cosine-lat-weighted global RMSE time series vs ERA5  →  CSV + plot per experiment dir
  2. Spatial propagation maps (experiment - control) at day 30 / 45 / 60  →  plot per exp dir
  3. % RMSE improvement bar chart across all experiments  →  CC_aurora/rmse_improvement.png

Run (CPU only):
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python analyze_osse.py
"""

import warnings; warnings.filterwarnings("ignore")
import gc, sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import netCDF4 as nc4

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
CC_DIR    = BASE / "CC_aurora"
DATA_DIR  = BASE / "AURORA_DATA"
ERA5_DIR  = DATA_DIR / "rmse_data"
CTRL_FILE = BASE / "auroraoutput_1year_1stfeb23.nc"

# ── Experiments ────────────────────────────────────────────────────────────────
# (name, nc_file, output_dir, insertion_type)
EXPERIMENTS = [
    ("land_temp",    CC_DIR / "T_LAND/output/aurora_t_land.nc",       CC_DIR / "T_LAND",    "temp"),
    ("land_uv",      CC_DIR / "UV_LAND/output/aurora_uv_land.nc",     CC_DIR / "UV_LAND",   "uv"),
    ("ocean_temp",   CC_DIR / "T_OCEAN/output/aurora_t_ocean.nc",     CC_DIR / "T_OCEAN",   "temp"),
    ("ocean_uv",     CC_DIR / "UV_OCEAN/output/aurora_uv_ocean.nc",   CC_DIR / "UV_OCEAN",  "uv"),
    ("coastal_temp", CC_DIR / "T_COASTAL/output/aurora_t_coastal.nc", CC_DIR / "T_COASTAL", "temp"),
    ("coastal_uv",   CC_DIR / "UV_COASTAL/output/aurora_uv_coastal.nc", CC_DIR / "UV_COASTAL", "uv"),
]

# ── Constants ─────────────────────────────────────────────────────────────────
N_STEPS    = 240
INJECT_STEP = 120   # injection starts at step index 120 (day 30)
# base_time stored in file; all CC_aurora share 2023-02-01 12:00 UTC
BASE_TIME  = datetime(2023, 2, 1, 12, 0)

# Day indices (0-based): day 30 = step 119 (lead_time=720h), day 45=179, day 60=239
DAY30_IDX, DAY45_IDX, DAY60_IDX = 119, 179, 239
# Corresponding lead_time in hours
DAY30_H, DAY45_H, DAY60_H = 720, 1080, 1440

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

LEV_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PLOT_LEVEL = 500   # hPa for RMSE plots and maps

LABEL_MAP = {
    "land_temp": "Land T Ins", "land_uv": "Land UV Ins",
    "ocean_temp": "Ocean T Ins", "ocean_uv": "Ocean UV Ins",
    "coastal_temp": "Coastal T Ins", "coastal_uv": "Coastal UV Ins",
}
COLOR_MAP = {
    "land_temp": "#d62728", "land_uv": "#ff7f0e",
    "ocean_temp": "#1f77b4", "ocean_uv": "#17becf",
    "coastal_temp": "#2ca02c", "coastal_uv": "#9467bd",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def cos_weights_1d(lats_1d):
    """Return (n_lat, 1) cosine-latitude weight array."""
    return np.cos(np.deg2rad(lats_1d))[:, np.newaxis]

def global_rmse(fc, truth, w2d):
    """Cosine-weighted RMSE.  fc/truth: (lat,lon), w2d: (lat,1)."""
    w_total = float(w2d.sum()) * fc.shape[1]
    return float(np.sqrt(np.sum(w2d * (fc - truth) ** 2) / w_total))

def make_lead_times(nc_file):
    """Return numpy array of lead_time hours from a CC_aurora nc file."""
    with nc4.Dataset(str(nc_file)) as ds:
        return ds.variables["lead_time"][:].data.copy()

def read_base_time(nc_file):
    # The 6 experiment files here (T_LAND/UV_LAND/T_OCEAN/UV_OCEAN/T_COASTAL/
    # UV_COASTAL) had a +6h base_time mislabeling bug (base_time was captured
    # from the first forecast's own valid time instead of the true
    # pre-rollout reference time); they were patched in place to store the
    # correct value, so this is read per-file rather than assumed from
    # BASE_TIME. CTRL_FILE never had this bug -- its own valid-time
    # computations below still use BASE_TIME directly, unchanged.
    with nc4.Dataset(str(nc_file)) as ds:
        return datetime.utcfromtimestamp(int(ds.variables["base_time"][...]))

def make_valid_times(lead_h_arr, base=BASE_TIME):
    return [base + timedelta(hours=int(h)) for h in lead_h_arr]

def load_era5_var(shortname, level_hpa, grib_files):
    """
    Load ERA5 variable at a single pressure level for all grib_files.
    Returns (time_dt64_arr, data_721x1440).
    ERA5 lat: 90→-90 (721 rows); slice [:-1] for Aurora's 720-row grid.
    """
    print(f"  Loading ERA5 {shortname} @ {level_hpa} hPa ...")
    dsets = []
    for f in grib_files:
        if not f.exists():
            print(f"    WARNING: {f} not found, skipping.")
            continue
        ds = xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": shortname,
                                             "typeOfLevel": "isobaricInhPa",
                                             "level": int(level_hpa)},
                             indexpath=None)
        dsets.append(ds)
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data  = combined[shortname].values.astype(np.float32)  # (n_t, 721, 1440)
    for ds in dsets: ds.close()
    del combined
    print(f"    → shape {data.shape}, {len(times)} timesteps")
    return times, data

def load_era5_sfc(shortname, grib_files):
    """Load ERA5 surface variable."""
    print(f"  Loading ERA5 sfc {shortname} ...")
    dsets = []
    for f in grib_files:
        if not f.exists():
            continue
        ds = xr.open_dataset(str(f), engine="cfgrib", indexpath=None)
        dsets.append(ds)
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data  = combined[shortname].values.astype(np.float32)
    for ds in dsets: ds.close()
    del combined
    print(f"    → shape {data.shape}")
    return times, data

def era5_idx(era5_times, valid_time_dt):
    """Find nearest ERA5 time index for a given datetime."""
    vt = np.datetime64(valid_time_dt)
    deltas = np.abs(era5_times - vt)
    idx = int(np.argmin(deltas))
    if deltas[idx] > np.timedelta64(3, "h"):
        return None   # no match within 3h
    return idx

def discrete_cmap_norm(vmin, vmax, n=12):
    """Discrete diverging colormap + BoundaryNorm for clean step coloring."""
    cmap  = plt.cm.get_cmap("RdBu_r", n)
    bounds = np.linspace(vmin, vmax, n + 1)
    norm  = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def save_rmse_csv(exp_name, lead_hours, rmse_dict, out_dir):
    """Save RMSE values to CSV."""
    df = pd.DataFrame({"lead_hour": lead_hours, **rmse_dict})
    csv_path = out_dir / f"rmse_{exp_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")
    return df

# ── Load land-sea mask (Aurora static.nc) ─────────────────────────────────────
print("Loading Aurora LSM ...")
static_ds = xr.open_dataset(str(DATA_DIR / "static.nc"), engine="netcdf4")
lsm_721  = static_ds["lsm"].values[0]   # (721, 1440) 0=ocean 1=land
lsm_aurora = lsm_721[:-1]               # (720, 1440) skip south-pole row
land_mask  = lsm_aurora > 0.5
ocean_mask = ~land_mask
# coastal: simple 2-step dilation boundary
from scipy.ndimage import binary_dilation
land_dil   = binary_dilation(land_mask,  iterations=4)
ocean_dil  = binary_dilation(ocean_mask, iterations=4)
coastal_mask = land_dil & ocean_dil
static_ds.close()
print(f"  Masks: land={land_mask.sum()}, ocean={ocean_mask.sum()}, coastal={coastal_mask.sum()}")

# Aurora lat grid (720, descending 90→-89.75)
aurora_lats = np.arange(90.0, -90.0, -0.25)    # 720 pts
aurora_lons = np.arange(0.0, 360.0, 0.25)       # 1440 pts
cos_w_aurora = cos_weights_1d(aurora_lats)       # (720,1)

# ── Load Aurora 1yr control ───────────────────────────────────────────────────
print("Loading Aurora 1yr control ...")
ctrl_ds   = nc4.Dataset(str(CTRL_FILE))
ctrl_lt   = ctrl_ds.variables["lead_time"][:].data  # hours, 12h steps
ctrl_lats = ctrl_ds.variables["lat"][:].data         # (720,)
# Day30/45/60 indices in 1yr file (lead_time=720,1080,1440h)
ctrl_d30  = int(np.where(ctrl_lt == DAY30_H)[0][0])
ctrl_d45  = int(np.where(ctrl_lt == DAY45_H)[0][0])
ctrl_d60  = int(np.where(ctrl_lt == DAY60_H)[0][0])
print(f"  Control indices day30={ctrl_d30}, day45={ctrl_d45}, day60={ctrl_d60}")

# ── RMSE storage (indexed by experiment name) ─────────────────────────────────
all_rmse = {}   # {exp_name: {varname: (240,) array}}

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1: Compute RMSE for T500 (used for temp experiments and as comparison)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Loading ERA5 T500 ===")
era5_t_times, era5_t500 = load_era5_var("t", PLOT_LEVEL, ERA5_PRES)
# era5_t500: (n_t, 721, 1440) → slice [:-1] for Aurora 720-lat grid
era5_t500_aurora = era5_t500[:, :-1, :]   # (n_t, 720, 1440)

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if ins_type != "temp":
        continue
    print(f"\n--- RMSE T500 for {exp_name} ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    lead_h = make_lead_times(nc_file)
    vtimes  = make_valid_times(lead_h, base=read_base_time(nc_file))
    rmse_t500 = np.full(N_STEPS, np.nan)

    with nc4.Dataset(str(nc_file)) as ds:
        t500_fc = ds.variables["t500"][:].data.astype(np.float32)  # (240,720,1440)

    for s in range(N_STEPS):
        ei = era5_idx(era5_t_times, vtimes[s])
        if ei is None:
            continue
        rmse_t500[s] = global_rmse(t500_fc[s], era5_t500_aurora[ei], cos_w_aurora)

    if exp_name not in all_rmse:
        all_rmse[exp_name] = {}
    all_rmse[exp_name]["T500"] = rmse_t500
    del t500_fc

# Also compute control T500 RMSE for injection period (days 30-60 = lead 720-1440h)
ctrl_t500 = ctrl_ds.variables["t500"][:, :, :].data.astype(np.float32)  # (730,720,1440)
print("\nComputing control T500 RMSE for injection period ...")
ctrl_t500_rmse_inj = []
for ci in range(ctrl_d30, ctrl_d60 + 1):
    h = ctrl_lt[ci]
    valid_dt = BASE_TIME + timedelta(hours=int(h))
    ei = era5_idx(era5_t_times, valid_dt)
    if ei is not None:
        ctrl_t500_rmse_inj.append(global_rmse(ctrl_t500[ci], era5_t500_aurora[ei], cos_w_aurora))
ctrl_t500_mean_rmse = np.mean(ctrl_t500_rmse_inj) if ctrl_t500_rmse_inj else np.nan
print(f"  Control T500 mean RMSE (inj period): {ctrl_t500_mean_rmse:.4f}")
del ctrl_t500
del era5_t500_aurora, era5_t500; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2: Compute RMSE for U500, V500 (UV experiments)
# ═══════════════════════════════════════════════════════════════════════════════
ctrl_u500_mean_rmse = np.nan
ctrl_v500_mean_rmse = np.nan
ctrl_spd500_mean_rmse = np.nan

for shortname in ["u", "v"]:
    print(f"\n=== Loading ERA5 {shortname}500 ===")
    era5_uv_times, era5_uv500 = load_era5_var(shortname, PLOT_LEVEL, ERA5_PRES)
    era5_uv500_aurora = era5_uv500[:, :-1, :]  # (n_t, 720, 1440)

    var_key = f"{shortname.upper()}500"
    for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
        if ins_type != "uv":
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        lead_h = make_lead_times(nc_file)
        vtimes  = make_valid_times(lead_h, base=read_base_time(nc_file))
        rmse_arr = np.full(N_STEPS, np.nan)

        fc_var_name = f"{shortname}500"
        with nc4.Dataset(str(nc_file)) as ds:
            fc_data = ds.variables[fc_var_name][:].data.astype(np.float32)

        for s in range(N_STEPS):
            ei = era5_idx(era5_uv_times, vtimes[s])
            if ei is None:
                continue
            rmse_arr[s] = global_rmse(fc_data[s], era5_uv500_aurora[ei], cos_w_aurora)

        if exp_name not in all_rmse:
            all_rmse[exp_name] = {}
        all_rmse[exp_name][var_key] = rmse_arr
        del fc_data

    # Control UV RMSE
    ctrl_uv500 = ctrl_ds.variables[f"{shortname}500"][:, :, :].data.astype(np.float32)
    ctrl_uv_rmse_inj = []
    for ci in range(ctrl_d30, ctrl_d60 + 1):
        h = ctrl_lt[ci]
        valid_dt = BASE_TIME + timedelta(hours=int(h))
        ei = era5_idx(era5_uv_times, valid_dt)
        if ei is not None:
            ctrl_uv_rmse_inj.append(global_rmse(ctrl_uv500[ci], era5_uv500_aurora[ei], cos_w_aurora))
    ctrl_uv_mean = np.mean(ctrl_uv_rmse_inj) if ctrl_uv_rmse_inj else np.nan
    if shortname == "u":
        ctrl_u500_mean_rmse = ctrl_uv_mean
    else:
        ctrl_v500_mean_rmse = ctrl_uv_mean
    del ctrl_uv500
    del era5_uv500_aurora, era5_uv500; gc.collect()

ctrl_spd500_mean_rmse = np.sqrt(
    ctrl_u500_mean_rmse**2 + ctrl_v500_mean_rmse**2) if not (
    np.isnan(ctrl_u500_mean_rmse) or np.isnan(ctrl_v500_mean_rmse)) else np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 3: Compute RMSE for T2M surface (temp experiments)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Loading ERA5 T2M (surface) ===")
era5_sfc_times, era5_t2m = load_era5_sfc("t2m", ERA5_SURF)
era5_t2m_aurora = era5_t2m[:, :-1, :] if era5_t2m is not None else None

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if ins_type != "temp" or era5_t2m_aurora is None:
        continue
    lead_h = make_lead_times(nc_file)
    vtimes  = make_valid_times(lead_h, base=read_base_time(nc_file))
    rmse_t2m = np.full(N_STEPS, np.nan)

    with nc4.Dataset(str(nc_file)) as ds:
        t2m_fc = ds.variables["2t"][:].data.astype(np.float32)

    for s in range(N_STEPS):
        ei = era5_idx(era5_sfc_times, vtimes[s])
        if ei is None:
            continue
        rmse_t2m[s] = global_rmse(t2m_fc[s], era5_t2m_aurora[ei], cos_w_aurora)

    all_rmse[exp_name]["T2M"] = rmse_t2m
    del t2m_fc

del era5_t2m_aurora, era5_t2m; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 4: Compute RMSE for U10M/V10M surface (UV experiments)
# ═══════════════════════════════════════════════════════════════════════════════
for era5_sfc_name, aurora_sfc_name, var_key in [("u10", "10u", "U10M"), ("v10", "10v", "V10M")]:
    print(f"\n=== Loading ERA5 {era5_sfc_name} (surface) ===")
    era5_sfc_times, era5_sfc_data = load_era5_sfc(era5_sfc_name, ERA5_SURF)
    if era5_sfc_data is None:
        continue
    era5_sfc_aurora = era5_sfc_data[:, :-1, :]

    for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
        if ins_type != "uv":
            continue
        lead_h = make_lead_times(nc_file)
        vtimes  = make_valid_times(lead_h, base=read_base_time(nc_file))
        rmse_arr = np.full(N_STEPS, np.nan)

        with nc4.Dataset(str(nc_file)) as ds:
            fc_data = ds.variables[aurora_sfc_name][:].data.astype(np.float32)

        for s in range(N_STEPS):
            ei = era5_idx(era5_sfc_times, vtimes[s])
            if ei is None:
                continue
            rmse_arr[s] = global_rmse(fc_data[s], era5_sfc_aurora[ei], cos_w_aurora)

        all_rmse[exp_name][var_key] = rmse_arr
        del fc_data

    del era5_sfc_aurora, era5_sfc_data; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# Save CSVs and plot RMSE time series
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Saving CSVs and RMSE time series plots ===")

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if exp_name not in all_rmse:
        continue
    lead_h = make_lead_times(nc_file)
    lead_days = lead_h / 24.0
    rmse_d = all_rmse[exp_name]

    # Save CSV
    df = pd.DataFrame({"lead_hour": lead_h, **rmse_d})
    csv_path = out_dir / f"rmse_{exp_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    # Plot RMSE time series
    fig, ax = plt.subplots(figsize=(12, 5))
    for var_key, rmse_vals in rmse_d.items():
        ax.plot(lead_days, rmse_vals, linewidth=2, label=var_key)
    ax.axvline(x=30, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Injection start (day 30)")
    ax.set_xlabel("Lead time (days from IC 2023-02-01 12:00 UTC)", fontsize=11)
    ax.set_ylabel("Cosine-lat-weighted global RMSE", fontsize=11)
    ax.set_title(f"Aurora  |  {LABEL_MAP.get(exp_name, exp_name)}  |  RMSE vs ERA5", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 60)
    fig.tight_layout()
    fig.savefig(out_dir / f"rmse_timeseries_{exp_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_dir}/rmse_timeseries_{exp_name}.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PROPAGATION MAPS — per-grid |error| vs ERA5 at day 30 / 45 / 60
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Generating propagation maps (per-grid RMSE vs ERA5) ===")
lons_2d, lats_2d = np.meshgrid(aurora_lons, aurora_lats)

INJ_START_IDX = 121   # first saved step with injection applied (step 121 in rollout, lead=732h)
INJ_START_H   = 732   # 122 × 6h

# These indices are read from the experiment files below (all 6 share the
# same corrected base_time), not from CTRL_FILE -- use the experiment-side
# reference time, not BASE_TIME (which is CTRL_FILE's, and 6h later).
EXP_BASE_TIME = read_base_time(EXPERIMENTS[0][1])
DAY_CONFIGS = [
    ("First Injection (Day 30+12h)", INJ_START_IDX, EXP_BASE_TIME + timedelta(hours=INJ_START_H)),
    ("Day 45", DAY45_IDX, EXP_BASE_TIME + timedelta(hours=DAY45_H)),
    ("Day 60", DAY60_IDX, EXP_BASE_TIME + timedelta(hours=DAY60_H)),
]

def rmse_cmap_norm(vmax, n=10):
    """Discrete sequential colormap: light (low RMSE) → dark (high RMSE)."""
    cmap   = plt.cm.get_cmap("YlOrRd", n)
    bounds = np.linspace(0.0, vmax, n + 1)
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

# Load ERA5 T500 and U500 — only 3 snapshots needed
print("  Reloading ERA5 T500 + U500 for map snapshots ...")
era5_tm, era5_t500m = load_era5_var("t", PLOT_LEVEL, ERA5_PRES)
era5_um, era5_u500m = load_era5_var("u", PLOT_LEVEL, ERA5_PRES)
t500_snaps = np.stack([era5_t500m[era5_idx(era5_tm, vt), :-1, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
u500_snaps = np.stack([era5_u500m[era5_idx(era5_um, vt), :-1, :] for _, _, vt in DAY_CONFIGS]).astype(np.float32)
del era5_t500m, era5_u500m; gc.collect()

for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    print(f"  Propagation map: {exp_name} ...")
    var_fc      = "t500" if ins_type == "temp" else "u500"
    var_label   = "T @ 500 hPa (K)" if ins_type == "temp" else "U @ 500 hPa (m/s)"
    truth_snaps = t500_snaps if ins_type == "temp" else u500_snaps

    with nc4.Dataset(str(nc_file)) as ds:
        exp_slices = np.stack([
            ds.variables[var_fc][idx].data.astype(np.float32) for _, idx, _ in DAY_CONFIGS
        ])  # (3, 720, 1440)

    errors = np.abs(exp_slices - truth_snaps)   # per-grid absolute error vs ERA5

    vmax = max(float(np.nanpercentile(errors, 97)), 0.1)
    vmax = round(vmax, 1)
    cmap, norm, bounds = rmse_cmap_norm(vmax, n=10)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7),
                              subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, (day_label, _, _), err in zip(axes, DAY_CONFIGS, errors):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="k")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="gray")
        im = ax.pcolormesh(lons_2d, lats_2d, err, cmap=cmap, norm=norm,
                           transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_title(day_label, fontsize=12, fontweight="bold")
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    fig.subplots_adjust(bottom=0.18, top=0.90, left=0.03, right=0.97, wspace=0.05)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_label(f"|Error| vs ERA5  —  {var_label}", fontsize=11)
    cbar.set_ticks(bounds[::2])
    fig.suptitle(
        f"Aurora  |  {LABEL_MAP.get(exp_name, exp_name)}  |  Per-Grid |Error| vs ERA5  ({var_label})",
        fontsize=13, fontweight="bold")
    out_path = out_dir / f"propagation_map_{exp_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")
    del exp_slices, errors

del t500_snaps, u500_snaps, era5_tm, era5_um; gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# % RMSE IMPROVEMENT BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Computing % RMSE improvement and bar chart ===")

improvements = {}
for exp_name, nc_file, out_dir, ins_type in EXPERIMENTS:
    if exp_name not in all_rmse:
        improvements[exp_name] = np.nan
        continue
    rmse_d = all_rmse[exp_name]

    if ins_type == "temp" and "T500" in rmse_d:
        exp_rmse_inj = np.nanmean(rmse_d["T500"][INJECT_STEP:])
        ctrl_ref     = ctrl_t500_mean_rmse
    elif ins_type == "uv" and "U500" in rmse_d and "V500" in rmse_d:
        exp_spd_rmse = np.sqrt(
            np.nanmean(rmse_d["U500"][INJECT_STEP:]) ** 2 +
            np.nanmean(rmse_d["V500"][INJECT_STEP:]) ** 2
        )
        exp_rmse_inj = exp_spd_rmse
        ctrl_ref     = ctrl_spd500_mean_rmse
    else:
        improvements[exp_name] = np.nan
        continue

    if ctrl_ref > 0 and not np.isnan(ctrl_ref):
        improvements[exp_name] = (ctrl_ref - exp_rmse_inj) / ctrl_ref * 100.0
    else:
        improvements[exp_name] = np.nan

print("  Improvements:", {k: f"{v:.2f}%" for k, v in improvements.items()})

# Save improvement CSV
imp_df = pd.DataFrame({"experiment": list(improvements.keys()),
                        "pct_rmse_improvement": list(improvements.values())})
imp_csv = CC_DIR / "rmse_improvement_aurora.csv"
imp_df.to_csv(imp_csv, index=False)
print(f"  Saved: {imp_csv}")

# Bar chart
fig, ax = plt.subplots(figsize=(11, 6))
names   = [LABEL_MAP.get(k, k) for k in improvements]
vals    = [improvements[k] for k in improvements]
colors  = [COLOR_MAP.get(k, "steelblue") for k in improvements]
bars = ax.bar(names, vals, color=colors, edgecolor="k", linewidth=0.7, width=0.6)
for bar, v in zip(bars, vals):
    if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.5 if v >= 0 else -1.5),
                f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_ylabel("% RMSE Improvement over Control (days 30–60)", fontsize=11)
ax.set_title("Aurora CC  |  % RMSE Improvement by Insertion Experiment\n"
             "(T500 for temp-ins, Wind speed 500 hPa for UV-ins)", fontsize=11)
ax.set_ylim(min(min(v for v in vals if not np.isnan(v)) - 5, -5),
            max(max(v for v in vals if not np.isnan(v)) + 5, 10))
plt.xticks(rotation=20, ha="right", fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout()
bar_path = CC_DIR / "rmse_improvement_aurora.png"
fig.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {bar_path}")

ctrl_ds.close()
print("\n=== Aurora OSSE analysis complete ===")
