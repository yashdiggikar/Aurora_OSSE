#!/usr/bin/env python3
"""
Section 5.8  Aurora Spatial Bias Analysis  (memory-efficient)
Mature phase: lead day 30 to end of run (steps 60–119).
Processes one Aurora run at a time to keep peak RAM ~3–4 GB.

Grid: 720×1440 (0.25°, 90°N to -89.75°; South Pole excluded)
IC: 2023-02-01 12:00 UTC
Control: first 120 steps of auroraoutput_1year_1stfeb23.nc
"""

import sys, gc, warnings, logging
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files")
ERA5_DIR  = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")

CTRL_FILE  = Path("/umbc/rs/pi_halem/users/yashd1/auroraoutput_1year_1stfeb23.nc")
TINS_FILE  = BASE / "auroraoutput_temp_inserted.nc"
UVINS_FILE = BASE / "auroraoutput_uv_inserted.nc"
ZINS_FILE  = BASE / "auroraoutput_geopotential_inserted.nc"

# Mature phase covers day 30–60 = steps 60–119 = times from 2023-03-02 12Z onward
# ERA5 files needed: March + April only
ERA5_MATURE_FILES = [
    ERA5_DIR / "era5_pressure_2023_03.grib",
    ERA5_DIR / "era5_pressure_2023_04.grib",
]

OUT_DIR  = BASE / "bias"
VAL_DIR  = OUT_DIR / "values"
PLOT_DIR = OUT_DIR / "plots"
LOG_DIR  = OUT_DIR / "logs"
for d in [VAL_DIR, PLOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "section_5_8_aurora_debug_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

log.info("=" * 72)
log.info("Section 5.8 Aurora Spatial Bias Analysis  [memory-efficient]")
log.info("=" * 72)

# ─── Aurora run map ───────────────────────────────────────────────────────────
AUR_RUNS = {
    "control":  (CTRL_FILE,  slice(0, 120)),
    "t_ins":    (TINS_FILE,  slice(None)),
    "uv_ins":   (UVINS_FILE, slice(None)),
    "z_ins":    (ZINS_FILE,  slice(None)),
}

UNIT_MAP  = {"t500": "K",      "u500": "m s⁻¹", "v500": "m s⁻¹",
             "z500": "m²s⁻²",  "wind500": "m s⁻¹"}
LABEL_MAP = {"t500": "T500",   "u500": "U500",   "v500": "V500",
             "z500": "Z500",   "wind500": "Wind500"}

saved_csvs  = []
saved_plots = []

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Identify grid and mature-phase indices
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 1: Grid and mature-phase indices")

# Aurora IC = 2023-02-01 12Z; lead_times stored as hours [0, 12, 24, …]
IC          = pd.Timestamp("2023-02-01 12:00:00")
lead_hours  = np.arange(0, 120 * 12, 12)             # 0, 12, …, 1428
aur_times   = pd.to_datetime([IC + pd.Timedelta(hours=int(h)) for h in lead_hours])
lead_days   = lead_hours / 24.0

NLAT, NLON = 720, 1440
aur_lat    = np.arange(90.0, -90.0, -0.25)           # 90.0 → -89.75
aur_lon    = np.linspace(0, 359.75, NLON)

mature_start = IC + pd.Timedelta(days=30)
mature_mask  = aur_times >= mature_start
mature_idx   = np.where(mature_mask)[0]
mature_times = aur_times[mature_idx]

log.info(f"  Grid: {NLAT}×{NLON}  lat: {aur_lat[0]:.2f}→{aur_lat[-1]:.2f}")
log.info(f"  IC: {IC}  |  mature start: {mature_start}")
log.info(f"  Mature indices: {mature_idx[0]}…{mature_idx[-1]}  "
         f"({len(mature_idx)} steps, "
         f"{mature_times[0]} → {mature_times[-1]})")

LAT2D, LON2D = np.meshgrid(aur_lat, aur_lon, indexing="ij")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Load ERA5 truth for mature phase (March + April only)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 2: Loading ERA5 truth (mature phase: March–April 2023)")

target_era5_times = mature_times


def load_era5_var(short_name: str, level: int, target_times) -> np.ndarray:
    """Load ERA5 at one pressure level, subset to target_times.
    Returns (n_times, 720, 1440) float32 aligned to Aurora lat/lon."""
    parts = []
    for f in ERA5_MATURE_FILES:
        try:
            ds = xr.open_dataset(
                str(f), engine="cfgrib",
                backend_kwargs={"filter_by_keys": {
                    "shortName": short_name,
                    "typeOfLevel": "isobaricInhPa",
                    "level": level,
                }},
            )
            drop_these = [c for c in ds.coords
                          if ds.coords[c].dims == () or
                          c in ("step", "number", "isobaricInhPa", "valid_time",
                                "surface", "heightAboveGround")]
            ds   = ds.drop_vars(drop_these, errors="ignore")
            dvar = list(ds.data_vars)[0]
            da   = ds[dvar]
            if "time" not in da.dims and "valid_time" in da.dims:
                da = da.rename({"valid_time": "time"})
            rmap = {}
            if "latitude" in da.dims: rmap["latitude"] = "lat"
            if "longitude" in da.dims: rmap["longitude"] = "lon"
            if rmap: da = da.rename(rmap)
            # Ensure descending lat order
            if da.lat.values[0] < da.lat.values[-1]:
                da = da.isel(lat=slice(None, None, -1))
            parts.append(da)
            log.info(f"    {short_name}@{level}: {Path(f).name} shape={da.shape}")
        except Exception as exc:
            log.warning(f"    SKIP {Path(f).name} for {short_name}: {exc}")

    combined = xr.concat(parts, dim="time").sortby("time")
    if combined.lat.size == 721:
        combined = combined.isel(lat=slice(0, 720))   # drop South Pole

    era5_pd  = pd.DatetimeIndex(combined.time.values)
    hit      = era5_pd.isin(target_times)
    selected = combined.isel(time=hit).sel(time=target_times)
    arr      = selected.values.astype(np.float32)
    log.info(f"    {short_name}@{level} selected: shape={arr.shape}  "
             f"range=[{arr.min():.4f}, {arr.max():.4f}]")
    return arr


log.info("  Loading T500 …")
era5_t500 = load_era5_var("t", 500, target_era5_times)
log.info("  Loading U500 …")
era5_u500 = load_era5_var("u", 500, target_era5_times)
log.info("  Loading V500 …")
era5_v500 = load_era5_var("v", 500, target_era5_times)
log.info("  Loading Z500 …")
era5_z500 = load_era5_var("z", 500, target_era5_times)

era5_wind500 = np.sqrt(era5_u500**2 + era5_v500**2).astype(np.float32)
log.info(f"  ERA5 wind500 range=[{era5_wind500.min():.3f}, {era5_wind500.max():.3f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load each Aurora run, compute mature-phase bias
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 3: Computing spatial bias for each Aurora run")


def load_aurora_mature(run_name: str, var_name: str) -> np.ndarray:
    fpath, sl = AUR_RUNS[run_name]
    with h5py.File(str(fpath), "r") as f:
        if sl == slice(0, 120):
            arr = f[var_name][mature_idx, :, :]
        else:
            arr = f[var_name][mature_idx, :, :]
    return arr.astype(np.float32)


def calc_bias(aur_arr, era5_arr):
    return np.nanmean(aur_arr - era5_arr, axis=0).astype(np.float64)


ERA5_TRUTH = {
    "t500":    era5_t500,
    "u500":    era5_u500,
    "v500":    era5_v500,
    "z500":    era5_z500,
    "wind500": era5_wind500,
}

bias = {}

for run_name in AUR_RUNS:
    log.info(f"\n  ── Run: {run_name} ──")
    bias[run_name] = {}
    for var in ["t500", "u500", "v500", "z500"]:
        log.info(f"    Loading {var} …")
        aur_arr = load_aurora_mature(run_name, var)
        b = calc_bias(aur_arr, ERA5_TRUTH[var])
        bias[run_name][var] = b
        log.info(f"    {var} bias: min={b.min():.4f}  max={b.max():.4f}  "
                 f"mean={b.mean():.4f}  MAB={np.abs(b).mean():.4f}")
        del aur_arr

    log.info(f"    Deriving wind500 …")
    aur_u  = load_aurora_mature(run_name, "u500")
    aur_v  = load_aurora_mature(run_name, "v500")
    aur_ws = np.sqrt(aur_u**2 + aur_v**2).astype(np.float32)
    del aur_u, aur_v
    b_ws = calc_bias(aur_ws, ERA5_TRUTH["wind500"])
    bias[run_name]["wind500"] = b_ws
    log.info(f"    wind500 bias: min={b_ws.min():.4f}  max={b_ws.max():.4f}  "
             f"mean={b_ws.mean():.4f}  MAB={np.abs(b_ws).mean():.4f}")
    del aur_ws
    gc.collect()

del era5_t500, era5_u500, era5_v500, era5_z500, era5_wind500
gc.collect()
log.info("\n  ERA5 arrays released.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Save individual bias CSVs
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 4: Saving bias CSVs")


def save_bias_csv(arr, run, var, suffix="bias"):
    fpath = str(VAL_DIR / f"Aurora_{run}_{var}_{suffix}.csv")
    pd.DataFrame({
        "lat": LAT2D.ravel(), "lon": LON2D.ravel(), suffix: arr.ravel()
    }).to_csv(fpath, index=False)
    log.info(f"  Saved: {fpath}")
    saved_csvs.append(fpath)


for run in AUR_RUNS:
    for var in bias[run]:
        save_bias_csv(bias[run][var], run, var, "bias")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Bias reduction  (|ctrl| - |ins|)
# ═══════════════════════════════════════════════════════════════════════════════
log.info("\nSTEP 5: Calculating bias reduction")

bias_red = {}

for run in ["t_ins", "uv_ins", "z_ins"]:
    bias_red[run] = {}
    for var in bias.get(run, {}):
        if var not in bias.get("control", {}): continue
        cb = bias["control"][var]
        ib = bias[run][var]
        br = np.abs(cb) - np.abs(ib)
        bias_red[run][var] = br
        improved = br.mean() > 0
        log.info(f"  [{run}] {var}: ctrl MAB={np.abs(cb).mean():.4f}  "
                 f"ins MAB={np.abs(ib).mean():.4f}  mean BR={br.mean():.4f}  "
                 f"→ {'↑ IMPROVED' if improved else '↓ WORSENED'}")
        fpath = str(VAL_DIR / f"Aurora_{run}_{var}_combined.csv")
        pd.DataFrame({
            "lat": LAT2D.ravel(), "lon": LON2D.ravel(),
            "control_bias":   cb.ravel(),
            "insertion_bias": ib.ravel(),
            "bias_reduction": br.ravel(),
        }).to_csv(fpath, index=False)
        saved_csvs.append(fpath)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
PROJ = ccrs.PlateCarree()
COL_TTLS = ["Control Bias", "Insertion Bias", "Bias Reduction  (+= improved)"]

mature_label = (f"Mature phase: {mature_times[0].strftime('%b %d')}–"
                f"{mature_times[-1].strftime('%b %d %Y')}  "
                f"({len(mature_times)} × 12-hr steps)")


def _panel(ax, data, lat, lon, title, vmax=None, cmap="RdBu_r"):
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(data), 97))
    vmax = max(vmax, 1e-9)
    im = ax.pcolormesh(lon, lat, data, transform=PROJ,
                       cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.5)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                 alpha=0.5, linestyle="--")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    return im


def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      pad=0.04, shrink=0.8, aspect=30)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        fp = str(PLOT_DIR / f"{stem}.{ext}")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        log.info(f"  Saved: {fp}")
        saved_plots.append(fp)
    plt.close(fig)

# ─── FIGURE 5.43 — Control mature-phase bias maps ────────────────────────────
log.info("\nFIGURE 5.43: Control run mature-phase bias maps")
plot_543 = [v for v in ["t500", "u500", "z500", "wind500"]
            if v in bias.get("control", {})]

fig, axes = plt.subplots(len(plot_543), 1,
                          figsize=(15, 4.5 * len(plot_543)),
                          subplot_kw={"projection": PROJ})
if len(plot_543) == 1: axes = [axes]

for ax, var in zip(axes, plot_543):
    b    = bias["control"][var]
    unit = UNIT_MAP[var]
    im   = _panel(ax, b, aur_lat, aur_lon,
                  f"Control – {LABEL_MAP[var]} Mature-Phase Bias  [{unit}]")
    _cbar(fig, ax, im, unit)

fig.suptitle(f"Figure 5.43 – Aurora Control Run: Mature-Phase Mean Bias\n{mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_43_Aurora_control_mature_phase_bias_maps")

# ─── FIGURE 5.44 — Temperature-insertion: U500 & Z500 ────────────────────────
log.info("\nFIGURE 5.44: Temperature-insertion bias reduction (U500, Z500)")
plot_544 = [v for v in ["u500", "z500"]
            if v in bias.get("control", {}) and v in bias.get("t_ins", {})]

fig, axes = plt.subplots(len(plot_544), 3,
                          figsize=(19, 4.5 * len(plot_544)),
                          subplot_kw={"projection": PROJ})
if len(plot_544) == 1: axes = axes[np.newaxis, :]

for row, var in enumerate(plot_544):
    cb_, ib_, br_ = bias["control"][var], bias["t_ins"][var], bias_red["t_ins"][var]
    unit = UNIT_MAP[var]; lbl = LABEL_MAP[var]
    vb   = max(float(np.nanpercentile(np.abs(cb_), 97)),
               float(np.nanpercentile(np.abs(ib_), 97)))
    vr   = float(np.nanpercentile(np.abs(br_), 97))
    im0  = _panel(axes[row,0], cb_, aur_lat, aur_lon, f"Control {lbl}  [{unit}]", vmax=vb)
    im1  = _panel(axes[row,1], ib_, aur_lat, aur_lon, f"T-Ins {lbl}  [{unit}]",   vmax=vb)
    im2  = _panel(axes[row,2], br_, aur_lat, aur_lon, f"{lbl} Bias Reduction  [{unit}]", vmax=vr)
    for im_, ax_ in zip([im0, im1, im2], axes[row]):
        _cbar(fig, ax_, im_, unit)

for ax_, ct in zip(axes[0], COL_TTLS):
    ax_.set_title(ct, fontsize=10, fontweight="bold", pad=6)

fig.suptitle(f"Figure 5.44 – Aurora Temperature-Insertion vs. Control\n"
             f"Mature-Phase Bias: U500 & Z500  |  {mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_44_Aurora_temperature_insertion_bias_reduction")

# ─── FIGURE 5.45 — Wind-insertion: T500 & Z500 ───────────────────────────────
log.info("\nFIGURE 5.45: Wind-insertion bias reduction (T500, Z500)")
plot_545 = [v for v in ["t500", "z500"]
            if v in bias.get("control", {}) and v in bias.get("uv_ins", {})]

fig, axes = plt.subplots(len(plot_545), 3,
                          figsize=(19, 4.5 * len(plot_545)),
                          subplot_kw={"projection": PROJ})
if len(plot_545) == 1: axes = axes[np.newaxis, :]

for row, var in enumerate(plot_545):
    cb_, ib_, br_ = bias["control"][var], bias["uv_ins"][var], bias_red["uv_ins"][var]
    unit = UNIT_MAP[var]; lbl = LABEL_MAP[var]
    vb   = max(float(np.nanpercentile(np.abs(cb_), 97)),
               float(np.nanpercentile(np.abs(ib_), 97)))
    vr   = float(np.nanpercentile(np.abs(br_), 97))
    im0  = _panel(axes[row,0], cb_, aur_lat, aur_lon, f"Control {lbl}  [{unit}]", vmax=vb)
    im1  = _panel(axes[row,1], ib_, aur_lat, aur_lon, f"UV-Ins {lbl}  [{unit}]",  vmax=vb)
    im2  = _panel(axes[row,2], br_, aur_lat, aur_lon, f"{lbl} Bias Reduction  [{unit}]", vmax=vr)
    for im_, ax_ in zip([im0, im1, im2], axes[row]):
        _cbar(fig, ax_, im_, unit)

for ax_, ct in zip(axes[0], COL_TTLS):
    ax_.set_title(ct, fontsize=10, fontweight="bold", pad=6)

fig.suptitle(f"Figure 5.45 – Aurora Wind-Insertion vs. Control\n"
             f"Mature-Phase Bias: T500 & Z500  |  {mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_45_Aurora_wind_insertion_bias_reduction")

# ─── FIGURE 5.46 — Geopotential-insertion: T500 & Wind500 ────────────────────
log.info("\nFIGURE 5.46: Geopotential-insertion bias reduction (T500, Wind500)")
plot_546 = [v for v in ["t500", "wind500"]
            if v in bias.get("control", {}) and v in bias.get("z_ins", {})]

fig, axes = plt.subplots(len(plot_546), 3,
                          figsize=(19, 4.5 * len(plot_546)),
                          subplot_kw={"projection": PROJ})
if len(plot_546) == 1: axes = axes[np.newaxis, :]

for row, var in enumerate(plot_546):
    cb_, ib_, br_ = bias["control"][var], bias["z_ins"][var], bias_red["z_ins"][var]
    unit = UNIT_MAP[var]; lbl = LABEL_MAP[var]
    vb   = max(float(np.nanpercentile(np.abs(cb_), 97)),
               float(np.nanpercentile(np.abs(ib_), 97)))
    vr   = float(np.nanpercentile(np.abs(br_), 97))
    im0  = _panel(axes[row,0], cb_, aur_lat, aur_lon, f"Control {lbl}  [{unit}]", vmax=vb)
    im1  = _panel(axes[row,1], ib_, aur_lat, aur_lon, f"Z-Ins {lbl}  [{unit}]",   vmax=vb)
    im2  = _panel(axes[row,2], br_, aur_lat, aur_lon, f"{lbl} Bias Reduction  [{unit}]", vmax=vr)
    for im_, ax_ in zip([im0, im1, im2], axes[row]):
        _cbar(fig, ax_, im_, unit)

for ax_, ct in zip(axes[0], COL_TTLS):
    ax_.set_title(ct, fontsize=10, fontweight="bold", pad=6)

fig.suptitle(f"Figure 5.46 – Aurora Geopotential-Insertion vs. Control\n"
             f"Mature-Phase Bias: T500 & Wind500  |  {mature_label}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
_save(fig, "Figure_5_46_Aurora_geopotential_insertion_bias_reduction")

# ─── FIGURE 5.47 — Failure-mode summary map ──────────────────────────────────
log.info("\nFIGURE 5.47: Failure-mode summary map")
diag_var = "z500" if "z500" in bias.get("control", {}) else "t500"
cb_diag  = bias["control"][diag_var]

t_impr = bias_red.get("t_ins",  {}).get(diag_var, np.zeros_like(cb_diag))
w_impr = bias_red.get("uv_ins", {}).get(diag_var, np.zeros_like(cb_diag))
g_impr = bias_red.get("z_ins",  {}).get(diag_var, np.zeros_like(cb_diag))

T_thresh = 0.10 * float(np.percentile(np.abs(cb_diag), 90))
log.info(f"  Classification threshold: {T_thresh:.4f}")

ny, nx    = cb_diag.shape
class_arr = np.zeros((ny, nx), dtype=np.int8)
stk       = np.stack([t_impr, w_impr, g_impr], axis=-1)
best_idx  = np.argmax(stk, axis=-1)
worst_idx = np.argmin(stk, axis=-1)
max_val   = stk.max(axis=-1)
min_val   = stk.min(axis=-1)

all_improve = (t_impr > T_thresh) & (w_impr > T_thresh) & (g_impr > T_thresh)
all_worsen  = (t_impr < -T_thresh) & (w_impr < -T_thresh) & (g_impr < -T_thresh)

class_arr[(best_idx == 0) & (max_val >  T_thresh)] = 3
class_arr[(best_idx == 1) & (max_val >  T_thresh)] = 4
class_arr[(best_idx == 2) & (max_val >  T_thresh)] = 5
class_arr[(worst_idx== 0) & (min_val < -T_thresh)] = 6
class_arr[(worst_idx== 1) & (min_val < -T_thresh)] = 7
class_arr[(worst_idx== 2) & (min_val < -T_thresh)] = 8
class_arr[all_improve] = 1
class_arr[all_worsen]  = 2

CLASS_LABELS = {
    0: "Neutral / mixed",
    1: "All insertions improve",
    2: "All insertions worsen",
    3: "Temp-insertion best",
    4: "Wind-insertion best",
    5: "Geo-insertion best",
    6: "Temp-insertion worst",
    7: "Wind-insertion worst",
    8: "Geo-insertion worst",
}

cos_lat = np.cos(np.deg2rad(aur_lat))
wt_2d   = np.tile(cos_lat[:, np.newaxis], (1, nx))
total_w = float(wt_2d.sum())
class_stats = {}
for cid, cname in CLASS_LABELS.items():
    pct = 100.0 * float(wt_2d[class_arr == cid].sum()) / total_w
    class_stats[cid] = {"name": cname, "pct": pct}
    log.info(f"  Class {cid} ({cname:30s}): {pct:6.2f}%")

pd.DataFrame({
    "lat": LAT2D.ravel(), "lon": LON2D.ravel(),
    "class_id":   class_arr.ravel(),
    "class_name": [CLASS_LABELS[c] for c in class_arr.ravel()],
}).to_csv(VAL_DIR / "Aurora_failure_mode_map.csv", index=False)

pd.DataFrame([
    {"class_id": cid, "class_name": v["name"], "area_pct": round(v["pct"], 4)}
    for cid, v in class_stats.items()
]).to_csv(VAL_DIR / "Aurora_failure_mode_area_summary.csv", index=False)

COLORS  = ["#b0b0b0", "#2ecc71", "#e74c3c",
           "#3498db", "#9b59b6", "#f39c12",
           "#1a5276", "#7d3c98", "#784212"]
cmap_fm = ListedColormap(COLORS)
norm_fm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap_fm.N)

fig, ax = plt.subplots(1, 1, figsize=(16, 7), subplot_kw={"projection": PROJ})
im = ax.pcolormesh(aur_lon, aur_lat, class_arr,
                   cmap=cmap_fm, norm=norm_fm, transform=PROJ, shading="auto")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS,   linewidth=0.3, linestyle=":")
ax.set_global()
ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
cb.set_ticks(range(9))
cb.set_ticklabels([f"{i}: {CLASS_LABELS[i]}" for i in range(9)], fontsize=7.5)
cb.set_label("Failure Mode Class", fontsize=9)
ax.set_title(
    f"Figure 5.47 – Aurora Insertion Failure-Mode Summary ({LABEL_MAP[diag_var]})\n"
    f"{mature_label}  |  threshold={T_thresh:.3f}",
    fontsize=11, fontweight="bold",
)
ax.text(0.01, 0.01,
        "\n".join([f"{v['name']}: {v['pct']:.1f}%" for v in class_stats.values()]),
        transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75))
plt.tight_layout()
_save(fig, "Figure_5_47_Aurora_failure_mode_summary_map")

# ─── Final summary ────────────────────────────────────────────────────────────
log.info("\n" + "=" * 72)
log.info("ALL SAVED CSV FILES:")
for f in saved_csvs: log.info(f"  {f}")
log.info("\nALL SAVED PLOT FILES:")
for f in sorted(saved_plots): log.info(f"  {f}")
log.info("\nBIAS-REDUCTION SUMMARY:")
for run in ["t_ins", "uv_ins", "z_ins"]:
    for var in bias_red.get(run, {}):
        br_mean  = float(bias_red[run][var].mean())
        ctrl_mab = float(np.abs(bias["control"][var]).mean())
        ins_mab  = float(np.abs(bias[run][var]).mean())
        log.info(f"  {run:12s} {LABEL_MAP[var]:8s}: "
                 f"ctrl MAB={ctrl_mab:.4f}  ins MAB={ins_mab:.4f}  "
                 f"mean BR={br_mean:+.4f}  "
                 f"{'↑ IMPROVED' if br_mean > 0 else '↓ WORSENED'}")
log.info("\n" + "=" * 72)
log.info("Section 5.8 Aurora spatial bias analysis completed successfully.")
log.info("=" * 72)
print("\nSection 5.8 Aurora spatial bias analysis completed successfully.")
