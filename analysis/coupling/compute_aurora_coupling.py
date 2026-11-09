#!/usr/bin/env python3
"""
Section 5.6 — Cross-Variable Coupling Analysis for Aurora
Coupling score C(A→B):
  For each pressure level of variable group B:
      pct_level = 100 × (RMSE_ctrl(B_level) − RMSE_ins_A(B_level)) / RMSE_ctrl(B_level)
  Average across all 13 pressure levels (mature phase day 30–60, steps 60–119):
      C(A→B) = mean(pct_level)

Variable groups: T, U, V, Z, Q  (13 levels: 1000–50 hPa each)
Insertion runs: t_ins (T), uv_ins (U+V), z_ins (Z)

Aurora files store each level as a named dataset (e.g. "t500").
RMSE is computed here directly (no pre-computed CSVs needed).

Grid: 720×1440  |  IC: 2023-02-01 12Z  |  120 steps × 12 hr
Control: first 120 steps of auroraoutput_1year_1stfeb23.nc
"""

import warnings; warnings.filterwarnings("ignore")
import gc
from pathlib import Path
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path("/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files")
ERA5DIR   = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data")
OUTDIR    = BASE / "coupling"
VALDIR    = OUTDIR / "values"
PLOTDIR   = OUTDIR / "plots"
LOGFILE   = OUTDIR / "logs" / "compute_aurora_coupling_log.txt"

for d in [VALDIR, PLOTDIR, OUTDIR / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

log = open(LOGFILE, "w", buffering=1)
def L(msg): print(msg); log.write(msg + "\n")

# ─── Aurora run files ─────────────────────────────────────────────────────────
CTRL_FILE  = Path("/umbc/rs/pi_halem/users/yashd1/auroraoutput_1year_1stfeb23.nc")
TINS_FILE  = BASE / "auroraoutput_temp_inserted.nc"
UVINS_FILE = BASE / "auroraoutput_uv_inserted.nc"
ZINS_FILE  = BASE / "auroraoutput_geopotential_inserted.nc"

AUR_FILES = {
    "Control": (CTRL_FILE,  slice(0, 120)),
    "temp_ins": (TINS_FILE,  slice(None)),
    "wind_ins": (UVINS_FILE, slice(None)),
    "geo_ins":  (ZINS_FILE,  slice(None)),
}

ERA5_PRES = [ERA5DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]

# ─── Time axis ────────────────────────────────────────────────────────────────
AURORA_IC    = datetime(2023, 2, 1, 12, 0)
N_STEPS      = 120
aur_times    = [AURORA_IC + timedelta(hours=12 * k) for k in range(N_STEPS)]
lead_days    = np.array([0.5 * k for k in range(N_STEPS)])
MATURE_STEPS = np.where(lead_days >= 30)[0]   # steps 60–119

# ─── Grid ─────────────────────────────────────────────────────────────────────
NLAT, NLON = 720, 1440
lats_1d    = np.arange(90.0, -90.0, -0.25)
cos_w_2d   = np.cos(np.deg2rad(lats_1d))[:, None] * np.ones((1, NLON))
total_w    = cos_w_2d.sum()

# ─── Variable groups and levels ───────────────────────────────────────────────
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
GROUPS = {
    "T": {"vars": [f"t{l}" for l in LEVELS], "label": "Temperature\n(T)",
          "unit": "K",      "color": "#e74c3c"},
    "U": {"vars": [f"u{l}" for l in LEVELS], "label": "Zonal Wind\n(U)",
          "unit": "m s⁻¹",  "color": "#3498db"},
    "V": {"vars": [f"v{l}" for l in LEVELS], "label": "Meridional Wind\n(V)",
          "unit": "m s⁻¹",  "color": "#2980b9"},
    "Z": {"vars": [f"z{l}" for l in LEVELS], "label": "Geopotential\n(Z)",
          "unit": "m²s⁻²", "color": "#f39c12"},
    "Q": {"vars": [f"q{l}" for l in LEVELS], "label": "Specific Humidity\n(Q)",
          "unit": "kg kg⁻¹", "color": "#27ae60"},
}

INS_RUNS = {
    "temp_ins": {
        "label":   "Temperature Insertion\n(All T levels → ERA5)",
        "short":   "T-Ins",
        "color":   "#c0392b",
        "trivial": ["T"],
    },
    "wind_ins": {
        "label":   "Wind Insertion\n(All U+V levels → ERA5)",
        "short":   "Wind-Ins",
        "color":   "#2471a3",
        "trivial": ["U", "V"],
    },
    "geo_ins": {
        "label":   "Geopotential Insertion\n(All Z levels → ERA5)",
        "short":   "Geo-Ins",
        "color":   "#d68910",
        "trivial": ["Z"],
    },
}

# ─── ERA5 loading (for RMSE computation) ─────────────────────────────────────
def load_era5_group_all_levels(shortname, target_times):
    """Load all 13 pressure levels for one variable group in a single grib open per month.
    Returns dict: level(int) -> np.ndarray(N_STEPS, 720, 1440) float32.
    One file open per month instead of 13 opens per month.
    """
    result = {lev: np.full((N_STEPS, NLAT, NLON), np.nan, dtype=np.float32)
              for lev in LEVELS}

    for fpath in ERA5_PRES:
        if not fpath.exists(): continue
        L(f"    [{shortname}] opening {fpath.name} (all levels) …")
        try:
            ds = xr.open_dataset(str(fpath), engine="cfgrib",
                                 filter_by_keys={"typeOfLevel": "isobaricInhPa",
                                                 "shortName": shortname},
                                 backend_kwargs={"indexpath": ""})
        except Exception as e:
            L(f"      SKIP {fpath.name}: {e}"); continue

        era5_ts  = pd.to_datetime(ds.time.values)
        lat_name = "latitude" if "latitude" in ds.dims else "lat"
        lev_name = "isobaricInhPa" if "isobaricInhPa" in ds.dims else "level"
        lat_vals = ds[lat_name].values
        flip     = lat_vals[0] < lat_vals[-1]

        varname  = list(ds.data_vars)[0]
        raw      = ds[varname].values.astype(np.float32)   # (time, level, lat, lon)
        lev_vals = ds[lev_name].values.astype(int)
        ds.close(); gc.collect()

        if flip:
            raw = raw[:, :, ::-1, :]          # flip lat axis
        if raw.shape[2] == 721:
            raw = raw[:, :, :720, :]           # drop South Pole

        for li, lev in enumerate(lev_vals):
            if lev not in result: continue
            for k, tgt in enumerate(target_times):
                if not np.all(np.isnan(result[lev][k])): continue
                deltas = np.abs(era5_ts - pd.Timestamp(tgt))
                idx    = deltas.argmin()
                if deltas[idx].total_seconds() <= 3600:
                    result[lev][k] = raw[idx, li, :, :]

        del raw; gc.collect()

    matched = {lev: int(np.sum(~np.all(np.isnan(result[lev]), axis=(1,2))))
               for lev in LEVELS}
    L(f"    [{shortname}] ERA5 matched: {matched}")
    return result


def load_aurora_var(run_key, var_name):
    fpath, sl = AUR_FILES[run_key]
    with h5py.File(str(fpath), "r") as f:
        arr = f[var_name][sl, :, :]
    return arr.astype(np.float32)


def global_rmse_ts(aur_arr, era5_arr):
    sq   = (aur_arr - era5_arr) ** 2
    rmse = np.sqrt((sq * cos_w_2d).sum(axis=(1, 2)) / total_w)
    return rmse.astype(np.float64)

# ─── Compute per-level RMSE for all runs and all variable groups ──────────────
L("=" * 70)
L("AURORA COUPLING ANALYSIS")
L("IC=2023-02-01 12Z  |  120 steps  |  Mature phase = steps 60–119")
L("=" * 70)

# ERA5 shortname map for each group variable name
ERA5_SHORT = {
    "t": "t", "u": "u", "v": "v", "z": "z", "q": "q"
}

# rmse_dict[run_key][var_name] = array shape (N_STEPS,)
rmse_dict = {rk: {} for rk in AUR_FILES}

L("\nComputing RMSE time series for all groups and levels …")

for gkey, gmeta in GROUPS.items():
    L(f"\n  Group {gkey}:")
    era5_short = gkey.lower()   # t, u, v, z, q

    era5_levels = load_era5_group_all_levels(era5_short, aur_times)

    for var_name in gmeta["vars"]:
        level = int(var_name[1:])   # e.g. "t500" → 500
        L(f"    {var_name} (lev={level})")

        era5_arr = era5_levels[level]
        matched = int(np.sum(~np.all(np.isnan(era5_arr), axis=(1, 2))))
        L(f"      ERA5 matched {matched}/{N_STEPS} steps")

        for run_key in AUR_FILES:
            aur_arr  = load_aurora_var(run_key, var_name)
            rmse_ts  = global_rmse_ts(aur_arr, era5_arr)
            rmse_dict[run_key][var_name] = rmse_ts
            mature_m = rmse_ts[MATURE_STEPS].mean()
            L(f"      {run_key:15s} mature RMSE={mature_m:.5f}")
            del aur_arr; gc.collect()

    del era5_levels; gc.collect()

L("\nAll RMSE time series computed.")

# ─── Compute coupling matrix ──────────────────────────────────────────────────
L("\n" + "=" * 70)
L("COUPLING MATRIX  (avg % improvement per level, mature phase day 30–60)")
L("=" * 70)

ctrl_mature = {var: rmse_dict["Control"][var][MATURE_STEPS].mean()
               for rk in ["Control"] for var in rmse_dict["Control"]}

pct_per_level = {}   # (ins_key, gkey) → array(13,)
coupling_avg  = {}   # (ins_key, gkey) → scalar

for ins_key, ins_meta in INS_RUNS.items():
    L(f"\n  {ins_meta['short']}")
    for gkey, gmeta in GROUPS.items():
        pcts = []
        for var_name in gmeta["vars"]:
            ctrl_m = ctrl_mature.get(var_name, np.nan)
            ins_m  = rmse_dict[ins_key][var_name][MATURE_STEPS].mean()
            if ctrl_m > 1e-8:
                pcts.append(100.0 * (ctrl_m - ins_m) / ctrl_m)
            else:
                pcts.append(np.nan)
        arr = np.array(pcts)
        pct_per_level[(ins_key, gkey)] = arr
        avg = float(np.nanmean(arr))
        coupling_avg[(ins_key, gkey)] = avg
        trivial_mark = " [TRIVIAL]" if gkey in ins_meta["trivial"] else ""
        L(f"    → {gkey}: {avg:6.1f}%{trivial_mark}")

# ─── Save coupling matrix CSV ──────────────────────────────────────────────────
rows = []
for ins_key, ins_meta in INS_RUNS.items():
    for gkey, gmeta in GROUPS.items():
        rows.append({
            "insertion_run":   ins_key,
            "insertion_short": ins_meta["short"],
            "response_group":  gkey,
            "response_label":  gmeta["label"].replace("\n", " "),
            "coupling_score":  coupling_avg[(ins_key, gkey)],
            "trivial":         gkey in ins_meta["trivial"],
        })
df_coupling = pd.DataFrame(rows)
df_coupling.to_csv(VALDIR / "Aurora_coupling_matrix.csv", index=False)
L("\nSaved: Aurora_coupling_matrix.csv")

# ─── Save per-level detail CSV ─────────────────────────────────────────────────
level_rows = []
for ins_key, ins_meta in INS_RUNS.items():
    for gkey in GROUPS:
        arr = pct_per_level[(ins_key, gkey)]
        for i, lev in enumerate(LEVELS):
            level_rows.append({
                "insertion_run":   ins_key,
                "insertion_short": ins_meta["short"],
                "response_group":  gkey,
                "pressure_hPa":    lev,
                "coupling_score":  None if np.isnan(arr[i]) else arr[i],
                "trivial":         gkey in ins_meta["trivial"],
            })
pd.DataFrame(level_rows).to_csv(VALDIR / "Aurora_coupling_per_level.csv", index=False)
L("Saved: Aurora_coupling_per_level.csv")

# ─── Save summary table ────────────────────────────────────────────────────────
summary_rows = []
for ins_key, ins_meta in INS_RUNS.items():
    non_triv = [(gkey, coupling_avg[(ins_key, gkey)])
                for gkey in GROUPS
                if gkey not in ins_meta["trivial"]]
    non_triv.sort(key=lambda x: x[1], reverse=True)
    summary_rows.append({
        "model":         "Aurora",
        "insertion":     ins_meta["short"],
        "top_A_B":       non_triv[0][0]  if non_triv       else "—",
        "top_score":     round(non_triv[0][1], 1)  if non_triv else np.nan,
        "second_A_B":    non_triv[1][0]  if len(non_triv)>1 else "—",
        "second_score":  round(non_triv[1][1], 1)  if len(non_triv)>1 else np.nan,
        "weakest_A_B":   non_triv[-1][0] if non_triv       else "—",
        "weakest_score": round(non_triv[-1][1], 1) if non_triv else np.nan,
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(VALDIR / "Aurora_coupling_summary.csv", index=False)
L("Saved: Aurora_coupling_summary.csv")
L(df_summary.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5.30 — Main Coupling Heatmap
# ──────────────────────────────────────────────────────────────────────────────
L("\nPlotting Figure 5.30 …")

ins_keys = list(INS_RUNS.keys())
grp_keys = list(GROUPS.keys())
n_ins    = len(ins_keys)
n_grp    = len(grp_keys)

matrix     = np.array([[coupling_avg[(ir, gk)] for gk in grp_keys] for ir in ins_keys])
is_trivial = np.array([[gk in INS_RUNS[ir]["trivial"] for gk in grp_keys] for ir in ins_keys])

cmap = LinearSegmentedColormap.from_list("coupling", [
    (0.00, "#8B0000"), (0.25, "#D9534F"), (0.45, "#FAEBD7"),
    (0.50, "#FFFFFF"), (0.55, "#D5E8D4"), (0.75, "#28A745"), (1.00, "#145A32")
])
norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=100)

fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[14, 1], wspace=0.04)
ax  = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])

non_triv_mat = np.where(is_trivial, np.nan, matrix)
im = ax.imshow(non_triv_mat, cmap=cmap, norm=norm, aspect="auto",
               interpolation="nearest", zorder=1)

for i in range(n_ins):
    for j in range(n_grp):
        if is_trivial[i, j]:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                facecolor="#E0E0E0", edgecolor="white", lw=2.5, zorder=2))
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                facecolor="none", hatch="////", edgecolor="#AAAAAA", lw=0.6, zorder=3))
            ax.text(j, i, "Self-inserted\n(100% trivial)",
                    ha="center", va="center", fontsize=8.5,
                    color="#666666", fontstyle="italic", zorder=4)

for i in range(n_ins):
    for j in range(n_grp):
        if is_trivial[i, j]: continue
        val = matrix[i, j]
        if np.isnan(val): continue
        normed  = norm(val)
        txt_col = "white" if (normed < 0.28 or normed > 0.72) else "#111111"
        weight  = "bold"  if val >= 50 else "normal"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=14, fontweight=weight, color=txt_col, zorder=5)

for i in range(n_ins):
    for j in range(n_grp):
        edge = "#CC0000" if (not is_trivial[i,j] and matrix[i,j] < 0) else "#CCCCCC"
        lw   = 2.0      if (not is_trivial[i,j] and matrix[i,j] < 0) else 1.0
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
            facecolor="none", edgecolor=edge, lw=lw, zorder=6))

for x in np.arange(-0.5, n_grp, 1): ax.axvline(x, color="white", lw=2.5, zorder=7)
for y in np.arange(-0.5, n_ins, 1): ax.axhline(y, color="white", lw=2.5, zorder=7)

ax.set_xticks(range(n_grp))
ax.set_xticklabels([GROUPS[gk]["label"] + f"\n[{GROUPS[gk]['unit']}]"
                    for gk in grp_keys], fontsize=12, fontweight="bold")
ax.set_yticks(range(n_ins))
ax.set_yticklabels([INS_RUNS[ik]["label"] for ik in ins_keys],
                   fontsize=12, fontweight="bold")
ax.tick_params(length=0, pad=8)
ax.set_xlabel("Response Variable  (B — downstream field measured)",
              fontsize=13, fontweight="bold", labelpad=12)
ax.set_ylabel("Inserted Variable  (A — field forced to ERA5 truth)",
              fontsize=13, fontweight="bold", labelpad=12)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Coupling Score  C(A→B)  [%]",
               fontsize=11, fontweight="bold", labelpad=12)
cbar.set_ticks([-10, 0, 20, 40, 60, 80, 100])
cbar.ax.tick_params(labelsize=10)

trivial_patch = mpatches.Patch(facecolor="#E0E0E0", hatch="////",
    edgecolor="#AAAAAA", label="Self-insertion — variable forced = ERA5 truth")
ax.legend(handles=[trivial_patch], loc="lower right",
          fontsize=9.5, framealpha=0.95, edgecolor="#bbbbbb")

ax.set_title(
    "Figure 5.30 — Aurora: Cross-Variable Coupling Matrix\n"
    r"$C(A \rightarrow B)\ =\ 100\times"
    r"\frac{\overline{RMSE}_{ctrl}(B)\ -\ \overline{RMSE}_{ins\,A}(B)}"
    r"{\overline{RMSE}_{ctrl}(B)}$"
    "     averaged across all 13 pressure levels  |  Mature phase (day 30–60)",
    fontsize=12, fontweight="bold", pad=14
)
fig.savefig(str(PLOTDIR / "Figure_5_30_Aurora_coupling_heatmap.png"),
            dpi=220, bbox_inches="tight")
L("Saved: Figure_5_30_Aurora_coupling_heatmap.png")
plt.close(fig)

# ─── Coupling Bar Chart ────────────────────────────────────────────────────────
L("Plotting coupling bar chart …")

bar_w = 0.24
x     = np.arange(n_grp)
fig, ax = plt.subplots(figsize=(14, 6.5))

for k, (ins_key, ins_meta) in enumerate(INS_RUNS.items()):
    vals   = [np.nan if gk in ins_meta["trivial"] else coupling_avg[(ins_key, gk)]
              for gk in grp_keys]
    offset = (k - 1) * bar_w
    bars   = ax.bar(x + offset, vals, bar_w,
                    color=ins_meta["color"], label=ins_meta["short"],
                    edgecolor="white", linewidth=0.9, zorder=3, alpha=0.92)
    for bar_obj, v in zip(bars, vals):
        if np.isnan(v): continue
        ypos = v + 1.5 if v >= 0 else v - 3.5
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, ypos,
                f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold", color="#111111")

for k, (ins_key, ins_meta) in enumerate(INS_RUNS.items()):
    offset = (k - 1) * bar_w
    for j, gk in enumerate(grp_keys):
        if gk in ins_meta["trivial"]:
            ax.text(j + offset, 2.5, "✕", ha="center", va="bottom",
                    fontsize=11, color="#AAAAAA", fontweight="bold")

ax.axhline(0,  color="black",   lw=1.2, zorder=4)
ax.axhline(50, color="#888888", lw=0.9, ls="--", zorder=2, alpha=0.6)
ax.text(n_grp - 0.5, 51.5, "50% threshold", fontsize=9,
        color="#888888", ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels([GROUPS[gk]["label"] + f"\n[{GROUPS[gk]['unit']}]"
                    for gk in grp_keys], fontsize=12, fontweight="bold")
ax.set_ylabel("Coupling Score  C(A→B)  [% RMSE reduction]",
              fontsize=12, fontweight="bold")
ax.set_xlabel("Response Variable  B", fontsize=12, fontweight="bold", labelpad=8)
ax.tick_params(axis="y", labelsize=10)
ax.set_ylim(-5, 112)
ax.grid(axis="y", alpha=0.3, lw=0.6, zorder=0)
ax.legend(title="Insertion Experiment  (A)", title_fontsize=10.5,
          fontsize=11, loc="upper right", framealpha=0.95, edgecolor="#bbbbbb")
ax.set_title(
    "Aurora — Cross-Variable Coupling Scores (Non-Trivial Only)\n"
    "Average % RMSE reduction across all 13 pressure levels  |  "
    "Mature phase (day 30–60)  |  ✕ = self-inserted variable",
    fontsize=11.5, fontweight="bold", pad=10
)
fig.tight_layout(pad=1.8)
fig.savefig(str(PLOTDIR / "Figure_Aurora_coupling_bar.png"),
            dpi=220, bbox_inches="tight")
L("Saved: Figure_Aurora_coupling_bar.png")
plt.close(fig)

# ─── Per-level Coupling Profiles ──────────────────────────────────────────────
L("Plotting per-level coupling profiles …")

fig, axes = plt.subplots(1, n_grp, figsize=(22, 7),
                          gridspec_kw={"wspace": 0.28})

for j, gkey in enumerate(grp_keys):
    ax    = axes[j]
    gmeta = GROUPS[gkey]
    for ins_key, ins_meta in INS_RUNS.items():
        if gkey in ins_meta["trivial"]: continue
        arr = pct_per_level[(ins_key, gkey)]
        ax.plot(arr, LEVELS, color=ins_meta["color"], lw=2.2, ls="-",
                label=ins_meta["short"], marker="o", markersize=4, zorder=4)

    ax.axvline(0,  color="black",   lw=1.0, zorder=3)
    ax.axvline(50, color="#888888", lw=0.8, ls="--", alpha=0.6, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(1050, 45)
    ax.set_yticks(LEVELS)
    ax.set_yticklabels([str(l) for l in LEVELS], fontsize=8.5)
    ax.set_xlabel("Coupling Score  C(A→B)  [%]", fontsize=10, fontweight="bold")
    if j == 0:
        ax.set_ylabel("Pressure Level  [hPa]", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, which="major", alpha=0.3, lw=0.5)
    ax.set_title(f"Response: {gmeta['label'].replace(chr(10), ' ')}\n[{gmeta['unit']}]",
                 fontsize=11, fontweight="bold", pad=6, color=gmeta["color"])
    if j == 0:
        ax.legend(title="Inserted Variable", title_fontsize=9.5,
                  fontsize=10, loc="lower right", framealpha=0.95)

fig.suptitle(
    "Aurora — Coupling Score Vertical Profiles\n"
    "Each panel: one response variable  |  y-axis: pressure level  |  "
    "x-axis: % RMSE reduction  |  Mature phase (day 30–60)",
    fontsize=12, fontweight="bold", y=1.02
)
fig.savefig(str(PLOTDIR / "Figure_Aurora_coupling_vertical_profiles.png"),
            dpi=220, bbox_inches="tight")
L("Saved: Figure_Aurora_coupling_vertical_profiles.png")
plt.close(fig)

# ─── Table 5.8 — Coupling Summary ─────────────────────────────────────────────
L("Plotting Table 5.8 …")

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis("off")
hdrs = ["Insertion\nExperiment", "Levels\nInserted",
        "Strongest\nCoupling A→B", "Score\n(%)",
        "Second\nCoupling A→B", "Score\n(%)",
        "Weakest\nCoupling A→B", "Score\n(%)"]
levels_inserted = {
    "temp_ins":  "T: 1000–50 hPa",
    "wind_ins":  "U+V: 1000–50 hPa",
    "geo_ins":   "Z: 1000–50 hPa",
}
tdata = []
for _, r in df_summary.iterrows():
    ins_key = [k for k in INS_RUNS if INS_RUNS[k]["short"] == r["insertion"]][0]
    tdata.append([
        r["insertion"],   levels_inserted[ins_key],
        r["top_A_B"],     f"{r['top_score']:.1f}",
        r["second_A_B"],  f"{r['second_score']:.1f}",
        r["weakest_A_B"], f"{r['weakest_score']:.1f}",
    ])

tbl = ax.table(cellText=tdata, colLabels=hdrs,
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)

for j in range(len(hdrs)):
    c = tbl[0, j]; c.set_facecolor("#2c3e50")
    c.set_text_props(color="white", fontweight="bold")

row_bg = ["#fef9f0", "#f0f4fb", "#f0faf4"]
for i in range(len(INS_RUNS)):
    for j in range(len(hdrs)):
        tbl[i+1, j].set_facecolor(row_bg[i])

score_cols = [3, 5, 7]
for i in range(len(INS_RUNS)):
    for sc in score_cols:
        try:
            val = float(tdata[i][sc])
            if val >= 65:    tbl[i+1, sc].set_facecolor("#c3e6cb")
            elif val >= 40:  tbl[i+1, sc].set_facecolor("#fff3cd")
            else:            tbl[i+1, sc].set_facecolor("#f8d7da")
        except ValueError:
            pass

ax.set_title("Table 5.8 — Aurora: Cross-Variable Coupling Summary\n"
             "(Average % RMSE improvement across all 13 pressure levels, mature phase day 30–60)",
             fontsize=11.5, fontweight="bold", pad=14)
fig.tight_layout(pad=1.5)
fig.savefig(str(PLOTDIR / "Table_5_8_Aurora_coupling_summary.png"),
            dpi=220, bbox_inches="tight")
L("Saved: Table_5_8_Aurora_coupling_summary.png")
plt.close(fig)

L("\n" + "=" * 70)
L("DONE — all Aurora coupling CSVs and figures written.")
L("=" * 70)
log.close()
