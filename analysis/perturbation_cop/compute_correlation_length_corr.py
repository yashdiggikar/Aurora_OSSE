#!/usr/bin/env python3
"""
Empirical correlation-length verification for the Aurora Correlated OSSE
Perturbation experiments (T_PERTURBED_CORR, UV_PERTURBED_CORR).

See ../CORRELATION_LENGTH_VERIFICATION.md for the full method description
and rationale — this script follows the exact same method as
FCN/compute_correlation_length_corr.py, adapted to Aurora's grid (720x1440,
no south-pole row) and its saved-output convention (each level is its own
top-level NetCDF variable, no `channel` dimension to index through).

Full variable grid (matching this project's other 9 "vs truth" scripts, per
user request): T, U, V, Z, Q (specific humidity) -- computed for BOTH
experiments regardless of which family was directly inserted (cross-variable
coupling included), at 3 representative pressure levels (300/500/1000 hPa)
for the horizontal check, all 13 native levels for the vertical check.

Output per experiment (into PERTURBATION_EXPERIMENTS/Aurora/<experiment>/):
    corr_length_horizontal_<experiment>.csv
    corr_length_vertical_<experiment>.csv
    corr_length_horizontal_map_<experiment>.png
    corr_length_vertical_map_<experiment>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_correlation_length_corr.py
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

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
CC_DIR    = BASE / "CC_aurora"
ERA5_DIR  = BASE / "AURORA_DATA" / "rmse_data"
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "Aurora"

EXPERIMENTS = [
    ("T_PERTURBED_CORR",  CC_DIR / "T_PERTURBED_CORR"  / "output" / "aurora_t_perturbed_corr.nc",  "temp"),
    ("UV_PERTURBED_CORR", CC_DIR / "UV_PERTURBED_CORR" / "output" / "aurora_uv_perturbed_corr.nc", "uv"),
]

BASE_TIME     = datetime(2023, 2, 1, 12, 0)
TARGET_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # hPa
REPR_HPA      = [300, 500, 1000]   # representative levels for the horizontal check
INJECT_STEP   = 120
MAX_LAG_PX    = 40
VERT_L_LOGP   = 0.5

ERA5_PRES = [ERA5_DIR / f"era5_pressure_2023_{m:02d}.grib" for m in [2, 3, 4]]
ERA5_SURF = [ERA5_DIR / f"era5_surface_2023_{m:02d}.grib"  for m in [2, 3, 4]]

AURORA_LATS = np.arange(90.0, -90.0, -0.25)   # 720 pts, no south-pole row
AURORA_LONS = np.arange(0.0, 360.0, 0.25)
GRID_DEG = 0.25
KM_PER_DEG_LAT = 111.0

BANDS = {"equator_5N": 5.0, "midlat_45N": 45.0, "highlat_70N": 70.0}
BAND_ROWS = {name: int(np.argmin(np.abs(AURORA_LATS - lat))) for name, lat in BANDS.items()}

# Full variable grid: label -> (ERA5 shortName == Aurora's own saved variable prefix, derived-non-divergent flag)
UPPER_VARS = {"T": ("t", False), "U": ("u", True), "V": ("v", True), "Z": ("z", False), "Q": ("q", False)}
# label -> (Aurora's own saved surface variable name, ERA5 surface shortName)
SFC_VARS = {"T2M": ("2t", "t2m"), "U10": ("10u", "u10"), "V10": ("10v", "v10")}


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


def load_era5_full_levels(shortname, grib_files, level_list=TARGET_LEVELS):
    """Loop over levels with per-level filtering (proven fast/reliable
    pattern already used elsewhere in this project) rather than one
    unfiltered multi-level open (tried first, found slow/unreliable)."""
    per_level_times = None
    data_by_level = []
    good_levels = []
    for lv in level_list:
        dsets = []
        try:
            for f in grib_files:
                if not f.exists():
                    continue
                dsets.append(xr.open_dataset(str(f), engine="cfgrib",
                             filter_by_keys={"shortName": shortname, "typeOfLevel": "isobaricInhPa",
                                              "level": int(lv)}, indexpath=None))
            if not dsets:
                raise RuntimeError("no grib files opened")
            combined = xr.concat(dsets, dim="time").sortby("time")
            if per_level_times is None:
                per_level_times = combined.time.values
            data_by_level.append(combined[shortname].values.astype(np.float32)[:, :-1, :])   # drop south-pole row
            good_levels.append(lv)
        except Exception as e:
            print(f"    WARNING: level {lv}hPa unavailable for '{shortname}' ({e!r}), skipping this level.")
        finally:
            for ds in dsets: ds.close()
    if not data_by_level:
        return None, [], None
    data = np.stack(data_by_level, axis=1)   # (time, level, lat, lon)
    return per_level_times, good_levels, data


def load_era5_sfc(shortname, grib_files):
    dsets = []
    for f in grib_files:
        if not f.exists():
            continue
        dsets.append(xr.open_dataset(str(f), engine="cfgrib", indexpath=None))
    if not dsets:
        return None, None
    combined = xr.concat(dsets, dim="time").sortby("time")
    times = combined.time.values
    data = combined[shortname].values.astype(np.float32)[:, :-1, :]
    for ds in dsets: ds.close()
    return times, data


def nearest_time_idx(times, target_dt, tol_hours=3):
    vt = np.datetime64(target_dt)
    deltas = np.abs(times - vt)
    idx = int(np.argmin(deltas))
    return idx if deltas[idx] <= np.timedelta64(tol_hours, "h") else None


def lag_correlation_1d(line, max_lag, wrap):
    n = len(line)
    line = line - line.mean()
    denom = np.sum(line ** 2)
    corrs = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if wrap:
            shifted = np.roll(line, -k)
            corrs[k] = np.sum(line * shifted) / denom if denom > 0 else np.nan
        else:
            a, b = (line, line) if k == 0 else (line[:n - k], line[k:])
            corrs[k] = np.sum(a * b) / denom if denom > 0 else np.nan
    return corrs


def efold_length_km(lags_km, corrs):
    target = 1.0 / np.e
    below = np.where(corrs <= target)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return 0.0
    x0, x1 = lags_km[i - 1], lags_km[i]
    y0, y1 = corrs[i - 1], corrs[i]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def horizontal_check(field2d, label, level_hpa, derived_field, out_rows, ax, plot_it):
    n_lat = field2d.shape[0]
    for band_name, row in BAND_ROWS.items():
        lat = AURORA_LATS[row]
        zonal_line = field2d[row, :]
        zonal_corr = lag_correlation_1d(zonal_line, MAX_LAG_PX, wrap=True)
        km_per_px_zonal = KM_PER_DEG_LAT * np.cos(np.deg2rad(lat)) * GRID_DEG
        zonal_lags_km = np.arange(MAX_LAG_PX + 1) * km_per_px_zonal
        L_zonal = efold_length_km(zonal_lags_km, zonal_corr)

        col = 0
        merid_col = field2d[max(0, row - MAX_LAG_PX):min(n_lat, row + MAX_LAG_PX + 1), col]
        merid_corr = lag_correlation_1d(merid_col[len(merid_col) // 2:], MAX_LAG_PX, wrap=False)
        km_per_px_merid = KM_PER_DEG_LAT * GRID_DEG
        merid_lags_km = np.arange(len(merid_corr)) * km_per_px_merid
        L_merid = efold_length_km(merid_lags_km, merid_corr)

        for direction, lags_km, corr, L in [
            ("zonal", zonal_lags_km, zonal_corr, L_zonal),
            ("meridional", merid_lags_km, merid_corr, L_merid),
        ]:
            out_rows.append({
                "variable": label, "level_hpa": level_hpa, "band": band_name, "latitude_deg": lat,
                "direction": direction, "derived_field": derived_field,
                "empirical_L_km": L, "target_L_km": 250.0,
                "ratio_empirical_over_target": L / 250.0 if not np.isnan(L) else np.nan,
            })
            if ax is not None and plot_it and band_name == "midlat_45N":
                ax.plot(lags_km, corr, marker="o", ms=3,
                        label=f"{label} {direction} (L≈{L:.0f} km)" if not np.isnan(L) else f"{label} {direction}")


def vertical_check(cube, levels_hpa, label, out_rows):
    n = len(levels_hpa)
    flat = cube.reshape(n, -1)
    emp_corr = np.corrcoef(flat)
    logp = np.log(levels_hpa)
    theo_corr = np.exp(-np.abs(logp[:, None] - logp[None, :]) / VERT_L_LOGP)
    for i in range(n):
        for j in range(n):
            out_rows.append({
                "variable": label, "level_i": levels_hpa[i], "level_j": levels_hpa[j],
                "empirical_corr": emp_corr[i, j], "theoretical_corr": theo_corr[i, j],
                "diff": emp_corr[i, j] - theo_corr[i, j],
            })


for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    lead_h = make_lead_times(nc_file)
    valid_times = make_valid_times(lead_h, base=read_base_time(nc_file))
    inject_dt = valid_times[INJECT_STEP]
    exp_ds = nc4.Dataset(str(nc_file))

    horiz_rows, vert_rows = [], []
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    theta = np.linspace(0, MAX_LAG_PX * KM_PER_DEG_LAT * GRID_DEG, 200)
    ax_h.plot(theta, np.exp(-theta / 250.0), "k--", lw=1, label="theoretical exp(-d/250km)")
    ax_h.plot(theta, np.exp(-theta / (250.0 * np.sqrt(2))), "k:", lw=1,
              label="theoretical exp(-d/(250*sqrt(2))km)")

    def read_level_cube(prefix, levels):
        cube = np.empty((len(levels), *AURORA_LATS.shape, len(AURORA_LONS)), dtype=np.float32)
        for i, lv in enumerate(levels):
            varname = f"{prefix}{lv}"
            if varname not in exp_ds.variables:
                cube[i] = np.nan
                continue
            cube[i] = exp_ds.variables[varname][INJECT_STEP, :, :]
        return cube

    # Every upper-air variable/level is evaluated for every experiment (not
    # just the directly-inserted family) -- cross-variable coupling.
    for label, (era5_sn, derived) in UPPER_VARS.items():
        era5_times, era5_levels, era5_cube_all = load_era5_full_levels(era5_sn, ERA5_PRES)
        if era5_times is None:
            print(f"  [{exp_name}] {label}: no ERA5 data available at any level, skipping.")
            continue
        ti = nearest_time_idx(era5_times, inject_dt)
        if ti is None:
            print(f"  [{exp_name}] {label}: no ERA5 truth near injection time, skipping.")
            continue
        truth_cube = era5_cube_all[ti]

        exp_cube = read_level_cube(era5_sn, era5_levels)
        pert_cube = exp_cube - truth_cube

        for hpa in REPR_HPA:
            if hpa not in era5_levels:
                continue
            lvl_idx = era5_levels.index(hpa)
            horizontal_check(pert_cube[lvl_idx], label, hpa, derived, horiz_rows, ax_h, plot_it=(hpa == 500))
        if len(era5_levels) >= 2:
            vertical_check(pert_cube, era5_levels, label, vert_rows)
        print(f"  [{exp_name}] {label}: horizontal ({[h for h in REPR_HPA if h in era5_levels]}hPa) "
              f"+ vertical ({len(era5_levels)} levels) checks done")

        pd.DataFrame(horiz_rows).to_csv(out_dir / f"corr_length_horizontal_{exp_name}.csv", index=False)
        pd.DataFrame(vert_rows).to_csv(out_dir / f"corr_length_vertical_{exp_name}.csv", index=False)

    # sfc_exp_varname = Aurora's OWN saved-output surface variable name
    # (2t/10u/10v); sfc_era5_name = the ERA5 truth-side name (t2m/u10/v10) --
    # these are two different naming conventions and must NOT be the same
    # string (this was a real, silent bug fixed earlier this project: using
    # "t2m" for both meant `sfc_varname in exp_ds.variables` was always
    # False since Aurora's own output uses "2t").
    for sfc_label, (sfc_exp_varname, sfc_era5_name) in SFC_VARS.items():
        if sfc_exp_varname not in exp_ds.variables:
            print(f"  [{exp_name}] {sfc_label} (surface): '{sfc_exp_varname}' not in experiment output, skipping.")
            continue
        sfc_exp = np.asarray(exp_ds.variables[sfc_exp_varname][INJECT_STEP, :, :])
        sfc_times, sfc_truth_all = load_era5_sfc(sfc_era5_name, ERA5_SURF)
        if sfc_times is None:
            print(f"  [{exp_name}] {sfc_label} (surface): no ERA5 surface truth available, skipping.")
            continue
        sfc_ti = nearest_time_idx(sfc_times, inject_dt)
        if sfc_ti is not None:
            sfc_pert = sfc_exp - sfc_truth_all[sfc_ti]
            horizontal_check(sfc_pert, sfc_label, None, False, horiz_rows, None, plot_it=False)
            print(f"  [{exp_name}] {sfc_label} (surface): horizontal check done")

    pd.DataFrame(horiz_rows).to_csv(out_dir / f"corr_length_horizontal_{exp_name}.csv", index=False)
    pd.DataFrame(vert_rows).to_csv(out_dir / f"corr_length_vertical_{exp_name}.csv", index=False)

    ax_h.set_xlabel("lag distance (km)"); ax_h.set_ylabel("correlation")
    ax_h.set_title(f"{exp_name}: horizontal lag-correlation at 45N, level=500hPa")
    ax_h.legend(fontsize=7); ax_h.grid(alpha=0.3)
    fig_h.savefig(out_dir / f"corr_length_horizontal_map_{exp_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig_h)

    if vert_rows:
        df_v = pd.DataFrame(vert_rows)
        first_var = df_v["variable"].iloc[0]
        sub = df_v[df_v["variable"] == first_var]
        levels_sorted = sorted(sub["level_i"].unique())
        n = len(levels_sorted)
        emp_m = sub.pivot(index="level_i", columns="level_j", values="empirical_corr").loc[levels_sorted, levels_sorted].values
        theo_m = sub.pivot(index="level_i", columns="level_j", values="theoretical_corr").loc[levels_sorted, levels_sorted].values
        fig_v, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for a, m, title in zip(axes, [emp_m, theo_m, emp_m - theo_m],
                                [f"empirical ({first_var})", "theoretical", "difference"]):
            im = a.imshow(m, vmin=-1 if title != "difference" else -0.3,
                          vmax=1 if title != "difference" else 0.3, cmap="RdBu_r")
            a.set_title(title, fontsize=10)
            a.set_xticks(range(n)); a.set_xticklabels(levels_sorted, rotation=90, fontsize=6)
            a.set_yticks(range(n)); a.set_yticklabels(levels_sorted, fontsize=6)
            plt.colorbar(im, ax=a, shrink=0.8)
        fig_v.suptitle(f"{exp_name}: vertical correlation matrix, levels in hPa")
        fig_v.savefig(out_dir / f"corr_length_vertical_map_{exp_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig_v)

    print(f"  Saved: {out_dir / f'corr_length_horizontal_{exp_name}.csv'}")
    print(f"  Saved: {out_dir / f'corr_length_vertical_{exp_name}.csv'}")
    exp_ds.close()

print("\nDone.")
