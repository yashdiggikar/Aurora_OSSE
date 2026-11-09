#!/usr/bin/env python3
"""
Spectral blurring diagnostic for the Aurora Correlated OSSE Perturbation
experiments (T_PERTURBED_CORR, UV_PERTURBED_CORR).

See ../SPECTRAL_BLURRING_ANALYSIS.md for the full method description and
rationale — same method as FCN/compute_spectral_blur_corr.py, adapted to
Aurora's grid (720x1440, no south-pole row) and lead_time-indexed output
(no channel dimension, each level its own top-level NetCDF variable).

Full variable/level grid (matching this project's other "vs truth" scripts,
per user request): T, U, V, Z, Q (specific humidity) at 300/500/1000 hPa,
plus surface T2M/U10/V10 -- computed for BOTH experiments regardless of
which family was directly inserted (cross-variable coupling included). No
ERA5 truth needed here (experiment-minus-control, both Aurora's own output).

Output per experiment (into PERTURBATION_EXPERIMENTS/Aurora/<experiment>/):
    spectral_blur_<experiment>.csv
    spectral_blur_summary_<experiment>.csv
    spectral_blur_map_<experiment>_<band>.png

Run:
    source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
    python compute_spectral_blur_corr.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import netCDF4 as nc4
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = Path("/umbc/rs/pi_halem/users/yashd1")
CC_DIR    = BASE / "CC_aurora"
CTRL_FILE = BASE / "auroraoutput_1year_1stfeb23.nc"
OUT_DIR   = BASE / "PERTURBATION_EXPERIMENTS" / "Aurora"

EXPERIMENTS = [
    ("T_PERTURBED_CORR",  CC_DIR / "T_PERTURBED_CORR"  / "output" / "aurora_t_perturbed_corr.nc",  "temp"),
    ("UV_PERTURBED_CORR", CC_DIR / "UV_PERTURBED_CORR" / "output" / "aurora_uv_perturbed_corr.nc", "uv"),
]

BASE_TIME      = datetime(2023, 2, 1, 12, 0)
SNAPSHOT_STEPS = [120, 140, 160, 180, 200, 220, 239]
LEVELS = [300, 500, 1000]

AURORA_LATS = np.arange(90.0, -90.0, -0.25)   # 720 pts
AURORA_LONS = np.arange(0.0, 360.0, 0.25)
GRID_DEG = 0.25

BANDS = {"equator_5N": 5.0, "midlat_45N": 45.0, "highlat_70N": 70.0}
BAND_ROWS = {name: int(np.argmin(np.abs(AURORA_LATS - lat))) for name, lat in BANDS.items()}
WINDOW_DEG = 5.0

UPPER_PREFIXES = {"T": "t", "U": "u", "V": "v", "Z": "z", "Q": "q"}
SFC_VARNAMES = {"T2M": "2t", "U10": "10u", "V10": "10v"}
CH_NAMES = {f"{label}{lev}": f"{prefix}{lev}" for label, prefix in UPPER_PREFIXES.items() for lev in LEVELS}
CH_NAMES.update(SFC_VARNAMES)
ALL_LABELS = list(CH_NAMES)


def zonal_power_spectrum(field2d, row_center, window_rows):
    r0 = max(0, row_center - window_rows)
    r1 = min(field2d.shape[0], row_center + window_rows + 1)
    powers = []
    for r in range(r0, r1):
        line = field2d[r, :] - field2d[r, :].mean()
        fft = np.fft.rfft(line)
        powers.append(np.abs(fft) ** 2)
    return np.mean(powers, axis=0)


def effective_resolution_wavenumber(power, frac=0.01):
    ref = power[1]
    if ref <= 0:
        return np.nan
    below = np.where(power[1:] <= frac * ref)[0]
    return float(below[0] + 1) if len(below) else np.nan


for exp_name, nc_file, ins_type in EXPERIMENTS:
    out_dir = OUT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_file.exists():
        print(f"[{exp_name}] SKIP — output not yet present at {nc_file} (job still running?).")
        continue

    print(f"\n=== {exp_name} ===")
    exp_ds = nc4.Dataset(str(nc_file))
    ctrl_ds = nc4.Dataset(str(CTRL_FILE))

    # The control file (auroraoutput_1year_1stfeb23.nc) is 12-hourly/730
    # steps over ~1 year, NOT the same cadence/length as the 6-hourly/240-
    # step experiment output -- raw step-index alignment silently pulls the
    # WRONG valid time. Match by actual valid datetime instead, same pattern
    # already proven correct in compute_global_rmse_corr.py.
    #
    # Base time is read per-file, not assumed from the shared BASE_TIME
    # constant: exp_ds (CC_aurora "new family") had a +6h base_time
    # mislabeling bug and was patched to store the true reference time under
    # "base_time"; ctrl_ds (the old auroraoutput_1year_1stfeb23.nc) never had
    # that bug and stores its correct reference time under "time" instead.
    def _base_time_of(ds):
        var = "base_time" if "base_time" in ds.variables else "time"
        return datetime.utcfromtimestamp(int(ds.variables[var][...] if ds.variables[var].ndim == 0
                                              else ds.variables[var][0]))

    exp_base_time = _base_time_of(exp_ds)
    ctrl_base_time = _base_time_of(ctrl_ds)
    exp_valid_times = [exp_base_time + timedelta(hours=int(h))
                        for h in exp_ds.variables["lead_time"][:].data]
    ctrl_lead_h = ctrl_ds.variables["lead_time"][:].data
    ctrl_dts_np = np.array([np.datetime64(ctrl_base_time + timedelta(hours=int(h))) for h in ctrl_lead_h])

    spectrum_rows, summary_rows = [], []
    band_figs = {b: plt.subplots(figsize=(7, 5)) for b in BANDS}
    cmap = plt.get_cmap("viridis")
    first_label_plotted = None

    for label in ALL_LABELS:
        varname = CH_NAMES[label]
        if varname not in exp_ds.variables or varname not in ctrl_ds.variables:
            print(f"  [{exp_name}] {label}: variable unavailable, skipping.")
            continue
        if first_label_plotted is None:
            first_label_plotted = label

        for si, step in enumerate(SNAPSHOT_STEPS):
            vt = np.datetime64(exp_valid_times[step])
            deltas = np.abs(ctrl_dts_np - vt)
            ci = int(np.argmin(deltas))
            if deltas[ci] > np.timedelta64(6, "h"):
                print(f"  [{exp_name}] {label} step {step}: no control valid time within tolerance, skipping.")
                continue
            exp_field = np.asarray(exp_ds.variables[varname][step, :, :])
            ctrl_field = np.asarray(ctrl_ds.variables[varname][ci, :, :])
            diff = exp_field - ctrl_field

            for band_name, row in BAND_ROWS.items():
                window_rows = int(WINDOW_DEG / GRID_DEG)
                power = zonal_power_spectrum(diff, row, window_rows)
                wavenumbers = np.arange(len(power))
                for wn, p in zip(wavenumbers, power):
                    spectrum_rows.append({"variable": label, "step": step, "band": band_name,
                                           "wavenumber": int(wn), "power": float(p)})
                eff_res = effective_resolution_wavenumber(power)
                summary_rows.append({"variable": label, "step": step, "band": band_name,
                                      "effective_resolution_wavenumber": eff_res})

                fig, ax = band_figs[band_name]
                color = cmap(si / max(1, len(SNAPSHOT_STEPS) - 1))
                ax.loglog(wavenumbers[1:], power[1:], color=color, lw=1.2,
                          label=f"{label} step {step}" if label == first_label_plotted else None)

        print(f"  [{exp_name}] {label}: spectral blur computed for {len(SNAPSHOT_STEPS)} lead times x {len(BANDS)} bands")

        pd.DataFrame(spectrum_rows).to_csv(out_dir / f"spectral_blur_{exp_name}.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out_dir / f"spectral_blur_summary_{exp_name}.csv", index=False)

    for band_name, (fig, ax) in band_figs.items():
        ax.set_xlabel("zonal wavenumber (cycles / global ring)")
        ax.set_ylabel("power")
        ax.set_title(f"{exp_name}: zonal power spectrum, band={band_name}\n(experiment - control, color = lead time)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3, which="both")
        fig.savefig(out_dir / f"spectral_blur_map_{exp_name}_{band_name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {out_dir / f'spectral_blur_{exp_name}.csv'}")
    print(f"  Saved: {out_dir / f'spectral_blur_summary_{exp_name}.csv'}")
    exp_ds.close()
    ctrl_ds.close()

print("\nDone.")
