# Aurora Global NRMSE Analysis — Memory File
## Reference for the global NRMSE analysis for Aurora

---

## Overview

Computes cosine-latitude-weighted **global RMSE** time series for Aurora
(4 runs: control, t_ins, uv_ins, z_ins) vs ERA5, then normalises by the
**ERA5 temporal standard deviation** to produce dimensionless NRMSE.

IC: 2023-02-01 12:00 UTC  |  120 steps × 0.5 day = 60 days  |  Grid: 720×1440 (0.25°)

---

## Critical Differences from FCN

| Property | FCN | Aurora |
|----------|-----|--------|
| Grid | 721×1440 | 720×1440 (no South Pole at -90°) |
| IC | 2023-02-01 00Z | 2023-02-01 12Z |
| File format | h5py channels | h5py named variables (e.g. `t500`) |
| Wind insertion | `wind_ins` | `uv_ins` |
| Control file | `forecast_normal_60days_12hr.nc` | First 120 steps of `auroraoutput_1year_1stfeb23.nc` |

**Note on control file:** `auroraoutput_60day_12hr.nc` is a truncated write (18 GB instead of expected 34 GB, same HDF5 superblock). The 1-year file starts at the same IC (2023-02-01 12Z, unix=1675252800) and shares the same variable structure; steps 0–119 are the Aurora control run.

**ERA5 lat alignment:** ERA5 grib files return 721 lat rows (90 to -90 including South Pole). Aurora has 720 rows (90 to -89.75). Code drops `arr[:, :720, :]` to align.

---

## Directory Structure

```
Aurora_Pred_Files/nrmse/
├── values/
│   ├── Aurora_{var}_global_rmse.csv    # 5 files — step, lead_day, control, t_ins, uv_ins, z_ins
│   ├── Aurora_{var}_global_nrmse.csv   # 5 files — same, RMSE/σ
│   └── Aurora_era5_std_normalization.csv
├── logs/
│   └── compute_aurora_nrmse_log.txt
├── compute_aurora_nrmse.py
├── nrmse_results_analysis.txt
└── nrmse_memory.md   ← this file
```

---

## Normalization Formula

```
NRMSE(t) = RMSE(t) / σ_ERA5(var)
```

σ_ERA5 = cosine-lat weighted global mean of ERA5 temporal std over 120 steps.
NRMSE < 1.0 = skillful. NRMSE > 1.0 = worse than climatological variance.

---

## ERA5 Files

```
era5_pressure_2023_{02,03,04}.grib  →  /umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/
era5_surface_2023_{02,03,04}.grib
```

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/nrmse/compute_aurora_nrmse.py
```

Runtime: ~45–60 min. Output: 11 CSVs (5 RMSE + 5 NRMSE + 1 normalization).

---

## Variables Computed

| Key | ERA5 Source | Aurora Dataset |
|-----|-------------|----------------|
| t500 | pressure 't' lev 500 | `t500` |
| u500 | pressure 'u' lev 500 | `u500` |
| z500 | pressure 'z' lev 500 | `z500` |
| wind500 | sqrt(u²+v²) lev 500 | sqrt(`u500`²+`v500`²) |
| wind10 | surface '10u','10v' | sqrt(`10u`²+`10v`²) |

---

## Insertion Activation Note

Steps 0–59 (day 0–29.5): all 4 runs may be very similar.
Steps 60–119 (day 30–60): insertion runs diverge — this is the **mature phase**.
