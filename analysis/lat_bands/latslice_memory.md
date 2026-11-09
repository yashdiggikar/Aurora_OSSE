# Aurora Latitude-Slice RMSE Analysis — Memory File
## Section 5.7 (fine-grained): 4 specific latitude slices

---

## Overview

Latitude-slice variant of the band analysis: computes RMSE in narrow (±5°) bands
centred on 4 representative latitudes rather than broad dynamical bands.
Provides finer spatial resolution of insertion impacts.

Slice centres: 2°N (deep tropics), 14°N (subtropics), 40°N (midlat storm track), 60°N (subpolar)
Each slice: ±5° around centre, NH only

---

## Directory Structure

```
Aurora_Pred_Files/lat_bands/
├── values/
│   ├── Aurora_{run}_{var}_latslice_rmse.csv   — 4 runs × 5 vars = 20 CSVs
│   └── Aurora_latslice_pct_improvement_summary.csv
├── plots/
│   ├── Figure_LS1_Aurora_latslice_setup.png
│   ├── Figure_LS2_Aurora_t_insertion_latslice_rmse.png
│   ├── Figure_LS3_Aurora_wind_insertion_latslice_rmse.png
│   ├── Figure_LS4_Aurora_z_insertion_latslice_rmse.png
│   └── Figure_LS5_Aurora_latslice_bar_summary.png
├── logs/
│   └── section_5_7_aurora_latslice_debug_log.txt
├── section_5_7_aurora_latslice_analysis.py
├── latslice_memory.md   ← this file
└── section_5_7_latslice_results_analysis.txt
```

---

## Slice Definitions

| Slice | Lat Range | Regime |
|-------|-----------|--------|
| 2°N | -3° to 7° | Deep tropics (ITCZ) |
| 14°N | 9° to 19° | Sub-tropics / monsoon belt |
| 40°N | 35° to 45° | Midlatitude storm track |
| 60°N | 55° to 65° | Subpolar / jet exit |

---

## Figure Map

| Figure | Description |
|--------|-------------|
| LS.1 | Slice setup map (NH bands + SH mirror) |
| LS.2 | T-insertion RMSE: U500 & Z500, 4 slices |
| LS.3 | UV-insertion RMSE: T500 & Z500, 4 slices |
| LS.4 | Z-insertion RMSE: T500 & Wind500, 4 slices |
| LS.5 | Bar chart: % improvement per slice per insertion |

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/lat_bands/section_5_7_aurora_latslice_analysis.py
```
