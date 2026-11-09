# Aurora Latitude-Band RMSE Analysis — Memory File
## Section 5.7: Latitude-Band / Dynamical-Regime Analysis

---

## Overview

Computes cosine-latitude-weighted RMSE time series for 3 broad dynamical bands
(Tropics, Midlatitudes, Polar) and generates Figures 5.35–5.39.

IC: 2023-02-01 12Z  |  120 steps × 12 hr = 60 days
Mature phase: day 30–60 (steps 60–119)

---

## Latitude Band Definitions

| Band | Lat Ranges |
|------|-----------|
| Tropics | -30° to 30° |
| Midlatitudes | -60° to -30° AND 30° to 60° |
| Polar | -90° to -60° AND 60° to 90° |

Slice centres (for Figure 5.35 display): 2°N, 14°N, 40°N, 60°N

---

## Directory Structure

```
Aurora_Pred_Files/lats/
├── values/
│   ├── Aurora_{run}_{var}_latband_rmse.csv   — 4 runs × 5 vars = 20 CSVs
│   └── Aurora_latband_pct_improvement_summary.csv
├── plots/
│   ├── Figure_5_35_Aurora_latband_setup.png
│   ├── Figure_5_36_Aurora_t_insertion_latband_rmse.png
│   ├── Figure_5_37_Aurora_wind_insertion_latband_rmse.png
│   ├── Figure_5_38_Aurora_z_insertion_latband_rmse.png
│   └── Figure_5_39_Aurora_latband_bar_summary.png
├── logs/
│   └── section_5_7_aurora_latband_debug_log.txt
├── section_5_7_aurora_latband_analysis.py
├── lats_memory.md   ← this file
├── section_5_7_methods_and_interpretation.txt
└── section_5_7_theoretical_analysis.txt
```

---

## Figure Map

| Figure | Description |
|--------|-------------|
| 5.35 | Latitude-band setup map (shaded regions + slice centres) |
| 5.36 | T-insertion RMSE vs lead day: U500 & Z500 in 3 bands |
| 5.37 | UV-insertion RMSE vs lead day: T500 & Z500 in 3 bands |
| 5.38 | Z-insertion RMSE vs lead day: T500 & Wind500 in 3 bands |
| 5.39 | Grouped bar chart: % RMSE improvement by band, all insertions |

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/lats/section_5_7_aurora_latband_analysis.py
```

---

## Key Notes

- All RMSE values are cosine-latitude weighted within each band
- Mature phase shaded in gray (day 30–60) on time series panels
- Figure 5.39 shows both initial (day 0–30, hatched) and mature (day 30–60) phases
- uv_ins = Wind insertion (U+V together, same as FCN wind_ins but renamed for Aurora)
