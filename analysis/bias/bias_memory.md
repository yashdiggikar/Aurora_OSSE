# Aurora Spatial Bias Analysis — Memory File
## Section 5.8: Spatial Bias Diagnosis

---

## Overview

Computes mature-phase (day 30–60) mean bias maps (model − ERA5) for 4 Aurora runs,
then calculates bias reduction for each insertion type vs control.
Generates Figures 5.43–5.47.

IC: 2023-02-01 12Z  |  Mature phase: day 30–60 (steps 60–119, 60 maps)
Mature-phase ERA5: March + April 2023 only (February not needed for mature phase)

---

## Key Difference from FCN

| Property | FCN | Aurora |
|----------|-----|--------|
| Grid | 721×1440 | 720×1440 |
| Mature start | 2023-03-03 00Z | 2023-03-02 12Z (IC+30days) |
| ERA5 lat rows | 721 (includes -90°) | 720 (drops South Pole row) |
| Wind insertion label | `wind_ins` | `uv_ins` |

---

## Directory Structure

```
Aurora_Pred_Files/bias/
├── values/
│   ├── Aurora_control_{var}_bias.csv          — 5 var × 1 run = 5 files
│   ├── Aurora_{ins}_{var}_bias.csv            — 5 var × 3 ins = 15 files
│   ├── Aurora_{ins}_{var}_combined.csv        — control_bias, insertion_bias, bias_reduction
│   ├── Aurora_failure_mode_map.csv
│   └── Aurora_failure_mode_area_summary.csv
├── plots/
│   ├── Figure_5_43_Aurora_control_mature_phase_bias_maps.png/.pdf
│   ├── Figure_5_44_Aurora_temperature_insertion_bias_reduction.png/.pdf
│   ├── Figure_5_45_Aurora_wind_insertion_bias_reduction.png/.pdf
│   ├── Figure_5_46_Aurora_geopotential_insertion_bias_reduction.png/.pdf
│   └── Figure_5_47_Aurora_failure_mode_summary_map.png/.pdf
├── logs/
│   └── section_5_8_aurora_debug_log.txt
├── section_5_8_aurora_bias_analysis.py
├── bias_memory.md   ← this file
├── section_5_8_methods_and_interpretation.txt
└── section_5_8_theoretical_analysis.txt
```

---

## Bias Definition

```
bias(lat, lon) = mean_t( Aurora(lat,lon,t) − ERA5(lat,lon,t) )
```
Averaged over all 60 mature-phase steps.

Bias reduction = |control_bias| − |insertion_bias|
  Positive = insertion reduces absolute bias (improved)
  Negative = insertion increases absolute bias (failure mode)

---

## Failure Mode Classification (Figure 5.47)

Based on Z500 bias reduction (diagnostic variable):
  0 = Neutral/mixed
  1 = All 3 insertions improve
  2 = All 3 insertions worsen
  3 = Temperature insertion best
  4 = Wind insertion best
  5 = Geo insertion best
  6 = Temperature insertion worst
  7 = Wind insertion worst
  8 = Geo insertion worst

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/bias/section_5_8_aurora_bias_analysis.py
```

Runtime: ~30–45 min (mature phase only = 60 time steps × 4 vars).
