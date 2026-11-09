# Aurora Cross-Variable Coupling Analysis — Memory File
## Section 5.6: Coupling Heatmaps and Table 5.8

---

## Overview

Computes coupling score C(A→B) for each insertion experiment (A) and each
response variable group (B), averaged across all 13 pressure levels.

C(A→B) = mean over 13 levels of: 100 × (RMSE_ctrl(B_lev) − RMSE_ins_A(B_lev)) / RMSE_ctrl(B_lev)

All computations in the MATURE phase (day 30–60, steps 60–119).

---

## Key Difference from FCN Coupling

FCN coupling script read pre-computed CSVs from `/umbc/rs/pi_halem/users/yashd1/fcn_rmse/FCNv2/`.
Aurora coupling script computes RMSE **directly from Aurora HDF5 files** since
Aurora stores all 13 pressure levels as named variables (`t500`, `u500`, ..., `z50`).

5 variable groups × 13 levels × 4 runs = 260 RMSE time series computed.

---

## Variable Groups

| Group | Variables | Levels |
|-------|-----------|--------|
| T | t1000, t925, ..., t50 | 13 levels |
| U | u1000, u925, ..., u50 | 13 levels |
| V | v1000, v925, ..., v50 | 13 levels |
| Z | z1000, z925, ..., z50 | 13 levels |
| Q | q1000, q925, ..., q50 | 13 levels |

Pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

---

## Insertion Experiments

| Key | Inserted | Trivial Groups |
|-----|----------|----------------|
| temp_ins | All T levels | T |
| wind_ins | All U+V levels | U, V |
| geo_ins | All Z levels | Z |

**Note:** Q group is not inserted in any experiment → always shows cross-variable response.

---

## Directory Structure

```
Aurora_Pred_Files/coupling/
├── values/
│   ├── Aurora_coupling_matrix.csv
│   ├── Aurora_coupling_per_level.csv
│   └── Aurora_coupling_summary.csv
├── plots/
│   ├── Figure_5_30_Aurora_coupling_heatmap.png
│   ├── Figure_Aurora_coupling_bar.png
│   ├── Figure_Aurora_coupling_vertical_profiles.png
│   └── Table_5_8_Aurora_coupling_summary.png
├── logs/
│   └── compute_aurora_coupling_log.txt
├── compute_aurora_coupling.py
├── coupling_memory.md   ← this file
└── coupling_results_analysis.txt
```

---

## Run Command

```bash
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python3 -u /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/coupling/compute_aurora_coupling.py
```

**Runtime warning:** This script loads ERA5 at each of 13 levels × 5 groups = 65 ERA5 loads.
Expected runtime: 2–4 hours. Run with `nohup` or in a screen session.

```bash
nohup python3 -u compute_aurora_coupling.py > /tmp/aurora_coupling_run.log 2>&1 &
```
