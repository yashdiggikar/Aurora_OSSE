# Hard-coded paths

The scripts in this repository are released **exactly as they were run** on the UMBC HPC
cluster (chip / `pi_halem` allocation), so they contain absolute paths to the original
locations of data, model weights and outputs. To re-run them elsewhere, replace the
prefixes below with your own locations (e.g. with `sed` or by editing the constants at the
top of each script). No other code changes are needed.

Root prefix used everywhere: `/umbc/rs/pi_halem/users/yashd1` (some legacy Aurora drivers use `/xace/d1/yashd1`).

| Original path prefix | Occurrences | Example file |
|---|---|---|
| `/umbc/rs/pi_halem/users/yashd1/CC_aurora` | 166 | `analysis/osse_core/run_analysis.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/AURORA_DATA` | 70 | `analysis/bias/section_5_8_aurora_bias_analysis.py` |
| `/umbc/rs/pi_halem/users/yashd1/venvs` | 41 | `analysis/osse_core/_recompute_perturbed_pipeline.sh` |
| `/umbc/rs/pi_halem/users/yashd1/auroraoutput_t_swath_inserted.nc` | 15 | `experiments/01_control_matched_60d/model_overrides/aurora/inf.py` |
| `/umbc/rs/pi_halem/users/yashd1/auroraoutput_1year_1stfeb23.nc` | 10 | `analysis/bias/section_5_8_aurora_bias_analysis.py` |
| `/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files` | 8 | `analysis/bias/section_5_8_aurora_bias_analysis.py` |
| `/umbc/rs/pi_halem/users/yashd1` | 7 | `analysis/osse_core/analyze_osse.py` |
| `/umbc/rs/pi_halem/users/yashd1/perturbation_results` | 4 | `analysis/osse_core/_recompute_perturbed_pipeline.sh` |
| `/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS` | 3 | `analysis/osse_core/_recompute_perturbed_pipeline.sh` |
| `/xace/d1/yashd1/aurora_temp_ins` | 3 | `experiments/00_global_runs/drivers_as_found/bhag.py` |
| `/xace/d1/yashd1/auroraoutput_geopotential_inserted.nc` | 3 | `experiments/00_global_runs/drivers_as_found/bhag.py` |
| `/xace/d1/yashd1/era5_geopotential_aurora.grib` | 3 | `experiments/00_global_runs/drivers_as_found/bhag.py` |
| `/umbc/rs/pi_halem/users/yashd1/auroraoutput_uv_inserted_LANDONLY.nc` | 1 | `experiments/00_global_runs/drivers_as_found/inf.py` |
| `/umbc/rs/pi_halem/users/yashd1/auroraoutput_t_inserted_LANDONLY.nc` | 1 | `experiments/00_global_runs/drivers_as_found/gppu.py` |
