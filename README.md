# Aurora_OSSE

Observing System Simulation Experiment (OSSE) **truth-insertion** framework for Microsoft's
**Aurora** atmospheric foundation model (0.25°, 13 pressure levels, `aurora-0.25-pretrained`
checkpoint), released with the manuscript *An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast* (Diggikar, Halem, and Raman) (submitted to AGU *JGR: Machine
Learning and Computation*).

Aurora is rolled out autoregressively for 60 days (240 × 6 h steps) from the **2023-02-01**
ERA5 initial condition. Steps 1–120 run free. From step 121 (day 30) onward, selected fields
are **hard-replaced with ERA5 "truth"** (everywhere or inside a spatial mask), and the
corrected state is fed back into the model. Forecasts are verified against ERA5.

This is one of five companion repositories: **`Aurora_OSSE`**, `FourCastNetv2_OSSE`,
`GraphCast_OSSE`, `PrithviWxC_OSSE`, `OSSE_CrossModel_Analysis`.

---

## Repository layout

```
Aurora_OSSE/
├── model/                      Full microsoft/aurora source tree (fork) incl. aurora/ package
│   └── aurora/rollout.py       truth-insertion-capable rollout (see model_overrides)
├── experiments/
│   ├── 00_global_runs/                  control and global T / U+V / Z insertion (T, Z reconstructed + verified)
│   ├── 01_control_matched_60d/          60-day free run from the identical IC and code path
│   ├── 05_t_insertion_land/   06_t_insertion_ocean/   07_t_insertion_coastal/
│   ├── 08_t_insertion_lower3/           T at 1000/925/850 hPa only
│   ├── 09_t_insertion_12h/              T every 12 h (configured, not run; no output)
│   ├── 10_t_insertion_perturbed_gaussian/  11_t_insertion_perturbed_cop/
│   ├── 12_uv_insertion_land/  13_uv_insertion_ocean/  14_uv_insertion_coastal/
│   ├── 15_uv_insertion_lower3/  16_uv_insertion_12h/ (not run)  17_uv_insertion_perturbed_cop/
│   └── 18_tuv_insertion_global/         T + U + V
│       └── model_overrides/aurora/      files that differ from model/aurora for this run
├── preprocessing/              ERA5 download (initial condition, truth)
├── analysis/
│   ├── osse_core/              RMSE time series, propagation maps, % improvement (analyze_osse.py)
│   ├── nrmse/ lats/ lat_bands/ bias/ coupling/   Global-run analyses (normalized RMSE,
│   │                           latitude bands/slices, spatial bias, cross-variable coupling)
│   └── perturbation_cop/       Global, lat-band, bias, correlation-length and spectral analyses for COP runs
├── environment/  tools/assemble_experiment.sh  docs/
```

### How the per-experiment model code is stored
Each original experiment folder held its own full copy of the `aurora` package. `model/aurora/`
holds the package once. `experiments/<exp>/model_overrides/aurora/` holds the exact files
(chiefly `rollout.py`) that differed in that experiment's copy, so the as-run state is fully
recoverable:

```bash
tools/assemble_experiment.sh experiments/05_t_insertion_land /scratch/aurora_t_land
```

The assembled tree is identical to the original as-run package, with two exceptions. It also
contains four unused alternative rollouts from the base package (`roll.py`, `roll2.py`,
`roll_landonly.py`, `rollout_uv_land.py`; never imported by the extended experiments). It
leaves out a stray `xace/` folder of old output PNGs.

Each experiment's own `inf.py` (driver) and `run_*.slurm` sit at the top of its folder. Every
file's original path and MD5 checksum are listed in `docs/PROVENANCE.tsv`.

---

## Experiments

| Folder | Inserted | Levels | Region / mask | Window |
|---|---|---|---|---|
| `00_global_runs/control_1year` | — | — | — | none (first 120 × 12 h outputs = 60 days used as control) |
| `00_global_runs/t_insertion_global` | T | 13 | global | steps ≥ 121 |
| `00_global_runs/uv_insertion_global` | U, V | 13 | global | steps ≥ 121 |
| `00_global_runs/z_insertion_global` | Z | 13 | global | steps ≥ 121 |
| `01_control_matched_60d` | — | — | — | none |
| `05/12_*_land` | T / U,V | 13 | land (land-only truth, NaN over sea) | 121–240 |
| `06/13_*_ocean` | T / U,V | 13 | ocean (LSM ≤ 0.5) | 121–240 |
| `07/14_*_coastal` | T / U,V | 13 | ~100 km coastal buffer | 121–240 |
| `08/15_*_lower3` | T / U,V | 1000/925/850 hPa | global | 121–240 |
| `09/16_*_12h` | T / U,V | 13 | global, every 12 h | 121, 123, … (not run) |
| `10_t_insertion_perturbed_gaussian` | T + i.i.d. Gaussian obs. error | 13 | global | 121–240 |
| `11/17_*_perturbed_cop` | T / U,V + correlated obs. error (COP) | 13 | global | 121–240 |
| `18_tuv_insertion_global` | T, U, V | 13 | global | 121–240 |

## Reconstructed files (global T and Z runs)
The global T- and Z-insertion runs were made on a different machine. Their as-run
`rollout.py` files, and the T/UV drivers, were not preserved, and have been **reconstructed**
in `experiments/00_global_runs/`. The surviving U+V rollout was taken with only
`inject_vars` changed. The reconstruction is **verified against the original output files**
(`experiments/00_global_runs/verify_reconstruction.py`): the inserted variable equals ERA5
exactly at all 13 levels, globally, from step 121 onward, and nothing else is altered. See
`experiments/00_global_runs/README.md`.

All extended experiments (01, 05–18) are fully preserved as run.

---

## Reproducing the results
1. **Environment:** Python 3.10, `pip install -r environment/requirements-frozen.txt`
   (torch 2.5.1+cu121). The drivers import the `aurora` package from the experiment folder
   via `sys.path`.
2. **Weights (not redistributed):** Hugging Face `microsoft/aurora`, file
   `aurora-0.25-pretrained.ckpt` (downloaded automatically by `Aurora.load_checkpoint`).
3. **Data:** ERA5 (Hersbach et al., 2020) via the CDS. Use `preprocessing/download_era5_ic.py`
   for the 2023-02-01 IC and static fields, and `preprocessing/download_era5_truth.py` with
   `download_era5_pressure_febapr2023.py` for the Feb–Apr 2023 truth and verification fields.
   Set `CDSAPI_KEY` or `~/.cdsapirc`.
4. **Run:** assemble the experiment (above), then `sbatch run_aurora_<exp>.slurm`. Scripts
   contain the original absolute cluster paths; see `docs/PATHS.md`.
5. **Analysis:** `analysis/osse_core/analyze_osse.py` (RMSE time series, propagation maps, %
   improvement), `extract_propagation_arrays.py`, the `analysis/*` global-run analyses, and
   `analysis/perturbation_cop/`. Cross-model figures are in `OSSE_CrossModel_Analysis`.

## License
MIT (`LICENSE`). `model/` is a modified fork of microsoft/aurora (MIT, `model/LICENSE.txt`);
see `NOTICE`.

## Citation
See `CITATION.cff`. Please also cite Aurora: Bodnar, C., et al. (2025), A foundation model for
the Earth system, *Nature*, 641, 1180–1187, doi:10.1038/s41586-025-09005-y.
