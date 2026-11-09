# Aurora — Correlated OSSE Perturbation Implementation

See `../METHOD_OVERVIEW.md` for the cross-model method description and
citations. This file documents the Aurora-specific code, paths, and
hyperparameters.

---

## 1. Location

| | Path |
|---|---|
| Temperature experiment | `/umbc/rs/pi_halem/users/yashd1/CC_aurora/T_PERTURBED_CORR/` |
| Wind experiment | `/umbc/rs/pi_halem/users/yashd1/CC_aurora/UV_PERTURBED_CORR/` |
| SLURM job scripts | `run_aurora_t_perturbed_corr.slurm`, `run_aurora_uv_perturbed_corr.slurm` (same dirs) |
| Base template (uncorrelated baseline, unmodified) | `CC_aurora/T_PERTURBED/` |

Each new directory has its own full copy of the `aurora/` package (Microsoft
Aurora, editable install), matching this project's existing
per-experiment-copy convention (same pattern as `T_LAND`, `T_OCEAN`, etc.).

## 2. Where the perturbation logic actually lives

`aurora/rollout.py`, function `rollout(model, batch, steps, truth_atmos,
start_inject_step)` — the SAME function signature/generator structure as
`T_PERTURBED/aurora/rollout.py`, but with `_correlated_level_fields()` (T
experiment) or `_generate_correlated_wind_perturbation()` +
`_streamfunction_to_uv()` (wind experiment) replacing the old
i.i.d.-per-point `rng.normal(...)` loop.

Unlike FourCastNetv2 (Section 2 of `FCN/IMPLEMENTATION.md`), Aurora's
non-perturbed insertion experiments (`T_LAND`, `T_OCEAN`, `T_COASTAL`) all
share a **single, parameterized** `rollout()` function
(`truth_atmos`/`spatial_mask` args) — but the perturbed variants
(`T_PERTURBED`, and now `T_PERTURBED_CORR`/`UV_PERTURBED_CORR`) still need
their own `rollout.py` copy, because the perturbation-generation logic
itself (not just the mask) differs and isn't expressible as a parameter to
the shared function without a larger refactor that was out of scope here.

`inf.py` in each directory is structurally identical to `T_PERTURBED/inf.py`
— only the output filename and `root.experiment` description string differ
(and, for the wind experiment, loading `u`/`v` GRIB variables instead of
`t`, merged into one `truth_atmos` dataset via `xr.merge([u_ds, v_ds])` so
`rollout.py` can do `sl["u"]`/`sl["v"]` on a single object, matching
`UV_LAND`'s existing convention).

## 3. Grid and channel conventions (Aurora 0.25 deg pretrained, verified)

- Grid: after `batch.crop(model.patch_size)`, H=720, W=1440 (no south-pole
  row — Aurora's crop drops to a multiple of patch size). Lat/lon arrays
  read dynamically from `batch.metadata.lat`/`.lon` inside `rollout()`
  (not hardcoded), so this is robust to any future patch-size change.
- Aurora's own atmos-variable level order is **descending pressure**:
  index 0 = 1000 hPa, ..., index 12 = 50 hPa — the OPPOSITE order from
  FCNv2's channel scheme. `LEVELS_HPA = [1000, 925, 850, 700, 600, 500,
  400, 300, 250, 200, 150, 100, 50]` is defined to match this, and the
  vertical correlation matrix / sigma tables below are ordered accordingly.
- Insertion targets `pred.atmos_vars["t"]` (temperature experiment) or
  `pred.atmos_vars["u"]`/`["v"]` (wind experiment) directly — full global
  replacement (`pred.atmos_vars["t"] = truth_t`), no spatial mask, matching
  `T_PERTURBED`'s existing "global, no mask" convention.

## 4. Hyperparameters used

Identical horizontal/vertical correlation hyperparameters to FCN (same 0.25
deg grid resolution):

```
HORIZ_CORR_LENGTH_KM = 250.0
SIGMA_PX              ≈ 9.0 grid cells
VERT_L_LOGP           = 0.5
```

Temperature sigma-by-level (K), **unchanged from `T_PERTURBED/`**:

```
[1000,925,850,700,600,500,400,300,250,200,150,100,50] hPa ->
[1.2,1.0,1.0,0.8,0.8,0.7,0.7,0.8,1.0,1.2,1.5,1.5,2.0] K
```

Wind sigma-by-level (m/s) — newly constructed, same physical reasoning as
FCN's table (larger near jet-stream levels), reordered to Aurora's
descending-pressure convention:

```
[1000,925,850,700,600,500,400,300,250,200,150,100,50] hPa ->
[1.3,1.1,1.0,1.0,1.1,1.2,1.4,1.8,2.0,2.2,1.8,1.6,1.5] m/s
```

## 5. IC and truth source

- IC: same as `T_PERTURBED/` and every other Aurora experiment —
  `AURORA_DATA/static.nc`, `2023-02-01-surface-level.nc`,
  `2023-02-01-atmospheric.nc` (fixed 2023-02-01 start, real ERA5-derived
  IC, **not** a control-forecast file like FCN's perturbed variants use).
- Truth: same ERA5 GRIB files as FCN (`AURORA_DATA/rmse_data/
  era5_pressure_2023_{02,03}.grib` — note `T_PERTURBED`'s existing pattern
  only concatenates Feb+Mar, not Feb+Mar+Apr like FCN's `t_perturbed`;
  kept identical to the existing `T_PERTURBED/inf.py` pattern for direct
  comparability against that baseline).

## 6. Environment

`venvs/aurora310` (Python 3.10, torch 2.5.1+cu121, scipy 1.15.3 already
present — no environment changes needed for Aurora, unlike FCN).

## 7. SLURM submission (2026-07-22)

```
sbatch CC_aurora/T_PERTURBED_CORR/run_aurora_t_perturbed_corr.slurm    # job 203460
sbatch CC_aurora/UV_PERTURBED_CORR/run_aurora_uv_perturbed_corr.slurm  # job 203461
```

Both requested 128G mem, 1 L40S GPU (`--constraint=L40S`), 72h walltime,
`--requeue` set, `pi_halem` account, `chip-gpu` qos — same resource request
as `T_PERTURBED/`'s original job. Note the much longer walltime budget
(72h) than FCN's (8h) — Aurora's own per-step wall-clock has historically
run longer than FCNv2's in this project's prior benchmarking.
