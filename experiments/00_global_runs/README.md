# Global runs (control, T, U+V, Z insertion)

These four runs produced the global Aurora results: 240 × 6 h steps from 2023-02-01, with
insertion from step 121 (day 30) at all 13 pressure levels and output saved at 00/12 UTC.
They were run on a different machine (`/xace/d1`), so some of the as-run files did not survive
and have been **reconstructed**. Each file's status is listed here and in `docs/PROVENANCE.tsv`.

| Run | Output file | Rollout (`model_overrides/aurora/rollout.py`) | Driver |
|---|---|---|---|
| Control | `auroraoutput_1year_1stfeb23.nc` (first 120 × 12 h records) | original, no injection | `control_1year/1yr_inf.py` (original) |
| U+V insertion | `auroraoutput_uv_inserted.nc` | original logic (`roll.py` on disk, renamed) | `uv_insertion_global/inf_uv.py` (**reconstructed**) |
| T insertion | `auroraoutput_temp_inserted.nc` | **reconstructed** (`inject_vars = ["t"]`) | `t_insertion_global/inf_t.py` (**reconstructed**) |
| Z insertion | `auroraoutput_geopotential_inserted.nc` | **reconstructed** (`inject_vars = ["z"]`) | `z_insertion_global/inf_z.py` (original) |

**How the files were reconstructed.** The surviving U+V rollout injects by looping over
`inject_vars` and hard-replacing each variable's 13-level field with the time-matched ERA5
field. That mechanism does not depend on the variable. The T and Z rollouts are therefore
the U+V file with only `inject_vars` (and docstring wording) changed. The T and U+V drivers
are the surviving Z driver with only the output filename and the truth GRIB filename changed.
The truth filenames are assumed; any ERA5 13-level GRIB for Feb–Apr 2023 containing the
variable works.

**Verification (passed).** `verify_reconstruction.py` checks the original output files
against ERA5 directly, with no model re-run. All three global runs are identical before
insertion (lead ≤ 720 h). From lead 732 h onward, the inserted variable equals ERA5 exactly
(max |pred − ERA5| = 0) at all 13 levels over the whole globe: `t` in the T run, `z` in the Z
run, `u` and `v` in the U+V run. The non-inserted variables keep nonzero errors. That is
exactly the behaviour of the reconstructed rollouts: global, all-level hard replacement of
only the listed variable from step 121.

`drivers_as_found/` holds the global drivers exactly as last saved on disk. They had been edited
in place between runs and all now point to the Z run.
