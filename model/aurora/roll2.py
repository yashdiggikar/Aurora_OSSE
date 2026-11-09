import dataclasses
from typing import Generator

import torch
from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]

def rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    truth_ds=None,
    start_inject_step: int = 121
) -> Generator[Batch, None, None]:
    """
    Perform a roll-out with optional truth temperature (t) injection at 3 pressure levels (850, 925, 1000 hPa).
    Everything else stays the same as your original rollout.
    """

    # Prepare batch for model
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # Pressure level → index mapping for the 13 canonical Aurora/ERA5 levels
    level_map = {
        50: 0, 100: 1, 150: 2, 200: 3, 250: 4, 300: 5, 400: 6,
        500: 7, 600: 8, 700: 9, 850: 10, 925: 11, 1000: 12
    }

    # Inject only temperature at these three levels
    inject_vars = ["t"]
    levels_to_inject = [850, 925, 1000]
    level_indices = [level_map[L] for L in levels_to_inject]

    def _inject_var(var_name: str, pred_batch: Batch, truth_slice, device, dtype, step: int):
        if var_name not in pred_batch.atmos_vars:
            raise KeyError(f"❌ '{var_name}' not found in pred.atmos_vars at step {step}")
        if var_name not in truth_slice:
            raise KeyError(
                f"❌ '{var_name}' not found in truth_slice at step {step}. "
                f"Available: {list(truth_slice.keys())}"
            )

        da = truth_slice[var_name]  # xarray.DataArray

        # Auto-detect pressure-level dimension name
        candidate_level_dims = ("level", "isobaricInhPa", "plev", "pressure", "lev")
        lvl_dim = next((d for d in candidate_level_dims if d in da.dims or d in da.coords), None)
        if lvl_dim is None:
            raise RuntimeError(
                f"❌ Could not find a pressure-level dimension among {candidate_level_dims}. "
                f"Found dims={da.dims} and coords={list(da.coords)}"
            )

        # Prefer label-based selection; fall back to nearest-index or fixed indices
        try:
            da_levels = da.sel({lvl_dim: levels_to_inject})
        except Exception:
            import numpy as np
            if lvl_dim in da.coords:
                avail = np.asarray(da.coords[lvl_dim].values)
                idxs = [int(np.abs(avail - L).argmin()) for L in levels_to_inject]
                da_levels = da.isel({lvl_dim: idxs})
                print(f"ℹ️ '{lvl_dim}' label match failed; used nearest indices {idxs} for {levels_to_inject}")
            else:
                # No coordinate labels—assume canonical 13-level order
                da_levels = da.isel({lvl_dim: level_indices})
                print(f"ℹ️ '{lvl_dim}' has no coord labels; fell back to fixed indices {level_indices}")

        # Match Aurora grid (model uses 720×1440; ERA5 often has 721 latitudes)
        if "latitude" in da_levels.dims:
            da_levels = da_levels.isel(latitude=slice(0, 720))
        elif "lat" in da_levels.dims:
            da_levels = da_levels.isel(lat=slice(0, 720))
        else:
            raise RuntimeError(f"❌ Could not find a latitude dimension in {da_levels.dims}")

        # Convert to torch: (3, 720, 1440) → (1, 1, 3, 720, 1440)
        truth_tensor = torch.from_numpy(da_levels.values).to(device).type(dtype)
        if truth_tensor.ndim != 3:
            raise RuntimeError(
                f"❌ Expected truth slice (levels, lat, lon). Got shape {truth_tensor.shape} with dims {da_levels.dims}"
            )
        truth_tensor = truth_tensor.unsqueeze(0).unsqueeze(0)

        pred_tensor = pred_batch.atmos_vars[var_name]  # (1, 1, 13, 720, 1440)

        # Sanity checks
        if pred_tensor.shape[-2:] != truth_tensor.shape[-2:]:
            raise ValueError(
                f"❌ Spatial shape mismatch for '{var_name}' at step {step}: "
                f"truth {truth_tensor.shape[-2:]} vs pred {pred_tensor.shape[-2:]}"
            )
        if pred_tensor.shape[2] < max(level_indices) + 1:
            raise ValueError(
                f"❌ Pred tensor has only {pred_tensor.shape[2]} levels; expected >= {max(level_indices)+1}"
            )

        # Inject only the selected pressure levels
        for k, idx in enumerate(level_indices):
            pred_tensor[:, :, idx, :, :] = truth_tensor[:, :, k, :, :]

        # Write back (explicit)
        pred_batch.atmos_vars[var_name] = pred_tensor

        print(
            f"✅ Injected truth for '{var_name}' at levels {levels_to_inject} "
            f"({lvl_dim}) at step {step} | truth slice shape: {truth_tensor.shape}"
        )

    for step in range(steps):
        # Forward one step
        pred = model.forward(batch)

        # Start injection from the configured step
        if truth_ds is not None and step >= start_inject_step:
            if step == start_inject_step:
                print(f"\n🚨 Step {step}: Starting truth injection for {inject_vars} at levels {levels_to_inject}")
                print("🔍 pred.atmos_vars.keys():", list(pred.atmos_vars.keys()))
                print("🔍 Expecting truth var(s):", inject_vars)
                if truth_ds is None:
                    raise RuntimeError("❌ truth_ds is None!")

            # Use forecast time for exact/nearest match; try 'time' then 'valid_time'
            forecast_time = pred.metadata.time[0]
            try:
                truth_slice = truth_ds.sel(time=forecast_time, method="nearest")
            except Exception:
                try:
                    truth_slice = truth_ds.sel(valid_time=forecast_time, method="nearest")
                except Exception as e2:
                    raise RuntimeError(
                        f"❌ Failed to match forecast time {forecast_time} in truth_ds on 'time' or 'valid_time': {e2}"
                    )

            # Inject temperature at the three requested levels
            for vname in inject_vars:
                _inject_var(vname, pred, truth_slice, p.device, p.dtype, step)

        # Yield current prediction
        yield pred

        # Autoregressive update: shift context window and append new forecast
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
