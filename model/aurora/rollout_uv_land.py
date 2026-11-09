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
    """Perform a roll-out with optional truth wind (u,v) injection at 13 pressure levels.

    NOTE: For land-only insertion, pass a LAND-ONLY truth_ds where sea points are NaN.
          (e.g., era5_uv_aurora_LANDONLY.nc created earlier.)
    """
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    level_map = {50: 0, 100: 1, 150: 2, 200: 3, 250: 4, 300: 5, 400: 6,
                 500: 7, 600: 8, 700: 9, 850: 10, 925: 11, 1000: 12}

    inject_vars = ["u", "v"]

    def _inject_var(var_name: str, pred_batch: Batch, truth_slice, device, dtype, step: int):
        if var_name not in pred_batch.atmos_vars:
            raise KeyError(f"❌ '{var_name}' not found in pred.atmos_vars at step {step}")
        if var_name not in truth_slice:
            raise KeyError(f"❌ '{var_name}' not found in truth_slice at step {step}. Available: {list(truth_slice.keys())}")

        # Slice to match Aurora grid (lat 720, lon 1440) and add (batch=1, time=1)
        truth_tensor = torch.from_numpy(
            truth_slice[var_name].isel(latitude=slice(0, 720)).values
        ).to(device).type(dtype)  # (13, 720, 1440)
        truth_tensor = truth_tensor.unsqueeze(0).unsqueeze(0)  # → (1, 1, 13, 720, 1440)

        pred_tensor = pred_batch.atmos_vars[var_name]         # (1, 1, 13, 720, 1440)
        if truth_tensor.shape != pred_tensor.shape:
            raise ValueError(
                f"❌ Shape mismatch for '{var_name}' at step {step}: "
                f"truth {truth_tensor.shape} vs pred {pred_tensor.shape}"
            )

        # === Land-only injection: replace ONLY where truth is finite; keep sea as prediction ===
        land_mask = torch.isfinite(truth_tensor)              # True on land, False on sea (NaN in truth)
        updated = torch.where(land_mask, truth_tensor, pred_tensor)
        pred_batch.atmos_vars[var_name] = updated

        print(f"✅ Land-only injected truth for '{var_name}' at step {step} | shape: {truth_tensor.shape}")

    for step in range(steps):
        pred = model.forward(batch)

        if truth_ds is not None and step >= start_inject_step:
            if step == start_inject_step:
                print(f"\n🚨 Step {step}: Starting truth injection for {inject_vars}")
                print("🔍 pred.atmos_vars.keys():", list(pred.atmos_vars.keys()))
                print("🔍 Expecting truth vars:", inject_vars)
                if truth_ds is None:
                    raise RuntimeError("❌ truth_ds is None!")
                # Sanity hint: truth_ds should be the LAND-ONLY file (sea is NaN)

            forecast_time = pred.metadata.time[0]
            try:
                truth_slice = truth_ds.sel(time=forecast_time, method="nearest")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to match forecast time {forecast_time} in truth_ds: {e}")

            for vname in inject_vars:
                _inject_var(vname, pred, truth_slice, p.device, p.dtype, step)

        yield pred

        # Autoregressive roll
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
