"""rollout.py — ERA5 U+V wind injection at lower 3 pressure levels (1000, 925, 850 hPa).

Aurora level ordering: [0]=1000, [1]=925, [2]=850, [3]=700, ... [12]=50 hPa.
Inserts globally (no spatial mask) at indices [0, 1, 2] only.
Injection starts at step >= start_inject_step (default 121 = day 30).
"""
import dataclasses
from typing import Generator

import numpy as np
import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]

INSERTED_LEVEL_INDICES = [0, 1, 2]   # 1000, 925, 850 hPa


def rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    truth_atmos=None,
    start_inject_step: int = 121,
) -> Generator[Batch, None, None]:
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype).crop(model.patch_size).to(p.device)

    _s = next(iter(batch.atmos_vars.values()))
    H, W = _s.shape[3], _s.shape[4]
    print(f"[rollout_uv_lower] H={H} W={W} | inject levels idx={INSERTED_LEVEL_INDICES} (1000/925/850 hPa) global")

    for step in range(steps):
        pred = model.forward(batch)

        if truth_atmos is not None and step >= start_inject_step:
            if step == start_inject_step:
                print(f"\n[rollout_uv_lower] Step {step}: U+V lower-3 injection STARTED")
            forecast_time = np.datetime64(pred.metadata.time[0], "ns")
            try:
                sl = truth_atmos.sel(time=forecast_time, method="nearest")
                for var in ["u", "v"]:
                    truth_np = sl[var].isel(latitude=slice(0, H)).values.astype("float32")  # (13,H,W)
                    truth_t  = (torch.from_numpy(truth_np).to(p.device).type(p.dtype)
                                .unsqueeze(0).unsqueeze(0))
                    pred_var = pred.atmos_vars[var].clone()
                    for li in INSERTED_LEVEL_INDICES:
                        pred_var[:, :, li, :, :] = truth_t[:, :, li, :, :]
                    pred.atmos_vars[var] = pred_var
                if step % 40 == 0 or step == start_inject_step:
                    print(f"  [step {step:3d}] U+V lower-3 injected (global) | {forecast_time}")
            except Exception as e:
                print(f"  [step {step}] UV_LOWER error: {e}")

        yield pred

        batch = dataclasses.replace(
            pred,
            surf_vars={k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in pred.surf_vars.items()},
            atmos_vars={k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1) for k, v in pred.atmos_vars.items()},
        )
