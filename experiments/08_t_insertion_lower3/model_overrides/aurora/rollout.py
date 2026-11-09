"""rollout.py — ERA5 temperature injection at lower 3 pressure levels (1000, 925, 850 hPa).

Aurora level ordering: [0]=1000, [1]=925, [2]=850, [3]=700, ... [12]=50 hPa.
Inserts globally (no spatial mask) at indices [0, 1, 2] only.
All other levels and all other variables are left untouched.
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
    print(f"[rollout_t_lower] H={H} W={W} | inject levels idx={INSERTED_LEVEL_INDICES} (1000/925/850 hPa) global")

    for step in range(steps):
        pred = model.forward(batch)

        if truth_atmos is not None and step >= start_inject_step:
            if step == start_inject_step:
                print(f"\n[rollout_t_lower] Step {step}: T lower-3 injection STARTED")
            forecast_time = np.datetime64(pred.metadata.time[0], "ns")
            try:
                sl       = truth_atmos.sel(time=forecast_time, method="nearest")
                truth_np = sl["t"].isel(latitude=slice(0, H)).values.astype("float32")  # (13,H,W)
                truth_t  = (torch.from_numpy(truth_np).to(p.device).type(p.dtype)
                            .unsqueeze(0).unsqueeze(0))  # (1,1,13,H,W)
                pred_t   = pred.atmos_vars["t"].clone()
                for li in INSERTED_LEVEL_INDICES:
                    pred_t[:, :, li, :, :] = truth_t[:, :, li, :, :]
                pred.atmos_vars["t"] = pred_t
                if step % 40 == 0 or step == start_inject_step:
                    print(f"  [step {step:3d}] T lower-3 injected (global) | {forecast_time}")
            except Exception as e:
                print(f"  [step {step}] T_LOWER error: {e}")

        yield pred

        batch = dataclasses.replace(
            pred,
            surf_vars={k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in pred.surf_vars.items()},
            atmos_vars={k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1) for k, v in pred.atmos_vars.items()},
        )
