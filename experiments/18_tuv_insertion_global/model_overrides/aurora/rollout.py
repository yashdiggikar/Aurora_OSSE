"""rollout.py — ERA5 T + U + V injection, all 13 pressure levels, global.

Aurora level ordering (descending): [0]=1000, [1]=925, [2]=850, ..., [12]=50 hPa.
All 13 levels of T, U, V replaced with ERA5 truth at every grid point globally.
No spatial mask — direct replacement (no NaN, no torch.where needed).
Injection starts at step >= start_inject_step (default 121 = day 30).
"""
import dataclasses
from typing import Generator

import numpy as np
import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]


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
    print(f"[rollout_tuv_global] H={H} W={W} | global all-level T+U+V | start_step={start_inject_step}")

    for step in range(steps):
        pred = model.forward(batch)

        if truth_atmos is not None and step >= start_inject_step:
            if step == start_inject_step:
                print(f"\n[rollout_tuv_global] Step {step}: T+U+V global injection STARTED")
            forecast_time = np.datetime64(pred.metadata.time[0], "ns")
            try:
                sl = truth_atmos.sel(time=forecast_time, method="nearest")
                for var in ["t", "u", "v"]:
                    truth_np = sl[var].isel(latitude=slice(0, H)).values.astype("float32")  # (13,H,W)
                    truth_t = (torch.from_numpy(truth_np).to(p.device).type(p.dtype)
                               .unsqueeze(0).unsqueeze(0))  # (1,1,13,H,W)
                    pred.atmos_vars[var] = truth_t   # global: replace all points directly
                if step % 40 == 0 or step == start_inject_step:
                    print(f"  [step {step:3d}] T+U+V injected globally ({H}x{W}) | {forecast_time}")
            except Exception as e:
                print(f"  [step {step}] TUV error: {e}")

        yield pred

        batch = dataclasses.replace(
            pred,
            surf_vars={k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in pred.surf_vars.items()},
            atmos_vars={k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1) for k, v in pred.atmos_vars.items()},
        )
