"""rollout.py — ERA5 U+V wind injection (all 13 pressure levels).

spatial_mask=None  → rely on NaN in truth file (e.g. LANDONLY nc)
spatial_mask       → numpy bool (721,1440); True=inject, False=keep model pred
Injection starts at step >= start_inject_step (default 121 = day 30).
"""
import dataclasses
from typing import Generator, Optional

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
    spatial_mask: Optional[np.ndarray] = None,
) -> Generator[Batch, None, None]:
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype).crop(model.patch_size).to(p.device)

    _s = next(iter(batch.atmos_vars.values()))
    H, W = _s.shape[3], _s.shape[4]
    mask_info = "file NaN" if spatial_mask is None else f"spatial_mask {spatial_mask.sum()} pts"
    print(f"[rollout_uv] H={H} W={W} | mask={mask_info} | start_step={start_inject_step}")

    for step in range(steps):
        pred = model.forward(batch)

        if truth_atmos is not None and step >= start_inject_step:
            if step == start_inject_step:
                print(f"\n[rollout_uv] Step {step}: U+V injection STARTED")
            forecast_time = np.datetime64(pred.metadata.time[0], "ns")
            try:
                sl = truth_atmos.sel(time=forecast_time, method="nearest")
                for var in ["u", "v"]:
                    truth_np = sl[var].isel(latitude=slice(0, H)).values.astype("float32")  # (13,H,W)
                    if spatial_mask is not None:
                        truth_np[:, ~spatial_mask[:H, :]] = np.nan
                    truth_t = (torch.from_numpy(truth_np).to(p.device).type(p.dtype)
                               .unsqueeze(0).unsqueeze(0))  # (1,1,13,H,W)
                    mask = torch.isfinite(truth_t)
                    pred.atmos_vars[var] = torch.where(mask, truth_t, pred.atmos_vars[var])
                if step % 40 == 0 or step == start_inject_step:
                    n_pts = int(mask[0, 0, 0].sum())
                    print(f"  [step {step:3d}] U+V injected | pts={n_pts} | {forecast_time}")
            except Exception as e:
                print(f"  [step {step}] UV error: {e}")

        yield pred

        batch = dataclasses.replace(
            pred,
            surf_vars={k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in pred.surf_vars.items()},
            atmos_vars={k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1) for k, v in pred.atmos_vars.items()},
        )
