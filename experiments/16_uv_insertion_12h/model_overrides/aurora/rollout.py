"""rollout.py — ERA5 U+V injection, all 13 levels, global, every 12 h (alternate 6-h steps).

Injects at steps 121, 123, 125, ... i.e. (step - start_inject_step) % 2 == 0.
Odd-offset steps keep the model's own forecast unmodified.
No perturbation — clean truth insertion.
"""
import dataclasses
import datetime
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

    _s = batch.surf_vars[next(iter(batch.surf_vars))]
    H, W = _s.shape[3], _s.shape[4]
    forecast_time = batch.metadata.time[0]

    for step in range(steps):
        forecast_time_next = forecast_time + datetime.timedelta(hours=6)

        import contextlib
        amp = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if p.is_cuda else contextlib.nullcontext())
        with amp:
            pred = model.forward(batch)

        inject_this_step = (
            truth_atmos is not None
            and step >= start_inject_step
            and (step - start_inject_step) % 2 == 0
        )

        if inject_this_step:
            try:
                sl = truth_atmos.sel(
                    time=np.datetime64(forecast_time_next, "ns"), method="nearest"
                )
                for var in ["u", "v"]:
                    truth_np = sl[var].isel(latitude=slice(0, H)).values.astype("float32")
                    truth_t = (
                        torch.from_numpy(truth_np)
                        .to(p.device).type(p.dtype)
                        .unsqueeze(0).unsqueeze(0)
                    )
                    pred.atmos_vars[var] = truth_t

                if step % 40 == 0:
                    print(f"[UV-12H] step={step} (inject)  time={forecast_time_next}")
            except Exception as e:
                print(f"[UV-12H] injection error step={step}: {e}")
        elif truth_atmos is not None and step >= start_inject_step and step % 40 == 0:
            print(f"[UV-12H] step={step} (skip)    time={forecast_time_next}")

        yield pred
        forecast_time = forecast_time_next

        batch = dataclasses.replace(
            pred,
            surf_vars={k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in pred.surf_vars.items()},
            atmos_vars={k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1) for k, v in pred.atmos_vars.items()},
        )
