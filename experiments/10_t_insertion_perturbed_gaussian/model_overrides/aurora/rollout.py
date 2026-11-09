"""rollout.py — ERA5 T injection, all 13 levels, global, with observation-error perturbation.

Perturbation: ε ~ N(0, σ²) per grid point, level-dependent σ (Ingleby 2017 / ECMWF).
Aurora level order: index 0 = 1000 hPa … index 12 = 50 hPa.
Spatially uncorrelated; seed = step index for exact reproducibility.
"""
import dataclasses
import datetime
from typing import Generator

import numpy as np
import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]

# σ_T (K) per Aurora level: [1000,925,850,700,600,500,400,300,250,200,150,100,50] hPa
# Source: Ingleby (2017) ECMWF radiosonde error assessment; Errico et al. (2013) OSSE framework
SIGMA_T = np.array([
    1.2,   # 1000 hPa
    1.0,   # 925
    1.0,   # 850
    0.8,   # 700
    0.8,   # 600
    0.7,   # 500  (free-troposphere minimum)
    0.7,   # 400
    0.8,   # 300
    1.0,   # 250
    1.2,   # 200  (tropopause region)
    1.5,   # 150
    1.5,   # 100
    2.0,   # 50   (stratosphere — solar heating bias)
], dtype=np.float32)


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
    forecast_time = batch.metadata.time[0]

    for step in range(steps):
        forecast_time_next = forecast_time + datetime.timedelta(hours=6)

        import contextlib
        amp = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if p.is_cuda else contextlib.nullcontext())
        with amp:
            pred = model.forward(batch)

        if truth_atmos is not None and step >= start_inject_step:
            try:
                sl = truth_atmos.sel(
                    time=np.datetime64(forecast_time_next, "ns"), method="nearest"
                )
                truth_np = sl["t"].isel(latitude=slice(0, H)).values.astype("float32")  # (13, H, W)

                # Observation-error perturbation: ε ~ N(0, σ²_lev), spatially uncorrelated
                rng = np.random.default_rng(seed=step)
                noise = np.stack([
                    rng.normal(0.0, float(SIGMA_T[li]), size=truth_np.shape[1:]).astype("float32")
                    for li in range(len(SIGMA_T))
                ], axis=0)  # (13, H, W)
                perturbed_np = truth_np + noise

                truth_t = (
                    torch.from_numpy(perturbed_np)
                    .to(p.device).type(p.dtype)
                    .unsqueeze(0).unsqueeze(0)  # (1,1,13,H,W)
                )
                pred.atmos_vars["t"] = truth_t

                if step % 40 == 0:
                    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
                    print(f"[T-PERTURBED] step={step}  time={forecast_time_next}  noise_rms={noise_rms:.3f}K")

            except Exception as e:
                print(f"[T-PERTURBED] injection error step={step}: {e}")

        yield pred
        forecast_time = forecast_time_next

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
