"""rollout.py — ERA5 T injection, all 13 levels, global, with spatially- and
vertically-CORRELATED observation-error perturbation (upgrade over
T_PERTURBED/, which used spatially-uncorrelated i.i.d. noise per level).

See PERTURBATION_EXPERIMENTS/Aurora/IMPLEMENTATION.md for the full method
writeup (citations, hyperparameter justification):
  1. Horizontal: each level's noise field is a Gaussian Random Field
     (Gaussian-filtered white noise), correlation length ~250 km.
  2. Vertical: exponential-decay correlation in log-pressure space between
     levels, applied via Cholesky-mixing of independently-smoothed fields.

Aurora level order: index 0 = 1000 hPa ... index 12 = 50 hPa.
Seed = step index for exact reproducibility (unchanged from T_PERTURBED/).
"""
import dataclasses
import datetime
from typing import Generator

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]

# σ_T (K) per Aurora level: [1000,925,850,700,600,500,400,300,250,200,150,100,50] hPa
# Source: Ingleby (2017) ECMWF radiosonde error assessment; Errico et al. (2013) OSSE framework
LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
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

# ── Correlation-structure hyperparameters (documented, citable defaults) ────
HORIZ_CORR_LENGTH_KM = 250.0
GRID_SPACING_DEG     = 0.25
KM_PER_DEG_LAT        = 111.0
SIGMA_PX = HORIZ_CORR_LENGTH_KM / (KM_PER_DEG_LAT * GRID_SPACING_DEG)   # ≈ 9.0 grid cells

VERT_L_LOGP = 0.5
_logp = np.log(np.array(LEVELS_HPA, dtype=np.float64))
_vert_dist = np.abs(_logp[:, None] - _logp[None, :])
VERT_CORR_T = np.exp(-_vert_dist / VERT_L_LOGP)
VERT_CHOL_T = np.linalg.cholesky(VERT_CORR_T).astype(np.float32)   # (13,13)


def _correlated_level_fields(n_lev, shape_hw, vert_chol, rng):
    """Return (n_lev, H, W) fields, each unit-variance, spatially smooth
    (Gaussian-filtered white noise, correlation length SIGMA_PX pixels) AND
    correlated across levels via vert_chol. Longitude axis (last) periodic
    (mode="wrap"); latitude axis (mode="nearest")."""
    H, W = shape_hw
    raw = np.stack([
        gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=SIGMA_PX, mode=("nearest", "wrap"))
        for _ in range(n_lev)
    ], axis=0)
    raw = raw / raw.std(axis=(1, 2), keepdims=True)
    mixed = np.einsum("ij,jhw->ihw", vert_chol, raw)
    mixed = mixed / mixed.std(axis=(1, 2), keepdims=True)
    return mixed.astype(np.float32)


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
    lat_deg = batch.metadata.lat.cpu().numpy().astype(np.float64)
    lon_deg = batch.metadata.lon.cpu().numpy().astype(np.float64)
    forecast_time = batch.metadata.time[0]

    for step in range(steps):
        forecast_time_next = forecast_time + datetime.timedelta(hours=6)

        # fp32 throughout -- bf16 autocast here was found to cause a ~1K RMS
        # (max ~2K) divergence from the fp32 runs the rest of the project's
        # Aurora RMSE (Multimodel-study) is based on, purely from precision,
        # not from the injection/perturbation logic itself.
        pred = model.forward(batch)

        if truth_atmos is not None and step >= start_inject_step:
            try:
                sl = truth_atmos.sel(
                    time=np.datetime64(forecast_time_next, "ns"), method="nearest"
                )
                truth_np = sl["t"].isel(latitude=slice(0, H)).values.astype("float32")  # (13, H, W)

                rng = np.random.default_rng(seed=step)
                corr_fields = _correlated_level_fields(13, (H, W), VERT_CHOL_T, rng)
                noise = corr_fields * SIGMA_T[:, None, None]
                perturbed_np = truth_np + noise

                truth_t = (
                    torch.from_numpy(perturbed_np)
                    .to(p.device).type(p.dtype)
                    .unsqueeze(0).unsqueeze(0)  # (1,1,13,H,W)
                )
                pred.atmos_vars["t"] = truth_t

                if step % 40 == 0:
                    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
                    print(f"[T-PERTURBED-CORR] step={step}  time={forecast_time_next}  noise_rms={noise_rms:.3f}K")

            except Exception as e:
                print(f"[T-PERTURBED-CORR] injection error step={step}: {e}")

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
