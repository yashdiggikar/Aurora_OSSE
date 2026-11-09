"""rollout.py — ERA5 U+V wind injection, all 13 levels, global, with
spatially- and vertically-CORRELATED, non-divergent (streamfunction-derived)
wind perturbation noise.

u and v are NOT perturbed independently: both are derived from ONE random
streamfunction field psi (horizontally- and vertically-correlated, same
construction as T_PERTURBED_CORR), via the standard non-divergent/
rotational-wind relation:
    u' = -(1/R) dpsi/dphi         v' = (1/(R cos(phi))) dpsi/dlambda
(Holton, "An Introduction to Dynamic Meteorology"). This avoids injecting
spurious pointwise divergence/convergence that independently-perturbed u,v
would create (Parrish & Derber 1992; Ingleby 2001 wind covariance
discussion). See PERTURBATION_EXPERIMENTS/Aurora/IMPLEMENTATION.md.

Aurora level order: index 0 = 1000 hPa ... index 12 = 50 hPa.
Seed = step index for exact reproducibility.
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

LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

# Wind observation error by level (m/s), Ingleby (2017)-style radiosonde wind
# error assessment. u and v share the same magnitude table (isotropic
# turbulence assumption) — documented simplification, see IMPLEMENTATION.md.
SIGMA_UV = np.array([
    1.3,   # 1000 hPa (surface friction / representativeness)
    1.1,   # 925
    1.0,   # 850
    1.0,   # 700
    1.1,   # 600
    1.2,   # 500
    1.4,   # 400
    1.8,   # 300
    2.0,   # 250
    2.2,   # 200  (near jet core)
    1.8,   # 150
    1.6,   # 100
    1.5,   # 50
], dtype=np.float32)

HORIZ_CORR_LENGTH_KM = 250.0
GRID_SPACING_DEG     = 0.25
KM_PER_DEG_LAT        = 111.0
SIGMA_PX = HORIZ_CORR_LENGTH_KM / (KM_PER_DEG_LAT * GRID_SPACING_DEG)

VERT_L_LOGP = 0.5
_logp = np.log(np.array(LEVELS_HPA, dtype=np.float64))
_vert_dist = np.abs(_logp[:, None] - _logp[None, :])
VERT_CORR_UV = np.exp(-_vert_dist / VERT_L_LOGP)
VERT_CHOL_UV = np.linalg.cholesky(VERT_CORR_UV).astype(np.float32)

EARTH_RADIUS_M = 6371000.0


def _correlated_level_fields(n_lev, shape_hw, vert_chol, rng):
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


def _streamfunction_to_uv(psi, lat_deg, lon_deg):
    """psi: (n_lev, H, W). Non-divergent u,v via u=-(1/R)dpsi/dphi,
    v=(1/(R cos(phi)))dpsi/dlambda."""
    lat_rad = np.deg2rad(lat_deg.astype(np.float64))
    lon_rad = np.deg2rad(lon_deg.astype(np.float64))
    dlambda = float(np.mean(np.diff(lon_rad)))

    dpsi_dlambda = (np.roll(psi, -1, axis=2) - np.roll(psi, 1, axis=2)) / (2.0 * dlambda)
    dpsi_dphi    = np.gradient(psi, lat_rad, axis=1)

    cosphi = np.clip(np.cos(lat_rad), np.cos(np.deg2rad(85.0)), None)[None, :, None]

    u = -(1.0 / EARTH_RADIUS_M) * dpsi_dphi
    v = (1.0 / (EARTH_RADIUS_M * cosphi)) * dpsi_dlambda
    return u.astype(np.float32), v.astype(np.float32)


def _generate_correlated_wind_perturbation(n_lev, lat_deg, lon_deg, vert_chol, sigma_by_level, rng):
    psi = _correlated_level_fields(n_lev, (len(lat_deg), len(lon_deg)), vert_chol, rng)
    u_raw, v_raw = _streamfunction_to_uv(psi, lat_deg, lon_deg)
    u_std = u_raw.std(axis=(1, 2), keepdims=True)
    v_std = v_raw.std(axis=(1, 2), keepdims=True)
    u = (u_raw / u_std) * sigma_by_level[:, None, None]
    v = (v_raw / v_std) * sigma_by_level[:, None, None]
    return u.astype(np.float32), v.astype(np.float32)


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
                u_np = sl["u"].isel(latitude=slice(0, H)).values.astype("float32")  # (13,H,W)
                v_np = sl["v"].isel(latitude=slice(0, H)).values.astype("float32")

                rng = np.random.default_rng(seed=step)
                u_noise, v_noise = _generate_correlated_wind_perturbation(
                    13, lat_deg, lon_deg, VERT_CHOL_UV, SIGMA_UV, rng
                )
                u_perturbed = u_np + u_noise
                v_perturbed = v_np + v_noise

                pred.atmos_vars["u"] = (torch.from_numpy(u_perturbed).to(p.device).type(p.dtype)
                                          .unsqueeze(0).unsqueeze(0))
                pred.atmos_vars["v"] = (torch.from_numpy(v_perturbed).to(p.device).type(p.dtype)
                                          .unsqueeze(0).unsqueeze(0))

                if step % 40 == 0:
                    print(f"[UV-PERTURBED-CORR] step={step}  time={forecast_time_next}  "
                          f"u_noise_rms={float(np.sqrt(np.mean(u_noise**2))):.3f}  "
                          f"v_noise_rms={float(np.sqrt(np.mean(v_noise**2))):.3f} m/s")

            except Exception as e:
                print(f"[UV-PERTURBED-CORR] injection error step={step}: {e}")

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
