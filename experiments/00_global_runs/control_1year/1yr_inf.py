import sys, time, os, contextlib
sys.path.insert(0, "/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/auroa_ins")

import torch
import xarray as xr
import numpy as np
import netCDF4 as nc
from pathlib import Path
import aurora
from aurora import Aurora, rollout, Batch, Metadata

print(f"✅ Aurora package is from: {aurora.__file__}")

# --- device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔌 device:", device)

# === CONFIG ===
DATA_DIR = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA")
OUTFILE = Path("/umbc/rs/pi_halem/users/yashd1/auroraoutput_1year_1stfeb23.nc")
STEPS = 1460  # 60 days × 4 steps/day (6-hour steps)

# === LOAD ERA5 FILES ===
static_ds = xr.open_dataset(DATA_DIR / "static.nc", engine="netcdf4")
surf_ds   = xr.open_dataset(DATA_DIR / "2023-02-01-surface-level.nc", engine="netcdf4")
atmos_ds  = xr.open_dataset(DATA_DIR / "2023-02-01-atmospheric.nc", engine="netcdf4")

# === PREPARE BATCH ===
toTf = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)

batch = Batch(
    surf_vars={
        "2t":  toTf(surf_ds["t2m"].values[:2][None]),
        "10u": toTf(surf_ds["u10"].values[:2][None]),
        "10v": toTf(surf_ds["v10"].values[:2][None]),
        "msl": toTf(surf_ds["msl"].values[:2][None]),
    },
    static_vars={
        "z":   toTf(static_ds["z"].values[0]),
        "slt": toTf(static_ds["slt"].values[0]),
        "lsm": toTf(static_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": toTf(atmos_ds["t"].values[:2][None]),
        "u": toTf(atmos_ds["u"].values[:2][None]),
        "v": toTf(atmos_ds["v"].values[:2][None]),
        "q": toTf(atmos_ds["q"].values[:2][None]),
        "z": toTf(atmos_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_ds.latitude.values),
        lon=torch.from_numpy(surf_ds.longitude.values),
        time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(p) for p in atmos_ds.pressure_level.values),
    ),
)
print("✅ Batch created (official style)")

# === LOAD MODEL ===
model = Aurora(use_lora=False)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval().to(device)
print("🧠 Aurora model loaded on", next(model.parameters()).device)

# === helpers ===
def t2np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()

# Save only 00/12 UTC leads (given 6-hour step size)
lead_times = [6 * i for i in range(STEPS) if ((6 * i) % 24) in (0, 12)]
n_save = len(lead_times)

if os.path.exists(OUTFILE):
    os.remove(OUTFILE)

print(f"💾 Creating NetCDF for streamed writes → {OUTFILE}")
root = nc.Dataset(OUTFILE, "w", format="NETCDF4")
created = False
saved_idx = 0

use_amp = torch.cuda.is_available()
t0 = time.time()
step_start = time.time()

print(f"🚀 Running {STEPS}-step inference with AMP={use_amp} on {device} …")

with torch.inference_mode():
    ctx = (torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp
           else contextlib.nullcontext())
    with ctx:
        for step_idx, pred in enumerate(
            rollout(model, batch, steps=STEPS),   # ✅ NORMAL ROLLOUT (no injection args)
            start=1
        ):
            # lazy create structure from first pred
            if not created:
                lat = pred.metadata.lat.cpu().numpy()
                lon = pred.metadata.lon.cpu().numpy()

                levels = tuple(pred.metadata.atmos_levels)
                base_time = pred.metadata.time[0]

                root.createDimension("time", 1)
                root.createDimension("lead_time", n_save)
                root.createDimension("lat", len(lat))
                root.createDimension("lon", len(lon))

                v_time = root.createVariable("time", "i8", ("time",))
                v_lead = root.createVariable("lead_time", "i4", ("lead_time",))
                v_lat  = root.createVariable("lat", "f4", ("lat",))
                v_lon  = root.createVariable("lon", "f4", ("lon",))

                v_time[:] = np.array([np.datetime64(base_time).astype("datetime64[s]").astype("int64")])
                v_lead[:] = np.array(lead_times, dtype=np.int32)
                v_lat[:]  = lat.astype(np.float32)
                v_lon[:]  = lon.astype(np.float32)

                def mkvar(name):
                    return root.createVariable(
                        name, "f4", ("lead_time", "lat", "lon"),
                        zlib=True, complevel=1,
                        chunksizes=(1, len(lat), len(lon))
                    )

                surf_keys  = list(pred.surf_vars.keys())
                atmos_keys = list(pred.atmos_vars.keys())

                surf_nc = {k: mkvar(k) for k in surf_keys}
                atmos_nc = {}
                for k in atmos_keys:
                    for i, lev in enumerate(levels):
                        atmos_nc[(k, i, lev)] = mkvar(f"{k}{lev}")

                root.sync()
                created = True
                print("✅ NetCDF structure ready. Streaming every saved step (00 & 12 UTC).")
                print("   surf_keys :", surf_keys)
                print("   atmos_keys:", atmos_keys)

            # log timing/GPU
            elapsed = time.time() - step_start
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                print(f"[step {step_idx}/{STEPS}] {elapsed:.2f}s | GPU alloc={alloc:.0f} MB reserved={reserved:.0f} MB")
            else:
                print(f"[step {step_idx}/{STEPS}] {elapsed:.2f}s")
            step_start = time.time()

            # save only 00/12 UTC
            hour = (6 * (step_idx - 1)) % 24
            if hour in (0, 12):
                for k in surf_keys:
                    surf_nc[k][saved_idx, :, :] = t2np(pred.surf_vars[k][0, 0])
                for k in atmos_keys:
                    for i, lev in enumerate(levels):
                        atmos_nc[(k, i, lev)][saved_idx, :, :] = t2np(pred.atmos_vars[k][0, 0, i])

                if (saved_idx % 10) == 0:
                    root.sync()
                    print(f"📝 wrote lead index {saved_idx}/{n_save-1} (hour={hour})")
                saved_idx += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

root.close()
print(f"✅ Forecast + streamed save complete in {(time.time()-t0)/60:.2f} min → {OUTFILE}")
