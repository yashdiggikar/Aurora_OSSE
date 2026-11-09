import sys
sys.path.insert(0, "/xace/d1/yashd1/aurora_temp_ins")

import aurora
print(f"✅ Aurora package is from: {aurora.__file__}")

import time
import torch
import xarray as xr
import numpy as np
from pathlib import Path
from aurora import Aurora, rollout, Batch, Metadata
import netCDF4 as nc  # streaming NetCDF writes

# === CONFIG ===
DATA_DIR = Path("/xace/d1")
OUTFILE = Path("/xace/d1/yashd1/auroraoutput_geopotential_inserted.nc")
STEPS = 240  # 60 days × 4 steps/day

# === LOAD ERA5 FILES ===
static_ds = xr.open_dataset(DATA_DIR / "static.nc", engine="netcdf4")
surf_ds   = xr.open_dataset(DATA_DIR / "2023-02-01-surface-level.nc", engine="netcdf4")
atmos_ds  = xr.open_dataset(DATA_DIR / "2023-02-01-atmospheric.nc",   engine="netcdf4")

# === LOAD TRUTH FOR INJECTION ===
truth_ds = xr.open_dataset("/xace/d1/yashd1/era5_geopotential_aurora.grib", engine="cfgrib")

# === PREPARE BATCH ===
batch = Batch(
    surf_vars={
        "2t": torch.from_numpy(surf_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_ds["msl"].values[:2][None]),
    },
    static_vars={
        "z":   torch.from_numpy(static_ds["z"].values[0]),
        "slt": torch.from_numpy(static_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_ds["v"].values[:2][None]),
        "q": torch.from_numpy(atmos_ds["q"].values[:2][None]),
        "z": torch.from_numpy(atmos_ds["z"].values[:2][None]),
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
# be explicit about device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"🧠 Aurora model loaded on device: {next(model.parameters()).device}")

# === STREAM INFERENCE DIRECTLY TO PREALLOCATED NETCDF ===
print(f"🚀 Running {STEPS}-step inference and streaming to NetCDF: {OUTFILE}")

ds_nc = nc.Dataset(str(OUTFILE), "w", format="NETCDF4")
ds_nc.set_auto_maskandscale(False)

# Force model grid sizes (Aurora outputs 720x1440)
LAT_N = 720
LON_N = 1440
LEV_N = len(atmos_ds.pressure_level.values)

# Dimensions (unlimited time)
ds_nc.createDimension("time", None)
ds_nc.createDimension("lat",  LAT_N)
ds_nc.createDimension("lon",  LON_N)
ds_nc.createDimension("level", LEV_N)

# Coordinate variables
var_time  = ds_nc.createVariable("lead_time_hours", "i4", ("time",))
var_lat   = ds_nc.createVariable("lat", "f4", ("lat",))
var_lon   = ds_nc.createVariable("lon", "f4", ("lon",))
var_level = ds_nc.createVariable("level", "i4", ("level",))

# Slice latitude to match 720 rows; longitude already 1440
var_lat[:]   = surf_ds.latitude.values[:LAT_N].astype("float32")
var_lon[:]   = surf_ds.longitude.values.astype("float32")
var_level[:] = atmos_ds.pressure_level.values.astype("int32")

# Output variables (no compression for faster per-record writes; chunk to one record)
surf_keys = ["2t", "10u", "10v", "msl"]
atmos_keys = ["t", "u", "v", "q", "z"]

surf_vars_out = {
    k: ds_nc.createVariable(
        k, "f4", ("time", "lat", "lon"),
        zlib=False, chunksizes=(1, LAT_N, LON_N)
    )
    for k in surf_keys
}
atmos_vars_out = {
    k: ds_nc.createVariable(
        k, "f4", ("time", "level", "lat", "lon"),
        zlib=False, chunksizes=(1, LEV_N, LAT_N, LON_N)
    )
    for k in atmos_keys
}

# Metadata
try:
    base_time = surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1]
    ds_nc.setncattr("base_time", str(base_time))
except Exception:
    ds_nc.setncattr("base_time_note", "base time not set from input files")

# === Rollout iterator so we can *time* each step's forward pass accurately ===
t_idx = 0
with torch.inference_mode():
    roll_iter = iter(rollout(model, batch, steps=STEPS, truth_ds=truth_ds, start_inject_step=121))

    for i in range(STEPS):
        # time the next() call which triggers model.forward inside rollout
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        pred = next(roll_iter)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd = time.time() - t0

        if i % 5 == 0:
            print(f"🔄 step {i}/{STEPS}: forward+inject = {t_fwd:.2f}s", flush=True)

        hour = (6 * i) % 24
        if hour in (0, 12):
            print(f"📝 writing step {i} (UTC hour {hour}) → time index {t_idx}")
            t_w0 = time.time()

            var_time[t_idx] = 6 * i

            # surface vars (time, lat, lon)
            for k, v in pred.surf_vars.items():
                if k in surf_vars_out:
                    arr = v[0, 0].to("cpu", non_blocking=True).numpy().astype("float32")
                    surf_vars_out[k][t_idx, :, :] = arr

            # atmospheric vars (time, level, lat, lon)
            for k, v in pred.atmos_vars.items():
                if k in atmos_vars_out:
                    arr = v[0, 0].to("cpu", non_blocking=True).numpy().astype("float32")
                    atmos_vars_out[k][t_idx, :, :, :] = arr

            t_write = time.time() - t_w0
            print(f"   💾 write took {t_write:.2f}s", flush=True)
            t_idx += 1

        # free GPU asap
        del pred
        if i % 8 == 0:
            torch.cuda.empty_cache()

ds_nc.close()
print(f"✅ Stream complete: wrote {t_idx} timestamps (00/12 UTC only) → {OUTFILE}")
