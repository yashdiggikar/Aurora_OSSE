import sys
sys.path.insert(0, "/xace/d1/yashd1/aurora_temp_ins")

import aurora
print(f"✅ Aurora package is from: {aurora.__file__}")

import torch
import xarray as xr
import numpy as np
from pathlib import Path
from aurora import Aurora, rollout, Batch, Metadata
import netCDF4 as nc  # ✅ added for streaming NetCDF writes

# === CONFIG ===
DATA_DIR = Path("/xace/d1")
OUTFILE = Path("/xace/d1/yashd1/auroraoutput_geopotential_inserted.nc")
STEPS = 240  # 60 days × 4 steps/day

# === LOAD ERA5 FILES ===
static_ds = xr.open_dataset(DATA_DIR / "static.nc", engine="netcdf4")
surf_ds = xr.open_dataset(DATA_DIR / "2023-02-01-surface-level.nc", engine="netcdf4")
atmos_ds = xr.open_dataset(DATA_DIR / "2023-02-01-atmospheric.nc", engine="netcdf4")

# === LOAD TRUTH FOR INJECTION & RMSE PRINTS ===
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
        "z": torch.from_numpy(static_ds["z"].values[0]),
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
model.eval()
print("🧠 Aurora model loaded")

# === STREAM INFERENCE DIRECTLY TO PREALLOCATED NETCDF ===
print(f"🚀 Running {STEPS}-step inference and streaming to NetCDF...")
ds_nc = nc.Dataset(str(OUTFILE), "w", format="NETCDF4")

# Define dimensions (unlimited time)
ds_nc.createDimension("time", None)
ds_nc.createDimension("lat", 720)
ds_nc.createDimension("lon", surf_ds.dims["longitude"])
ds_nc.createDimension("level", len(atmos_ds.pressure_level.values))

# Coordinate variables
var_time  = ds_nc.createVariable("lead_time_hours", "i4", ("time",))
var_lat   = ds_nc.createVariable("lat", "f4", ("lat",))
var_lon   = ds_nc.createVariable("lon", "f4", ("lon",))
var_level = ds_nc.createVariable("level", "i4", ("level",))

var_lat[:] = surf_ds.latitude.values[:720].astype("float32")
var_lon[:]   = surf_ds.longitude.values.astype("float32")
var_level[:] = atmos_ds.pressure_level.values.astype("int32")

# Output variables
surf_keys = ["2t", "10u", "10v", "msl"]
atmos_keys = ["t", "u", "v", "q", "z"]

surf_vars_out = {
    k: ds_nc.createVariable(k, "f4", ("time", "lat", "lon"), zlib=True, complevel=1)
    for k in surf_keys
}
atmos_vars_out = {
    k: ds_nc.createVariable(k, "f4", ("time", "level", "lat", "lon"), zlib=True, complevel=1)
    for k in atmos_keys
}

# Metadata
try:
    base_time = surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1]
    ds_nc.setncattr("base_time", str(base_time))
except Exception:
    ds_nc.setncattr("base_time_note", "base time not set from input files")

# Run rollout and write only 00/12 UTC snapshots
t_idx = 0
with torch.inference_mode():
    for i, pred in enumerate(rollout(model, batch, steps=STEPS, truth_ds=truth_ds, start_inject_step=121)):
        # progress prints
        if i % 10 == 0:
            print(f"⏱️ step {i}/{STEPS}")

        hour = (6 * i) % 24
        if hour in (0, 12):
            print(f"📝 writing step {i} (UTC hour {hour}) → time index {t_idx}")
            var_time[t_idx] = 6 * i

            # surface vars
            for k, v in pred.surf_vars.items():
                if k in surf_vars_out:
                    surf_vars_out[k][t_idx, :, :] = (
                        v[0, 0].to("cpu", non_blocking=True).numpy().astype("float32")
                    )

            # atmospheric vars (levels)
            for k, v in pred.atmos_vars.items():
                if k in atmos_vars_out:
                    atmos_vars_out[k][t_idx, :, :, :] = (
                        v[0, 0].to("cpu", non_blocking=True).numpy().astype("float32")
                    )

            t_idx += 1

        # free GPU asap between steps
        del pred
        if i % 8 == 0:
            torch.cuda.empty_cache()

ds_nc.close()
print(f"✅ Stream complete: wrote {t_idx} timestamps (00/12 UTC only) → {OUTFILE}")
