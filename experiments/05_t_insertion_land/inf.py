"""T_LAND: ERA5 temperature injection, land only, all 13 levels, steps 121-240."""
import sys, time, contextlib
sys.path.insert(0, "/umbc/rs/pi_halem/users/yashd1/CC_aurora/T_LAND")

import aurora as _a; print(f"Aurora: {_a.__file__}")
import torch, xarray as xr, numpy as np, netCDF4 as nc
from pathlib import Path
from aurora import Aurora, Batch, Metadata, rollout

DATA_DIR  = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA")
LSM_DIR   = Path("/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/lsm_data_aurora")
OUT_DIR   = Path("/umbc/rs/pi_halem/users/yashd1/CC_aurora/T_LAND/output")
OUTFILE   = OUT_DIR / "aurora_t_land.nc"
STEPS, START_INJ = 240, 121

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

static_ds = xr.open_dataset(DATA_DIR / "static.nc",                   engine="netcdf4")
surf_ds   = xr.open_dataset(DATA_DIR / "2023-02-01-surface-level.nc", engine="netcdf4")
atmos_ds  = xr.open_dataset(DATA_DIR / "2023-02-01-atmospheric.nc",   engine="netcdf4")

# Land-only truth: NaN at ocean
truth_atmos = xr.open_dataset(LSM_DIR / "era5_temp_aurora_LANDONLY.nc", engine="netcdf4")
print(f"Truth T LAND: {truth_atmos.dims}  {truth_atmos.time.values[0]} → {truth_atmos.time.values[-1]}")

toT = lambda a: torch.from_numpy(a.astype("float32"))
batch = Batch(
    surf_vars={"2t": toT(surf_ds["t2m"].values[:2][None]), "10u": toT(surf_ds["u10"].values[:2][None]),
               "10v": toT(surf_ds["v10"].values[:2][None]), "msl": toT(surf_ds["msl"].values[:2][None])},
    static_vars={"z": toT(static_ds["z"].values[0]), "slt": toT(static_ds["slt"].values[0]),
                 "lsm": toT(static_ds["lsm"].values[0])},
    atmos_vars={"t": toT(atmos_ds["t"].values[:2][None]), "u": toT(atmos_ds["u"].values[:2][None]),
                "v": toT(atmos_ds["v"].values[:2][None]), "q": toT(atmos_ds["q"].values[:2][None]),
                "z": toT(atmos_ds["z"].values[:2][None])},
    metadata=Metadata(lat=torch.from_numpy(surf_ds.latitude.values),
                      lon=torch.from_numpy(surf_ds.longitude.values),
                      time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
                      atmos_levels=tuple(int(p) for p in atmos_ds.pressure_level.values)),
)
print("Batch ready")

model = Aurora(use_lora=False)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval().to(device)
print(f"Model on {device}")

def t2np(t): return t.detach().to(torch.float32).cpu().numpy()
if OUTFILE.exists(): OUTFILE.unlink()
root = nc.Dataset(str(OUTFILE), "w", format="NETCDF4")
created, saved_idx, t0 = False, 0, time.time()

with torch.inference_mode():
    amp = (torch.cuda.amp.autocast(dtype=torch.bfloat16) if torch.cuda.is_available() else contextlib.nullcontext())
    with amp:
        for step_idx, pred in enumerate(
            rollout(model, batch, steps=STEPS, truth_atmos=truth_atmos, start_inject_step=START_INJ, spatial_mask=None),
            start=1
        ):
            if not created:
                lat, lon = pred.metadata.lat.cpu().numpy(), pred.metadata.lon.cpu().numpy()
                levels, base_time = tuple(pred.metadata.atmos_levels), batch.metadata.time[0]  # true ref time -- pred.metadata.time[0] is already +6h, which mislabeled every lead_time
                root.createDimension("lead_time", STEPS); root.createDimension("lat", len(lat)); root.createDimension("lon", len(lon))
                v_lead = root.createVariable("lead_time", "i4", ("lead_time",))
                v_lat  = root.createVariable("lat",  "f4", ("lat",));  v_lon = root.createVariable("lon", "f4", ("lon",))
                v_base = root.createVariable("base_time", "i8", ())
                v_lead[:] = np.arange(1, STEPS+1, dtype=np.int32) * 6
                v_lat[:] = lat.astype(np.float32); v_lon[:] = lon.astype(np.float32)
                v_base[:] = np.datetime64(base_time).astype("datetime64[s]").astype("int64")
                v_base.units = "seconds since 1970-01-01"
                def mkv(n): return root.createVariable(n, "f4", ("lead_time","lat","lon"), zlib=True, complevel=2, chunksizes=(1,len(lat),len(lon)))
                surf_nc  = {k: mkv(k) for k in pred.surf_vars}
                atmos_nc = {(k,i,lv): mkv(f"{k}{lv}") for k in pred.atmos_vars for i,lv in enumerate(levels)}
                root.experiment = "T_LAND: ERA5 T all-level land-only injection"; root.start_inject_step = START_INJ; root.sync()
                created = True; print("NetCDF ready. Streaming…")
            for k in pred.surf_vars: surf_nc[k][saved_idx] = t2np(pred.surf_vars[k][0,0])
            for k in pred.atmos_vars:
                for i,lv in enumerate(levels): atmos_nc[(k,i,lv)][saved_idx] = t2np(pred.atmos_vars[k][0,0,i])
            if saved_idx % 20 == 0: root.sync(); print(f"  saved {saved_idx}/{STEPS}  elapsed={( time.time()-t0)/60:.1f}min")
            saved_idx += 1
            if torch.cuda.is_available(): torch.cuda.empty_cache()

root.close()
print(f"\nDone {(time.time()-t0)/3600:.2f}h → {OUTFILE}")
