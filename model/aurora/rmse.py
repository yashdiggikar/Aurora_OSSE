import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from itertools import chain

# ── Config ────────────────────────────────────────────────
forecast_file = "/xace/d1/yashd1/auroraoutput_temp_inserted_3_low_levels.nc"
truth_file = "/xace/d1/yashd1/era5_aurora.grib"
output_dir = "/xace/d1/yashd1/aurora_temp_ins/3level_temp/scoring_aurora_model"
os.makedirs(output_dir, exist_ok=True)

# ── Load forecast ──────────────────────────────────────────
print("📂 Loading forecast data...")
forecast_ds = xr.open_dataset(forecast_file)
forecast_channels = list(forecast_ds.data_vars.keys())

forecast_base_time = pd.to_datetime(forecast_ds.time.values[0])
forecast_times = forecast_ds.lead_time.values
forecast_datetimes = forecast_base_time + pd.to_timedelta(forecast_times, unit="h")
print(f"✅ Forecast loaded: {len(forecast_datetimes)} lead times, {len(forecast_channels)} channels.")

# ── Load ERA5 truth ────────────────────────────────────────
print("📂 Loading truth data...")
truth_ds = xr.open_dataset(truth_file, engine="cfgrib")
truth_times = pd.to_datetime(truth_ds.time.values)
pressure_levels = truth_ds.isobaricInhPa.values
print(f"✅ Truth loaded: {len(truth_times)} timestamps, {len(pressure_levels)} pressure levels.")

# ── Align lat/lon grids ────────────────────────────────────
aurora_lat = forecast_ds.lat.values
aurora_lon = forecast_ds.lon.values

if truth_ds.latitude[0] < truth_ds.latitude[-1]:
    print("🔁 Reversing ERA5 latitude to match Aurora...")
    truth_ds = truth_ds.sortby("latitude", ascending=False)

# Subset truth to Aurora’s grid — preserve order
truth_ds = truth_ds.sel(latitude=aurora_lat, longitude=aurora_lon)
forecast_ds = forecast_ds.sel(lat=aurora_lat, lon=aurora_lon)

lat_weights = np.cos(np.deg2rad(aurora_lat))

# ── Time labels ────────────────────────────────────────────
time_labels = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in forecast_datetimes]

# ── Define variable groups ─────────────────────────────────
var_groups = {
    "temperature": [f"t{int(p)}" for p in pressure_levels],
    "zonal_wind": [f"u{int(p)}" for p in pressure_levels],
    "meridional_wind": [f"v{int(p)}" for p in pressure_levels],
    "geopotential": [f"z{int(p)}" for p in pressure_levels],
    "specific_humidity": [f"q{int(p)}" for p in pressure_levels],
}
all_channels = list(chain.from_iterable(var_groups.values()))
rmse_df = pd.DataFrame(index=all_channels, columns=time_labels, dtype=float)
acc_df = pd.DataFrame(index=all_channels, columns=time_labels, dtype=float)

# ── Metric functions ───────────────────────────────────────
def compute_rmse_weighted(pred, truth, weights):
    mse = ((pred - truth) ** 2)
    weighted_mse = np.average(mse, axis=0, weights=weights)
    return np.sqrt(np.mean(weighted_mse))

def compute_acc(pred, truth):
    p = pred.ravel()
    t = truth.ravel()
    pm, tm = p.mean(), t.mean()
    num = ((p - pm) * (t - tm)).sum()
    den = np.sqrt(((p - pm) ** 2).sum() * ((t - tm) ** 2).sum())
    return num / den if den != 0 else np.nan

# ── Compute RMSE & ACC ─────────────────────────────────────
print("🔁 Starting metric computation loop...")
for i, dt in enumerate(forecast_datetimes):
    print(f"⏱️ Step {i+1}/{len(forecast_datetimes)} → {dt}")
    if dt not in truth_times:
        print(f"⚠️ No matching truth time for {dt}. Skipping.")
        continue

    truth_sel = truth_ds.sel(time=dt)

    for group_name, ch_list in var_groups.items():
        for ch in ch_list:
            var = ch[0]
            p = int(ch[1:])
            if ch not in forecast_channels:
                print(f"⚠️ Skipping missing forecast channel: {ch}")
                continue

            pred_field = forecast_ds[ch].isel(lead_time=i).values
            try:
                truth_field = truth_sel[var].sel(isobaricInhPa=p).values
            except KeyError:
                print(f"⚠️ Missing truth for {var}{p} at {dt}")
                continue

            rmse = compute_rmse_weighted(pred_field, truth_field, lat_weights)
            acc = compute_acc(pred_field, truth_field)
            rmse_df.loc[ch, time_labels[i]] = rmse
            acc_df.loc[ch, time_labels[i]] = acc

print("✅ Metric computation complete.")

# ── Save CSVs ──────────────────────────────────────────────
rmse_df.to_csv(f"{output_dir}/aurora_rmse_all_vars.csv")
acc_df.to_csv(f"{output_dir}/aurora_acc_all_vars.csv")
print("💾 CSVs saved.")

# ── Plotting ───────────────────────────────────────────────
print("📊 Starting plots...")
for metric_name, df in [("rmse", rmse_df), ("acc", acc_df)]:
    for group_name, ch_list in var_groups.items():
        print(f"📈 Plotting {metric_name.upper()} for {group_name}")
        fig, ax = plt.subplots(figsize=(14, 6))
        for ch in ch_list:
            if ch in df.index:
                y = df.loc[ch].astype(float)
                ax.plot(forecast_datetimes, y, label=ch)

        ax.set_xticks(forecast_datetimes[::12])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_xlabel("Forecast time (UTC)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()}: {group_name.replace('_',' ').title()}")
        ax.legend(title="Level", ncol=4)
        ax.grid(True)
        plt.tight_layout()
        out_path = f"{output_dir}/aurora_{metric_name}_{group_name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved: {out_path}")

print("🎉 Complete.")
