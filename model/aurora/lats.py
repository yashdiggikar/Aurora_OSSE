import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from itertools import chain

# ── Config ──────────────────────────────────────────────────────────────
forecast_file = "/xace/d1/yashd1/aurora_uv_ins/scoring_aurora_model_uv"
truth_file = "/xace/d1/yashd1/era5_aurora.grib"
output_dir = "xace/d1/yashd1/aurora_uv_ins/aurora_latbands_output"
os.makedirs(output_dir, exist_ok=True)

# Load forecast and truth
forecast_ds = xr.open_dataset(forecast_file)
forecast_datetimes = pd.to_datetime(forecast_ds.time.values[0]) + pd.to_timedelta(forecast_ds.lead_time.values, unit="h")
forecast_channels = list(forecast_ds.data_vars.keys())

truth_ds = xr.open_dataset(truth_file, engine="cfgrib")
pressure_levels = truth_ds.isobaricInhPa.values
if truth_ds.latitude[0] < truth_ds.latitude[-1]:
    truth_ds = truth_ds.sortby("latitude", ascending=False)

# Align grids
aurora_lat = forecast_ds.lat.values
aurora_lon = forecast_ds.lon.values
truth_ds = truth_ds.sel(latitude=aurora_lat, longitude=aurora_lon)
forecast_ds = forecast_ds.sel(lat=aurora_lat, lon=aurora_lon)

# Define lat bands
lat_bands = {
    "2deg": lambda lat: np.abs(lat - 2.0) < 0.25,
    "40deg": lambda lat: np.abs(lat - 40.0) < 0.25,
}
var_groups = {
    "temperature": [f"t{int(p)}" for p in pressure_levels],
    "zonal_wind": [f"u{int(p)}" for p in pressure_levels],
    "meridional_wind": [f"v{int(p)}" for p in pressure_levels],
    "geopotential": [f"z{int(p)}" for p in pressure_levels],
    "specific_humidity": [f"q{int(p)}" for p in pressure_levels],
}
levels_to_plot = [1000, 500, 300]
time_labels = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in forecast_datetimes]

# Metric
def compute_rmse_weighted(pred, truth, weights):
    mse = (pred - truth) ** 2
    weighted_mse = np.average(mse, axis=0, weights=weights)
    return np.sqrt(np.mean(weighted_mse))

# Process each lat band
for band_name, band_selector in lat_bands.items():
    lat_mask = band_selector(aurora_lat)
    weights = np.cos(np.deg2rad(aurora_lat[lat_mask]))

    rmse_df = pd.DataFrame(index=list(chain.from_iterable(var_groups.values())), columns=time_labels, dtype=float)

    for i, dt in enumerate(forecast_datetimes):
        if dt not in pd.to_datetime(truth_ds.time.values):
            continue
        truth_sel = truth_ds.sel(time=dt)

        for group_name, ch_list in var_groups.items():
            for ch in ch_list:
                var = ch[0]
                p = int(ch[1:])
                if ch not in forecast_channels:
                    continue
                pred_field = forecast_ds[ch].isel(lead_time=i).values[lat_mask, :]
                try:
                    truth_field = truth_sel[var].sel(isobaricInhPa=p).values[lat_mask, :]
                except KeyError:
                    continue
                rmse = compute_rmse_weighted(pred_field, truth_field, weights)
                rmse_df.loc[ch, time_labels[i]] = rmse

    # Save CSV
    rmse_df.to_csv(f"{output_dir}/aurora_rmse_{band_name}_lat.csv")

    # Plot selected levels
    for group_name, prefix in {
        "temperature": "t",
        "zonal_wind": "u",
        "meridional_wind": "v",
        "specific_humidity": "q",
        "geopotential": "z"
    }.items():
        plt.figure(figsize=(12, 5))
        for level in [1000, 500, 300]:
            var = f"{prefix}{level}"
            if var in rmse_df.index:
                y = rmse_df.loc[var].astype(float)
                plt.plot(forecast_datetimes, y, label=var)
        plt.title(f"RMSE at {band_name} Latitude - {group_name.replace('_',' ').title()}")
        plt.xlabel("Forecast Time")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rmse_{group_name}_{band_name}_lat.png", dpi=300)
        plt.close()
