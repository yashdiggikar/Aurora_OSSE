import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === CONFIG ===
rmse_csv = "/xace/d1/yashd1/aurora_temp_ins/scoring_aurora_model/aurora_rmse_all_vars.csv"
acc_csv = "/xace/d1/yashd1/aurora_temp_ins/scoring_aurora_model/aurora_acc_all_vars.csv"
output_dir = "/xace/d1/yashd1/aurora_temp_ins/the_final_plots"
levels_to_plot = [1000, 500, 300]  # hPa
var_groups = {
    "temperature": "t",
    "zonal_wind": "u",
    "meridional_wind": "v",
    "specific_humidity": "q",
    "geopotential": "z",
}

# === LOAD CSVs ===
rmse_df = pd.read_csv(rmse_csv, index_col=0)
acc_df = pd.read_csv(acc_csv, index_col=0)
time = pd.to_datetime(rmse_df.columns)

# === PLOT FUNCTION ===
def plot_metric(metric_df, metric_name):
    for group_name, var_prefix in var_groups.items():
        plt.figure(figsize=(12, 5))
        for level in levels_to_plot:
            var = f"{var_prefix}{level}"
            if var in metric_df.index:
                y = metric_df.loc[var].astype(float)
                plt.plot(time, y, label=f"{var}")
        plt.title(f"{metric_name.upper()} - {group_name.replace('_',' ').title()}")
        plt.xlabel("Forecast Time")
        plt.ylabel(metric_name.upper())
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/aurora_{metric_name}_{group_name}_selected_levels.png", dpi=300)
        plt.close()

# === PLOT RMSE AND ACC ===
plot_metric(rmse_df, "rmse")
plot_metric(acc_df, "acc")

print("✅ Plots saved for levels 1000, 500, 300.")
