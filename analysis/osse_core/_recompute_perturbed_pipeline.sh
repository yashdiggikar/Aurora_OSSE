#!/bin/bash
set -e

echo "$(date) Waiting for SLURM jobs 225407 (T_PERTURBED_CORR) and 225408 (UV_PERTURBED_CORR) to finish..."
while squeue -j 225407,225408 -h 2>/dev/null | grep -q .; do
  sleep 60
done

echo "$(date) Both jobs left the queue. Exit status:"
sacct -j 225407,225408 --format=JobID,State,ExitCode,Elapsed

T_STATE=$(sacct -j 225407 --format=State --noheader | head -1 | tr -d ' ')
UV_STATE=$(sacct -j 225408 --format=State --noheader | head -1 | tr -d ' ')
echo "T_PERTURBED_CORR state: $T_STATE"
echo "UV_PERTURBED_CORR state: $UV_STATE"

if [[ "$T_STATE" != "COMPLETED" || "$UV_STATE" != "COMPLETED" ]]; then
  echo "ERROR: one or both jobs did not complete successfully (T=$T_STATE, UV=$UV_STATE). Aborting recompute."
  exit 1
fi

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
cd /umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/Aurora

echo "$(date) Running compute_global_rmse_corr.py ..."
python compute_global_rmse_corr.py

echo "$(date) Running compute_latband_rmse_corr.py ..."
python compute_latband_rmse_corr.py

echo "$(date) Running compute_spatial_bias_corr.py ..."
python compute_spatial_bias_corr.py

echo "$(date) Syncing curated CSVs to perturbation_results/Aurora/..."
for exp in T_PERTURBED_CORR UV_PERTURBED_CORR; do
  mkdir -p "/umbc/rs/pi_halem/users/yashd1/perturbation_results/Aurora/${exp}"
  for kind in global_rmse latband_rmse spatial_bias; do
    cp -v "/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/Aurora/${exp}/${kind}_${exp}.csv" \
          "/umbc/rs/pi_halem/users/yashd1/perturbation_results/Aurora/${exp}/"
  done
done

echo "$(date) PIPELINE COMPLETE"
