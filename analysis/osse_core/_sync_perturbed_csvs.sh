#!/bin/bash
set -e

echo "$(date) global_rmse (225467) already COMPLETED. Waiting for latband_rmse (225487, 32G) and spatial_bias (225488, 32G) on chip-gpu..."
while squeue -M chip-gpu -j 225487,225488 -h 2>/dev/null | grep -q .; do
  sleep 60
done

echo "$(date) Both jobs left the queue. Exit status:"
sacct -M chip-gpu -j 225487,225488 --format=JobID,JobName,State,ExitCode,Elapsed

FAIL=0
for j in 225487 225488; do
  ST=$(sacct -M chip-gpu -j "$j" --format=State --noheader | head -1 | tr -d ' ')
  echo "Job $j state: $ST"
  [[ "$ST" == "COMPLETED" ]] || FAIL=1
done

if [[ "$FAIL" == "1" ]]; then
  echo "ERROR: one or more analysis jobs did not complete successfully. Not syncing CSVs -- check logs in PERTURBATION_EXPERIMENTS/logs/."
  exit 1
fi

echo "$(date) Syncing curated CSVs to perturbation_results/Aurora/..."
for exp in T_PERTURBED_CORR UV_PERTURBED_CORR; do
  mkdir -p "/umbc/rs/pi_halem/users/yashd1/perturbation_results/Aurora/${exp}"
  for kind in global_rmse latband_rmse spatial_bias; do
    cp -v "/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/Aurora/${exp}/${kind}_${exp}.csv" \
          "/umbc/rs/pi_halem/users/yashd1/perturbation_results/Aurora/${exp}/"
  done
done

echo "$(date) PIPELINE COMPLETE"
