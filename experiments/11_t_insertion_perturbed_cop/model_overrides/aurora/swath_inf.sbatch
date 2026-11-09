#!/bin/bash
#SBATCH --job-name=aurora_t_swath_ins
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/CC_aurora/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/CC_aurora/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --constraint=L40S

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
WORK_DIR="/umbc/rs/pi_halem/users/yashd1/CC_aurora/auroa_ins"
INFERENCE_SCRIPT="${WORK_DIR}/aurora/inf.py"
LOG_DIR="/umbc/rs/pi_halem/users/yashd1/CC_aurora/logs"
mkdir -p "${LOG_DIR}"

# =============================================================================
# ACTIVATE VENV
# =============================================================================
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate

# =============================================================================
# CACHE DIRS ON SCRATCH (avoids HF "no space left" on /home)
# =============================================================================
RUN_ROOT="/umbc/scratch/${USER}/aurora_swath/${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}/hf" "${RUN_ROOT}/xdg" "${RUN_ROOT}/tmp"

export HF_HOME="${RUN_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export XDG_CACHE_HOME="${RUN_ROOT}/xdg"
export TMPDIR="${RUN_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export TOKENIZERS_PARALLELISM=false

# Disable HDF5 POSIX file locking — required for NFS mounts (pi_halem is NFS).
# Without this, xarray/netcdf4 can throw "RuntimeError: NetCDF: HDF error"
# when writing or appending files across nodes.
export HDF5_USE_FILE_LOCKING=FALSE

# =============================================================================
# JOB INFO
# =============================================================================
echo "========================================================"
echo "Job        : ${SLURM_JOB_NAME}  (ID: ${SLURM_JOB_ID})"
echo "Start      : $(date)"
echo "Host       : $(hostname)"
echo "Node list  : ${SLURM_JOB_NODELIST:-n/a}"
echo "Partition  : ${SLURM_JOB_PARTITION:-n/a}"
echo "CUDA devs  : ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Python     : $(which python)"
echo "Work dir   : ${WORK_DIR}"
echo "Script     : ${INFERENCE_SCRIPT}"
echo "HF_HOME    : ${HF_HOME}"
echo "RUN_ROOT   : ${RUN_ROOT}"
echo "========================================================"

echo ""
echo "=== GPU INFO ==="
nvidia-smi -L  || true
nvidia-smi     || true
echo "================"

python - <<'PYEOF'
import os, torch, sys
print("Python           :", sys.version.split()[0])
print("torch            :", torch.__version__)
print("CUDA avail       :", torch.cuda.is_available())
print("HF_HOME          :", os.environ.get("HF_HOME"))
print("HDF5_LOCKING     :", os.environ.get("HDF5_USE_FILE_LOCKING"))
if torch.cuda.is_available():
    print("GPU name         :", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM GiB         :", round(props.total_memory / 1024**3, 2))
PYEOF

# =============================================================================
# RUN SWATH INFERENCE
# =============================================================================
# cd into the auroa_ins parent so that Python finds aurora/ as a package.
# inf.py also does sys.path.insert(0, CC_aurora/auroa_ins) explicitly.
cd "${WORK_DIR}"

echo ""
echo "=== Running swath-masked temperature injection inference ==="
python -u "${INFERENCE_SCRIPT}"

echo ""
echo "========================================================"
echo "Finished   : $(date)"
echo "========================================================"
