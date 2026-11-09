#!/bin/bash
#SBATCH --job-name=aurora_t_perturbed_corr
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/CC_aurora/T_PERTURBED_CORR/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/CC_aurora/T_PERTURBED_CORR/logs/%x_%j.err

WORK_DIR="/umbc/rs/pi_halem/users/yashd1/CC_aurora/T_PERTURBED_CORR"
mkdir -p "${WORK_DIR}/logs"

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate

RUN_ROOT="/umbc/scratch/${USER}/aurora_t_perturbed_corr/${SLURM_JOB_ID}"
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
export HDF5_USE_FILE_LOCKING=FALSE

cd "${WORK_DIR}"
python -u "${WORK_DIR}/inf.py"
