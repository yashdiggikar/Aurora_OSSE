#!/bin/bash
#SBATCH --job-name=aurora_osse_analysis
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/CC_aurora/logs/analysis_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/CC_aurora/logs/analysis_%j.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/CC_aurora/logs
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python /umbc/rs/pi_halem/users/yashd1/CC_aurora/analyze_osse.py
