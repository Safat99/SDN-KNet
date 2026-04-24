#!/bin/bash
#SBATCH --job-name=tune_linear_gru
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --output=slurm_logs/tuning/linear/%x-%j.out
#SBATCH --error=slurm_logs/tuning/linear/%x-%j.err

echo "===== TUNING GRU (LINEAR) STARTED ====="

# ---------------- ENV ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate sdn_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p slurm_logs/tuning/linear
mkdir -p results/tuning_linear

echo "Node: $(hostname)"

# =========================================================
#                RUN TUNING
# =========================================================

python experiments/tune_linear_gru.py

echo "===== DONE ====="