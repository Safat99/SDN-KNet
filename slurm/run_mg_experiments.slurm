#!/bin/bash
#SBATCH --job-name=mg_gru
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

#SBATCH --array=0-59%20
#SBATCH --output=slurm_logs/mg/%x-%A_%a.out
#SBATCH --error=slurm_logs/mg/%x-%A_%a.err

echo "===== MG GRU Experiment Started ====="

# ---------------- ENV ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate sdn_knet_env

export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs/mg

echo "Node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# =========================================================
#         DEFINE EXPERIMENT GRID
# =========================================================

DELAYS=(0 2 5 10)
NOISES=(0.0 0.05 0.1)
SEEDS=(0 1 2 3 4)

# total combinations = 4 * 3 * 5 = 60

IDX=$SLURM_ARRAY_TASK_ID

D=${DELAYS[$((IDX % 4))]}
N=${NOISES[$(( (IDX / 4) % 3 ))]}
S=${SEEDS[$(( (IDX / (4*3)) % 5 ))]}

echo "Delay: $D"
echo "Noise: $N"
echo "Seed: $S"

# =========================================================
#                RUN EXPERIMENT
# =========================================================

python experiments/run_mg.py \
    --config configs/mackey_glass.yaml \
    --delay $D \
    --noise $N \
    --seed $S \
    --model gru

echo "===== DONE ====="
