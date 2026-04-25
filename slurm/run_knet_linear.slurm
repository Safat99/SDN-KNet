#!/bin/bash
#SBATCH --job-name=knet_linear
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH --array=0-59%10
#SBATCH --output=slurm_logs/knet/%x-%A_%a.out
#SBATCH --error=slurm_logs/knet/%x-%A_%a.err

echo "===== KalmanNet Experiment Started ====="

# ---------------- ENV ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate sdn_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p slurm_logs/knet

echo "Node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# =========================================================
#         DEFINE EXPERIMENT GRID
# =========================================================

DELAYS=(0 2 5 10)
NOISES=(0.0 0.05 0.1)
SEEDS=(0 1 2 3 4)

# total = 4 * 3 * 5 = 60

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

python -m experiments.run_kalmannet_linear_final \
    --config configs/linear.yaml \
    --delay $D \
    --noise $N \
    --seed $S

echo "===== DONE ====="