#!/bin/bash
#SBATCH --job-name=linear_kf_em_only
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

#SBATCH --array=0-59%10

#SBATCH --output=slurm_logs/linear_synthetic_ssm/kf_em/%x-%A_%a.out
#SBATCH --error=slurm_logs/linear_synthetic_ssm/kf_em/%x-%A_%a.err

echo "===== KF + EM Experiment Started ====="

# ---------------- ENV ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate sdn_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p slurm_logs/linear_synthetic_ssm/kf_em

echo "Node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# =========================================================
#         DEFINE EXPERIMENT GRID (KF + EM ONLY)
# =========================================================

DELAYS=(0 2 5 10)
NOISES=(0.0 0.05 0.1)
SEEDS=(0 1 2 3 4)

IDX=$SLURM_ARRAY_TASK_ID

D=${DELAYS[$((IDX % 4))]}
N=${NOISES[$(( (IDX / 4) % 3 ))]}
S=${SEEDS[$(( (IDX / (4*3)) % 5 ))]}

M="kf"
EXTRA="--use_em"

echo "Delay: $D"
echo "Noise: $N"
echo "Seed: $S"
echo "Model: $M"
echo "Use EM: 1"

# =========================================================
#                RUN EXPERIMENT
# =========================================================

echo "Running command:"
echo python experiments/run_linear.py \
    --config configs/linear.yaml \
    --delay $D \
    --noise $N \
    --seed $S \
    --model $M \
    $EXTRA

python experiments/run_linear.py \
    --config configs/linear.yaml \
    --delay $D \
    --noise $N \
    --seed $S \
    --model $M \
    $EXTRA

echo "===== DONE ====="