#!/bin/bash
#SBATCH --job-name=linear_synthetic_system_test
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
# #SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

## SBATCH --array=0-0 # now for test one job
#SBATCH --array=0-119%10

#SBATCH --output=slurm_logs/linear_synthetic_ssm/updated/%x-%A_%a.out
#SBATCH --error=slurm_logs/linear_synthetic_ssm/updated/%x-%A_%a.err

echo "===== Linear Experiment Started ====="

# ---------------- ENV ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate sdn_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p slurm_logs/linear

echo "Node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# =========================================================
#         DEFINE EXPERIMENT GRID
# =========================================================

DELAYS=(0 2 5 10)
NOISES=(0.0 0.05 0.1)
SEEDS=(0 1 2 3 4)
MODELS=("gru" "kf")
USE_EM=(0 1)   # only relevant for KF

# total = 4 * 3 * 5 * 2 * 2 = 240
# BUT we restrict EM to KF only → effectively 120 runs

IDX=$SLURM_ARRAY_TASK_ID

D=${DELAYS[$((IDX % 4))]}
N=${NOISES[$(( (IDX / 4) % 3 ))]}
S=${SEEDS[$(( (IDX / (4*3)) % 5 ))]}
M=${MODELS[$(( (IDX / (4*3*5)) % 2 ))]}
E=${USE_EM[$(( (IDX / (4*3*5*2)) % 2 ))]}

echo "Delay: $D"
echo "Noise: $N"
echo "Seed: $S"
echo "Model: $M"
echo "Use EM: $E"

# =========================================================
#                RUN EXPERIMENT
# =========================================================

EXTRA=""

if [ "$M" == "kf" ] && [ "$E" -eq 1 ]; then
    EXTRA="--use_em"
fi

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
