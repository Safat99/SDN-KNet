#!/bin/bash
#SBATCH --job-name=jena_sdn_knet
#SBATCH --account=priority-johnsheppard
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH --array=0-35%10

#SBATCH --output=slurm_logs/jena_knet/%x-%A_%a.out
#SBATCH --error=slurm_logs/jena_knet/%x-%A_%a.err

echo "===== Jena SDN-KalmanNet Experiment Started ====="

# ---------------- ENV ----------------
module load CUDA/12.2.0
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate sdn_tf

export PYTHONPATH="$SLURM_SUBMIT_DIR"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p slurm_logs/jena_knet

echo "Node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# =========================================================
#         EXPERIMENT GRID
# =========================================================

DELAYS=(0 2 5 10)
SEEDS=(0 1 2)

IDX=$SLURM_ARRAY_TASK_ID

D=${DELAYS[$((IDX % 4))]}
S=${SEEDS[$(( (IDX / 4) % 3 ))]}

echo "Delay: $D"
echo "Seed: $S"

# =========================================================
#                RUN
# =========================================================

python experiments/run_knet_jena.py \
    --config configs/jena.yaml \
    --delay $D \
    --seed $S

echo "===== DONE ====="