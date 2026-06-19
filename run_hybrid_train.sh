#!/bin/bash
#SBATCH --job-name=gvp_hybrid
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --account=GOV114009
#SBATCH --output=../logs/hybrid_%j.out
#SBATCH --error=../logs/hybrid_%j.err

cd /home/leo07010/DNA/cleveland_clinic
mkdir -p ../logs ../outputs/hybrid

echo "=== GVP-Hybrid: BCE + CTQW communicability ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_hybrid.py \
    --processed_dir data/asd_processed \
    --epochs 150 \
    --lr 1e-3 \
    --hidden_s 64 \
    --hidden_v 8 \
    --n_layers 4 \
    --K 32 \
    --lambda_ctqw 0.1 \
    --beta 0.5 \
    --topk 5 \
    --eval_every 10 \
    --outdir ../outputs/hybrid

echo "=== Done ==="
