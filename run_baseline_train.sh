#!/bin/bash
#SBATCH --job-name=gvp_baseline
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=GOV114009
#SBATCH --output=/home/leo07010/DNA/logs/baseline_%j.out
#SBATCH --error=/home/leo07010/DNA/logs/baseline_%j.err

cd /home/leo07010/DNA/cleveland_clinic
mkdir -p ../logs ../outputs/baseline

echo "=== GVP-GNN Baseline (no CTQW) ==="
date

python -u train_baseline.py \
    --processed_dir data/asd_processed \
    --epochs        150 \
    --lr            1e-3 \
    --hidden_s      64 \
    --hidden_v      8 \
    --n_layers      4 \
    --topk          5 \
    --eval_every    10 \
    --device        cuda \
    --outdir        ../outputs/baseline \
    --seed          42

echo "=== Done ==="
date
