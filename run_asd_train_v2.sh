#!/bin/bash
#SBATCH --job-name=ctqw_v2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=GOV114009
#SBATCH --output=/home/leo07010/DNA/logs/ctqw_v2_%j.out
#SBATCH --error=/home/leo07010/DNA/logs/ctqw_v2_%j.err

cd /home/leo07010/DNA/cleveland_clinic
mkdir -p ../logs ../outputs/train_asd_v2

echo "=== CTQW v2 (communicability loss) ==="
date

python -u train_asd_v2.py \
    --processed_dir data/asd_processed \
    --epochs        200 \
    --lr            1e-3 \
    --hidden_s      64 \
    --hidden_v      8 \
    --n_layers      4 \
    --topk          5 \
    --eval_every    10 \
    --device        cuda \
    --outdir        ../outputs/train_asd_v2 \
    --seed          42

echo "=== Done ==="
date
