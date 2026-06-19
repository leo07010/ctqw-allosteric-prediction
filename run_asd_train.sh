#!/bin/bash
#SBATCH --job-name=asd_train
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=GOV114009
#SBATCH --output=/home/leo07010/DNA/logs/asd_train_%j.out
#SBATCH --error=/home/leo07010/DNA/logs/asd_train_%j.err

cd /home/leo07010/DNA/cleveland_clinic
mkdir -p ../logs ../outputs/train_asd

echo "=== ASD Training Start ==="
date

python -u train_asd.py \
    --processed_dir data/asd_processed \
    --epochs        150 \
    --lr            1e-3 \
    --hidden_s      64 \
    --hidden_v      8 \
    --n_layers      4 \
    --t_max         50.0 \
    --ctqw_steps    100 \
    --neg_weight    0.1 \
    --rank_weight   1.0 \
    --margin        0.05 \
    --topk          5 \
    --eval_every    10 \
    --device        cuda \
    --outdir        ../outputs/train_asd \
    --seed          42

echo "=== Done ==="
date
