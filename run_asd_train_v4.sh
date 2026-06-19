#!/bin/bash
#SBATCH --job-name=ctqw_enaqt_v4
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --account=GOV114009
#SBATCH --output=../logs/train_v4_%j.out
#SBATCH --error=../logs/train_v4_%j.err

cd /home/leo07010/DNA/cleveland_clinic
mkdir -p ../logs ../outputs/train_asd_v4

echo "=== CTQW v4: ENAQT (Rebentrost 2009) ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_asd_v4.py \
    --processed_dir data/asd_processed \
    --epochs 200 \
    --lr 1e-3 \
    --hidden_s 64 \
    --hidden_v 8 \
    --n_layers 4 \
    --K 32 \
    --alpha 0.15 \
    --topk 5 \
    --eval_every 10 \
    --outdir ../outputs/train_asd_v4

echo "=== Done ==="
