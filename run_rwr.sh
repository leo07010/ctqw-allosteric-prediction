#!/bin/bash
#SBATCH --job-name=rwr_eval
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=GOV114009
#SBATCH --output=/home/leo07010/DNA/logs/rwr_%j.out
#SBATCH --error=/home/leo07010/DNA/logs/rwr_%j.err

cd /home/leo07010/DNA/cleveland_clinic

echo "=== RWR Evaluation ==="
date

for alpha in 0.05 0.10 0.15 0.20 0.30 0.50; do
    echo ""
    echo "--- alpha=$alpha ---"
    python -u eval_rwr.py --topk 5 --alpha $alpha --cutoff 12.0
done

echo ""
echo "=== Done ==="
date
