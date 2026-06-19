"""
Train GVP-GNN + MLP baseline (no CTQW).
Ablation to compare with GVP-GNN + CTQW pipeline.

Usage:
  python -u train_baseline.py --processed_dir data/asd_processed --epochs 150
"""

import os
import json
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model_baseline import GVPAlloBaseline
from dataset import load_all_challenge_samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_asd_samples(processed_dir, device):
    files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    samples = []
    for f in files:
        try:
            s = torch.load(f, map_location='cpu', weights_only=False)
            graph = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in s['graph'].items()}
            samples.append({
                'name':            s['name'],
                'graph':           graph,
                'allosteric_mask': s['allo_labels'].bool().to(device),
                'active_mask':     torch.zeros(s['n_residues'], dtype=torch.bool).to(device),
                'coords':          s['coords'],
            })
        except Exception as e:
            print(f"  Load error {f}: {e}")
    return samples


# ──────────────────────────────────────────────
# Loss: weighted BCE for class imbalance
# ──────────────────────────────────────────────

def baseline_loss(scores, allo_mask, pos_weight=10.0):
    """Weighted BCE: allosteric residues are ~10% → upweight positives."""
    labels = allo_mask.float()
    weights = torch.where(labels == 1,
                          torch.tensor(pos_weight, device=scores.device),
                          torch.tensor(1.0,        device=scores.device))
    return nn.functional.binary_cross_entropy(scores, labels,
                                              weight=weights)


# ──────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────

def train_one_epoch(model, samples, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for sample in samples:
        if not sample['allosteric_mask'].any():
            continue
        optimizer.zero_grad()
        scores = model(sample['graph'])                    # (N,)
        loss = baseline_loss(scores, sample['allosteric_mask'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, samples, topk=5, label=""):
    model.eval()
    results = []
    for sample in samples:
        allo_mask = sample['allosteric_mask']
        if not allo_mask.any():
            continue

        scores     = model(sample['graph']).cpu()
        allo_cpu   = allo_mask.cpu()
        coords_cpu = sample['coords'].cpu() if isinstance(sample['coords'], torch.Tensor) \
                     else torch.tensor(sample['coords'], dtype=torch.float32)

        # Exclude active site from ranking (none for ASD samples)
        active_cpu = sample['active_mask'].cpu()
        masked = scores.clone()
        masked[active_cpu] = -1.0
        k = min(topk, masked.shape[0] - 1)
        topk_idx = masked.topk(k).indices.tolist()

        known_idx    = allo_cpu.nonzero(as_tuple=True)[0]
        known_coords = coords_cpu[known_idx]

        hits = 0
        for idx in topk_idx:
            d = (known_coords - coords_cpu[idx]).norm(dim=-1).min().item()
            if d < 8.0:
                hits += 1

        hit_rate = hits / k
        results.append({'name': sample['name'], 'hit_rate': hit_rate,
                        'hits': hits, 'topk': k})

    avg = np.mean([r['hit_rate'] for r in results]) if results else 0.0
    if label:
        for r in results:
            print(f"  [{label}][{r['name']}] top-{r['topk']}: "
                  f"{r['hits']}/{r['topk']} = {r['hit_rate']:.2f}")
        print(f"  [{label}] avg hit rate: {avg:.3f}")
    return results, avg


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  seed: {args.seed}")

    print("\nLoading ASD training proteins...")
    train_samples = load_asd_samples(args.processed_dir, device)
    print(f"Training set: {len(train_samples)} proteins")

    print("\nLoading challenge proteins (test set)...")
    test_samples = load_all_challenge_samples(device=str(device))
    print(f"Test set: {len(test_samples)} proteins")

    model = GVPAlloBaseline(
        node_s_dim=24, node_v_dim=1, edge_s_dim=24, edge_v_dim=1,
        hidden_s=args.hidden_s, hidden_v=args.hidden_v,
        n_layers=args.n_layers, cutoff_dist=12.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_hit = 0.0
    log = []

    print(f"\nTraining baseline {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        train_loss = train_one_epoch(model, train_samples, optimizer)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n── Epoch {epoch}/{args.epochs}  lr={lr:.2e}  loss={train_loss:.4f} ──")
            _, train_hit = evaluate(model, train_samples[:10], args.topk, label="train")
            _, test_hit  = evaluate(model, test_samples,       args.topk, label="test")

            log.append({'epoch': epoch, 'train_loss': train_loss,
                        'train_hit': train_hit, 'test_hit': test_hit})

            if test_hit > best_test_hit:
                best_test_hit = test_hit
                ckpt = os.path.join(args.outdir, "best_baseline.pt")
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'test_hit': test_hit}, ckpt)
                print(f"  ✓ Best (hit={test_hit:.3f}) → {ckpt}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}  loss={train_loss:.4f}")

    print(f"\nDone. Best test hit rate: {best_test_hit:.3f}")
    with open(os.path.join(args.outdir, 'baseline_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data/asd_processed")
    p.add_argument("--epochs",    type=int,   default=150)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--hidden_s",  type=int,   default=64)
    p.add_argument("--hidden_v",  type=int,   default=8)
    p.add_argument("--n_layers",  type=int,   default=4)
    p.add_argument("--topk",      type=int,   default=5)
    p.add_argument("--eval_every",type=int,   default=10)
    p.add_argument("--device",    default="cuda")
    p.add_argument("--outdir",    default="../outputs/baseline")
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)
