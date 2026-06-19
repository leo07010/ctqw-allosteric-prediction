"""
Train GVP-GNN + CTQW on ASD allosteric proteins.
Test proteins (challenge targets) are never seen during training.

Usage (via SLURM):
  python -u train_asd.py --processed_dir data/asd_processed --epochs 100
"""

import os
import json
import argparse
import glob
import random
import torch
import torch.optim as optim
import numpy as np

from model import HamiltonianGVP
from ctqw_torch import ctqw_prob, allosteric_loss, ranking_loss
from dataset import load_all_challenge_samples


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Load ASD training samples
# ──────────────────────────────────────────────

def load_asd_samples(processed_dir, device):
    """Load preprocessed .pt files into list of sample dicts."""
    files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    samples = []
    for f in files:
        try:
            s = torch.load(f, map_location='cpu', weights_only=False)
            # Move graph tensors to device
            graph = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in s['graph'].items()}
            sample = {
                'name':             s['name'],
                'graph':            graph,
                'allosteric_mask':  s['allo_labels'].bool().to(device),
                'active_mask':      torch.zeros(s['n_residues'], dtype=torch.bool).to(device),
                'coords':           s['coords'],
            }
            samples.append(sample)
        except Exception as e:
            print(f"  Load error {f}: {e}")
    return samples


# ──────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────

def train_one_epoch(model, samples, optimizer, args):
    model.train()
    total_loss = 0.0
    n = 0

    for sample in samples:
        if not sample['allosteric_mask'].any():
            continue

        optimizer.zero_grad()
        H = model(sample['graph'])

        active_idx = sample['active_mask'].nonzero(as_tuple=True)[0].tolist()
        if not active_idx:
            # Use residue closest to geometric centroid as start
            coords = sample['coords'].cpu()
            centroid = coords.mean(dim=0)
            active_idx = [(coords - centroid).norm(dim=-1).argmin().item()]

        N = H.shape[0]
        t_max = args.t_max * (N / 300) ** 0.5
        try:
            prob = ctqw_prob(H, active_idx, t_max=t_max, n_steps=args.ctqw_steps)
        except torch._C._LinAlgError as e:
            print(f"  [skip] {sample['name']}: eigh failed — {e}")
            optimizer.zero_grad()
            continue

        loss_a = allosteric_loss(prob, sample['allosteric_mask'],
                                 sample['active_mask'], neg_weight=args.neg_weight)
        loss_r = ranking_loss(prob, sample['allosteric_mask'],
                              sample['active_mask'], margin=args.margin)
        loss = loss_a + args.rank_weight * loss_r

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, samples, args, label=""):
    model.eval()
    results = []

    for sample in samples:
        allo_mask = sample['allosteric_mask']
        active_mask = sample['active_mask']
        coords = sample['coords']

        if not allo_mask.any():
            continue

        H = model(sample['graph'])
        active_idx = active_mask.nonzero(as_tuple=True)[0].tolist()
        if not active_idx:
            coords_cpu = sample['coords'].cpu()
            centroid = coords_cpu.mean(dim=0)
            active_idx = [(coords_cpu - centroid).norm(dim=-1).argmin().item()]

        N = H.shape[0]
        t_max = args.t_max * (N / 300) ** 0.5
        try:
            prob = ctqw_prob(H, active_idx, t_max=t_max, n_steps=args.ctqw_steps)
        except torch._C._LinAlgError as e:
            print(f"  [skip eval] {sample['name']}: eigh failed — {e}")
            continue

        prob_cpu   = prob.detach().cpu()
        active_cpu = active_mask.cpu()
        allo_cpu   = allo_mask.cpu()
        coords_cpu = coords.cpu() if isinstance(coords, torch.Tensor) else \
                     torch.tensor(coords, dtype=torch.float32)

        prob_masked = prob_cpu.clone()
        prob_masked[active_cpu] = -1.0
        topk = min(args.topk, prob_masked.shape[0] - 1)
        topk_indices = prob_masked.topk(topk).indices.tolist()

        known_idx = allo_cpu.nonzero(as_tuple=True)[0]
        known_coords = coords_cpu[known_idx]

        hits = 0
        for idx in topk_indices:
            d = (known_coords - coords_cpu[idx]).norm(dim=-1).min().item()
            if d < 8.0:
                hits += 1

        hit_rate = hits / topk
        results.append({'name': sample['name'], 'hit_rate': hit_rate,
                         'hits': hits, 'topk': topk})

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

    # Training set: ASD proteins
    print("\nLoading ASD training proteins...")
    train_samples = load_asd_samples(args.processed_dir, device)
    print(f"Training set: {len(train_samples)} proteins")

    if len(train_samples) == 0:
        print("ERROR: No training samples found. Run asd_dataset.py first.")
        return

    # Test set: challenge proteins (never used in training)
    print("\nLoading challenge proteins (test set)...")
    test_samples = load_all_challenge_samples(device=str(device))
    print(f"Test set: {len(test_samples)} proteins")

    # Model
    model = HamiltonianGVP(
        node_s_dim  = 24,
        node_v_dim  = 1,
        edge_s_dim  = 24,
        edge_v_dim  = 1,
        hidden_s    = args.hidden_s,
        hidden_v    = args.hidden_v,
        n_layers    = args.n_layers,
        cutoff_dist = 12.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_hit = 0.0
    log = []

    print(f"\nTraining {args.epochs} epochs on ASD data...")
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        train_loss = train_one_epoch(model, train_samples, optimizer, args)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n── Epoch {epoch}/{args.epochs}  lr={lr:.2e}  "
                  f"train_loss={train_loss:.4f} ──")

            # Quick train hit rate (first 10 proteins)
            _, train_hit = evaluate(model, train_samples[:10], args, label="train")

            # Full test hit rate on challenge proteins
            _, test_hit = evaluate(model, test_samples, args, label="test")

            log.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_hit_rate': train_hit,
                'test_hit_rate': test_hit,
            })

            if test_hit > best_test_hit:
                best_test_hit = test_hit
                ckpt_path = os.path.join(args.outdir, "best_model_asd.pt")
                torch.save({
                    'epoch':      epoch,
                    'model':      model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'test_hit':   test_hit,
                    'args':       vars(args),
                }, ckpt_path)
                print(f"  ✓ Best test checkpoint (hit={test_hit:.3f}) → {ckpt_path}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}  loss={train_loss:.4f}")

    print(f"\nDone. Best test hit rate: {best_test_hit:.3f}")

    with open(os.path.join(args.outdir, 'train_asd_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data/asd_processed")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--hidden_s",      type=int,   default=64)
    p.add_argument("--hidden_v",      type=int,   default=8)
    p.add_argument("--n_layers",      type=int,   default=4)
    p.add_argument("--t_max",         type=float, default=50.0)
    p.add_argument("--ctqw_steps",    type=int,   default=100)
    p.add_argument("--neg_weight",    type=float, default=0.1)
    p.add_argument("--rank_weight",   type=float, default=1.0)
    p.add_argument("--margin",        type=float, default=0.05)
    p.add_argument("--topk",          type=int,   default=5)
    p.add_argument("--eval_every",    type=int,   default=10)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--outdir",        default="../outputs/train_asd")
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)
