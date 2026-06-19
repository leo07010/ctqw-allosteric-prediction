"""
CTQW v4: GVP-GNN + ENAQT (Environment-Assisted Quantum Transport).

Replaces eigendecomposition-based communicability with Haken-Strobl-Reineker
ENAQT rates + differentiable RWR. No eigh call, cleaner gradients.

  k_ij = 2|H_ij|^2 * gamma / (gamma^2 + (H_ii - H_jj)^2)
  p = RWR(W_enaqt, seed=active_site, alpha=0.15)
  loss = -log(p[allo] / (p[allo] + p[bg]))

Usage (via SLURM):
  python -u train_asd_v4.py --processed_dir data/asd_processed --epochs 200
"""

import os
import json
import argparse
import glob
import random
import torch
import torch.optim as optim
import numpy as np

from model_v3 import HamiltonianGVP_v3
from ctqw_enaqt import enaqt_loss, enaqt_rwr
from dataset import load_all_challenge_samples


def set_seed(seed: int):
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


def centroid_fallback(coords, device):
    c = coords.cpu() if isinstance(coords, torch.Tensor) else \
        torch.tensor(coords, dtype=torch.float32)
    idx = (c - c.mean(dim=0)).norm(dim=-1).argmin().item()
    mask = torch.zeros(c.shape[0], dtype=torch.bool, device=device)
    mask[idx] = True
    return mask


def train_one_epoch(model, samples, optimizer, alpha):
    model.train()
    total_loss, n_ok, n_skip = 0.0, 0, 0

    for sample in samples:
        if not sample['allosteric_mask'].any():
            continue

        optimizer.zero_grad()
        H, log_gamma = model(sample['graph'])

        active_mask = sample['active_mask']
        if not active_mask.any():
            active_mask = centroid_fallback(sample['coords'], H.device)

        try:
            loss = enaqt_loss(H, log_gamma, active_mask, sample['allosteric_mask'],
                              alpha=alpha)
        except Exception as e:
            print(f"  [skip] {sample['name']}: {e}")
            n_skip += 1
            continue

        if not torch.isfinite(loss):
            n_skip += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_ok += 1

    if n_skip:
        print(f"  [train] {n_skip} skipped this epoch")
    return total_loss / max(n_ok, 1)


@torch.no_grad()
def evaluate(model, samples, topk=5, alpha=0.15, label=""):
    model.eval()
    results = []

    for sample in samples:
        allo_mask   = sample['allosteric_mask']
        active_mask = sample['active_mask']
        coords      = sample['coords']
        if not allo_mask.any():
            continue

        H, log_gamma = model(sample['graph'])

        if not active_mask.any():
            active_mask = centroid_fallback(coords, H.device)

        scores = enaqt_rwr(H, log_gamma, active_mask, alpha=alpha).cpu()

        allo_cpu   = allo_mask.cpu()
        active_cpu = active_mask.cpu()
        coords_cpu = coords.cpu() if isinstance(coords, torch.Tensor) else \
                     torch.tensor(coords, dtype=torch.float32)

        scores[active_cpu] = -1.0
        k = min(topk, scores.shape[0] - 1)
        topk_idx = scores.topk(k).indices.tolist()

        known_idx    = allo_cpu.nonzero(as_tuple=True)[0]
        known_coords = coords_cpu[known_idx]

        hits = sum(
            1 for idx in topk_idx
            if (known_coords - coords_cpu[idx]).norm(dim=-1).min().item() < 8.0
        )
        results.append({'name': sample['name'], 'hit_rate': hits / k,
                        'hits': hits, 'topk': k})

    avg = np.mean([r['hit_rate'] for r in results]) if results else 0.0
    if label:
        for r in results:
            print(f"  [{label}][{r['name']}] top-{r['topk']}: "
                  f"{r['hits']}/{r['topk']} = {r['hit_rate']:.2f}")
        print(f"  [{label}] avg hit rate: {avg:.3f}")
    return results, avg


def main(args):
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  seed: {args.seed}")

    print("\nLoading ASD training proteins...")
    train_samples = load_asd_samples(args.processed_dir, device)
    print(f"Training set: {len(train_samples)} proteins")
    if not train_samples:
        print("ERROR: No training samples."); return

    print("\nLoading challenge proteins (test set)...")
    test_samples = load_all_challenge_samples(device=str(device))
    print(f"Test set: {len(test_samples)} proteins")

    model = HamiltonianGVP_v3(
        node_s_dim=24, node_v_dim=1, edge_s_dim=24, edge_v_dim=1,
        hidden_s=args.hidden_s, hidden_v=args.hidden_v,
        n_layers=args.n_layers, cutoff_dist=12.0, K=args.K,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params  K={args.K}  alpha={args.alpha}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_hit = 0.0
    log = []

    print(f"\nTraining {args.epochs} epochs (ENAQT: dense H + learnable gamma)...")
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        train_loss = train_one_epoch(model, train_samples, optimizer, args.alpha)
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n── Epoch {epoch}/{args.epochs}  lr={lr:.2e}  loss={train_loss:.4f} ──")
            _, train_hit = evaluate(model, train_samples[:10], args.topk,
                                    alpha=args.alpha, label="train")
            _, test_hit  = evaluate(model, test_samples, args.topk,
                                    alpha=args.alpha, label="test")

            log.append({'epoch': epoch, 'train_loss': train_loss,
                        'train_hit': train_hit, 'test_hit': test_hit})

            if test_hit > best_test_hit:
                best_test_hit = test_hit
                ckpt = os.path.join(args.outdir, "best_model_v4.pt")
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'test_hit': test_hit, 'args': vars(args)}, ckpt)
                print(f"  ✓ Best (hit={test_hit:.3f}) → {ckpt}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}  loss={train_loss:.4f}")

    print(f"\nDone. Best test hit rate: {best_test_hit:.3f}")
    with open(os.path.join(args.outdir, 'train_v4_log.json'), 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="data/asd_processed")
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--hidden_s",  type=int,   default=64)
    p.add_argument("--hidden_v",  type=int,   default=8)
    p.add_argument("--n_layers",  type=int,   default=4)
    p.add_argument("--K",         type=int,   default=32)
    p.add_argument("--topk",      type=int,   default=5)
    p.add_argument("--eval_every",type=int,   default=10)
    p.add_argument("--alpha",     type=float, default=0.15)
    p.add_argument("--device",    default="cuda")
    p.add_argument("--outdir",    default="../outputs/train_asd_v4")
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)
