"""
train_enaqt.py
==============
Train GVPHybrid with ENAQT contrastive loss (no ESM for now — PDB only).
Loss = bce_w * BCE + enaqt_w * ENAQT_contrastive

This replaces communicability loss with ENAQT to get H optimized for
quantum transport from active site → allosteric residues.

Key difference from model_hybrid training:
  - communicability_loss → enaqt_contrastive_loss
  - log_gamma per protein is a model parameter (single shared scalar)
  - Training teaches H to support directed quantum transport
"""

import os, sys, glob, random, warnings, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_hybrid import GVPHybrid
from evaluate_auprc import precision_recall_auc, hit_at_k


def load_asd_split(processed_dir, device="cpu", val_frac=0.15, seed=42):
    """Load ASD processed .pt files, split into train/val."""
    files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    random.seed(seed); random.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files, train_files = files[:n_val], files[n_val:]

    def _load(flist):
        samples = []
        for f in flist:
            try:
                s = torch.load(f, map_location="cpu", weights_only=False)
                graph = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in s["graph"].items()}
                samples.append({
                    "name":            s["name"],
                    "graph":           graph,
                    "allosteric_mask": s["allo_labels"].bool().to(device),
                    "active_mask":     torch.zeros(
                        s["n_residues"], dtype=torch.bool, device=device),
                    "coords":          s["coords"],
                })
            except Exception as e:
                print(f"  skip {f}: {e}")
        return samples

    return _load(train_files), _load(val_files)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    epochs       = 100,
    lr           = 1e-3,
    weight_decay = 1e-4,
    clip_grad    = 1.0,
    bce_w        = 0.5,
    enaqt_w      = 0.5,
    enaqt_n_iter = 50,      # RWR iterations (fewer for training speed)
    enaqt_alpha  = 0.15,
    eval_every   = 5,
    hidden_s     = 64,
    hidden_v     = 8,
    n_layers     = 4,
    K            = 32,
    cutoff_dist  = 12.0,
    batch_size   = 1,       # one protein at a time (proteins differ in N)
)


def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("CPU")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# ENAQT RWR (fast, fewer iterations for training)
# ─────────────────────────────────────────────────────────────────────────────

def enaqt_rwr_train(H, log_gamma, seed_mask, alpha=0.15, n_iter=50):
    """Differentiable RWR on ENAQT rates. Fewer iters for speed."""
    gamma  = torch.exp(log_gamma).clamp(min=1e-6, max=1e3)
    d      = torch.diag(H)
    delta2 = (d.unsqueeze(1) - d.unsqueeze(0)) ** 2
    H_off  = H - torch.diag_embed(d)
    K      = 2.0 * H_off**2 * gamma / (gamma**2 + delta2 + 1e-10)
    cols   = K.sum(0).clamp(min=1e-10)
    W      = K / cols.unsqueeze(0)
    p0     = seed_mask.float(); p0 = p0 / p0.sum().clamp(min=1e-10)
    p      = p0.clone()
    for _ in range(n_iter):
        p = alpha * p0 + (1 - alpha) * (W @ p)
    return p


def enaqt_loss_train(H, log_gamma, allo_mask, n_iter=50):
    """
    Uniform-seed ENAQT loss: seed from ALL residues equally.
    Physical interpretation: from any random starting point, ENAQT
    quantum walk should naturally concentrate at allosteric residues.
    Works without active_mask — suitable for ASD dataset.
    """
    if allo_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=H.device)
    N  = H.shape[0]
    seed = torch.ones(N, dtype=torch.bool, device=H.device)  # uniform seed
    p    = enaqt_rwr_train(H, log_gamma, seed, n_iter=n_iter)
    pos  = p[allo_mask].mean()
    neg  = p[~allo_mask].mean()
    return -torch.log(pos / (pos + neg + 1e-8) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, log_gamma, samples, n_iter=100):
    model.eval()
    auprcs, h5s = [], []
    for s in samples:
        graph       = s["graph"]
        allo_mask   = s["allosteric_mask"]
        active_mask = s["active_mask"]
        coords      = s["coords"].numpy() if isinstance(s["coords"], torch.Tensor) \
                      else np.array(s["coords"])

        if allo_mask.sum() == 0:
            continue

        logits, H = model(graph)
        bce_s   = torch.sigmoid(logits).cpu().numpy()
        N_prot  = H.shape[0]
        seed_u  = torch.ones(N_prot, dtype=torch.bool, device=H.device)
        p       = enaqt_rwr_train(H, log_gamma, seed_u, n_iter=n_iter)
        enaqt_s = p.cpu().numpy()

        def norm01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-9)

        bce_n = norm01(bce_s)
        enq_n = norm01(enaqt_s)
        combo = bce_n + enq_n

        labels = allo_mask.cpu().numpy().astype(int)
        auprcs.append(precision_recall_auc(combo, labels))
        h5s.append(hit_at_k(combo, labels, coords, k=5))
    return np.mean(auprcs), np.mean(h5s)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = get_device()

    # Load data
    processed_dir = "data/asd_processed"
    train_samples, val_samples = load_asd_split(processed_dir, device=device)
    print(f"Train: {len(train_samples)} proteins  Val: {len(val_samples)} proteins")

    # Model — start from best_hybrid.pt if it exists
    model = GVPHybrid(
        node_s_dim=24, node_v_dim=1, edge_s_dim=24, edge_v_dim=1,
        hidden_s=CFG["hidden_s"], hidden_v=CFG["hidden_v"],
        n_layers=CFG["n_layers"], cutoff_dist=CFG["cutoff_dist"], K=CFG["K"],
    ).to(device)

    ckpt_init = "../outputs/hybrid/best_hybrid.pt"
    if os.path.exists(ckpt_init):
        ck = torch.load(ckpt_init, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Warm-start from {ckpt_init}  epoch={ck.get('epoch')}")
    else:
        print("Training from scratch")

    # Shared log_gamma — one learnable scalar for all proteins
    # (protein-specific γ requires ESM; this is the first step)
    log_gamma = nn.Parameter(torch.tensor(-1.0, device=device))  # γ≈0.37 init

    optimizer = optim.Adam(
        list(model.parameters()) + [log_gamma],
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG["epochs"])

    best_ap, best_h5 = 0.0, 0.0
    os.makedirs("outputs/enaqt", exist_ok=True)
    log_rows = []

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        t0 = time.time()
        losses, bce_ls, enaqt_ls = [], [], []

        random.shuffle(train_samples)
        for s in train_samples:
            graph       = s["graph"]
            allo_mask   = s["allosteric_mask"]
            active_mask = s["active_mask"]

            if allo_mask.sum() == 0:
                continue

            logits, H = model(graph)

            # BCE loss
            labels = allo_mask.float()
            n_pos  = labels.sum().clamp(min=1)
            n_neg  = (len(labels) - n_pos).clamp(min=1)
            pw     = (n_neg / n_pos).unsqueeze(0)
            bce_l  = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pw)

            # ENAQT contrastive loss (uniform seed — no active_mask needed)
            enaqt_l = enaqt_loss_train(
                H, log_gamma, allo_mask,
                n_iter=CFG["enaqt_n_iter"])

            loss = CFG["bce_w"] * bce_l + CFG["enaqt_w"] * enaqt_l

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + [log_gamma], CFG["clip_grad"])
            optimizer.step()

            losses.append(loss.item())
            bce_ls.append(bce_l.item())
            enaqt_ls.append(enaqt_l.item())

        scheduler.step()

        avg_loss   = np.mean(losses)
        avg_bce    = np.mean(bce_ls)
        avg_enaqt  = np.mean(enaqt_ls)
        gamma_val  = torch.exp(log_gamma).item()
        elapsed    = time.time() - t0

        print(f"Ep {epoch:3d}/{CFG['epochs']}  "
              f"loss={avg_loss:.4f}  bce={avg_bce:.4f}  enaqt={avg_enaqt:.4f}  "
              f"γ={gamma_val:.4f}  ({elapsed:.1f}s)", flush=True)

        if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
            val_ap, val_h5 = evaluate(model, log_gamma, val_samples)
            print(f"  VAL  AUPRC={val_ap:.3f}  Hit@5={val_h5:.2f}", flush=True)
            log_rows.append({"epoch": epoch, "loss": avg_loss,
                             "auprc": val_ap, "hit5": val_h5,
                             "gamma": gamma_val})

            if val_ap > best_ap or (val_ap == best_ap and val_h5 > best_h5):
                best_ap, best_h5 = val_ap, val_h5
                torch.save({
                    "model": model.state_dict(),
                    "log_gamma": log_gamma.item(),
                    "epoch": epoch,
                    "val_auprc": val_ap,
                    "val_hit5": val_h5,
                    "cfg": CFG,
                }, "outputs/enaqt/best_enaqt.pt")
                print(f"  → saved best  AUPRC={val_ap:.3f}  Hit@5={val_h5:.2f}")

    with open("outputs/enaqt/train_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"\nDone. Best val AUPRC={best_ap:.3f}  Hit@5={best_h5:.2f}")


if __name__ == "__main__":
    main()
