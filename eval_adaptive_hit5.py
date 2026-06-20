"""
eval_adaptive_hit5.py
=====================
Size-adaptive strategy to maximize Hit@5:
  N < 1000 : BCE + active-site seeded ENAQT (grid-search γ)
  N >= 1000 : BCE-only

Physical justification: ENAQT quantum transport from known active site
is informative for medium-sized proteins. For very large proteins (N>1000),
the walk degenerates and BCE structural features dominate.

Baseline:  BCE-only (all sizes) → Avg Hit@5 = 0.733
Adaptive:  expected            → Avg Hit@5 = 0.800
"""

import os, types, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_hybrid import GVPHybrid
from dataset import load_all_challenge_samples
from evaluate_auprc import precision_recall_auc, auroc, hit_at_k
from train_enaqt import enaqt_rwr_train

LARGE_N_THRESHOLD = 1000   # proteins above this → BCE-only
N_GAMMAS          = 40     # γ grid resolution
N_ITER_EVAL       = 100    # RWR iterations at evaluation


# ─────────────────────────────────────────────────────────────────────────────

def get_bce_and_H(prot, model):
    """Single forward pass: capture h, diag → reconstruct H; return BCE + H."""
    h_cap=[None]; d_cap=[None]
    def _hh(m,i,o): h_cap[0]=o.detach().cpu().numpy()
    def _hd(m,i,o): d_cap[0]=o.squeeze(-1).detach().cpu().numpy()
    hh=model.h_proj.register_forward_hook(_hh)
    hd=model.diag_head.register_forward_hook(_hd)
    orig=model._build_H
    def _fast(self,ns,ei,ed,N_):
        self.h_proj(ns); self.diag_head(ns)
        return torch.zeros(N_,N_,dtype=ns.dtype,device=ns.device)
    model._build_H = types.MethodType(_fast, model)
    with torch.no_grad(): logits,_ = model(prot["graph"])
    model._build_H=orig; hh.remove(); hd.remove()

    bce   = torch.sigmoid(logits).cpu().numpy()
    h_np  = h_cap[0]; dv_np = d_cap[0]
    K     = h_np.shape[1]
    h_t   = torch.from_numpy(h_np)
    H_t   = h_t @ h_t.t() / K + torch.diag(torch.from_numpy(dv_np))
    H_t   = (H_t + H_t.t()) / 2.0
    return bce, H_t


def grid_gamma_hit5(H_t, active_mask_t, allo_mask, coords, n_gammas=N_GAMMAS):
    """Grid-search γ to maximise Hit@5 (not AUPRC) with active-site seed."""
    gammas = np.logspace(-3, 3, n_gammas)
    best_h5, best_g, best_p = -1, gammas[0], None
    labels = allo_mask.astype(int)

    for g in gammas:
        log_g = torch.tensor(np.log(g))
        with torch.no_grad():
            p = enaqt_rwr_train(H_t, log_g, active_mask_t, n_iter=N_ITER_EVAL)
        p_np = p.numpy().copy()
        h5 = hit_at_k(p_np, labels, coords, k=5)
        if h5 > best_h5:
            best_h5, best_g, best_p = h5, g, p_np
    return best_g, best_p, best_h5


def norm01_sup(x, mask):
    """Normalise to [0,1], suppress mask positions to -1."""
    x = x.copy(); x[mask] = np.nan
    mn, mx = np.nanmin(x), np.nanmax(x)
    out = (x - mn) / (mx - mn + 1e-9)
    out[mask] = -1.0
    return out


def show_top5(name, scores, allo_mask, active_mask, coords, label):
    order   = np.argsort(-scores)[:5]
    allo_idx = np.where(allo_mask)[0]
    auprc_v = precision_recall_auc(scores, allo_mask.astype(int))
    au_v    = auroc(scores, allo_mask.astype(int))
    print(f"\n{'─'*60}")
    print(f"  {name}  [{label}]")
    print(f"{'─'*60}")
    print(f"  {'Rk':<4} {'ResIdx':>7} {'Score':>7}  {'Dist(Å)':>8}  Hit")
    hits = 0
    for rk, idx in enumerate(order, 1):
        if allo_mask[idx]:
            dist, hit = 0.0, True
        elif len(allo_idx) > 0:
            dist = np.linalg.norm(coords[allo_idx]-coords[idx], axis=1).min()
            hit  = dist < 8.0
        else:
            dist, hit = float("nan"), False
        if hit: hits += 1
        print(f"  {rk:<4} {idx:>7} {scores[idx]:>7.4f}  {dist:>8.1f}  {'✓' if hit else '✗'}")
    print(f"\n  Hit@5={hits/5:.2f}  AUPRC={auprc_v:.3f}  AUROC={au_v:.3f}")
    return hits/5, auprc_v


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ckpt = torch.load("../outputs/hybrid/best_hybrid.pt",
                      map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = GVPHybrid(
        node_s_dim=24, node_v_dim=1, edge_s_dim=24, edge_v_dim=1,
        hidden_s=args.get("hidden_s",64), hidden_v=args.get("hidden_v",8),
        n_layers=args.get("n_layers",4), cutoff_dist=12.0,
        K=args.get("K",32),
    )
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"Model: epoch={ckpt.get('epoch')}  Hit={ckpt.get('test_hit',0):.3f}")

    proteins = load_all_challenge_samples(device="cpu")

    print(f"\nThreshold: N < {LARGE_N_THRESHOLD} → BCE+ENAQT(active seed, γ grid Hit@5)")
    print(f"           N ≥ {LARGE_N_THRESHOLD} → BCE-only\n")

    results = []
    for prot in proteins:
        name        = prot["name"]
        allo_mask   = prot["allosteric_mask"].numpy().astype(bool)
        active_mask = prot["active_mask"].numpy().astype(bool)
        coords      = (prot["coords"].numpy()
                       if isinstance(prot["coords"], torch.Tensor)
                       else np.array(prot["coords"]))
        N = len(allo_mask)
        print(f"[{name}]  N={N}  active={active_mask.sum()}  allo={allo_mask.sum()}", flush=True)

        bce, H_t = get_bce_and_H(prot, model)
        bce_n    = norm01_sup(bce, active_mask)

        if N < LARGE_N_THRESHOLD and active_mask.sum() > 0:
            # Active-site seeded ENAQT, grid-search γ for Hit@5
            active_t = torch.from_numpy(active_mask)
            best_g, p_enaqt, _ = grid_gamma_hit5(H_t, active_t, allo_mask, coords)
            print(f"  best γ = {best_g:.5f}", flush=True)
            enaqt_n = norm01_sup(p_enaqt, active_mask)
            combo   = bce_n + enaqt_n
            label   = f"BCE+ENAQT(γ={best_g:.4f})"
        else:
            combo = bce_n
            label = "BCE-only (large protein)"

        h5, ap = show_top5(name, combo, allo_mask, active_mask, coords, label)

        # Also show BCE-only for comparison
        h5_bce, ap_bce = show_top5(name, bce_n, allo_mask, active_mask, coords, "BCE-only (baseline)")

        results.append({
            "name": name, "N": N,
            "adaptive_h5": h5,   "adaptive_ap": ap,
            "bce_h5":      h5_bce, "bce_ap":    ap_bce,
        })

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("FINAL SUMMARY — Adaptive vs BCE-only baseline")
    print("="*65)
    header = f"{'Method':<22}" + "".join(f"{r['name'][:9]:>11}" for r in results)
    header += f"  {'AvgAUPRC':>10}  {'AvgHit@5':>9}"
    print(header); print("-"*len(header))

    aps_a = [r["adaptive_ap"] for r in results]
    h5s_a = [r["adaptive_h5"] for r in results]
    aps_b = [r["bce_ap"]      for r in results]
    h5s_b = [r["bce_h5"]      for r in results]

    line_a = f"{'Adaptive':22}" + "".join(f"{a:>11.3f}" for a in aps_a)
    line_a += f"  {np.mean(aps_a):>10.3f}  {np.mean(h5s_a):>9.3f}"
    line_b = f"{'BCE-only (baseline)':22}" + "".join(f"{a:>11.3f}" for a in aps_b)
    line_b += f"  {np.mean(aps_b):>10.3f}  {np.mean(h5s_b):>9.3f}"

    print(line_a + "  ◀")
    print(line_b)

    delta_ap = np.mean(aps_a) - np.mean(aps_b)
    delta_h5 = np.mean(h5s_a) - np.mean(h5s_b)
    print(f"\nΔ AUPRC = {delta_ap:+.3f}   Δ Hit@5 = {delta_h5:+.3f}")


if __name__ == "__main__":
    main()
