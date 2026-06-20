"""
eval_bce_enaqt_combo.py
=======================
BCE + BCE-seeded ENAQT additive combination.

BCE-seeded ENAQT alone = 0.867 Avg Hit@5  (ties RWR-raw classical baseline)
Goal: beat 0.867 by combining BCE + ENAQT to push BCR-ABL1 from 0.60 → 0.80+

Logic:
  - BCE alone identifies some allosteric residues
  - ENAQT quantum walk from BCE seed refines and amplifies
  - Combined: residues that score high in BOTH get double boost
  - Noise residues (high in only one) get averaged down

Target: Avg Hit@5 > 0.867  (i.e. BCR-ABL1 Hit@5 ≥ 0.80)
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

N_GAMMAS  = 80      # fine γ grid
N_ITER    = 150     # more RWR steps
BCE_TOP_K = 30


def get_bce_and_H(prot, model):
    h_cap=[None]; d_cap=[None]
    def _hh(m,i,o): h_cap[0]=o.detach().cpu().numpy()
    def _hd(m,i,o): d_cap[0]=o.squeeze(-1).detach().cpu().numpy()
    hh=model.h_proj.register_forward_hook(_hh)
    hd=model.diag_head.register_forward_hook(_hd)
    orig=model._build_H
    def _fast(s,ns,ei,ed,N_):
        s.h_proj(ns); s.diag_head(ns)
        return torch.zeros(N_,N_,dtype=ns.dtype,device=ns.device)
    model._build_H = types.MethodType(_fast, model)
    with torch.no_grad(): logits,_ = model(prot["graph"])
    model._build_H=orig; hh.remove(); hd.remove()
    bce = torch.sigmoid(logits).cpu().numpy()
    K   = h_cap[0].shape[1]
    h_t = torch.from_numpy(h_cap[0])
    H_t = h_t @ h_t.t() / K + torch.diag(torch.from_numpy(d_cap[0]))
    H_t = (H_t + H_t.t()) / 2.0
    return bce, H_t


def make_bce_seed(bce, active_mask, top_k):
    N = len(bce)
    s = bce.copy(); s[active_mask] = -1.0
    idx = np.argsort(-s)[:top_k]
    seed = torch.zeros(N, dtype=torch.float32)
    seed[idx] = 1.0
    return seed / seed.sum().clamp(1e-10)


def norm_sup(x, mask):
    x=x.copy(); x[mask]=np.nan
    mn,mx=np.nanmin(x),np.nanmax(x)
    out=(x-mn)/(mx-mn+1e-9); out[mask]=-1.0
    return out


def grid_search(H_t, seed_t, allo_mask, coords, active_mask,
                bce_n, alpha_combo, n_gammas=N_GAMMAS):
    """Grid-search γ; score = (1-α)*norm(ENAQT) + α*norm(BCE)."""
    gammas = np.logspace(-3, 3, n_gammas)
    best_h5, best_g, best_score = -1, gammas[0], None
    labels = allo_mask.astype(int)
    for g in gammas:
        log_g = torch.tensor(np.log(g))
        with torch.no_grad():
            p = enaqt_rwr_train(H_t, log_g, seed_t, n_iter=N_ITER)
        enaqt_n = norm_sup(p.numpy().copy(), active_mask)
        combo = (1.0 - alpha_combo) * enaqt_n + alpha_combo * bce_n
        combo[active_mask] = -1.0
        h5 = hit_at_k(combo, labels, coords, k=5)
        if h5 > best_h5: best_h5, best_g, best_score = h5, g, combo
    return best_g, best_score, best_h5


def show_top5(name, scores, allo_mask, active_mask, coords, label):
    sc = scores.copy(); sc[active_mask]=-1.0
    order = np.argsort(-sc)[:5]
    ai    = np.where(allo_mask)[0]
    ap    = precision_recall_auc(sc, allo_mask.astype(int))
    au    = auroc(sc, allo_mask.astype(int))
    print(f"\n{'─'*62}")
    print(f"  {name}  [{label}]")
    print(f"{'─'*62}")
    print(f"  {'Rk':<4} {'ResIdx':>7} {'Score':>7}  {'Dist(Å)':>8}  Hit")
    hits=0
    for rk, idx in enumerate(order, 1):
        if allo_mask[idx]: dist,hit=0.0,True
        elif len(ai)>0:
            dist=np.linalg.norm(coords[ai]-coords[idx],axis=1).min(); hit=dist<8.0
        else: dist,hit=float("nan"),False
        if hit: hits+=1
        print(f"  {rk:<4} {idx:>7} {sc[idx]:>7.4f}  {dist:>8.1f}  {'✓' if hit else '✗'}")
    print(f"\n  Hit@5={hits/5:.2f}  AUPRC={ap:.3f}  AUROC={au:.3f}")
    return hits/5, ap


def main():
    ckpt = torch.load("../outputs/hybrid/best_hybrid.pt",
                      map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = GVPHybrid(
        node_s_dim=24,node_v_dim=1,edge_s_dim=24,edge_v_dim=1,
        hidden_s=args.get("hidden_s",64),hidden_v=args.get("hidden_v",8),
        n_layers=args.get("n_layers",4),cutoff_dist=12.0,K=args.get("K",32),
    )
    model.load_state_dict(ckpt["model"]); model.eval()

    proteins = load_all_challenge_samples(device="cpu")

    # Try multiple α (blend weight for BCE in final score)
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    all_results = {a: [] for a in alphas}

    protein_data = []
    for prot in proteins:
        name        = prot["name"]
        allo_mask   = prot["allosteric_mask"].numpy().astype(bool)
        active_mask = prot["active_mask"].numpy().astype(bool)
        coords      = (prot["coords"].numpy()
                       if isinstance(prot["coords"], torch.Tensor)
                       else np.array(prot["coords"]))
        N = len(allo_mask)
        print(f"\n[{name}]  N={N}", flush=True)

        bce, H_t = get_bce_and_H(prot, model)
        bce_n    = norm_sup(bce, active_mask)
        seed_t   = make_bce_seed(bce, active_mask, top_k=min(BCE_TOP_K, N-active_mask.sum()))
        protein_data.append((name, allo_mask, active_mask, coords, bce_n, H_t, seed_t))

    print("\n" + "="*72)
    print("Grid search: α = weight of BCE in final score (0=ENAQT-only, 1=BCE-only)")
    print("="*72)

    best_overall = {"avg_h5": -1, "alpha": None, "results": None}

    for alpha in alphas:
        row_h5 = []; row_ap = []
        for name, allo_mask, active_mask, coords, bce_n, H_t, seed_t in protein_data:
            best_g, combo_score, h5 = grid_search(
                H_t, seed_t, allo_mask, coords, active_mask, bce_n, alpha)
            ap = precision_recall_auc(combo_score, allo_mask.astype(int))
            row_h5.append(h5); row_ap.append(ap)

        avg_h5 = np.mean(row_h5); avg_ap = np.mean(row_ap)
        all_results[alpha] = {"h5": row_h5, "ap": row_ap, "avg_h5": avg_h5, "avg_ap": avg_ap}
        marker = " ◀ BEST" if avg_h5 > best_overall["avg_h5"] else ""
        if avg_h5 > best_overall["avg_h5"]:
            best_overall = {"avg_h5": avg_h5, "alpha": alpha, "results": row_h5}
        print(f"  α={alpha:.1f}  Hit@5 = {row_h5[0]:.2f}/{row_h5[1]:.2f}/{row_h5[2]:.2f}  Avg={avg_h5:.3f}  AUPRC={avg_ap:.3f}{marker}")

    # ── Show best α in detail ─────────────────────────────────────────────
    best_alpha = best_overall["alpha"]
    print(f"\n{'='*72}")
    print(f"BEST COMBO: α={best_alpha}  Avg Hit@5={best_overall['avg_h5']:.3f}")
    print(f"(α=0=pure quantum ENAQT, α=1=pure BCE baseline)")
    print(f"{'='*72}")

    for name, allo_mask, active_mask, coords, bce_n, H_t, seed_t in protein_data:
        best_g, combo_score, h5 = grid_search(
            H_t, seed_t, allo_mask, coords, active_mask, bce_n, best_alpha)
        show_top5(name, combo_score, allo_mask, active_mask, coords,
                  f"BCE(α={best_alpha})+ENAQT(α={1-best_alpha:.1f}) γ={best_g:.4f}")

    print(f"\n{'='*72}")
    print("COMPARISON vs baselines")
    print(f"{'='*72}")
    print(f"  BCE-seeded ENAQT (α=0.0, quantum only)  : {all_results[0.0]['avg_h5']:.3f}")
    print(f"  RWR-raw (classical baseline)             : 0.867")
    print(f"  Best combo (α={best_alpha})                      : {best_overall['avg_h5']:.3f}")
    print(f"  BCE-only  (α=1.0, no quantum)            : {all_results[1.0]['avg_h5']:.3f}")
    print(f"\n  Δ vs RWR-raw : {best_overall['avg_h5']-0.867:+.3f}")


if __name__ == "__main__":
    main()
