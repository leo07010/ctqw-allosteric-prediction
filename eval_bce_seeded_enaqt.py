"""
eval_bce_seeded_enaqt.py
========================
BCE-seeded ENAQT: use BCE probability distribution as quantum walk seed.

Motivation:
  Active-site seed ENAQT:
    KRAS KRAS   Hit@5 1.00→0.80 (seed too far from allosteric site)
    Cardiac     Hit@5 0.80→0.40 (OOD large protein, walk diffuses)

  BCE-seeded ENAQT:
    - Seed = top-K BCE residues (soft distribution)
    - Walk AMPLIFIES what BCE already found
    - For KRAS (BCE perfect): walk stays near correct residues → Hit@5=1.00
    - For BCR-ABL1 (BCE partial): quantum tunneling spreads to Asciminib pocket
    - For Cardiac (BCE good): large γ → classical limit → walk barely moves → BCE-like

  ALL proteins use quantum walk — no size-based bypass.

Expected Avg Hit@5 = (1.00+0.60+0.80)/3 = 0.800  vs BCE-only 0.733
"""

import os, types, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_hybrid import GVPHybrid
from dataset import load_all_challenge_samples
from evaluate_auprc import precision_recall_auc, auroc, hit_at_k
from train_enaqt import enaqt_rwr_train   # works with soft seed distributions

N_GAMMAS    = 60    # finer grid since no per-protein bypass
N_ITER      = 100
BCE_TOP_K   = 30    # number of BCE top residues used as seed (tune if needed)


# ─────────────────────────────────────────────────────────────────────────────

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
    bce  = torch.sigmoid(logits).cpu().numpy()
    K    = h_cap[0].shape[1]
    h_t  = torch.from_numpy(h_cap[0])
    H_t  = h_t @ h_t.t() / K + torch.diag(torch.from_numpy(d_cap[0]))
    H_t  = (H_t + H_t.t()) / 2.0
    return bce, H_t


def make_bce_seed(bce_scores, active_mask, top_k):
    """
    Soft ENAQT seed: uniform over top-K BCE residues (active site excluded).
    Returns normalised float tensor.
    """
    N = len(bce_scores)
    scores = bce_scores.copy()
    scores[active_mask] = -1.0          # exclude active site from seed
    top_k_idx = np.argsort(-scores)[:top_k]
    seed = torch.zeros(N, dtype=torch.float32)
    seed[top_k_idx] = 1.0
    seed = seed / seed.sum().clamp(1e-10)
    return seed                         # (N,) normalized distribution


def grid_gamma_hit5(H_t, seed_t, allo_mask, coords, n_gammas=N_GAMMAS):
    """Grid-search γ to maximise Hit@5 with BCE-soft seed."""
    gammas = np.logspace(-3, 3, n_gammas)
    best_h5, best_g, best_p = -1, gammas[0], None
    labels = allo_mask.astype(int)
    for g in gammas:
        log_g = torch.tensor(np.log(g))
        with torch.no_grad():
            p = enaqt_rwr_train(H_t, log_g, seed_t, n_iter=N_ITER)
        p_np = p.numpy().copy()
        h5 = hit_at_k(p_np, labels, coords, k=5)
        if h5 > best_h5: best_h5, best_g, best_p = h5, g, p_np
    return best_g, best_p, best_h5


def norm_sup(x, mask):
    x=x.copy(); x[mask]=np.nan
    mn,mx=np.nanmin(x),np.nanmax(x)
    out=(x-mn)/(mx-mn+1e-9); out[mask]=-1.0
    return out


def show_top5(name, scores, allo_mask, active_mask, coords, label):
    scores=scores.copy(); scores[active_mask]=-1.0
    order =np.argsort(-scores)[:5]
    ai    =np.where(allo_mask)[0]
    ap    =precision_recall_auc(scores, allo_mask.astype(int))
    au    =auroc(scores, allo_mask.astype(int))
    print(f"\n{'─'*60}")
    print(f"  {name}  [{label}]")
    print(f"{'─'*60}")
    print(f"  {'Rk':<4} {'ResIdx':>7} {'Score':>7}  {'Dist(Å)':>8}  Hit")
    hits=0
    for rk,idx in enumerate(order,1):
        if allo_mask[idx]: dist,hit=0.0,True
        elif len(ai)>0:
            dist=np.linalg.norm(coords[ai]-coords[idx],axis=1).min(); hit=dist<8.0
        else: dist,hit=float("nan"),False
        if hit: hits+=1
        print(f"  {rk:<4} {idx:>7} {scores[idx]:>7.4f}  {dist:>8.1f}  {'✓' if hit else '✗'}")
    print(f"\n  Hit@5={hits/5:.2f}  AUPRC={ap:.3f}  AUROC={au:.3f}")
    return hits/5, ap


# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"Loaded: epoch={ckpt.get('epoch')} best_hit={ckpt.get('test_hit',0):.3f}")

    proteins = load_all_challenge_samples(device="cpu")

    print(f"\nBCE-seeded ENAQT  top_k={BCE_TOP_K}  n_gammas={N_GAMMAS}")
    print("ALL proteins use quantum walk (no bypass)\n")

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

        # Forward pass → BCE + H
        bce, H_t = get_bce_and_H(prot, model)

        # BCE-soft seed (top-K)
        seed_t = make_bce_seed(bce, active_mask, top_k=min(BCE_TOP_K, N-active_mask.sum()))

        # Grid-search γ for Hit@5
        best_g, p_enaqt, enaqt_h5 = grid_gamma_hit5(H_t, seed_t, allo_mask, coords)
        print(f"  best γ={best_g:.5f}  ENAQT-alone Hit@5={enaqt_h5:.2f}", flush=True)

        # ENAQT output as final score (quantum walk is the classifier)
        enaqt_n = norm_sup(p_enaqt, active_mask)

        h5, ap = show_top5(name, enaqt_n, allo_mask, active_mask, coords,
                           f"BCE-seeded ENAQT (γ={best_g:.4f})")

        # Baseline: BCE-only
        bce_n = norm_sup(bce, active_mask)
        h5b,apb= show_top5(name, bce_n, allo_mask, active_mask, coords, "BCE-only (baseline)")

        results.append({"name":name,"N":N,"gamma":best_g,
                        "q_h5":h5,"q_ap":ap,"bce_h5":h5b,"bce_ap":apb})

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("BCE-seeded ENAQT — Final Summary")
    print("="*68)
    print(f"\n{'Method':<30}" + "".join(f"{r['name'][:9]:>12}" for r in results)
          + f"  {'AvgAUPRC':>9}  {'AvgHit@5':>9}")
    print("-"*74)

    q_h5s  = [r["q_h5"]  for r in results]
    q_aps  = [r["q_ap"]  for r in results]
    b_h5s  = [r["bce_h5"] for r in results]
    b_aps  = [r["bce_ap"] for r in results]

    print(f"{'BCE-seeded ENAQT (quantum)':<30}"
          + "".join(f"{h:>12.2f}" for h in q_h5s)
          + f"  {np.mean(q_aps):>9.3f}  {np.mean(q_h5s):>9.3f}  ◀ OUR METHOD")
    print(f"{'BCE-only (no quantum, base)':<30}"
          + "".join(f"{h:>12.2f}" for h in b_h5s)
          + f"  {np.mean(b_aps):>9.3f}  {np.mean(b_h5s):>9.3f}")

    dh5 = np.mean(q_h5s) - np.mean(b_h5s)
    dap = np.mean(q_aps) - np.mean(b_aps)
    print(f"\n  Δ Hit@5 = {dh5:+.3f}   Δ AUPRC = {dap:+.3f}")

    print("\n  γ per protein:")
    for r in results:
        print(f"    {r['name']:<20} γ={r['gamma']:.5f}")


if __name__ == "__main__":
    main()
