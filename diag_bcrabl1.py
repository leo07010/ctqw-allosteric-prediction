"""
Diagnose BCR-ABL1 BCE-seeded ENAQT top-10 and active-site distances.
Find why residues 207 (9.5Å) and 321 (9.9Å) rank above true allosteric residues.
"""
import os, types, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_hybrid import GVPHybrid
from dataset import load_all_challenge_samples
from evaluate_auprc import precision_recall_auc, hit_at_k
from train_enaqt import enaqt_rwr_train

BCE_TOP_K = 30
N_ITER    = 150

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

def main():
    ckpt = torch.load("../outputs/hybrid/best_hybrid.pt", map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    model = GVPHybrid(
        node_s_dim=24,node_v_dim=1,edge_s_dim=24,edge_v_dim=1,
        hidden_s=args.get("hidden_s",64),hidden_v=args.get("hidden_v",8),
        n_layers=args.get("n_layers",4),cutoff_dist=12.0,K=args.get("K",32),
    )
    model.load_state_dict(ckpt["model"]); model.eval()
    proteins = load_all_challenge_samples(device="cpu")
    prot = {p["name"]: p for p in proteins}["BCR_ABL1"]

    def to_np(x): return x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
    allo_mask   = to_np(prot["allosteric_mask"]).astype(bool)
    active_mask = to_np(prot["active_mask"]).astype(bool)
    coords      = to_np(prot["coords"])
    N = len(allo_mask)

    bce, H_t = get_bce_and_H(prot, model)

    # Distance to active site
    active_coords = coords[active_mask]
    dist_to_active = np.array([
        np.linalg.norm(active_coords - coords[j], axis=1).min()
        for j in range(N)
    ])
    # Distance to allosteric mask
    allo_idx = np.where(allo_mask)[0]
    dist_to_allo = np.array([
        np.linalg.norm(coords[allo_idx] - coords[j], axis=1).min()
        for j in range(N)
    ])

    # BCE seed (top-30)
    bce_s = bce.copy(); bce_s[active_mask] = -1.0
    top_k_idx = np.argsort(-bce_s)[:BCE_TOP_K]
    seed = torch.zeros(N, dtype=torch.float32)
    seed[top_k_idx] = 1.0
    seed /= seed.sum()

    # ENAQT with best gamma (from previous run: 0.0016)
    log_g = torch.tensor(np.log(0.0016))
    with torch.no_grad():
        p = enaqt_rwr_train(H_t, log_g, seed, n_iter=N_ITER)
    enaqt_scores = p.numpy().copy()
    enaqt_scores[active_mask] = -1.0

    print("=" * 75)
    print("BCR-ABL1  BCE-seeded ENAQT  TOP-15 RESIDUES")
    print(f"  active site: {active_mask.sum()} residues  allo mask: {allo_mask.sum()} residues")
    print("=" * 75)
    print(f"{'Rk':<4} {'ResIdx':>7} {'ENAQTsc':>8} {'BCE':>7} {'InSeed':>7} "
          f"{'Dist→Allo':>10} {'Dist→Act':>10}  Hit  Notes")
    print("-" * 75)

    order = np.argsort(-enaqt_scores)[:15]
    for rk, idx in enumerate(order, 1):
        in_seed  = idx in top_k_idx
        da = dist_to_allo[idx]
        dact = dist_to_active[idx]
        hit  = allo_mask[idx] or da < 8.0
        act_flag = " [NEAR-ACTIVE]" if dact < 15.0 and not active_mask[idx] else ""
        seed_flag = " [seed]" if in_seed else ""
        print(f"  {rk:<4} {idx:>7} {enaqt_scores[idx]:>8.4f} {bce[idx]:>7.4f} "
              f"{'✓' if in_seed else '':>7} "
              f"{da:>10.1f} {dact:>10.1f}  "
              f"{'✓' if hit else '✗'}{act_flag}{seed_flag}")

    print("\nBCE SEED residues (top-30 BCE, active excluded):")
    print(f"{'ResIdx':>8} {'BCE':>7} {'Dist→Allo':>10} {'Dist→Act':>10}  Hit")
    for idx in top_k_idx:
        da = dist_to_allo[idx]
        dact = dist_to_active[idx]
        hit = allo_mask[idx] or da < 8.0
        print(f"  {idx:>7} {bce[idx]:>7.4f} {da:>10.1f} {dact:>10.1f}  {'✓' if hit else '✗'}")

    # Count how many seed residues are "near active" vs "remote"
    seed_near_active = sum(1 for i in top_k_idx if dist_to_active[i] < 15.0)
    seed_remote      = sum(1 for i in top_k_idx if dist_to_active[i] >= 15.0)
    print(f"\nSeed breakdown: {seed_near_active} near-active (<15Å), {seed_remote} remote (≥15Å)")

    # Remote BCE seed: exclude near-active residues
    print("\n--- REMOTE SEED (BCE top-30 excluding residues within 15Å of active site) ---")
    bce_remote = bce.copy()
    bce_remote[active_mask] = -1.0
    bce_remote[dist_to_active < 15.0] = -1.0
    remote_idx = np.argsort(-bce_remote)[:BCE_TOP_K]
    seed_r = torch.zeros(N, dtype=torch.float32)
    seed_r[remote_idx] = 1.0
    seed_r /= seed_r.sum()

    with torch.no_grad():
        p_r = enaqt_rwr_train(H_t, log_g, seed_r, n_iter=N_ITER)
    enaqt_r = p_r.numpy().copy(); enaqt_r[active_mask] = -1.0

    order_r = np.argsort(-enaqt_r)[:5]
    hits = 0
    print(f"{'Rk':<4} {'ResIdx':>7} {'Score':>8} {'Dist→Allo':>10} {'Dist→Act':>10}  Hit")
    for rk, idx in enumerate(order_r, 1):
        da = dist_to_allo[idx]; dact = dist_to_active[idx]
        hit = allo_mask[idx] or da < 8.0
        if hit: hits += 1
        print(f"  {rk:<4} {idx:>7} {enaqt_r[idx]:>8.4f} {da:>10.1f} {dact:>10.1f}  {'✓' if hit else '✗'}")
    print(f"Remote-seeded ENAQT BCR-ABL1 Hit@5 = {hits/5:.2f}")

if __name__ == "__main__":
    main()
