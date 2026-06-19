"""
Random Walk with Restart (RWR) for allosteric site prediction.

No training required — pure graph algorithm.
Seed nodes = active site residues.
p = alpha * p0 + (1-alpha) * W * p   (iterate until convergence)

Usage:
  python eval_rwr.py --topk 5 --alpha 0.2 --cutoff 12.0
"""

import argparse
import numpy as np
import torch

from pipeline import parse_pdb_ca, build_contact_graph
from dataset import load_all_challenge_samples, APO_ACTIVE_SITES


def rwr(A: np.ndarray, seed_idx: list, alpha: float = 0.2, max_iter: int = 1000, tol: float = 1e-8):
    """
    Random Walk with Restart.

    Args:
        A:        (N,N) weighted adjacency (symmetric, non-negative)
        seed_idx: list of seed node indices (active site)
        alpha:    restart probability
        max_iter: max iterations
        tol:      convergence threshold

    Returns:
        p: (N,) stationary distribution
    """
    N = A.shape[0]

    # Column-normalize A → transition matrix W
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    W = A / col_sums[np.newaxis, :]   # (N, N): W[:,j] sums to 1

    # Initial / restart distribution: uniform over seeds
    p0 = np.zeros(N)
    p0[seed_idx] = 1.0 / len(seed_idx)

    p = p0.copy()
    for _ in range(max_iter):
        p_new = alpha * p0 + (1.0 - alpha) * W @ p
        if np.abs(p_new - p).max() < tol:
            p = p_new
            break
        p = p_new

    return p


def evaluate_rwr(topk=5, alpha=0.2, cutoff=12.0):
    """Run RWR on all 3 challenge proteins and report hit rates."""
    challenge_configs = [
        {
            "name":      "KRAS_G12C",
            "pdb":       "pdbs/4OBE.pdb",
            "apo_key":   "4OBE",
            "holo_pdb":  "pdbs/6OIM.pdb",
            "allo_lig":  ["MOV"],
        },
        {
            "name":      "BCR_ABL1",
            "pdb":       "pdbs/1OPL.pdb",
            "apo_key":   "1OPL",
            "holo_pdb":  "pdbs/5MO4.pdb",
            "allo_lig":  ["AY7"],
        },
        {
            "name":      "CardiacMyosin",
            "pdb":       "pdbs/5TBY.pdb",
            "apo_key":   "5TBY",
            "holo_pdb":  "pdbs/6C1H.pdb",
            "allo_lig":  ["ADP", "MG"],
        },
    ]

    # Load challenge samples for allosteric labels
    test_samples = load_all_challenge_samples(device="cpu")
    sample_by_name = {s["name"]: s for s in test_samples}

    print(f"\nRWR Evaluation  alpha={alpha}  cutoff={cutoff}Å  top-{topk}")
    print("=" * 60)

    results = []
    for cfg in challenge_configs:
        name = cfg["name"]
        sample = sample_by_name[name]

        # Build contact graph from apo structure
        residues, coords = parse_pdb_ca(cfg["pdb"])
        N = len(residues)

        A, _ = build_contact_graph(coords, cutoff=cutoff)

        # Active site seed indices
        active_resnums = set(APO_ACTIVE_SITES.get(cfg["apo_key"], []))
        seed_idx = [r["index"] for r in residues if r["resnum"] in active_resnums]

        if not seed_idx:
            centroid = coords.mean(axis=0)
            seed_idx = [int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))]
            print(f"  [{name}] WARNING: no active site → using centroid node {seed_idx[0]}")

        # RWR
        p = rwr(A, seed_idx, alpha=alpha)

        # Mask out active site residues from ranking
        p_masked = p.copy()
        p_masked[seed_idx] = -1.0

        k = min(topk, N - len(seed_idx))
        topk_idx = np.argsort(p_masked)[::-1][:k]

        # Evaluate: hit if any known allosteric residue within 8Å
        allo_mask = sample["allosteric_mask"].numpy()
        known_idx = np.where(allo_mask)[0]
        coords_t = coords

        hits = 0
        for idx in topk_idx:
            dists = np.linalg.norm(coords_t[known_idx] - coords_t[idx], axis=1)
            if dists.min() < 8.0:
                hits += 1

        hit_rate = hits / k
        results.append({"name": name, "hits": hits, "topk": k, "hit_rate": hit_rate})
        print(f"  [{name}]  N={N}  seeds={len(seed_idx)}  top-{k}: {hits}/{k} = {hit_rate:.2f}")

    avg = np.mean([r["hit_rate"] for r in results])
    print(f"\n  avg hit rate: {avg:.3f}")
    return results, avg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topk",   type=int,   default=5)
    p.add_argument("--alpha",  type=float, default=0.2)
    p.add_argument("--cutoff", type=float, default=12.0)
    args = p.parse_args()

    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    evaluate_rwr(topk=args.topk, alpha=args.alpha, cutoff=args.cutoff)
