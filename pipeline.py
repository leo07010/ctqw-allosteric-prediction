"""
Cleveland Clinic Challenge: Allosteric Site Prediction via CTQW
Pipeline: PDB → PocketMiner node potential → Hamiltonian → CTQW → ranking

Usage:
    python pipeline.py --pdb 4OBE.pdb --scored 4OBE_pocketminer.pdb --active 10-17,25-40,57-76
    python pipeline.py --pdb 4OBE.pdb --mock   # use mock PocketMiner scores for testing
"""

import numpy as np
import argparse
import os
from Bio import PDB
from Bio.PDB.vectors import Vector


# ─────────────────────────────────────────────
# 1. PDB PARSING
# ─────────────────────────────────────────────

def parse_pdb_ca(pdb_file: str):
    """
    Extract Cα coordinates and residue metadata from PDB.
    Returns:
        residues: list of dicts {chain, resnum, resname, index}
        coords:   np.ndarray shape (N, 3)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues, coords = [], []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " " and "CA" in res:
                    ca_vec = res["CA"].get_vector().get_array()
                    coords.append(ca_vec)
                    residues.append({
                        "chain":   chain.id,
                        "resnum":  res.id[1],
                        "resname": res.resname,
                        "index":   len(residues),
                    })
        break  # first model only

    return residues, np.array(coords, dtype=float)


# ─────────────────────────────────────────────
# 2. CONTACT GRAPH
# ─────────────────────────────────────────────

def build_contact_graph(coords: np.ndarray, cutoff: float = 8.0):
    """
    Build Cα contact graph.
    Edge weight w_ij = 1/d_ij^2 if d_ij < cutoff else 0.
    Returns:
        A:    np.ndarray (N,N) weighted adjacency (symmetric)
        dist: np.ndarray (N,N) pairwise Cα distances
    """
    diff = coords[:, None, :] - coords[None, :, :]        # (N,N,3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))             # (N,N)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where((dist < cutoff) & (dist > 0.0), 1.0 / dist ** 2, 0.0)

    return w, dist


# ─────────────────────────────────────────────
# 3. POCKETMINER SCORES
# ─────────────────────────────────────────────

def load_pocketminer_bfactor(scored_pdb: str, n_residues: int):
    """
    Load PocketMiner per-residue scores from its output PDB.
    PocketMiner stores scores in the B-factor column (range 0–100).
    Returns np.ndarray shape (N,) in range [0, 1].
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("scored", scored_pdb)

    scores = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] == " " and "CA" in res:
                    scores.append(res["CA"].get_bfactor() / 100.0)
        break

    scores = np.array(scores, dtype=float)
    assert len(scores) == n_residues, (
        f"Score count ({len(scores)}) != residue count ({n_residues}). "
        "Make sure the scored PDB matches the input PDB."
    )
    return scores


def mock_pocketminer_scores(n: int, seed: int = 42):
    """
    Placeholder scores for pipeline testing (no real PocketMiner needed).
    Generates a smooth random field over residues to mimic realistic output.
    """
    rng = np.random.default_rng(seed)
    # Smooth out with a running average to avoid salt-and-pepper noise
    raw = rng.random(n)
    kernel = np.ones(7) / 7
    smooth = np.convolve(raw, kernel, mode="same")
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth


# ─────────────────────────────────────────────
# 4. HAMILTONIAN CONSTRUCTION
# ─────────────────────────────────────────────

def build_hamiltonian(
    A: np.ndarray,
    pocket_scores: np.ndarray,
    alpha: float = 1.0,
    beta:  float = 0.5,
):
    """
    H = -alpha * A  +  diag(beta * (1 - pocket_scores_norm))

    Off-diagonal (-alpha * A):
        Negative adjacency = structural coupling.
        Walker flows through high-weight edges.

    Diagonal (node potential):
        High PocketMiner score → low on-site energy → walker lingers longer.
        beta controls the strength of the pocket bias.

    H is real symmetric → valid Hermitian Hamiltonian for CTQW.
    """
    scores_norm = pocket_scores.copy()
    rng = scores_norm.max() - scores_norm.min()
    if rng > 1e-8:
        scores_norm = (scores_norm - scores_norm.min()) / rng

    V = beta * (1.0 - scores_norm)           # high score → low energy
    H = -alpha * A + np.diag(V)
    H = (H + H.T) / 2.0                      # enforce symmetry
    return H


# ─────────────────────────────────────────────
# 5. CTQW
# ─────────────────────────────────────────────

def ctqw_time_average(H: np.ndarray, initial_node: int, t_max: float = 50.0, n_steps: int = 300):
    """
    Run CTQW and return time-averaged probability distribution.

    |ψ(t)⟩ = e^{-iHt} |initial_node⟩
    P_avg(i) = (1/T) ∫_0^T |⟨i|ψ(t)⟩|² dt

    Uses eigendecomposition for efficiency: O(N³) once, then O(N) per time step.
    """
    n = H.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(H)   # eigh guarantees real eigenvalues

    psi_0 = np.zeros(n, dtype=complex)
    psi_0[initial_node] = 1.0
    c = eigenvectors.conj().T @ psi_0               # project to eigenbasis

    times = np.linspace(0.0, t_max, n_steps)
    prob_sum = np.zeros(n)

    for t in times:
        phase  = np.exp(-1j * eigenvalues * t)
        psi_t  = eigenvectors @ (c * phase)
        prob_sum += np.abs(psi_t) ** 2

    return prob_sum / n_steps


def build_connectivity_matrix(H: np.ndarray, t_max: float = 50.0):
    """
    Compute N×N time-averaged connectivity matrix using closed-form solution.

    ⟨|⟨j|e^{-iHt}|i⟩|²⟩_t = Σ_k |V_ik|² |V_jk|²

    This is the required output for the Cleveland Clinic Challenge.
    (Valid for non-degenerate eigenvalues; exact in the T→∞ limit.)

    Returns C shape (N,N), where C[i,j] = connectivity from i to j.
    """
    _, V = np.linalg.eigh(H)
    V2 = V ** 2                                     # |V_ik|² shape (N, n_eigen)
    C  = V2 @ V2.T                                  # (N,N)
    return C


# ─────────────────────────────────────────────
# 6. RANKING
# ─────────────────────────────────────────────

def rank_allosteric_sites(prob_avg, residues, active_indices, top_k=5):
    """
    Rank residues by CTQW probability (descending), excluding active site.
    Returns list of dicts with rank, residue info, and score.
    """
    exclude = set(active_indices)
    ranked = sorted(
        [(i, float(prob_avg[i])) for i in range(len(prob_avg)) if i not in exclude],
        key=lambda x: -x[1],
    )
    result = []
    for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
        r = residues[idx]
        result.append({
            "rank":    rank,
            "index":   idx,
            "chain":   r["chain"],
            "resnum":  r["resnum"],
            "resname": r["resname"],
            "score":   score,
        })
    return result


# ─────────────────────────────────────────────
# 7. VALIDATION
# ─────────────────────────────────────────────

def validate_against_known(top_sites, known_allosteric_resnums,
                            residues, coords, dist_threshold=8.0):
    """
    Check how many of the top predicted sites are within dist_threshold Å
    of any known allosteric residue.

    Returns:
        hits: number of predictions within threshold
        details: per-prediction info
    """
    known_indices = [i for i, r in enumerate(residues)
                     if r["resnum"] in known_allosteric_resnums]

    details = []
    hits = 0
    for site in top_sites:
        pred_idx = site["index"]
        min_dist = min(
            np.linalg.norm(coords[pred_idx] - coords[ki])
            for ki in known_indices
        ) if known_indices else float("inf")

        is_hit = min_dist <= dist_threshold
        hits += int(is_hit)
        details.append({**site, "min_dist_to_known": min_dist, "hit": is_hit})

    return hits, details


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    pdb_file:             str,
    scored_pdb:           str  = None,
    active_site_resnums:  list = None,
    known_allosteric:     list = None,
    cutoff:               float = 8.0,
    alpha:                float = 1.0,
    beta:                 float = 0.5,
    t_max:                float = 50.0,
    n_steps:              int   = 300,
    top_k:                int   = 5,
    use_mock_scores:      bool  = False,
):
    print(f"\n{'='*55}")
    print(f" PDB → PocketMiner → H → CTQW → Ranking")
    print(f"{'='*55}")

    # Step 1: parse
    print(f"\n[1/5] Parsing {os.path.basename(pdb_file)}")
    residues, coords = parse_pdb_ca(pdb_file)
    n = len(residues)
    print(f"      {n} residues")

    # Step 2: contact graph
    print(f"[2/5] Building contact graph  (cutoff={cutoff} Å)")
    A, dist_matrix = build_contact_graph(coords, cutoff)
    n_edges = int(np.sum(A > 0)) // 2
    print(f"      {n_edges} edges")

    # Step 3: PocketMiner scores
    print(f"[3/5] Loading PocketMiner scores")
    if use_mock_scores:
        scores = mock_pocketminer_scores(n)
        print(f"      Using mock scores (testing mode)")
    else:
        assert scored_pdb, "Provide --scored <pocketminer_output.pdb> or use --mock"
        scores = load_pocketminer_bfactor(scored_pdb, n)
    print(f"      Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Step 4: Hamiltonian
    print(f"[4/5] Building Hamiltonian  (α={alpha}, β={beta})")
    H = build_hamiltonian(A, scores, alpha, beta)
    eigvals = np.linalg.eigvalsh(H)
    print(f"      Eigenvalue range: [{eigvals.min():.3f}, {eigvals.max():.3f}]")

    # Step 5: CTQW
    print(f"[5/5] Running CTQW  (t_max={t_max}, steps={n_steps})")

    # Map active site residue numbers → indices
    if active_site_resnums:
        active_indices = [i for i, r in enumerate(residues)
                          if r["resnum"] in active_site_resnums]
        if not active_indices:
            raise ValueError(f"None of {active_site_resnums} found in PDB residues.")
    else:
        # Fallback: highest-degree node
        active_indices = [int(np.argmax(np.sum(A > 0, axis=1)))]
        print(f"      No active site given → using residue "
              f"{residues[active_indices[0]]['resname']}"
              f"{residues[active_indices[0]]['resnum']}")

    # Average CTQW over multiple active-site starting nodes
    prob_avg = np.zeros(n)
    for idx in active_indices:
        prob_avg += ctqw_time_average(H, idx, t_max, n_steps)
    prob_avg /= len(active_indices)

    # Connectivity matrix (challenge output)
    print("      Computing N×N connectivity matrix (closed-form)...")
    C = build_connectivity_matrix(H)

    # Ranking
    top_sites = rank_allosteric_sites(prob_avg, residues, active_indices, top_k)

    # Validation (optional)
    val_details = None
    if known_allosteric:
        hits, val_details = validate_against_known(
            top_sites, known_allosteric, residues, coords
        )
        hit_rate = hits / len(top_sites)
    else:
        hits, hit_rate = 0, None

    # ── Print results ──
    print(f"\n{'─'*55}")
    print(f" Top-{top_k} Predicted Allosteric Sites")
    print(f"{'─'*55}")
    header = f"{'Rank':<5} {'Residue':<12} {'Chain':<6} {'CTQW Score':<12}"
    if known_allosteric:
        header += f"{'Dist to Known':<14} {'Hit'}"
    print(header)
    print("─" * (55 if not known_allosteric else 75))

    rows = val_details if val_details else top_sites
    for row in rows:
        line = (f"  {row['rank']:<4} {row['resname']}{row['resnum']:<8} "
                f"{row['chain']:<6} {row['score']:.6f}   ")
        if known_allosteric:
            line += f"{row['min_dist_to_known']:>8.2f} Å      {'✓' if row['hit'] else '✗'}"
        print(line)

    if known_allosteric:
        print(f"\n  Hit rate: {hits}/{len(top_sites)} within 8 Å of known allosteric site")

    return {
        "residues":             residues,
        "coords":               coords,
        "hamiltonian":          H,
        "pocket_scores":        scores,
        "ctqw_prob":            prob_avg,
        "connectivity_matrix":  C,
        "top_sites":            top_sites,
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_resnum_list(s: str):
    """Parse '10-17,25-40,57-76' or '12,15,20' into a flat list of ints."""
    nums = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            nums.extend(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(part))
    return nums


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allosteric site prediction via CTQW")
    parser.add_argument("--pdb",     required=True,  help="Input apo PDB file")
    parser.add_argument("--scored",  default=None,   help="PocketMiner output PDB (B-factor = score)")
    parser.add_argument("--mock",    action="store_true", help="Use mock PocketMiner scores (no real PocketMiner needed)")
    parser.add_argument("--active",  default=None,   help="Active site residues, e.g. '10-17,25-40'")
    parser.add_argument("--known",   default=None,   help="Known allosteric residues for validation")
    parser.add_argument("--cutoff",  type=float, default=8.0)
    parser.add_argument("--alpha",   type=float, default=1.0, help="Structural coupling strength")
    parser.add_argument("--beta",    type=float, default=0.5, help="Pocket bias strength")
    parser.add_argument("--tmax",    type=float, default=50.0)
    parser.add_argument("--steps",   type=int,   default=300)
    parser.add_argument("--topk",    type=int,   default=5)
    parser.add_argument("--outdir",  default=".",    help="Directory to save outputs")
    args = parser.parse_args()

    active = parse_resnum_list(args.active) if args.active else None
    known  = parse_resnum_list(args.known)  if args.known  else None

    result = run_pipeline(
        pdb_file            = args.pdb,
        scored_pdb          = args.scored,
        active_site_resnums = active,
        known_allosteric    = known,
        cutoff              = args.cutoff,
        alpha               = args.alpha,
        beta                = args.beta,
        t_max               = args.tmax,
        n_steps             = args.steps,
        top_k               = args.topk,
        use_mock_scores     = args.mock,
    )

    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pdb))[0]
    np.save(f"{args.outdir}/{base}_connectivity.npy", result["connectivity_matrix"])
    np.save(f"{args.outdir}/{base}_ctqw_prob.npy",    result["ctqw_prob"])
    print(f"\nSaved: {base}_connectivity.npy, {base}_ctqw_prob.npy → {args.outdir}/")
