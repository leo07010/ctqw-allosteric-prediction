"""
Differentiable Continuous-Time Quantum Walk (CTQW) in PyTorch.

|ψ(t)⟩ = e^{-iHt} |ψ₀⟩
P(i,t) = |⟨i|ψ(t)⟩|²

Uses torch.linalg.eigh for differentiable eigendecomposition,
so gradients flow back through H to the GVP-GNN.
"""

import torch
import torch.nn.functional as F


def ctqw_prob(H: torch.Tensor, initial_nodes: list, t_max: float = 50.0, n_steps: int = 200):
    """
    Time-averaged CTQW probability distribution.

    Args:
        H:             (N,N) symmetric Hamiltonian (real, on any device)
        initial_nodes: list of node indices = active site (start state)
        t_max:         total evolution time
        n_steps:       number of time points to average

    Returns:
        prob_avg: (N,) time-averaged probability, differentiable w.r.t. H
    """
    N = H.size(0)
    device = H.device

    # Symmetrize and cast to float64 — LAPACK dsyevd converges far more
    # reliably on double precision for sparse/near-degenerate Hamiltonians
    H64 = 0.5 * (H + H.T).double()
    eye64 = torch.eye(N, device=device, dtype=torch.float64)

    # Increasing regularization fallback
    for eps in (1e-4, 1e-3, 1e-2, 5e-2):
        try:
            eigenvalues64, V64 = torch.linalg.eigh(H64 + eps * eye64)
            break
        except torch._C._LinAlgError:
            if eps == 5e-2:
                raise

    # Keep eigenvectors in float32 for downstream ops; eigenvalues stay float64
    eigenvalues = eigenvalues64.float()
    V = V64.float()

    # Initial state: equal superposition over active site nodes
    psi0 = torch.zeros(N, dtype=torch.float32, device=device)
    psi0[initial_nodes] = 1.0 / (len(initial_nodes) ** 0.5)
    psi0 = psi0.to(torch.complex64)

    # Project to eigenbasis: c_k = ⟨φ_k|ψ₀⟩
    c = V.to(torch.complex64).T @ psi0              # (N,)

    # Time evolution and averaging
    times = torch.linspace(0.0, t_max, n_steps, device=device)
    prob_avg = torch.zeros(N, device=device)

    for t in times:
        phase  = torch.exp(-1j * eigenvalues.to(torch.complex64) * t)  # (N,)
        psi_t  = V.to(torch.complex64) @ (c * phase)                    # (N,)
        prob_t = psi_t.abs() ** 2                                        # (N,)
        prob_avg = prob_avg + prob_t.real

    prob_avg = prob_avg / n_steps
    return prob_avg                                   # (N,) real, grad flows through H


def ctqw_connectivity_matrix(H: torch.Tensor):
    """
    Closed-form time-averaged connectivity matrix.

    C[i,j] = ⟨|⟨j|e^{-iHt}|i⟩|²⟩_t = Σ_k |V_ik|² |V_jk|²

    Valid for non-degenerate eigenvalues (T→∞ limit).
    This is the N×N matrix required by the Cleveland Clinic Challenge.

    Returns: C (N,N), differentiable w.r.t. H.
    """
    H64 = 0.5 * (H + H.T).double()
    N = H.size(0)
    eye64 = torch.eye(N, device=H.device, dtype=torch.float64)
    for eps in (1e-4, 1e-3, 1e-2, 5e-2):
        try:
            _, V64 = torch.linalg.eigh(H64 + eps * eye64)
            break
        except torch._C._LinAlgError:
            if eps == 5e-2:
                raise
    V = V64.float()
    V2 = V ** 2                          # |V_ik|² shape (N, N)
    C  = V2 @ V2.T                       # (N, N)
    return C


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────

def communicability_loss(
    H:               torch.Tensor,   # (N,N) Hamiltonian
    active_mask:     torch.Tensor,   # (N,) bool — active site seeds
    allosteric_mask: torch.Tensor,   # (N,) bool — known allosteric residues
):
    """
    Communicability-based loss using closed-form C[i,j] = Σ_k |V_ik|² |V_jk|².

    Directly maximizes quantum communication from active site to allosteric
    residues vs background — one eigh call, no time loop.

    Loss = -log( mean_C(active→allo) / (mean_C(active→allo) + mean_C(active→bg)) )
    """
    C = ctqw_connectivity_matrix(H)   # (N, N)

    active_idx = active_mask.nonzero(as_tuple=True)[0]
    neg_mask   = ~(allosteric_mask | active_mask)

    C_row = C[active_idx]             # (n_active, N) — rows from active site nodes

    pos = C_row[:, allosteric_mask].mean()   # mean communicability to allo sites
    neg = C_row[:, neg_mask].mean()          # mean communicability to background

    # Normalized contrastive loss
    ratio = pos / (pos + neg + 1e-8)
    return -torch.log(ratio + 1e-8)

def allosteric_loss(
    prob_avg:        torch.Tensor,   # (N,) CTQW probability
    allosteric_mask: torch.Tensor,   # (N,) bool — known allosteric residues
    active_mask:     torch.Tensor,   # (N,) bool — active site (excluded from neg)
    neg_weight:      float = 0.1,
):
    """
    End-to-end loss: maximize QW probability at allosteric sites.

    Positive term: -log P(allosteric residues)  → pull probability up
    Negative term: -log(1 - P(background))      → push background down
    """
    eps = 1e-8

    # Positive: allosteric residues should accumulate probability
    pos = prob_avg[allosteric_mask]
    loss_pos = -torch.log(pos + eps).mean()

    # Negative: neither active site nor allosteric → should have low prob
    neg_mask = (~allosteric_mask) & (~active_mask)
    if neg_mask.any():
        neg = prob_avg[neg_mask]
        loss_neg = -torch.log(1.0 - neg + eps).mean()
    else:
        loss_neg = torch.tensor(0.0, device=prob_avg.device)

    return loss_pos + neg_weight * loss_neg


def ranking_loss(
    prob_avg:        torch.Tensor,
    allosteric_mask: torch.Tensor,
    active_mask:     torch.Tensor,
    margin:          float = 0.1,
):
    """
    Margin ranking loss: allosteric residues should rank above background.

    For each allosteric residue a and background residue b:
        loss += max(0, margin - P(a) + P(b))
    """
    pos_probs = prob_avg[allosteric_mask]
    neg_mask  = (~allosteric_mask) & (~active_mask)
    neg_probs = prob_avg[neg_mask]

    if neg_probs.numel() == 0 or pos_probs.numel() == 0:
        return torch.tensor(0.0, device=prob_avg.device)

    # All pairs: (n_pos, n_neg)
    diff = pos_probs.unsqueeze(1) - neg_probs.unsqueeze(0)      # (n_pos, n_neg)
    loss = F.relu(margin - diff).mean()
    return loss
