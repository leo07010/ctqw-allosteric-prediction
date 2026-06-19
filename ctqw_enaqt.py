"""
ENAQT: Environment-Assisted Quantum Transport on protein contact graphs.

Based on: Rebentrost et al., arXiv:0807.0929
Haken-Strobl-Reineker transition rates:
    k_ij = 2 |H_ij|^2 * gamma / (gamma^2 + (H_ii - H_jj)^2)

Key advantage over pure CTQW communicability:
  - No eigendecomposition → no near-degenerate eigenvalue plateau
  - ENAQT rates are sparse/directed → sharp allosteric prediction signal
  - gamma is learnable → model finds optimal quantum-classical balance
  - Direct gradient path: loss → p → W → K → H_ij (no eigenvector intermediary)
"""

import torch


def enaqt_rates(H: torch.Tensor, log_gamma: torch.Tensor) -> torch.Tensor:
    """
    Haken-Strobl-Reineker ENAQT transition rate matrix.

    Args:
        H:         (N, N) symmetric Hamiltonian
        log_gamma: scalar, log of dephasing rate (learnable)

    Returns:
        K: (N, N) rate matrix, K[i,j] = rate from j → i
    """
    gamma = torch.exp(log_gamma)
    diag = torch.diag(H)
    energy_gap_sq = (diag.unsqueeze(1) - diag.unsqueeze(0)) ** 2  # (N, N)
    off_diag = H - torch.diag_embed(diag)
    coupling_sq = off_diag ** 2
    K = 2.0 * coupling_sq * gamma / (gamma ** 2 + energy_gap_sq + 1e-10)
    return K


def enaqt_rwr(
    H: torch.Tensor,
    log_gamma: torch.Tensor,
    seed_mask: torch.Tensor,
    alpha: float = 0.15,
    n_iter: int = 100,
) -> torch.Tensor:
    """
    Differentiable RWR on ENAQT transition matrix.

    Args:
        H:         (N, N) Hamiltonian
        log_gamma: scalar log dephasing rate
        seed_mask: (N,) bool — active site seed nodes
        alpha:     restart probability (0.15 default)
        n_iter:    power iteration steps

    Returns:
        p: (N,) steady-state probability distribution
    """
    K = enaqt_rates(H, log_gamma)
    col_sums = K.sum(0).clamp(min=1e-10)
    W = K / col_sums.unsqueeze(0)            # column-stochastic (N, N)

    n_seeds = seed_mask.float().sum().clamp(min=1.0)
    p0 = seed_mask.float() / n_seeds
    p = p0.clone()

    for _ in range(n_iter):
        p = alpha * p0 + (1.0 - alpha) * (W @ p)

    return p


def enaqt_loss(
    H: torch.Tensor,
    log_gamma: torch.Tensor,
    active_mask: torch.Tensor,
    allosteric_mask: torch.Tensor,
    alpha: float = 0.15,
) -> torch.Tensor:
    """
    Contrastive loss on ENAQT steady-state distribution.

    Maximizes p[allosteric] / (p[allosteric] + p[background]).

    Args:
        H:               (N, N) Hamiltonian
        log_gamma:       scalar log dephasing rate
        active_mask:     (N,) bool — orthosteric site (CTQW seed)
        allosteric_mask: (N,) bool — known allosteric residues (training labels)
        alpha:           RWR restart probability

    Returns:
        loss: scalar
    """
    p = enaqt_rwr(H, log_gamma, active_mask, alpha=alpha)
    neg_mask = ~(allosteric_mask | active_mask)
    pos = p[allosteric_mask].mean()
    neg = p[neg_mask].mean()
    ratio = pos / (pos + neg + 1e-8)
    return -torch.log(ratio + 1e-8)
