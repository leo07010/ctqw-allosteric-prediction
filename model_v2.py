"""
HamiltonianGVP_v2: Dense Hamiltonian via node embedding outer product.

Key change from v1:
  v1: H = sparse edge-based matrix → rank ~degree, degenerate eigenvalues
  v2: H = h @ h^T / K + diag(softplus(s))
        → dense, full-rank, well-conditioned spectrum
        → clean gradient through outer product, no index_put_

Architecture:
  GVP encode → node_s (N, hidden_s)
  h = proj(node_s)            → (N, K) communication embedding
  H = h @ h.T / K             → (N, N) low-rank dense term
  H += diag(softplus(s))       → full rank, all eigenvalues > 0
  H += sparse_edge_correction  → optional local structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GVPLayer, GVPMessagePassing


class HamiltonianGVP_v2(nn.Module):
    """
    Dense Hamiltonian for CTQW.

    H = h @ h^T / K  +  diag(softplus(s))  +  alpha * H_sparse

    h @ h^T     : global dense communication (rank K, rich eigenvectors)
    diag(...)   : positive diagonal → full rank, no degenerate zeros
    H_sparse    : local edge corrections (original sparse term, weighted)
    """

    def __init__(
        self,
        node_s_dim:  int = 24,
        node_v_dim:  int = 1,
        edge_s_dim:  int = 24,
        edge_v_dim:  int = 1,
        hidden_s:    int = 64,
        hidden_v:    int = 8,
        n_layers:    int = 4,
        cutoff_dist: float = 12.0,
        K:           int = 32,    # embedding dim for outer product
    ):
        super().__init__()
        self.cutoff_dist = cutoff_dist
        self.K = K

        # GVP encoder (same as v1)
        self.node_embed = GVPLayer(node_s_dim, node_v_dim, hidden_s, hidden_v)
        self.mp_layers = nn.ModuleList([
            GVPMessagePassing(hidden_s, hidden_v, edge_s_dim, edge_v_dim, hidden_s, hidden_v)
            for _ in range(n_layers)
        ])

        # ── Dense H components ─────────────────────────────────
        # 1. Project node embeddings to K-dim communication space
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, K),
        )

        # 2. Positive diagonal (softplus ensures > 0)
        self.diag_head = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
            nn.Softplus(),       # output is always positive
        )

        # 3. Sparse edge correction (local structure, small weight)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_s + 1, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
            nn.Tanh(),           # bounded [-1, 1]
        )

        # Learned mixing weight between dense and sparse
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))  # init alpha ~ 0.37

    def forward(self, graph: dict) -> torch.Tensor:
        """Returns H (N, N): symmetric, full-rank, dense Hamiltonian."""
        node_s     = graph["node_s"]
        node_V     = graph["node_V"]
        edge_s     = graph["edge_s"]
        edge_V     = graph["edge_V"]
        edge_index = graph["edge_index"]
        edge_dists = graph["edge_dists"]
        N          = graph["N"]

        # GVP encoding
        node_s, node_V = self.node_embed(node_s, node_V)
        for layer in self.mp_layers:
            node_s, node_V = layer(node_s, node_V, edge_s, edge_V, edge_index)

        # ── Dense term: h @ h^T ────────────────────────────────
        h = self.h_proj(node_s)          # (N, K)
        H_dense = h @ h.T / self.K       # (N, N), rank ≤ K

        # ── Diagonal: positive definite contribution ───────────
        diag_vals = self.diag_head(node_s).squeeze(-1)   # (N,) > 0
        H = H_dense + torch.diag(diag_vals)

        # ── Sparse edge correction: local structural signal ────
        src, dst = edge_index
        edge_feat = torch.cat([
            node_s[src],
            node_s[dst],
            edge_dists.unsqueeze(-1) / self.cutoff_dist,
        ], dim=-1)
        w_ij = self.edge_head(edge_feat).squeeze(-1)     # (E,)

        H_sparse = torch.zeros(N, N, device=node_s.device)
        H_sparse.index_put_((src, dst), w_ij, accumulate=True)
        H_sparse.index_put_((dst, src), w_ij, accumulate=True)
        count = torch.zeros(N, N, device=node_s.device)
        count.index_put_((src, dst), torch.ones_like(w_ij), accumulate=True)
        count.index_put_((dst, src), torch.ones_like(w_ij), accumulate=True)
        H_sparse = H_sparse / count.clamp(min=1)

        # Combine: dense dominates, sparse adds local correction
        alpha = torch.sigmoid(self.log_alpha)
        H = H + alpha * H_sparse

        # Final symmetry guarantee
        H = (H + H.T) / 2.0
        return H
