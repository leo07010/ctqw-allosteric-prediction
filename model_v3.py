"""
ENAQT model: GVP-GNN + per-protein learnable dephasing rate gamma.

Key difference from v2 (global gamma scalar):
  gamma is now output by a small MLP reading the mean-pooled node embeddings.
  Each protein gets its own optimal dephasing rate, allowing the model to
  balance quantum coherence vs classical diffusion per-protein.

forward() returns (H, log_gamma) where log_gamma is a scalar Tensor with grad.
"""

import torch
import torch.nn as nn
from model import GVPLayer, GVPMessagePassing


class HamiltonianGVP_v3(nn.Module):
    """
    Dense Hamiltonian + per-protein ENAQT dephasing rate.

    H = h @ h^T / K  +  diag(softplus(s))  +  alpha * H_sparse
    log_gamma = gamma_head(mean(node_s))   -- per-protein, not global
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
        K:           int = 32,
    ):
        super().__init__()
        self.cutoff_dist = cutoff_dist
        self.K = K

        # ── GVP encoder (identical to v2) ──────────────────────
        self.node_embed = GVPLayer(node_s_dim, node_v_dim, hidden_s, hidden_v)
        self.mp_layers = nn.ModuleList([
            GVPMessagePassing(hidden_s, hidden_v, edge_s_dim, edge_v_dim, hidden_s, hidden_v)
            for _ in range(n_layers)
        ])

        # ── Dense H components (identical to v2) ───────────────
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, K),
        )
        self.diag_head = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
            nn.Softplus(),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_s + 1, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
            nn.Tanh(),
        )
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))

        # ── Per-protein gamma head (NEW) ────────────────────────
        # Reads mean-pooled node embeddings → scalar log_gamma
        # Init bias = -1.5 → gamma_init ≈ 0.22 (slightly more quantum than v4's 0.37)
        self.gamma_head = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 4),
            nn.SiLU(),
            nn.Linear(hidden_s // 4, 1),
        )
        # Bias init for last layer
        nn.init.constant_(self.gamma_head[-1].bias, -1.5)

    def forward(self, graph: dict):
        """
        Returns:
            H:         (N, N) symmetric dense Hamiltonian
            log_gamma: scalar, per-protein log dephasing rate (differentiable)
        """
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

        # ── Per-protein log_gamma ──────────────────────────────
        # Mean-pool across all residues → single protein-level embedding
        graph_emb = node_s.mean(dim=0)                       # (hidden_s,)
        log_gamma = self.gamma_head(graph_emb).squeeze(-1)   # scalar

        # ── Dense H (same as v2) ───────────────────────────────
        h = self.h_proj(node_s)
        H_dense = h @ h.T / self.K

        diag_vals = self.diag_head(node_s).squeeze(-1)
        H = H_dense + torch.diag(diag_vals)

        src, dst = edge_index
        edge_feat = torch.cat([
            node_s[src],
            node_s[dst],
            edge_dists.unsqueeze(-1) / self.cutoff_dist,
        ], dim=-1)
        w_ij = self.edge_head(edge_feat).squeeze(-1)

        H_sparse = torch.zeros(N, N, device=node_s.device)
        H_sparse.index_put_((src, dst), w_ij, accumulate=True)
        H_sparse.index_put_((dst, src), w_ij, accumulate=True)
        count = torch.zeros(N, N, device=node_s.device)
        count.index_put_((src, dst), torch.ones_like(w_ij), accumulate=True)
        count.index_put_((dst, src), torch.ones_like(w_ij), accumulate=True)
        H_sparse = H_sparse / count.clamp(min=1)

        alpha = torch.sigmoid(self.log_alpha)
        H = H + alpha * H_sparse
        H = (H + H.T) / 2.0

        return H, log_gamma
