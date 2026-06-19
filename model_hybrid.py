"""
GVP-Hybrid: shared encoder → BCE classification head + Hamiltonian head.

Training:  loss = BCE(score_head) + λ * communicability_loss(H)
Inference: scores = sigmoid(logits) + β * C[active_site → i]

The BCE head provides dense per-node gradient (why baseline wins).
The CTQW communicability adds physical allosteric communication prior.
"""

import torch
import torch.nn as nn
from model import GVPLayer, GVPMessagePassing


class GVPHybrid(nn.Module):
    """
    Shared GVP encoder with two output heads:
      1. score_head  → (N,) logits for direct BCE classification
      2. H_head      → (N, N) Hamiltonian for CTQW communicability signal

    forward() returns (logits, H).
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

        # ── Shared GVP encoder ─────────────────────────────────
        self.node_embed = GVPLayer(node_s_dim, node_v_dim, hidden_s, hidden_v)
        self.mp_layers = nn.ModuleList([
            GVPMessagePassing(hidden_s, hidden_v, edge_s_dim, edge_v_dim,
                              hidden_s, hidden_v)
            for _ in range(n_layers)
        ])

        # ── Head 1: direct classification (same as baseline) ───
        self.score_head = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
        )

        # ── Head 2: Hamiltonian (same as v2 dense H) ───────────
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

    def _build_H(self, node_s, edge_index, edge_dists, N):
        h = self.h_proj(node_s)
        H = h @ h.T / self.K

        diag_vals = self.diag_head(node_s).squeeze(-1)
        H = H + torch.diag(diag_vals)

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
        return (H + H.T) / 2.0

    def forward(self, graph: dict):
        """
        Returns:
            logits: (N,) raw scores for BCE
            H:      (N, N) symmetric Hamiltonian for CTQW
        """
        node_s     = graph["node_s"]
        node_V     = graph["node_V"]
        edge_s     = graph["edge_s"]
        edge_V     = graph["edge_V"]
        edge_index = graph["edge_index"]
        edge_dists = graph["edge_dists"]
        N          = graph["N"]

        node_s, node_V = self.node_embed(node_s, node_V)
        for layer in self.mp_layers:
            node_s, node_V = layer(node_s, node_V, edge_s, edge_V, edge_index)

        logits = self.score_head(node_s).squeeze(-1)       # (N,)
        H      = self._build_H(node_s, edge_index, edge_dists, N)

        return logits, H
