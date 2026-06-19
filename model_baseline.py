"""
GVP-GNN + MLP baseline for allosteric site prediction.
No CTQW — GVP node embeddings fed directly into a sigmoid classifier.
Used as ablation: does CTQW add value beyond a plain GNN?
"""

import torch
import torch.nn as nn
from model import GVPLayer, GVPMessagePassing


class GVPAlloBaseline(nn.Module):
    """
    Same GVP encoder as HamiltonianGVP, but with a direct node
    classification head instead of CTQW.

    Output: (N,) sigmoid scores in [0, 1].
    """

    def __init__(
        self,
        node_s_dim: int = 24,
        node_v_dim: int = 1,
        edge_s_dim: int = 24,
        edge_v_dim: int = 1,
        hidden_s: int = 64,
        hidden_v: int = 8,
        n_layers: int = 4,
        cutoff_dist: float = 12.0,
    ):
        super().__init__()
        self.cutoff_dist = cutoff_dist

        # Input projection via GVPLayer (handles (N, v_dim, 3) correctly)
        self.node_embed = GVPLayer(node_s_dim, node_v_dim, hidden_s, hidden_v)

        # GVP message-passing layers
        self.layers = nn.ModuleList([
            GVPMessagePassing(hidden_s, hidden_v, edge_s_dim, edge_v_dim,
                              hidden_s, hidden_v)
            for _ in range(n_layers)
        ])

        # Direct allosteric score head
        self.score_head = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
        )

    def forward(self, graph: dict) -> torch.Tensor:
        node_s = graph['node_s']                          # (N, node_s_dim)
        node_V = graph['node_V']                          # (N, 1, 3)
        edge_s = graph['edge_s']                          # (E, edge_s_dim)
        edge_V = graph['edge_V']                          # (E, 1, 3)
        edge_index = graph['edge_index']                  # (2, E)
        edge_dists = graph['edge_dists']                  # (E,)

        # Input projection (GVPLayer handles vector features correctly)
        h_s, h_V = self.node_embed(node_s, node_V)

        # Message passing
        for layer in self.layers:
            h_s, h_V = layer(h_s, h_V, edge_s, edge_V, edge_index)

        # Direct score
        scores = self.score_head(h_s).squeeze(-1)         # (N,)
        return torch.sigmoid(scores)
