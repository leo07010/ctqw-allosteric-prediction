"""
GVP-GNN with Hamiltonian prediction head.

Architecture:
  PDB → residue graph → GVP-GNN encoder → per-residue embeddings
        → edge MLP → H_ij (off-diagonal)
        → node MLP → H_ii (diagonal / on-site energy)
        → H_predicted (N×N symmetric)

Then: CTQW on H_predicted → allosteric site ranking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────
# Simplified GVP layer (scalar + vector features)
# Reference: Jing et al. 2021 (Learning from Protein Structure)
# ─────────────────────────────────────────────────────────────

class GVPLayer(nn.Module):
    """
    Single Geometric Vector Perceptron layer.
    Processes (scalar, vector) feature pairs with equivariance.

    s: scalar features  (batch, n_scalar)
    V: vector features  (batch, n_vector, 3)
    """

    def __init__(self, s_in, v_in, s_out, v_out):
        super().__init__()
        self.s_in, self.v_in = s_in, v_in
        self.s_out, self.v_out = s_out, v_out

        # Equivariant vector transform: v_in → v_out
        self.W_V = nn.Linear(v_in, v_out, bias=False)

        # Scalar gate for vector output: s_in → v_out sigmoid gates
        self.W_gate = nn.Linear(s_in, v_out)

        # Scalar transform: [s, ||V_out||] → s_out
        self.W_s = nn.Sequential(
            nn.Linear(s_in + v_out, s_out),
            nn.SiLU(),
        )

    def forward(self, s, V):
        # V: (B, v_in, 3) → equivariant linear transform
        V_h = torch.einsum("bfd,ef->bed", V, self.W_V.weight)  # (B, v_out, 3)

        # Norms of transformed vectors: (B, v_out)
        v_norms = torch.norm(V_h, dim=-1)

        # Scalar branch: concat s with vector norms
        s_cat = torch.cat([s, v_norms], dim=-1)                 # (B, s_in+v_out)
        s_out = self.W_s(s_cat)                                  # (B, s_out)

        # Gate vectors by scalar-derived signal (sigmoid)
        gate  = torch.sigmoid(self.W_gate(s))                   # (B, v_out)
        V_out = V_h * gate.unsqueeze(-1)                        # (B, v_out, 3)

        return s_out, V_out


class GVPMessagePassing(nn.Module):
    """
    One round of GVP message-passing over a protein graph.
    Each node aggregates messages from its 30 nearest neighbors.
    """

    def __init__(self, node_s, node_v, edge_s, edge_v, hidden_s, hidden_v):
        super().__init__()
        # Message: concat(node_i, node_j, edge_ij) → message
        self.msg_gvp = GVPLayer(
            s_in  = 2 * node_s + edge_s,
            v_in  = 2 * node_v + edge_v,
            s_out = hidden_s,
            v_out = hidden_v,
        )
        # Update: concat(node_i, aggregated_msg) → new node
        self.upd_gvp = GVPLayer(
            s_in  = node_s + hidden_s,
            v_in  = node_v + hidden_v,
            s_out = node_s,
            v_out = node_v,
        )
        self.layer_norm_s = nn.LayerNorm(node_s)

    def forward(self, node_s, node_V, edge_s, edge_V, edge_index):
        """
        node_s:     (N, node_s_dim)
        node_V:     (N, node_v_dim, 3)
        edge_s:     (E, edge_s_dim)
        edge_V:     (E, edge_v_dim, 3)
        edge_index: (2, E)  row=src, col=dst
        """
        src, dst = edge_index

        # Build messages from src→dst
        msg_s_in = torch.cat([node_s[src], node_s[dst], edge_s], dim=-1)
        msg_V_in = torch.cat([node_V[src], node_V[dst], edge_V], dim=-2)
        msg_s, msg_V = self.msg_gvp(msg_s_in, msg_V_in)

        # Aggregate (sum) messages at each destination node
        N = node_s.size(0)
        agg_s = torch.zeros(N, msg_s.size(-1), device=node_s.device)
        agg_V = torch.zeros(N, msg_V.size(-2), 3, device=node_s.device)
        agg_s = agg_s.index_add(0, dst, msg_s)
        agg_V = agg_V.index_add(0, dst, msg_V)

        # Update node features
        upd_s_in = torch.cat([node_s, agg_s], dim=-1)
        upd_V_in = torch.cat([node_V, agg_V], dim=-2)
        new_s, new_V = self.upd_gvp(upd_s_in, upd_V_in)

        # Residual + LayerNorm on scalar branch
        new_s = self.layer_norm_s(new_s + node_s)

        return new_s, new_V


# ─────────────────────────────────────────────────────────────
# Protein graph feature builder
# ─────────────────────────────────────────────────────────────

AA_VOCAB = {
    "ALA":0,"ARG":1,"ASN":2,"ASP":3,"CYS":4,"GLN":5,"GLU":6,"GLY":7,
    "HIS":8,"ILE":9,"LEU":10,"LYS":11,"MET":12,"PHE":13,"PRO":14,
    "SER":15,"THR":16,"TRP":17,"TYR":18,"VAL":19,
}

def build_protein_graph(residues, coords, n_neighbors=30, device="cpu"):
    """
    Build protein graph tensors from residue list and Cα coordinates.

    Node scalar features:
      - AA one-hot (20 dim)
      - sin/cos of backbone dihedrals φ,ψ (4 dim)  ← approximated from Cα
      → total: 24 dim

    Node vector features:
      - Cβ direction (approximated as Cα[i+1]-Cα[i] unit vector): 1 vector

    Edge scalar features:
      - RBF encoding of distance (16 dim)
      - sequence distance sinusoidal (8 dim)
      → total: 24 dim

    Edge vector features:
      - unit vector Cα_j - Cα_i: 1 vector

    Returns dict of tensors ready for GVP-GNN.
    """
    N = len(residues)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)

    # ── Node scalars ──────────────────────────────────────────
    aa_onehot = torch.zeros(N, 20, device=device)
    for i, r in enumerate(residues):
        aa_idx = AA_VOCAB.get(r["resname"], 7)   # default GLY
        aa_onehot[i, aa_idx] = 1.0

    # Approximate φ/ψ from Cα positions (coarse proxy)
    # Forward/backward bond vectors
    fwd = F.normalize(
        torch.cat([coords_t[1:] - coords_t[:-1], torch.zeros(1,3,device=device)], 0), dim=-1)
    bwd = F.normalize(
        torch.cat([torch.zeros(1,3,device=device), coords_t[:-1] - coords_t[1:]], 0), dim=-1)
    dihe = torch.cat([fwd, bwd], dim=-1)            # (N, 6) as dihedral proxy
    # Pack to 4 dim: sin/cos of two angles
    angle1 = torch.atan2(dihe[:,1], dihe[:,0])
    angle2 = torch.atan2(dihe[:,4], dihe[:,3])
    dihedral_feats = torch.stack([angle1.sin(), angle1.cos(),
                                   angle2.sin(), angle2.cos()], dim=-1)  # (N,4)

    node_s = torch.cat([aa_onehot, dihedral_feats], dim=-1)   # (N, 24)

    # ── Node vectors ──────────────────────────────────────────
    node_V = fwd.unsqueeze(1)                                  # (N, 1, 3)

    # ── Build kNN graph ──────────────────────────────────────
    diff  = coords_t[:, None, :] - coords_t[None, :, :]       # (N,N,3)
    dists = torch.norm(diff, dim=-1)                           # (N,N)
    dists_filled = dists.clone()
    dists_filled.fill_diagonal_(1e9)
    k = min(n_neighbors, N - 1)
    _, knn_idx = dists_filled.topk(k, largest=False, dim=-1)   # (N, k)

    # Build edge_index (src→dst for all kNN)
    src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)                # (2, N*k)

    # ── Edge scalars: RBF distance ────────────────────────────
    edge_dists = dists[src, dst]                               # (E,)
    rbf_centers = torch.linspace(0, 20, 16, device=device)
    rbf_gamma   = 1.0
    rbf = torch.exp(-rbf_gamma * (edge_dists.unsqueeze(-1) - rbf_centers)**2)  # (E,16)

    # Sequence distance encoding
    seq_dist = (src - dst).float().abs().unsqueeze(-1)         # (E,1)
    seq_enc  = torch.cat([
        torch.sin(seq_dist / (10000 ** (torch.arange(0,8,2,device=device).float()/8))),
        torch.cos(seq_dist / (10000 ** (torch.arange(1,8,2,device=device).float()/8))),
    ], dim=-1)                                                  # (E,8)

    edge_s = torch.cat([rbf, seq_enc], dim=-1)                 # (E,24)

    # ── Edge vectors ──────────────────────────────────────────
    edge_vec = F.normalize(coords_t[dst] - coords_t[src], dim=-1)  # (E,3)
    edge_V   = edge_vec.unsqueeze(1)                               # (E,1,3)

    return {
        "node_s":     node_s,        # (N, 24)
        "node_V":     node_V,        # (N, 1, 3)
        "edge_s":     edge_s,        # (E, 24)
        "edge_V":     edge_V,        # (E, 1, 3)
        "edge_index": edge_index,    # (2, E)
        "coords":     coords_t,      # (N, 3)
        "edge_dists": edge_dists,    # (E,)
        "N": N, "E": edge_index.size(1),
    }


# ─────────────────────────────────────────────────────────────
# GVP-GNN Hamiltonian Predictor
# ─────────────────────────────────────────────────────────────

class HamiltonianGVP(nn.Module):
    """
    GVP-GNN that predicts a protein communication Hamiltonian H.

    H_ij (off-diagonal) = effective coupling between residues i and j.
    H_ii (diagonal)     = on-site energy (higher = QW scattered away).

    H is the Hamiltonian for CTQW:
        |ψ(t)⟩ = e^{-iHt} |active_site⟩
    High-probability residues at long times = predicted allosteric sites.
    """

    def __init__(
        self,
        node_s_dim:   int = 24,
        node_v_dim:   int = 1,
        edge_s_dim:   int = 24,
        edge_v_dim:   int = 1,
        hidden_s:     int = 64,
        hidden_v:     int = 8,
        n_layers:     int = 4,
        cutoff_dist:  float = 12.0,
    ):
        super().__init__()
        self.cutoff_dist = cutoff_dist

        # Input projection
        self.node_embed = GVPLayer(node_s_dim, node_v_dim, hidden_s, hidden_v)

        # Message-passing layers
        self.mp_layers = nn.ModuleList([
            GVPMessagePassing(hidden_s, hidden_v, edge_s_dim, edge_v_dim, hidden_s, hidden_v)
            for _ in range(n_layers)
        ])

        # ── Hamiltonian heads ──────────────────────────────────

        # Off-diagonal: H_ij = f(h_i, h_j, d_ij)
        # Output: one scalar per edge → coupling strength
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_s + 1, hidden_s),   # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
            nn.Tanh(),                                # bound to [-1, 1]
        )

        # Diagonal: H_ii = g(h_i)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_s, hidden_s // 2),
            nn.SiLU(),
            nn.Linear(hidden_s // 2, 1),
        )

    def forward(self, graph: dict):
        """
        graph: output of build_protein_graph()
        Returns H: (N, N) symmetric real Hamiltonian tensor.
        """
        node_s     = graph["node_s"]
        node_V     = graph["node_V"]
        edge_s     = graph["edge_s"]
        edge_V     = graph["edge_V"]
        edge_index = graph["edge_index"]
        edge_dists = graph["edge_dists"]
        N          = graph["N"]

        # Input projection
        node_s, node_V = self.node_embed(node_s, node_V)

        # Message-passing
        for layer in self.mp_layers:
            node_s, node_V = layer(node_s, node_V, edge_s, edge_V, edge_index)

        src, dst = edge_index

        # ── Off-diagonal H_ij ─────────────────────────────────
        # Concat embeddings of src and dst + distance
        edge_feat = torch.cat([
            node_s[src],
            node_s[dst],
            edge_dists.unsqueeze(-1) / self.cutoff_dist,   # normalized distance
        ], dim=-1)
        w_ij = self.edge_head(edge_feat).squeeze(-1)        # (E,) in [-1,1]

        # Symmetrize: w_ij and w_ji should be equal
        # We build the matrix by averaging both directions
        H = torch.zeros(N, N, device=node_s.device)
        H.index_put_((src, dst), w_ij, accumulate=True)
        H.index_put_((dst, src), w_ij, accumulate=True)
        # Divide by 2 only where both directions exist
        count = torch.zeros(N, N, device=node_s.device)
        count.index_put_((src, dst), torch.ones_like(w_ij), accumulate=True)
        count.index_put_((dst, src), torch.ones_like(w_ij), accumulate=True)
        count = count.clamp(min=1)
        H = H / count

        # ── Diagonal H_ii ─────────────────────────────────────
        diag_vals = self.node_head(node_s).squeeze(-1)      # (N,)
        H = H + torch.diag(diag_vals)

        # Final symmetry guarantee
        H = (H + H.T) / 2.0

        return H
