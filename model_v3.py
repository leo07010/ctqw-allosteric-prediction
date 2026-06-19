"""
ENAQT model: HamiltonianGVP_v2 + learnable dephasing rate gamma.

Reuses the full dense H construction from model_v2.py and adds a single
learnable scalar log_gamma for the ENAQT dephasing rate.
"""

import torch
import torch.nn as nn
from model_v2 import HamiltonianGVP_v2


class HamiltonianGVP_v3(HamiltonianGVP_v2):
    """
    GVP-GNN with dense Hamiltonian + learnable ENAQT dephasing rate.

    forward() returns (H, log_gamma) instead of just H.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # log(gamma), init -1.0 → gamma ≈ 0.37 (mid-range dephasing)
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))

    def forward(self, graph: dict):
        H = super().forward(graph)
        return H, self.log_gamma
