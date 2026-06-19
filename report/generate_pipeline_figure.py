"""
generate_pipeline_figure.py
Generates fig8: complete LLM+PDB+GVP-GNN+CTQW pipeline architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

MM = 1 / 25.4
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# Wong 2011 palette
BL="#0072B2"; OR="#E69F00"; GR="#009E73"; RD="#D55E00"
SK="#56B4E9"; PU="#CC79A7"; YL="#F0E442"; DK="#333333"
LG="#DDDDDD"  # light gray

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "font.family": "sans-serif",
})

fig, ax = plt.subplots(figsize=(183 * MM, 145 * MM))
ax.set_xlim(0, 183)
ax.set_ylim(0, 145)
ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1)

def rbox(cx, cy, w, h, fill, edge, text, fontsize=6.5, bold=False, rx=2.5):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rx}",
        fc=fill, ec=edge, linewidth=0.6, zorder=2,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=DK, fontweight=weight,
            wrap=True, zorder=3,
            multialignment="center")

def arr(x1, y1, x2, y2, col=DK, lw=0.7, style="->", rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=col, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=7, zorder=4,
    )
    ax.add_patch(patch)

def label(x, y, text, col=DK, fs=7, bold=False):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=col,
            fontweight="bold" if bold else "normal", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
label(91.5, 140, "LLM + PDB + GVP-GNN + CTQW Pipeline for Allosteric Pocket Prediction",
      fs=8.5, bold=True, col=DK)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1: Inputs  (y = 125)
# ─────────────────────────────────────────────────────────────────────────────
Y1 = 125
rbox(25,  Y1, 40, 11, fill="#EEF4FB", edge=BL,
     text="PDB Structure\n(.pdb / .mmcif)\nCα coordinates, edges", bold=False)
rbox(91.5, Y1, 40, 11, fill="#EEF4FB", edge=GR,
     text="Protein Sequence\n(FASTA from PDB)\n→ ESM-2 embeddings", bold=False)
rbox(158, Y1, 40, 11, fill="#EEF4FB", edge=PU,
     text="PDB Annotations\nB-factor, DSSP codes\nknown active sites", bold=False)

label(25,  Y1+7.5, "Structure", col=BL, fs=6.5)
label(91.5,Y1+7.5, "Sequence",  col=GR, fs=6.5)
label(158, Y1+7.5, "Metadata",  col=PU, fs=6.5)

# ─────────────────────────────────────────────────────────────────────────────
# Row 2: Feature extraction  (y = 107)
# ─────────────────────────────────────────────────────────────────────────────
Y2 = 107
# Left box: geometric
rbox(35, Y2, 56, 10, fill="#FDEBD0", edge=OR,
     text="Geometric features  (24-d)\ndist, angle, residue chem props")
# Middle box: ESM-2
rbox(91.5, Y2, 44, 10, fill="#E8F8F2", edge=GR,
     text="ESM-2  (480-d/residue)\nper-residue context", bold=True)
ax.text(91.5, Y2+7.8, "FREE  ·  local inference", ha="center",
        fontsize=5.5, color=GR, style="italic")
# Right box: PDB feats
rbox(148, Y2, 44, 10, fill="#F0EAF8", edge=PU,
     text="PDB features  (4-d)\nB-factor, DSSP, bindsite")

# Arrows Row1 → Row2
arr(25,  Y1-5.5, 35, Y2+5,  col=BL)
arr(91.5,Y1-5.5, 91.5, Y2+5, col=GR)
arr(158, Y1-5.5, 148, Y2+5,  col=PU)

# Merge arrow
arr(35,  Y2-5, 70, Y2-13, col=OR,  rad=0.2)
arr(91.5,Y2-5, 91.5, Y2-13, col=GR)
arr(148, Y2-5, 113, Y2-13, col=PU,  rad=-0.2)

# ─────────────────────────────────────────────────────────────────────────────
# Row 3: Node feature concat  (y = 90)
# ─────────────────────────────────────────────────────────────────────────────
Y3 = 90
rbox(91.5, Y3, 90, 9, fill="#FFF9E6", edge=OR,
     text="Node features = [ geometric (24-d)  ‖  ESM-2 (480-d)  ‖  PDB (4-d) ]  =  508-d per residue",
     fontsize=6)

# ─────────────────────────────────────────────────────────────────────────────
# Row 4: GVP-GNN  (y = 73)
# ─────────────────────────────────────────────────────────────────────────────
Y4 = 73
arr(91.5, Y3-4.5, 91.5, Y4+8, col=DK)
rbox(91.5, Y4, 100, 14, fill="#D6EAF8", edge=BL,
     text="GVP-GNN  (4 layers, geometric equivariant)\nNode embed → 4× Message Passing → Node embeddings  s ∈ ℝᴺˣ⁶⁴",
     bold=True, fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Row 5: Dual head output  (y = 55)
# ─────────────────────────────────────────────────────────────────────────────
Y5 = 55
arr(60,  Y4-7, 40, Y5+7,  col=OR,  rad=0.15)
arr(123, Y4-7, 143, Y5+7, col=SK, rad=-0.15)

rbox(40,  Y5, 60, 12, fill="#FDEBD0", edge=OR,
     text="H head → H ∈ ℝᴺˣᴺ\n(learnable Hamiltonian)", bold=True)
rbox(143, Y5, 60, 12, fill="#EAF4FD", edge=SK,
     text="Score head → logits ∈ ℝᴺ\n(BCE classification)", bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# Row 6: CTQW  (y = 37)
# ─────────────────────────────────────────────────────────────────────────────
Y6 = 37
arr(40,  Y5-6, 40, Y6+8,  col=OR)
arr(143, Y5-6, 143, Y6+8, col=SK)
# Merge
arr(40,  Y6-8, 91.5, Y6-16, col=OR, rad=0.2)
arr(143, Y6-8, 91.5, Y6-16, col=SK, rad=-0.2)

rbox(40, Y6, 60, 14, fill="#E8F8F2", edge=GR,
     text="CTQW  P(j, t*)\n= |⟨j|e⁻ⁱᴴᵗ*|active⟩|²\nt* learnable (per protein)", bold=True, fontsize=6.5)
rbox(143, Y6, 60, 11, fill="#EAF4FD", edge=SK,
     text="Training: BCE loss\nweighted pos_weight=10\ndense gradient signal", fontsize=6.5)

# ─────────────────────────────────────────────────────────────────────────────
# Row 7: Final scores + evaluation  (y = 18)
# ─────────────────────────────────────────────────────────────────────────────
Y7 = 18
rbox(91.5, Y7, 100, 11, fill="#E8F8F2", edge=GR,
     text="scores = α · sigmoid(logits)  +  β · P(j, t*)    →   Top-K allosteric candidates",
     bold=True, fontsize=7)
arr(91.5, Y6-16, 91.5, Y7+5.5, col=GR)

# Evaluation box
rbox(160, Y7, 38, 22, fill="#FDF2F8", edge=PU,
     text="Evaluation\nAUPRC\nAUROC\nHit@K\nEF10%", bold=False, fontsize=6.5)
arr(141.5, Y7, 141, Y7, col=PU)

# ─────────────────────────────────────────────────────────────────────────────
# Side annotation: WHY ESM-2?
# ─────────────────────────────────────────────────────────────────────────────
ax.text(5, 107, "WHY LLM?", fontsize=6.5, color=GR, fontweight="bold", rotation=90,
        va="center")
ax.text(5, 95,
        "• Cross-protein\n  generalization\n• Undruggable\n  detection\n• Evol. context",
        fontsize=5.5, color=GR, va="center")

# Side annotation: KEY FIX
ax.text(178, 37, "KEY FIX", fontsize=6.5, color=GR, fontweight="bold", rotation=90,
        va="center")
ax.text(178, 25,
        "P(j,t*) vs\nC[i,j]:\n32× better\ndiscrimination",
        fontsize=5.5, color=GR, va="center")

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc="#EEF4FB", ec=BL, label="Structure input"),
    mpatches.Patch(fc="#E8F8F2", ec=GR, label="LLM / CTQW"),
    mpatches.Patch(fc="#F0EAF8", ec=PU, label="PDB metadata"),
    mpatches.Patch(fc="#D6EAF8", ec=BL, label="GVP-GNN"),
    mpatches.Patch(fc="#FDEBD0", ec=OR, label="H Hamiltonian"),
    mpatches.Patch(fc="#EAF4FD", ec=SK, label="BCE head"),
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0),
          fontsize=5.5, ncol=3, frameon=True, edgecolor=LG)

fig.savefig(os.path.join(OUT, "fig8_pipeline_complete.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig8_pipeline_complete.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: fig8_pipeline_complete.pdf / .png")
