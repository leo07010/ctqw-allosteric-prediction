#!/usr/bin/env python3
"""
Publication-ready figures for CTQW allosteric prediction report.
Follows sai-tv/academic-figures skill:
  - Wong 2011 colorblind-safe palette
  - Nature-style dimensions (160 mm full width)
  - Named groups / layers, text as text elements
  - Arrow length >= 10 mm, 3 mm clearance
  - Outputs: PDF (for LaTeX) + SVG (editable)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

MM = 1 / 25.4          # mm → inch
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Wong 2011 colorblind-safe palette ────────────────────────────────────────
BL = "#0072B2"; BL_T = "#D9EBF7"   # blue    – input / data
OR = "#E69F00"; OR_T = "#FFF0CC"   # orange  – Hamiltonian / CTQW
GR = "#009E73"; GR_T = "#D1EFE7"   # green   – GVP / model
RD = "#D55E00"; RD_T = "#F9DDD4"   # red     – failure / limitation
SK = "#56B4E9"; SK_T = "#D9EFFA"   # sky     – active site
PU = "#CC79A7"; PU_T = "#F5E3EE"   # purple  – hybrid / combined
YL = "#F0E442"; YL_T = "#FFFDE7"   # yellow  – highlight
DK = "#333333"; WH = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

class F:
    """Figure with mm coordinate system; y=0 at top."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig, self.ax = plt.subplots(figsize=(w * MM, h * MM))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)

    # coordinate helpers
    def _y(self, y_top):
        return self.h - y_top

    def box(self, cx, cy, bw, bh, fill, edge, rx=1.5, lw=0.4, zorder=2):
        """Rounded box centred at (cx, cy) in from-top coords."""
        patch = FancyBboxPatch(
            (cx - bw / 2, self._y(cy) - bh / 2), bw, bh,
            boxstyle=f"round,pad=0,rounding_size={rx}",
            fc=fill, ec=edge, linewidth=lw, zorder=zorder,
        )
        self.ax.add_patch(patch)

    def txt(self, cx, cy, s, fs=6.5, wt="normal", col=DK, style="normal",
            ha="center"):
        self.ax.text(
            cx, self._y(cy), s,
            fontsize=fs, fontweight=wt, ha=ha, va="center",
            color=col, fontstyle=style, zorder=3,
        )

    def txt2(self, cx, cy, l1, l2, fs=6.5, wt="normal", col=DK, gap=3.2):
        self.txt(cx, cy - gap / 2, l1, fs=fs, wt=wt, col=col)
        self.txt(cx, cy + gap / 2, l2, fs=fs, col=col)

    def txt3(self, cx, cy, l1, l2, l3, fs=6.0, wt="normal", col=DK, gap=3.0):
        self.txt(cx, cy - gap, l1, fs=fs, wt=wt, col=col)
        self.txt(cx, cy,       l2, fs=fs, col=col)
        self.txt(cx, cy + gap, l3, fs=fs, col=col)

    def arr(self, x1, y1, x2, y2, col=DK, lw=0.55, dashed=False,
            rad=0.0, ms=7, zorder=2):
        """Arrow from (x1,y1) to (x2,y2) in from-top coords."""
        ls = (0, (3, 2)) if dashed else "solid"
        patch = FancyArrowPatch(
            (x1, self._y(y1)), (x2, self._y(y2)),
            arrowstyle="->",
            color=col, linewidth=lw, linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=ms, zorder=zorder,
        )
        self.ax.add_patch(patch)

    def label(self, x, y, s):
        """Panel label (bold, top-left corner, 2.5 mm above content)."""
        self.txt(x, y, s, fs=8, wt="bold")

    def save(self, name):
        for ext in ("pdf", "svg", "png"):
            path = os.path.join(OUT, f"{name}.{ext}")
            dpi = 150 if ext == "png" else 300
            self.fig.savefig(path, format=ext, dpi=dpi,
                             bbox_inches="tight", pad_inches=0.5)
        plt.close(self.fig)
        print(f"  saved {name}.pdf + .svg + .png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Overall Prediction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fig1():
    f = F(160, 215)

    # ── main column ────────────────────────────────
    CX, BW, BH = 80, 52, 9
    rows = [14, 35, 56, 77, 98, 119]
    labels = [
        ("PDB (apo structure)", BL, BL_T),
        ("Parse Cα atoms", BL, BL_T),
        ("Contact graph  G (cutoff 12 Å)", BL, BL_T),
        ("Node & edge features", GR, GR_T),
        ("GVP-GNN encoder  (4 layers)", GR, GR_T),
        ("Node embed  s ∈ ℝᴺˣ⁶⁴", GR, GR_T),
    ]
    for y, (txt_s, ec, fc) in zip(rows, labels):
        f.box(CX, y, BW, BH, fc, ec)
        f.txt(CX, y, txt_s, fs=6.5)

    # straight arrows between main-column rows
    for i in range(len(rows) - 1):
        f.arr(CX, rows[i] + BH / 2 + 0.8, CX, rows[i + 1] - BH / 2 - 0.8)

    # ── branches ──────────────────────────────────
    LX, RX, BR_BW, BR_BH = 38, 122, 44, 9

    # branch arrows from node embed
    ey = rows[-1] + BH / 2 + 0.8
    f.arr(CX, ey, LX, 141 - BR_BH / 2 - 0.8, col=OR, rad=-0.2)
    f.arr(CX, ey, RX, 141 - BR_BH / 2 - 0.8, col=PU, rad=0.2)

    # H head  +  score head
    f.box(LX, 141, BR_BW, BR_BH, OR_T, OR)
    f.txt2(LX, 141, "H head", "H ∈ ℝᴺˣᴺ", fs=6.5)

    f.box(RX, 141, BR_BW, BR_BH, PU_T, PU)
    f.txt2(RX, 141, "Score head", "logits ∈ ℝᴺ", fs=6.5)

    # CTQW  +  BCE
    f.box(LX, 162, BR_BW, BR_BH, OR_T, OR)
    f.txt2(LX, 162, "CTQW / ENAQT", "communicability", fs=6.2)

    f.box(RX, 162, BR_BW, BR_BH, PU_T, PU)
    f.txt2(RX, 162, "Weighted BCE", "loss", fs=6.2)

    # arrows within branches
    f.arr(LX, 141 + BR_BH / 2 + 0.8, LX, 162 - BR_BH / 2 - 0.8, col=OR)
    f.arr(RX, 141 + BR_BH / 2 + 0.8, RX, 162 - BR_BH / 2 - 0.8, col=PU)

    # active site seed (dashed, sky blue)
    f.box(80, 149, 30, 8, SK_T, SK, rx=1.2)
    f.txt2(80, 149, "Active site seed", "(APO_ACTIVE)", fs=5.8, col=SK)
    f.arr(80, 153, LX + BR_BW / 2 - 0.8, 162 - BR_BH / 2 - 0.8,
          col=SK, dashed=True, rad=0.1)

    # convergence arrows
    f.arr(LX, 162 + BR_BH / 2 + 0.8, CX, 188 - BH / 2 - 0.8,
          col=OR, rad=0.2)
    f.arr(RX, 162 + BR_BH / 2 + 0.8, CX, 188 - BH / 2 - 0.8,
          col=PU, rad=-0.2)

    # combined scores  +  top-K
    f.box(CX, 188, BW, BH, PU_T, PU)
    f.txt2(CX, 188, "Combined scores",
           "σ(logits) + β · C[active→i]", fs=6.2)

    f.box(CX, 208, BW, BH, GR_T, GR)
    f.txt(CX, 208, "Top-K residues  =  predicted pocket", fs=6.5, wt="bold")

    f.arr(CX, 188 + BH / 2 + 0.8, CX, 208 - BH / 2 - 0.8, col=GR)

    f.save("fig1_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — GVP-GNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

def fig2():
    f = F(160, 130)

    BH = 9
    # ── input features ────────────────────────────
    f.box(30, 20, 44, BH, BL_T, BL)
    f.txt2(30, 20, "Scalar  sᵢ ∈ ℝ²⁴", "Vector  Vᵢ ∈ ℝ¹ˣ³", fs=6.2)

    f.box(100, 20, 44, BH, BL_T, BL)
    f.txt2(100, 20, "Edge scalar  eᵢⱼ ∈ ℝ²⁴", "Edge vector  eᵢⱼ ∈ ℝ¹ˣ³", fs=6.2)

    # ── embedding ─────────────────────────────────
    f.arr(30, 24.5, 30, 38.5)
    f.arr(100, 24.5, 100, 38.5)

    f.box(30, 43, 44, BH, GR_T, GR)
    f.txt2(30, 43, "Node embed layer", "hₛ ∈ ℝ⁶⁴,  hᵥ ∈ ℝ⁸ˣ³", fs=6.2)

    f.box(100, 43, 44, BH, GR_T, GR)
    f.txt(100, 43, "Edge embed layer", fs=6.5)

    # ── 4× message passing ───────────────────────
    f.arr(65, 43, 70, 43, col=GR)  # node embed → MP block
    f.arr(100, 47.5, 80, 65.5, col=GR)

    f.box(80, 70, 80, 10, GR_T, GR, rx=2)
    f.txt(80, 70, "GVP Message Passing  ×4", fs=7, wt="bold")

    f.arr(30, 47.5, 30, 65, col=GR)
    # self-loop indicator
    f.ax.annotate("", xy=(30, f._y(70)), xytext=(30, f._y(65)),
                  arrowprops=dict(arrowstyle="->", color=GR,
                                  connectionstyle="arc3,rad=-0.5",
                                  lw=0.55, mutation_scale=7))
    f.txt(6, 68, "×4", fs=7, wt="bold", col=GR)

    # ── node embedding output ────────────────────
    f.arr(80, 75, 80, 87.5)
    f.box(80, 92, 52, BH, GR_T, GR)
    f.txt2(80, 92, "Node embeddings", "s ∈ ℝᴺˣ⁶⁴,  V ∈ ℝᴺˣ⁸ˣ³", fs=6.5)

    # ── two heads ────────────────────────────────
    f.arr(80, 96.5, 38, 108.5, col=OR, rad=-0.15)
    f.arr(80, 96.5, 122, 108.5, col=PU, rad=0.15)

    f.box(38, 113, 44, BH, OR_T, OR)
    f.txt2(38, 113, "H head  →  H ∈ ℝᴺˣᴺ",
           "(CTQW communicability)", fs=6.2)

    f.box(122, 113, 44, BH, PU_T, PU)
    f.txt2(122, 113, "Score head  →  logits ∈ ℝᴺ",
           "(BCE classification)", fs=6.2)

    f.label(4, 6, "")
    f.save("fig2_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Loss Function Evolution  (horizontal flow)
# ─────────────────────────────────────────────────────────────────────────────

def fig3():
    # 6 models, wider figure to avoid clipping
    FW = 180
    f = F(FW, 68)

    BW, BH = 21, 13
    gap = 7
    step = BW + gap   # 28
    n = 6
    total = n * BW + (n - 1) * gap   # 126 + 35 = 161
    offset = (FW - total) / 2         # (180 - 161)/2 = 9.5
    xs = [offset + i * step for i in range(n)]

    data = [
        ("CTQW v1", "Sparse H",      "0.533", RD, RD_T),
        ("CTQW v2", "Dense H",       "0.800", OR, OR_T),
        ("CTQW v3", "Dense H+ENAQT", "0.800", OR, OR_T),
        ("CTQW v4", "Per-protein γ", "0.800", OR, OR_T),
        ("Hybrid",  "BCE+CTQW",      "0.867", PU, PU_T),
        ("Baseline","BCE only",       "0.933", GR, GR_T),
    ]

    CY, BADGE_Y = 24, 48

    for i, (title, subtitle, score, ec, fc) in enumerate(data):
        cx = xs[i] + BW / 2

        # main box
        f.box(cx, CY, BW, BH, fc, ec, rx=1.5)
        f.txt(cx, CY - 2.5, title, fs=6.5, wt="bold")
        f.txt(cx, CY + 2.5, subtitle, fs=5.5)

        # score badge (circle in data coords only)
        circ = plt.Circle((cx, f._y(BADGE_Y)), 4.8,
                           fc=fc, ec=ec, lw=0.5, zorder=3)
        f.ax.add_patch(circ)
        f.txt(cx, BADGE_Y, score, fs=6.5, wt="bold", col=ec)

        # inter-model arrow
        if i < len(data) - 1:
            nx = xs[i + 1] + BW / 2
            f.arr(cx + BW / 2 + 0.8, CY, nx - BW / 2 - 0.8, CY,
                  col=DK, lw=0.4, ms=6)

    f.save("fig3_evolution")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — ENAQT Failure Analysis
# ─────────────────────────────────────────────────────────────────────────────

def fig4():
    f = F(130, 100)

    BW, BH = 44, 10

    # ── two proteins ──────────────────────────────
    f.box(32, 14, BW, BH, SK_T, SK)
    f.txt2(32, 14, "KRAS / Cardiac Myosin", "near-pocket  (< 15 Å)", fs=6.2)

    f.box(98, 14, BW, BH, RD_T, RD)
    f.txt2(98, 14, "BCR-ABL1", "far-pocket  (> 30 Å)", fs=6.2)

    # ── optimal gamma ─────────────────────────────
    f.arr(32, 19, 32, 31)
    f.arr(98, 19, 98, 31)

    f.box(32, 36, BW, BH, SK_T, SK)
    f.txt2(32, 36, "Optimal  γ ≈ 0.8", "(high dephasing)", fs=6.2)

    f.box(98, 36, BW, BH, RD_T, RD)
    f.txt2(98, 36, "Optimal  γ ≈ 0.05", "(near-coherent)", fs=6.2)

    # conflict arrow
    f.arr(54, 36, 76, 36, col=RD, lw=0.8)
    f.txt(65, 33.5, "conflict", fs=6, col=RD, wt="bold")

    # ── shared gamma head ─────────────────────────
    f.arr(32, 41, 65, 53.5)
    f.arr(98, 41, 65, 53.5)

    f.box(65, 58, 50, BH, RD_T, RD)
    f.txt2(65, 58, "Shared γ head", "learns compromise  γ ≈ 0.17", fs=6.2)

    f.arr(65, 63, 65, 74.5)

    # ── result ────────────────────────────────────
    f.box(65, 79, 50, 10, RD_T, RD)
    f.txt2(65, 79, "BCR-ABL1 hit rate = 0.00", "Cardiac = 0.60  →  plateau", fs=6.2,
           col=RD)

    # annotation
    f.txt(65, 93, "Root cause: conflicting γ requirements across proteins",
          fs=6.0, style="italic", col=DK)

    f.save("fig4_enaqt_failure")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Hybrid Model (Training vs Inference)
# ─────────────────────────────────────────────────────────────────────────────

def fig5():
    f = F(160, 100)

    BH = 9

    # ── shared encoder (centre) ───────────────────
    f.box(80, 18, 52, BH, GR_T, GR)
    f.txt2(80, 18, "Shared GVP-GNN encoder", "(4 layers, 100k params)", fs=6.5)

    # ── two heads ────────────────────────────────
    f.arr(80, 22.5, 38, 34.5, col=OR, rad=-0.15)
    f.arr(80, 22.5, 122, 34.5, col=PU, rad=0.15)

    f.box(38, 39, 44, BH, OR_T, OR)
    f.txt2(38, 39, "H head", "H ∈ ℝᴺˣᴺ", fs=6.5)

    f.box(122, 39, 44, BH, PU_T, PU)
    f.txt2(122, 39, "Score head", "logits ∈ ℝᴺ", fs=6.5)

    # ── training path (left) ─────────────────────
    f.arr(38, 43.5, 38, 55.5, col=OR)
    f.box(38, 60, 44, BH, OR_T, OR)
    f.txt2(38, 60, "Comm. loss", "−log(C⁺/ (C⁺+C⁻))", fs=6.2)

    f.arr(122, 43.5, 122, 55.5, col=PU)
    f.box(122, 60, 44, BH, PU_T, PU)
    f.txt2(122, 60, "Weighted BCE", "pos_weight=10", fs=6.2)

    # combined loss
    f.arr(38, 64.5, 80, 76.5, col=OR, rad=0.2)
    f.arr(122, 64.5, 80, 76.5, col=PU, rad=-0.2)

    f.box(80, 81, 52, BH, YL_T, "#C4A800")
    f.txt(80, 81, "loss = BCE  +  λ · L_comm", fs=6.5, wt="bold")

    # ── inference label ───────────────────────────
    # divider line between training / inference labels

    # inference annotation
    f.box(80, 93, 90, 7, "#F5F5F5", "#AAAAAA", rx=1)
    f.txt(38, 93, "Training", fs=6, col=OR, wt="bold")
    f.txt(122, 93, "Inference: scores = σ(logits) + β · C[active→i]",
          fs=5.8, col=PU)

    f.save("fig5_hybrid")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Three-layer Validation Framework
# ─────────────────────────────────────────────────────────────────────────────

def fig6():
    f = F(130, 160)

    BW, BH = 70, 11
    CX = 52

    # ── input ─────────────────────────────────────
    f.box(CX, 12, BW, BH, BL_T, BL)
    f.txt2(CX, 12, "Undruggable protein",
           "(apo structure only)", fs=6.5)

    f.arr(CX, 17.5, CX, 29.5)

    # ── layer 1 ───────────────────────────────────
    f.box(CX, 35, BW, 14, GR_T, GR, rx=2)
    f.txt(CX, 28.5, "Layer 1 — Geometry", fs=7, wt="bold", col=GR)
    f.txt(CX, 33.5, "fpocket / DoGSiteScorer", fs=6.2)
    f.txt(CX, 37.0, "volume > 300 Å³,  score > 0.5", fs=6.2)
    f.arr(CX, 42, CX, 54)

    # ── layer 2 ───────────────────────────────────
    f.box(CX, 60, BW, 16, GR_T, GR, rx=2)
    f.txt(CX, 53.5, "Layer 2 — Dynamics", fs=7, wt="bold", col=GR)
    f.txt(CX, 58.5, "ENM/PRS: effectivity↑", fs=6.2)
    f.txt(CX, 63.0, "CTQW: C[active→i]↑", fs=6.2)
    f.txt(CX, 67.0, "→ communication hub", fs=6.0, style="italic")
    f.arr(CX, 68, CX, 80)

    # ── layer 3 ───────────────────────────────────
    f.box(CX, 86, BW, 16, PU_T, PU, rx=2)
    f.txt(CX, 79.5, "Layer 3 — Evolution", fs=7, wt="bold", col=PU)
    f.txt(CX, 84.5, "ConSurf: conservation score↑", fs=6.2)
    f.txt(CX, 89.0, "EVcouplings: residue coevolution", fs=6.2)
    f.txt(CX, 93.5, "with active site residues", fs=6.2)
    f.arr(CX, 94, CX, 106)

    # ── candidate ─────────────────────────────────
    f.box(CX, 112, BW, 11, GR_T, GR)
    f.txt(CX, 109, "High-confidence candidate", fs=7, wt="bold")
    f.txt(CX, 114, "all three layers converge", fs=6.2)

    # ── cryptic pocket side branch ────────────────
    # dashed arrow from layer1 to critical limitation
    RX = 110
    f.box(RX, 35, 34, 11, RD_T, RD, rx=1.5)
    f.txt(RX, 31.5, "Critical limitation", fs=6.5, wt="bold", col=RD)
    f.txt(RX, 36.5, "Cryptic pockets not", fs=5.8, col=RD)
    f.txt(RX, 40.0, "visible in apo", fs=5.8, col=RD)

    f.arr(87, 35, 93, 35, col=RD, dashed=True, lw=0.4)

    f.arr(RX, 40.5, RX, 53.5, col=RD)

    f.box(RX, 59, 34, 13, OR_T, OR, rx=1.5)
    f.txt(RX, 54.5, "Solution: MDpocket", fs=6.5, wt="bold", col=OR)
    f.txt(RX, 59.5, "fpocket on 100 ns", fs=5.8)
    f.txt(RX, 63.5, "MD frames", fs=5.8)
    f.txt(RX, 67.0, "(high compute)", fs=5.5, style="italic")

    # dashed static structure annotation
    f.txt(88, 22, "Static structure", fs=5.5, style="italic", col="#888888",
          ha="center")
    f.arr(88, 24, 93, 29, col="#AAAAAA", dashed=True, lw=0.3, ms=5)

    f.save("fig6_validation")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — PocketMiner + CTQW + ENM/PRS Proposed Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fig7():
    f = F(155, 158)

    BH = 10

    # ── protein input ─────────────────────────────
    f.box(77.5, 12, 50, BH, BL_T, BL)
    f.txt2(77.5, 12, "Undruggable protein", "(apo structure)", fs=6.5)

    # branch arrows to three paths
    f.arr(77.5, 17, 33, 29, col=OR, rad=-0.15)
    f.arr(77.5, 17, 77.5, 29, col=PU)
    f.arr(77.5, 17, 122, 29, col=GR, rad=0.15)

    # ── path 1: PocketMiner ──────────────────────
    f.box(33, 35, 44, 11, OR_T, OR, rx=2)
    f.txt(33, 30, "PocketMiner", fs=7, wt="bold", col=OR)
    f.txt(33, 35.5, "GVP-GNN", fs=6.2)
    f.txt(33, 39.5, "trained on MD", fs=6.2)

    f.arr(33, 40.5, 33, 52.5, col=OR)
    f.box(33, 58, 44, 10, OR_T, OR)
    f.txt(33, 54.5, "P[cryptic] ∈ ℝᴺ", fs=6.5, wt="bold")
    f.txt(33, 59.5, "(geometric pocket", fs=5.8)
    f.txt(33, 63.5, " likelihood)", fs=5.8)

    # ── path 2: CTQW Hybrid ───────────────────────
    f.box(77.5, 35, 44, 11, PU_T, PU, rx=2)
    f.txt(77.5, 30, "CTQW Hybrid", fs=7, wt="bold", col=PU)
    f.txt(77.5, 35.5, "GVP-GNN +", fs=6.2)
    f.txt(77.5, 39.5, "BCE + CTQW", fs=6.2)

    f.arr(77.5, 40.5, 77.5, 52.5, col=PU)
    f.box(77.5, 58, 44, 10, PU_T, PU)
    f.txt(77.5, 54.5, "Allosteric score ∈ ℝᴺ", fs=6.5, wt="bold")
    f.txt(77.5, 59.5, "(communication", fs=5.8)
    f.txt(77.5, 63.5, " likelihood)", fs=5.8)

    # ── path 3: ENM/PRS ──────────────────────────
    f.box(122, 35, 44, 11, GR_T, GR, rx=2)
    f.txt(122, 30, "ENM / PRS", fs=7, wt="bold", col=GR)
    f.txt(122, 35.5, "Perturbation Response", fs=6.0)
    f.txt(122, 39.5, "Scan", fs=6.0)

    f.arr(122, 40.5, 122, 52.5, col=GR)
    f.box(122, 58, 44, 10, GR_T, GR)
    f.txt(122, 54.5, "effectivity[i] ∈ ℝᴺ", fs=6.5, wt="bold")
    f.txt(122, 59.5, "(communication", fs=5.8)
    f.txt(122, 63.5, " hub score)", fs=5.8)

    # ── intersection ─────────────────────────────
    f.arr(33, 63, 77.5, 75, col=OR, rad=0.2)
    f.arr(77.5, 63, 77.5, 75, col=PU)
    f.arr(122, 63, 77.5, 75, col=GR, rad=-0.2)

    f.box(77.5, 82, 58, 16, GR_T, GR, rx=2)
    f.txt(77.5, 75.5, "Intersection", fs=7, wt="bold")
    f.txt(77.5, 80.5, "high in all three  =", fs=6.5, wt="bold")
    f.txt(77.5, 85.5, "cryptic allosteric pocket", fs=6.2, style="italic")
    f.txt(77.5, 90.0, "candidate", fs=6.2, style="italic")

    # ── evolutionary validation ───────────────────
    f.arr(77.5, 90.5, 77.5, 103.5)

    f.box(77.5, 107, 58, 10, PU_T, PU)
    f.txt(77.5, 102.5, "ConSurf + EVcouplings", fs=6.5, wt="bold")
    f.txt(77.5, 108.5, "evolutionary validation", fs=6.2)

    f.arr(77.5, 112, 77.5, 124)

    # ── final candidate ───────────────────────────
    f.box(77.5, 130, 58, 11, GR_T, GR, rx=2)
    f.txt(77.5, 125.5, "Final candidate", fs=7.5, wt="bold")
    f.txt(77.5, 131.5, "for experimental validation", fs=6.2)
    f.txt(77.5, 136.5, "(HDX-MS, NMR, SPR)", fs=6.0, style="italic")

    # side labels
    f.txt(8, 47, "cryptic", fs=6, style="italic", col=OR, ha="center")
    f.txt(146, 47, "dynamic", fs=6, style="italic", col=GR, ha="center")

    f.save("fig7_pocketminer")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures …")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("Done — figures saved to", OUT)
