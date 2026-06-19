"""
quantum_walk_demo.py
====================
A self-contained demonstration of Continuous-Time Quantum Walk (CTQW)
for protein allosteric communication, showing:

  1. Ballistic vs diffusive spreading on a path graph  (σ ~ t vs √t)
  2. Time-dependent signal: P(allo, t) peak >> T→∞ communicability
  3. Real protein-like graph: CTQW finds long-range pockets better
  4. Why our current model (T→∞ C[i,j]) loses signal — and the fix

Run:  python3 quantum_walk_demo.py
Output: figures/quantum_walk_demo.pdf  figures/quantum_walk_demo.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import os

# ── Wong 2011 palette ────────────────────────────────────────────────────────
BL, OR, GR, RD = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SK, PU, YL, DK = "#56B4E9", "#CC79A7", "#F0E442", "#222222"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Math utilities
# ─────────────────────────────────────────────────────────────────────────────

def eigh_cached(H):
    """Eigendecompose a real symmetric matrix."""
    return np.linalg.eigh(H)


def ctqw_prob(evals, evecs, j0, t_arr):
    """
    P(j, t) for CTQW starting at node j0.

    |ψ(t)⟩ = e^{-iHt} |j0⟩
    P(j, t) = |⟨j|ψ(t)⟩|²

    Returns array of shape (N, len(t_arr)).
    """
    N = evecs.shape[0]
    psi0 = np.zeros(N, complex)
    psi0[j0] = 1.0
    # Project onto eigen-basis
    c = evecs.T.conj() @ psi0                      # (N,)
    # For each t: psi_t = evecs @ diag(e^{-i λ t}) c
    phases = np.exp(-1j * np.outer(evals, t_arr))   # (N, T)
    psi_t = evecs @ (phases * c[:, None])           # (N, T)
    return np.abs(psi_t) ** 2                       # (N, T)


def ctrw_prob(L, j0, t_arr):
    """
    p(t) for continuous-time classical random walk starting at j0.

    dp/dt = -L p   →   p(t) = e^{-Lt} p0

    L = D - A (graph Laplacian, not negative).
    Returns array of shape (N, len(t_arr)).
    """
    N = L.shape[0]
    p0 = np.zeros(N)
    p0[j0] = 1.0
    evals_L, evecs_L = np.linalg.eigh(L)
    evals_L = np.maximum(evals_L, 0)               # numerical safety
    decays = np.exp(-np.outer(evals_L, t_arr))     # (N, T)
    c = evecs_L.T @ p0                             # (N,)
    p_t = evecs_L @ (decays * c[:, None])          # (N, T)
    return np.maximum(p_t, 0)


def path_graph(N):
    """Adjacency matrix of path graph P_N."""
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i+1] = A[i+1, i] = 1.0
    return A


def laplacian(A):
    return np.diag(A.sum(1)) - A


def communicability(evecs):
    """C[i,j] = Σ_k V_ik² V_jk²  (T→∞ average of CTQW)."""
    V2 = evecs ** 2           # (N, N)
    return V2 @ V2.T          # C[i,j]


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1: Path graph  (1D allosteric chain)
# ─────────────────────────────────────────────────────────────────────────────
N1 = 30
A1 = path_graph(N1)
L1 = laplacian(A1)
ev1, evc1 = eigh_cached(A1)

T_arr = np.linspace(0, 25, 500)
j_src = 0                          # active site = left end
j_allo = N1 - 1                    # allosteric   = right end

Pq1 = ctqw_prob(ev1, evc1, j_src, T_arr)         # (N1, T)
Pc1 = ctrw_prob(L1,  j_src, T_arr)               # (N1, T)

# Spreading width σ(t)
nodes1 = np.arange(N1)
sigma_q = np.sqrt(np.sum((nodes1[:, None] - np.sum(nodes1[:, None] * Pq1, 0)) ** 2 * Pq1, 0))
sigma_c = np.sqrt(np.sum((nodes1[:, None] - np.sum(nodes1[:, None] * Pc1, 0)) ** 2 * Pc1, 0))

# Allosteric signal
P_allo_q = Pq1[j_allo, :]
P_allo_c = Pc1[j_allo, :]

C1 = communicability(evc1)
c_allo = C1[j_src, j_allo]                       # T→∞ value
t_star_idx = np.argmax(P_allo_q)
t_star      = T_arr[t_star_idx]
P_star      = P_allo_q[t_star_idx]

print(f"Path graph N={N1}")
print(f"  T→∞ communicability C[0,{N1-1}]     = {c_allo:.4f}")
print(f"  Peak quantum signal  P(allo, t*={t_star:.1f}) = {P_star:.4f}")
print(f"  Amplification at t*: {P_star / c_allo:.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2: Protein-like graph — BCR-ABL1 inspired
# ─────────────────────────────────────────────────────────────────────────────
#
#  Key insight: BCR-ABL1 allosteric site is >30Å from active site (graph distance ~10)
#  Decoy pocket branches off at graph distance ~3.
#
#  Topology:
#    0(active) ─ 1 ─ 2 ─ 3 ─ 4 ─ 5 ─ 6 ─ 7 ─ 8 ─ 9 ─ 10(allo)
#                    |
#                   11 ─ 12 ─ 13(decoy)
#
#  Quantum prediction: at t* ~ L_allo = 10, the quantum wave packet
#  has PASSED the decoy (L_decoy = 3) and is NOW at the allo site.
#  CTQW discriminates allo from decoy; CTRW cannot (decoy is closer).
#

def connect(A, i, j):
    A[i, j] = A[j, i] = 1.0

L_allo  = 10  # graph distance active → allosteric
L_decoy = 3   # graph distance branch-point → decoy
branch  = 2   # where the decoy branch attaches to the main chain

N2 = (L_allo + 1) + L_decoy   # 14 nodes total
A2 = np.zeros((N2, N2))

# Main chain: 0 – 1 – 2 – ... – L_allo
for k in range(L_allo):
    connect(A2, k, k + 1)

# Decoy branch: branch – (L_allo+1) – ... – (L_allo+L_decoy)
for k in range(L_decoy):
    connect(A2, branch + (L_allo + 1 + k - branch), L_allo + 1 + k)
# correct indexing for branch:
A2 = np.zeros((N2, N2))
for k in range(L_allo):
    connect(A2, k, k + 1)
for k in range(L_decoy):
    A2[branch, L_allo + 1 + k] = 0
    A2[L_allo + 1 + k, branch] = 0
connect(A2, branch, L_allo + 1)
for k in range(L_decoy - 1):
    connect(A2, L_allo + 1 + k, L_allo + 2 + k)

L2 = laplacian(A2)
ev2, evc2 = eigh_cached(A2)

j_act2  = 0                # active site
j_allo2 = L_allo           # allosteric site (node 10)
j_dec2  = N2 - 1           # decoy (last node of branch)

# Extended time range for this graph (wave packet reaches allo at t~L_allo)
T2_arr = np.linspace(0, 50, 1000)
Pq2 = ctqw_prob(ev2, evc2, j_act2, T2_arr)
Pc2 = ctrw_prob(L2,  j_act2, T2_arr)

P_allo2_q = Pq2[j_allo2, :]
P_allo2_c = Pc2[j_allo2, :]
P_dec2_q  = Pq2[j_dec2, :]
P_dec2_c  = Pc2[j_dec2, :]

C2 = communicability(evc2)
c2_allo = C2[j_act2, j_allo2]
c2_dec  = C2[j_act2, j_dec2]
t2_star_idx = np.argmax(P_allo2_q)
t2_star = T2_arr[t2_star_idx]

print(f"\nProtein-like graph N={N2}  (allo at dist {L_allo}, decoy at dist {L_decoy})")
print(f"  CTQW P(allo,  t*={t2_star:.1f}) = {P_allo2_q[t2_star_idx]:.4f}")
print(f"  CTQW P(decoy, t*={t2_star:.1f}) = {P_dec2_q[t2_star_idx]:.4f}")
print(f"  CTRW P(allo,  t*={t2_star:.1f}) = {P_allo2_c[t2_star_idx]:.4f}")
print(f"  CTRW P(decoy, t*={t2_star:.1f}) = {P_dec2_c[t2_star_idx]:.4f}")
print(f"  T→∞ C[act, allo]  = {c2_allo:.4f}")
print(f"  T→∞ C[act, decoy] = {c2_dec:.4f}")
print(f"  CTQW ratio allo/decoy at t* = {P_allo2_q[t2_star_idx]/max(P_dec2_q[t2_star_idx],1e-9):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure  (183 mm wide, 4-panel)
# ─────────────────────────────────────────────────────────────────────────────
MM = 1 / 25.4
fig = plt.figure(figsize=(183 * MM, 130 * MM))
gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.08, right=0.97,
                       top=0.92, bottom=0.12,
                       wspace=0.42, hspace=0.52)

ax_heat_q = fig.add_subplot(gs[0, 0])
ax_heat_c = fig.add_subplot(gs[0, 1])
ax_sigma  = fig.add_subplot(gs[0, 2])
ax_allo1  = fig.add_subplot(gs[1, 0])
ax_allo2  = fig.add_subplot(gs[1, 1])
ax_comm   = fig.add_subplot(gs[1, 2])


# ── Panel A: CTQW heatmap on path graph ──────────────────────────────────────
im_q = ax_heat_q.pcolormesh(T_arr, nodes1, Pq1,
                             cmap="Blues", shading="auto", vmin=0, vmax=0.4)
ax_heat_q.axhline(j_allo, color=OR, lw=0.8, ls="--", label=f"node {j_allo}")
ax_heat_q.set_xlabel("Time $t$")
ax_heat_q.set_ylabel("Node $j$")
ax_heat_q.set_title("(A)  CTQW — ballistic spreading", fontweight="bold")
ax_heat_q.set_xlim(0, 25)
fig.colorbar(im_q, ax=ax_heat_q, fraction=0.046, pad=0.03)


# ── Panel B: Classical heatmap ────────────────────────────────────────────────
im_c = ax_heat_c.pcolormesh(T_arr, nodes1, Pc1,
                             cmap="Oranges", shading="auto", vmin=0, vmax=0.4)
ax_heat_c.axhline(j_allo, color=BL, lw=0.8, ls="--")
ax_heat_c.set_xlabel("Time $t$")
ax_heat_c.set_title("(B)  CTRW — diffusive spreading", fontweight="bold")
ax_heat_c.set_xlim(0, 25)
fig.colorbar(im_c, ax=ax_heat_c, fraction=0.046, pad=0.03)


# ── Panel C: σ(t) comparison ──────────────────────────────────────────────────
t_plot = T_arr[1:]
ax_sigma.plot(t_plot, sigma_q[1:], color=BL, lw=1.2, label="CTQW  $\\sigma \\propto t$")
ax_sigma.plot(t_plot, sigma_c[1:], color=OR, lw=1.2, label="CTRW  $\\sigma \\propto \\sqrt{t}$")
ax_sigma.plot(t_plot, 0.95 * t_plot,         color=BL, lw=0.6, ls=":", alpha=0.6)
ax_sigma.plot(t_plot, 1.3 * np.sqrt(t_plot), color=OR, lw=0.6, ls=":", alpha=0.6)
ax_sigma.set_xlabel("Time $t$")
ax_sigma.set_ylabel("Width $\\sigma(t)$")
ax_sigma.set_title("(C)  Ballistic vs Diffusive", fontweight="bold")
ax_sigma.legend(fontsize=6.5, frameon=False)
ax_sigma.set_xlim(0, 25)


# ── Panel D: Allosteric signal on path graph ──────────────────────────────────
ax_allo1.plot(T_arr, P_allo_q, color=BL, lw=1.2, label="CTQW $P(\\mathrm{allo}, t)$")
ax_allo1.plot(T_arr, P_allo_c, color=OR, lw=1.2, label="CTRW $p(\\mathrm{allo}, t)$")
ax_allo1.axhline(c_allo, color=BL, lw=0.8, ls="--", alpha=0.7,
                 label=f"$C[0,{j_allo}]={c_allo:.3f}$ ($T\\to\\infty$)")
ax_allo1.axvline(t_star, color=GR, lw=0.8, ls="--", alpha=0.8)
ax_allo1.text(t_star + 0.5, P_star * 0.9,
              f"$t^*={t_star:.0f}$\n$P={P_star:.2f}$",
              fontsize=6, color=GR)
ax_allo1.set_xlabel("Time $t$")
ax_allo1.set_ylabel("$P(\\mathrm{allo}, t)$")
ax_allo1.set_title("(D)  Allosteric signal  (path graph)", fontweight="bold")
ax_allo1.legend(fontsize=6, frameon=False)
ax_allo1.set_xlim(0, 25)


# ── Panel E: Protein-like graph — allo vs decoy ───────────────────────────────
# Discrimination ratio: P(allo,t) / P(decoy,t)
eps = 1e-8
ratio_q = P_allo2_q / (P_dec2_q + eps)
ratio_c = P_allo2_c / (P_dec2_c + eps)
# Clip for visibility
ratio_q_plot = np.clip(ratio_q, 0, 200)
ratio_c_plot = np.clip(ratio_c, 0, 200)

ax_allo2.semilogy(T2_arr, ratio_q, color=BL, lw=1.2,
                  label="CTQW  $P(\\mathrm{allo})/P(\\mathrm{decoy})$")
ax_allo2.semilogy(T2_arr, ratio_c, color=OR, lw=1.2,
                  label="CTRW  $p(\\mathrm{allo})/p(\\mathrm{decoy})$")
ax_allo2.axhline(1.0, color=DK, lw=0.7, ls=":", alpha=0.5)
ax_allo2.axvline(t2_star, color=BL, lw=0.8, ls="--", alpha=0.7,
                 label=f"$t^*={t2_star:.0f}$")
ax_allo2.text(t2_star + 1, ratio_q[t2_star_idx] * 0.4,
              f"ratio={ratio_q[t2_star_idx]:.0f}×", fontsize=6, color=BL)
ax_allo2.set_xlabel("Time $t$")
ax_allo2.set_ylabel("$P(\\mathrm{allo}) / P(\\mathrm{decoy})$  (log)")
ax_allo2.set_title("(E)  Discrimination ratio", fontweight="bold")
ax_allo2.legend(fontsize=6, frameon=False)
ax_allo2.set_xlim(0, 50)
ax_allo2.set_ylim(1e-3, 1e4)


# ── Panel F: CTQW communicability vs optimal-time signal ─────────────────────
# Compare: C[src, j]  vs  P(j, t*)  for all nodes in protein graph
c_row = C2[j_act2, :]
P_row = Pq2[:, t2_star_idx]

c_row = C2[j_act2, :]
P_row = Pq2[:, t2_star_idx]
node_idx = np.arange(N2)

ax_comm.bar(node_idx, c_row,  alpha=0.6, color=OR, width=0.8,
            label=f"$C[\\mathrm{{act}}, j]$ ($T\\to\\infty$)")
ax_comm.bar(node_idx, P_row,  alpha=0.6, color=BL, width=0.45,
            label=f"$P(j,\\, t^*={t2_star:.0f})$")
ax_comm.axvline(j_allo2 - 0.5, color=GR, lw=1.0, ls="--", alpha=0.8)
ax_comm.axvline(j_dec2  - 0.5, color=RD, lw=1.0, ls="--", alpha=0.8)
ax_comm.text(j_allo2, max(P_row) * 0.88, "allo\n(dist 10)", fontsize=5.5, color=GR, ha="center")
ax_comm.text(j_dec2,  max(P_row) * 0.88, "decoy\n(dist 3)",  fontsize=5.5, color=RD, ha="center")
ax_comm.set_xlabel("Node $j$")
ax_comm.set_ylabel("Signal strength")
ax_comm.set_title("(F)  $C[i,j]$ vs $P(j,t^*)$  (protein graph)", fontweight="bold")
ax_comm.legend(fontsize=6, frameon=False)


# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "CTQW for Allosteric Prediction — Why $T\\to\\infty$ Communicability Loses Signal",
    fontsize=9, fontweight="bold", y=0.99,
)

fig.savefig(os.path.join(OUT, "quantum_walk_demo.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(OUT, "quantum_walk_demo.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: figures/quantum_walk_demo.pdf  figures/quantum_walk_demo.png")
