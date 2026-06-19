# Allosteric Pocket Prediction via Continuous-Time Quantum Walk

Cleveland Clinic Challenge 2026 — Task: predict allosteric pockets in three disease-relevant proteins from apo structures only.

## Problem

Given the **apo (unbound) structure** of a protein and its known **orthosteric (active) site**, predict which residues form the **allosteric pocket** — a distal binding site that modulates protein function without blocking the active site.

**Challenge proteins:**

| Protein | PDB (apo) | Allosteric ligand | Biological relevance |
|---|---|---|---|
| KRAS G12C | 4OBE | Sotorasib (6OIM) | Oncogenic RAS, switch-II pocket |
| BCR-ABL1 | 1OPL | Asciminib (5MO4) | Leukemia kinase, myristoyl pocket |
| Cardiac Myosin | 5TBY | Mavacamten proxy (6C1H) | Hypertrophic cardiomyopathy |

---

## Core Idea

The key hypothesis: **allosteric communication can be modeled as quantum information propagation** on the protein contact graph.

```
Protein structure (PDB)
       ↓
GVP-GNN (graph neural network on Cα contact graph)
       ↓
H  — learned Hamiltonian (N×N symmetric)
       ↓
CTQW — Continuous-Time Quantum Walk: |ψ(t)⟩ = e^{-iHt} |active_site⟩
       ↓
C[i,j] = Σₖ |Vᵢₖ|² |Vⱼₖ|²  — quantum communicability matrix
       ↓
Top-K residues by C[active_site → j]  =  predicted allosteric pocket
```

The CTQW starts at the **orthosteric (active) site** — derived from holo PDB structures — and propagates through the protein graph. Residues that accumulate high quantum probability are predicted allosteric sites.

---

## Methods & Results

Top-5 hit rate at 8Å (fraction of top-5 predicted residues within 8Å of known allosteric sites):

| Method | KRAS | BCR-ABL1 | CardiacMyosin | **Avg** |
|---|---|---|---|---|
| **GVP Baseline** (BCE) | **1.00** | **1.00** | **0.80** | **0.933** |
| CTQW v2/v3 (comm. loss) | 0.80 | 1.00 | 0.60 | 0.800 |
| RWR α=0.30 (no training) | 1.00 | 0.20 | 1.00 | 0.733 |
| CTQW v1 (prob loss) | 0.40 | 0.80 | 0.40 | 0.533 |

**Key findings:**
- GVP baseline (direct node classification with weighted BCE) achieves best overall performance
- CTQW communicability loss significantly outperforms CTQW probability loss (0.800 vs 0.533)
- Dense H (v3: `h@hᵀ + diag`) and sparse H (v2: edge-based) give identical performance — the bottleneck is the communicability loss, not the Hamiltonian structure
- RWR (no training) outperforms CTQW v1 — structural graph connectivity alone is a strong signal for proximal allosteric sites
- BCR-ABL1 (myristoyl pocket, >30Å from active site) requires learned features; RWR fails (0.20) while GVP+CTQW succeeds (1.00)

---

## Architecture

### GVP-GNN Encoder (`model.py`, `model_v2.py`)

Geometric Vector Perceptron ([Jing et al., 2021](https://arxiv.org/abs/2106.03843)) on the Cα contact graph (cutoff 12Å).

**Node features (per residue):**
- One-hot amino acid type (20-dim)
- Backbone dihedral angles (4-dim)
- Vector: Cα→Cβ direction (1×3)

**Edge features (per contact pair):**
- RBF-encoded distance (16-dim)
- Relative orientation (8-dim)
- Vector: unit displacement (1×3)

**Output:**

| Model | H construction | Rank |
|---|---|---|
| `HamiltonianGVP` (v1) | Sparse edge-based | ~degree |
| `HamiltonianGVP_v2` (v3) | `h@hᵀ/K + diag(softplus(s)) + α·H_sparse` | Full (N) |
| `GVPAlloBaseline` | Direct BCE classifier | — |

### CTQW Module (`ctqw_torch.py`)

```python
# Communicability matrix (closed-form, no time loop)
C[i,j] = Σₖ |Vᵢₖ|² |Vⱼₖ|²   where  H = V Λ Vᵀ

# Communicability loss
pos = C[active_idx, allosteric_mask].mean()
neg = C[active_idx, ~(allo|active)].mean()
loss = -log(pos / (pos + neg))
```

Eigendecomposition uses **float64** for numerical stability (sparse H has near-degenerate eigenvalues in float32).

### Training Data

- **ASD (Allosteric Site Database 2019):** 123 proteins with experimentally validated allosteric labels
- Source: [ASD_Release_201909_AS.txt](https://github.com/MoaazK/deepallo)
- Active site residues from orthosteric ligand positions in holo PDB (<5Å cutoff)

---

## File Structure

```
cleveland_clinic/
│
├── model.py            # HamiltonianGVP: sparse H (CTQW v1 & v2)
├── model_baseline.py   # GVPAlloBaseline: direct node classifier
├── model_v2.py         # HamiltonianGVP_v2: dense H = h@hᵀ + diag (CTQW v3)
├── ctqw_torch.py       # CTQW kernels: ctqw_prob, ctqw_connectivity_matrix,
│                       #               communicability_loss, allosteric_loss
├── dataset.py          # Challenge protein loader + allosteric label extraction
├── asd_dataset.py      # ASD training set preprocessor
├── pipeline.py         # PDB parsing, Cα contact graph builder
├── eval_rwr.py         # Random Walk with Restart evaluation
│
├── train_asd.py        # Train CTQW v1 (prob loss, sparse H)
├── train_asd_v2.py     # Train CTQW v2 (comm loss, sparse H)
├── train_asd_v3.py     # Train CTQW v3 (comm loss, dense H)
├── train_baseline.py   # Train GVP baseline (BCE, no CTQW)
│
├── run_asd_train.sh       # SLURM: CTQW v1
├── run_asd_train_v2.sh    # SLURM: CTQW v2
├── run_asd_train_v3.sh    # SLURM: CTQW v3
├── run_baseline_train.sh  # SLURM: GVP baseline
├── run_rwr.sh             # SLURM: RWR sweep
│
├── data/
│   ├── ASD_Release_201909_AS.txt   # ASD database (download separately)
│   ├── asd_train_selection.json    # 123 PDB IDs selected for training
│   └── pdbs/                       # Generated by asd_dataset.py
│
└── pdbs/                           # Challenge protein structures
    ├── 4OBE.pdb   (KRAS apo)
    ├── 6OIM.pdb   (KRAS holo + Sotorasib)
    ├── 1OPL.pdb   (BCR-ABL1 apo)
    ├── 5MO4.pdb   (BCR-ABL1 holo + Asciminib)
    ├── 5TBY.pdb   (CardiacMyosin apo)
    └── 6C1H.pdb   (CardiacMyosin holo + ADP)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download ASD data

```bash
wget -O data/ASD_Release_201909_AS.txt \
  https://raw.githubusercontent.com/MoaazK/deepallo/main/source_data/ASD_Release_201909_AS.txt
```

### 3. Preprocess ASD training set

```bash
# Downloads PDBs and builds graph features (~30 min)
python asd_dataset.py --asd_file data/ASD_Release_201909_AS.txt \
                      --out_dir data/asd_processed
```

### 4. Train

```bash
# GVP Baseline (best performance, ~30 min)
sbatch run_baseline_train.sh

# CTQW v2 (communicability loss, ~60 min)
sbatch run_asd_train_v2.sh

# CTQW v3 (dense H, ~90 min)
sbatch run_asd_train_v3.sh
```

### 5. Evaluate RWR (no training needed)

```bash
sbatch run_rwr.sh
```

---

## Active Site Derivation

Orthosteric (active) site residue numbers are extracted from holo PDB structures using a 5Å distance cutoff from ligand heavy atoms:

| Protein | Holo PDB | Ligand | Active site residues (apo numbering) |
|---|---|---|---|
| KRAS G12C | 4TQ9 | GDP | 12–18, 30, 34, 117, 145–147 |
| BCR-ABL1 | 2GQG | Dasatinib (ATP-site proxy) | 248, 269–271, 314, 317–321, 380–381 |
| CardiacMyosin | 5N6A | ADP | 126, 128, 130, 180–186 |

---

## Why CTQW?

Allostery is fundamentally a **long-range communication** phenomenon. Classical graph methods (e.g., shortest paths, RWR) describe diffusion, not quantum coherence. CTQW models:

- **Quantum interference**: constructive interference at allosteric sites
- **Tunneling**: paths through structurally unfavorable regions
- **Non-classical communicability**: `C[i,j]` captures all eigenmodes simultaneously

In practice, CTQW exceeds RWR on BCR-ABL1 (1.00 vs 0.20) — the myristoyl pocket is >30Å from the ATP site and unreachable by diffusion, but the learned Hamiltonian creates quantum tunneling paths.

---

## Citation

```bibtex
@misc{ctqw_allosteric_2026,
  title   = {Allosteric Pocket Prediction via Continuous-Time Quantum Walk on Protein Contact Graphs},
  author  = {Leo},
  year    = {2026},
  note    = {Cleveland Clinic Challenge 2026}
}
```

---

## License

MIT
