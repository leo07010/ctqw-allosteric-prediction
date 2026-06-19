"""
Dataset: extract allosteric site labels from holo PDB structures.

For each challenge protein:
  - Input:  apo PDB  (e.g. 4OBE) — what the model sees
  - Labels: allosteric residues derived from holo PDB (e.g. 6OIM)
            = residues in apo that are within LABEL_DIST Å of the
              ligand/drug in the holo structure (aligned by sequence)

Also supports ASD/CryptoSite for pre-training.
"""

import os
import numpy as np
from Bio import PDB, pairwise2
from Bio.SeqUtils import seq1
from pipeline import parse_pdb_ca, build_contact_graph
from model import build_protein_graph
import torch


LABEL_DIST = 10.0         # Å: residues within this distance of ligand = allosteric label

# Per-protein allosteric ligand names (HETATM residue names in holo PDB)
ALLO_LIGANDS = {
    "pdbs/6OIM.pdb": ["MOV"],        # KRAS: Sotorasib (AMG510)
    "pdbs/5MO4.pdb": ["AY7"],        # BCR-ABL1: Asciminib
    "pdbs/6C1H.pdb": ["ADP", "MG"],  # Cardiac Myosin: ADP binding site (Mavacamten site proxy)
}

# Manually curated allosteric residue numbers (fallback / supplement)
# From literature: residues lining the known allosteric pockets
KNOWN_ALLO_RESNUMS = {
    "KRAS_G12C":      list(range(12, 13)) + list(range(68, 78)),   # Switch-II pocket
    "BCR_ABL1":       list(range(309, 330)),                        # Myristoyl pocket
    "CardiacMyosin":  list(range(707, 730)) + list(range(777, 790)), # Converter domain
}

# Active site residue numbers derived from orthosteric ligand positions:
#   KRAS:          4TQ9 (GDP  < 5Å)
#   BCR-ABL1:      2GQG (dasatinib/1N1 < 5Å), residue numbers match 1OPL
#   CardiacMyosin: 5N6A (ADP  < 5Å), residue numbers match 5TBY
APO_ACTIVE_SITES = {
    "4OBE": [12, 13, 14, 15, 16, 17, 18, 30, 34, 117, 145, 146, 147],
    "1OPL": [248, 269, 270, 271, 314, 317, 318, 320, 321, 380, 381],
    "5TBY": [126, 128, 130, 180, 181, 182, 183, 184, 185, 186],
}


# ─────────────────────────────────────────────────────────────
# Extract allosteric labels from holo structure
# ─────────────────────────────────────────────────────────────

def get_ligand_resnums_from_holo(holo_pdb: str, allowed_names: list, dist: float = LABEL_DIST):
    """
    Find residue numbers of protein residues within `dist` Å of allosteric ligand
    atoms IN THE HOLO STRUCTURE.

    Returns set of (chain_id, resnum) tuples so we can map back to apo by residue number.
    This avoids needing to align apo and holo structures.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("holo", holo_pdb)

    lig_atoms = []
    protein_ca = {}

    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0].startswith("H_") and res.resname in allowed_names:
                    for atom in res:
                        if atom.element != "H":
                            lig_atoms.append(atom.get_vector().get_array())
                elif res.id[0] == " " and "CA" in res:
                    protein_ca[(chain.id, res.id[1])] = res["CA"].get_vector().get_array()
        break

    if not lig_atoms:
        return set()

    lig_arr  = np.array(lig_atoms)                              # (L, 3)
    close_residues = set()
    for (chain_id, resnum), ca_coord in protein_ca.items():
        dists = np.linalg.norm(lig_arr - ca_coord, axis=1)
        if dists.min() < dist:
            close_residues.add(resnum)

    return close_residues


def allosteric_labels_from_holo(apo_pdb: str, holo_pdb: str,
                                  manual_resnums: list = None,
                                  dist: float = LABEL_DIST):
    """
    Label allosteric residues in apo structure by:
      1. Finding which residue NUMBERS are near the allosteric ligand in holo
      2. Marking those same residue numbers in the apo structure
         (avoids coordinate alignment — works because residue numbering is conserved)
      3. Optionally adding manually curated residue numbers

    Returns: np.ndarray (N,) bool
    """
    apo_residues, _ = parse_pdb_ca(apo_pdb)

    # Get allosteric ligand name list for this holo PDB
    allowed = ALLO_LIGANDS.get(holo_pdb, [])
    close_resnums = get_ligand_resnums_from_holo(holo_pdb, allowed, dist)

    # Add manual residues
    if manual_resnums:
        close_resnums |= set(manual_resnums)

    if not close_resnums:
        print(f"  WARNING: No allosteric residues found for {holo_pdb}")
        return np.zeros(len(apo_residues), dtype=bool)

    apo_resnums = set(r["resnum"] for r in apo_residues)
    matched = close_resnums & apo_resnums

    labels = np.array([r["resnum"] in matched for r in apo_residues], dtype=bool)
    print(f"  Allosteric labels: {labels.sum()}/{len(apo_residues)} residues "
          f"(ligand-based: {len(close_resnums)}, matched in apo: {len(matched)})")
    return labels


# ─────────────────────────────────────────────────────────────
# Challenge dataset (3 proteins)
# ─────────────────────────────────────────────────────────────

CHALLENGE_PROTEINS = [
    {
        "name":    "KRAS_G12C",
        "apo":     "pdbs/4OBE.pdb",
        "holo":    "pdbs/6OIM.pdb",
        "active":  APO_ACTIVE_SITES["4OBE"],
    },
    {
        "name":    "BCR_ABL1",
        "apo":     "pdbs/1OPL.pdb",
        "holo":    "pdbs/5MO4.pdb",
        "active":  APO_ACTIVE_SITES["1OPL"],
    },
    {
        "name":    "CardiacMyosin",
        "apo":     "pdbs/5TBY.pdb",
        "holo":    "pdbs/6C1H.pdb",
        "active":  APO_ACTIVE_SITES["5TBY"],
    },
]


def load_challenge_sample(entry: dict, device: str = "cpu"):
    """
    Load one challenge protein as a training sample.

    Returns dict with:
        graph:            protein graph tensors for GVP-GNN
        allosteric_mask:  (N,) bool tensor
        active_mask:      (N,) bool tensor
        residues:         list of residue dicts
        coords:           (N,3) numpy array
        name:             protein name
    """
    residues, coords = parse_pdb_ca(entry["apo"])
    N = len(residues)

    print(f"\n[{entry['name']}] {N} residues from {entry['apo']}")

    # Graph for GVP-GNN
    graph = build_protein_graph(residues, coords, n_neighbors=30, device=device)

    # Allosteric labels from holo + manual curated residues
    allo_np = allosteric_labels_from_holo(
        entry["apo"], entry["holo"],
        manual_resnums=KNOWN_ALLO_RESNUMS.get(entry["name"], []),
    )
    allosteric_mask = torch.tensor(allo_np, dtype=torch.bool, device=device)

    # Active site mask
    active_set = set(entry["active"])
    active_np  = np.array([r["resnum"] in active_set for r in residues])
    active_mask = torch.tensor(active_np, dtype=torch.bool, device=device)

    return {
        "graph":            graph,
        "allosteric_mask":  allosteric_mask,
        "active_mask":      active_mask,
        "residues":         residues,
        "coords":           coords,
        "name":             entry["name"],
    }


def load_all_challenge_samples(device: str = "cpu"):
    """Load all 3 challenge proteins."""
    return [load_challenge_sample(e, device) for e in CHALLENGE_PROTEINS]


# ─────────────────────────────────────────────────────────────
# Quick label inspection
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for entry in CHALLENGE_PROTEINS:
        print(f"\n{'='*50}")
        print(f"Protein: {entry['name']}")
        labels = allosteric_labels_from_holo(
            entry["apo"], entry["holo"],
            manual_resnums=KNOWN_ALLO_RESNUMS.get(entry["name"], []),
        )
        residues, _ = parse_pdb_ca(entry["apo"])
        labeled = [r for r, l in zip(residues, labels) if l]
        names = [r['resname'] + str(r['resnum']) for r in labeled[:10]]
        suffix = '...' if len(labeled) > 10 else ''
        print(f"  Allosteric residues: {names}{suffix}")
