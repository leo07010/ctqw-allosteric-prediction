"""
Download ASD training proteins and extract allosteric site labels.

Usage:
  python asd_dataset.py --outdir data/asd_processed --n_workers 4
"""

import os
import json
import argparse
import urllib.request
import numpy as np
import torch
from Bio.PDB import PDBParser, is_aa
import warnings
warnings.filterwarnings("ignore")

from model import build_protein_graph

CUTOFF_ALLO = 8.0   # Angstrom: residues within this of ligand → allosteric

EXCLUDE_PDBS = {
    '4OBE', '6OIM',
    '1OPL', '5MO4',
    '5TBY', '6C1H',
    '1NKP',
}


def download_pdb(pdb_id, pdb_dir):
    path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"  DOWNLOAD FAILED {pdb_id}: {e}")
        return None


def parse_pdb_for_training(pdb_path, modname):
    """
    Extract protein Cα coords + allosteric labels.

    modname: residue name of the allosteric ligand (HETATM in PDB)
    Returns: (residues, coords, allo_labels) or None on failure
    """
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure('s', pdb_path)
    except Exception as e:
        print(f"  PARSE ERROR {pdb_path}: {e}")
        return None

    model = struct[0]

    # Collect protein Cα
    residues = []
    ca_coords = []
    for chain in model:
        for res in chain:
            if res.id[0] != ' ':
                continue  # skip HETATM + waters
            if not is_aa(res, standard=True):
                continue
            if 'CA' not in res:
                continue
            residues.append({'chain': chain.id, 'resname': res.resname, 'resnum': res.id[1]})
            ca_coords.append(res['CA'].get_vector().get_array())

    if len(residues) < 30:
        return None  # too short

    ca_coords = np.array(ca_coords, dtype=np.float32)

    # Find allosteric ligand atoms
    lig_coords = []
    for chain in model:
        for res in chain:
            hf = res.id[0]
            rname = res.resname.strip()
            # HETATM with matching name (case-insensitive)
            if hf != ' ' and hf != 'W' and rname.upper() == modname.upper():
                for atom in res.get_atoms():
                    lig_coords.append(atom.get_vector().get_array())

    if not lig_coords:
        return None  # ligand not found in structure

    lig_coords = np.array(lig_coords, dtype=np.float32)

    # Compute min distance from each Cα to any ligand atom
    diff = ca_coords[:, None, :] - lig_coords[None, :, :]  # (N, L, 3)
    dists = np.linalg.norm(diff, axis=-1).min(axis=1)      # (N,)
    allo_labels = (dists < CUTOFF_ALLO).astype(np.float32)

    n_allo = int(allo_labels.sum())
    if n_allo == 0:
        return None  # no positive labels

    return residues, ca_coords, allo_labels


def process_one(pdb_id, modname, pdb_dir, outdir):
    """Download PDB, extract labels, save tensor dict."""
    out_path = os.path.join(outdir, f"{pdb_id}.pt")
    if os.path.exists(out_path):
        return True  # already done

    pdb_path = download_pdb(pdb_id, pdb_dir)
    if pdb_path is None:
        return False

    result = parse_pdb_for_training(pdb_path, modname)
    if result is None:
        print(f"  SKIP {pdb_id}: no usable labels for modulator '{modname}'")
        return False

    residues, coords, allo_labels = result
    N = len(residues)

    if N > 1500:
        print(f"  SKIP {pdb_id}: {N} residues (too large)")
        return False

    # Build protein graph
    graph = build_protein_graph(residues, coords, n_neighbors=30)

    # Move graph tensors to CPU (will be moved to device during training)
    sample = {
        'name': pdb_id,
        'n_residues': N,
        'coords': torch.tensor(coords, dtype=torch.float32),
        'allo_labels': torch.tensor(allo_labels, dtype=torch.float32),
        'graph': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                  for k, v in graph.items()},
    }

    torch.save(sample, out_path)
    n_allo = int(allo_labels.sum())
    print(f"  OK {pdb_id}: {N} residues, {n_allo} allosteric ({100*n_allo/N:.1f}%)")
    return True


def main(args):
    os.makedirs(args.pdb_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.selection_json) as f:
        selection = json.load(f)

    print(f"Processing {len(selection)} proteins...")
    ok, fail = 0, 0
    for pdb_id, info in selection.items():
        modname = info['modname']
        success = process_one(pdb_id, modname, args.pdb_dir, args.outdir)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} OK, {fail} failed")
    print(f"Processed files in: {args.outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selection_json", default="data/asd_train_selection.json")
    p.add_argument("--pdb_dir",        default="data/asd_pdbs")
    p.add_argument("--outdir",         default="data/asd_processed")
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)
