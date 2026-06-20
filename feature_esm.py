"""
feature_esm.py
==============
Extract ESM-2 per-residue embeddings for proteins.
Model: esm2_t12_35M_UR50D  (480-dim, free, no API)

Usage:
  python feature_esm.py --pdb pdbs/4OBE.pdb --out esm_cache/KRAS_G12C.npy
  python feature_esm.py --all          # process all 3 challenge proteins
"""

import os, sys, argparse
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ESM_DIM   = 480          # esm2_t12_35M output dim
ESM_MODEL = "esm2_t12_35M_UR50D"
CACHE_DIR = "esm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
       "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
       "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# ─────────────────────────────────────────────────────────────────────────────

def parse_sequence_from_pdb(pdb_path, chain="A"):
    """Extract residue sequence from ATOM records (chain A, CA only)."""
    seq = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            c = line[21].strip()
            if c and c != chain: continue
            res3 = line[17:20].strip()
            resi  = int(line[22:26].strip())
            icode = line[26].strip()
            key   = (resi, icode)
            if key in seen: continue
            seen.add(key)
            aa = AA3.get(res3, "X")
            seq.append(aa)
    return "".join(seq)


def extract_esm(sequence: str, name: str, device="cpu") -> np.ndarray:
    """
    Returns per-residue ESM-2 embeddings: (N, 480) float32.
    """
    import esm as esm_lib
    model, alphabet = esm_lib.pretrained.esm2_t12_35M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    data = [(name, sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[12], return_contacts=False)

    # Layer 12 representations: shape (1, L+2, 480) — strip BOS/EOS
    reps = results["representations"][12][0, 1:-1, :]   # (N, 480)
    return reps.cpu().numpy().astype(np.float32)


def get_chain_ids(pdb_path):
    """Return ordered list of unique chain IDs found in ATOM records."""
    seen, chains = set(), []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            c = line[21].strip()
            if c and c not in seen:
                seen.add(c); chains.append(c)
    return chains


def get_esm_features(pdb_path: str, name: str,
                     chain: str = "A",
                     device: str = "cpu") -> np.ndarray:
    """
    Multi-chain aware ESM extraction.
    Extracts each chain independently, concatenates embeddings in chain order.
    Cache key = name_allchains.npy for multi-chain, name.npy for single-chain.
    Returns (N_total, 480) float32 matching GVP residue ordering.
    """
    chains = get_chain_ids(pdb_path)
    multi  = len(chains) > 1
    cache_key  = f"{name}_allchains" if multi else name
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")

    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        print(f"  [ESM] {name}: loaded from cache  shape={arr.shape}")
        return arr

    print(f"  [ESM] {name}: chains={chains}  extracting ({ESM_MODEL})...", flush=True)
    parts = []
    # Load ESM model once for all chains
    import esm as esm_lib
    esm_model, alphabet = esm_lib.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    for ch in chains:
        seq = parse_sequence_from_pdb(pdb_path, chain=ch)
        print(f"  [ESM]   chain {ch}: len={len(seq)}", flush=True)
        data = [(f"{name}_{ch}", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[12], return_contacts=False)
        reps = results["representations"][12][0, 1:-1, :].cpu().numpy().astype(np.float32)
        parts.append(reps)

    arr = np.concatenate(parts, axis=0)
    np.save(cache_path, arr)
    print(f"  [ESM] saved → {cache_path}  shape={arr.shape}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────

CHALLENGE_PROTEINS = [
    {"name": "KRAS_G12C",    "pdb": "pdbs/4OBE.pdb",  "chain": "A"},
    {"name": "BCR_ABL1",     "pdb": "pdbs/1OPL.pdb",  "chain": "A"},
    {"name": "CardiacMyosin","pdb": "pdbs/5TBY.pdb",  "chain": "A"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb",  default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--chain",default="A")
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.all:
        for p in CHALLENGE_PROTEINS:
            get_esm_features(p["pdb"], p["name"], p["chain"], args.device)
    elif args.pdb:
        name = args.name or os.path.splitext(os.path.basename(args.pdb))[0]
        get_esm_features(args.pdb, name, args.chain, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
