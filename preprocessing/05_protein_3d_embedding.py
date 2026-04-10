import pandas as pd
import numpy as np
import torch
from torch import nn
from se3_transformer_pytorch import SE3Transformer
import requests
from Bio.PDB import PDBParser
from io import StringIO
from tqdm import tqdm
import pickle
import os
import random
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_CSV = "../data/pairs_validated.csv"
OUTPUT_PKL = "../data/protein_3d_embeddings.pkl"
COORD_CACHE = "../data/esmfold_coords_cache.pkl"
RANDOM_SEED = 42

INTERNAL_DIM = 32
OUTPUT_DIM = 512
DEPTH = 2
HEADS = 2
DIM_HEAD = 16

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed to {seed}")

def get_residue_features(sequence: str):
    return torch.ones((len(sequence), 1), dtype=torch.float32)

def get_ca_coords(sequence, cache):
    if sequence in cache:
        return cache[sequence]

    api_url = 'https://api.esmatlas.com/foldSequence/v1/pdb/'
    for _ in range(3):
        try:
            response = requests.post(api_url, data=sequence, timeout=180)
            if response.status_code == 200: break
        except: time.sleep(1)
    else: return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pred", StringIO(response.text))
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue: coords.append(residue['CA'].get_coord())
        return np.array(coords)
    except: return None

class StructureEncoder(nn.Module):
    def __init__(self, input_feature_dim=1, dim=32, depth=2, heads=2, dim_head=16, output_dim=512):
        super().__init__()

        self.feature_projector = nn.Linear(input_feature_dim, dim)

        self.se3_transformer = SE3Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, num_degrees=2
        )
        self.final_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2),
            nn.ReLU(), nn.Linear(dim * 2, output_dim)
        )

    def forward(self, feats: torch.Tensor, coors: torch.Tensor, mask: torch.Tensor = None):
        device = coors.device
        if mask is None:
            batch_size, num_residues = coors.shape[0], coors.shape[1]
            mask = torch.ones(batch_size, num_residues, device=device).bool()

        projected_feats = self.feature_projector(feats)
        node_embeddings = self.se3_transformer(projected_feats, coors, mask=mask)

        masked_embeddings = node_embeddings.masked_fill(~mask[..., None], 0.)
        graph_sum = masked_embeddings.sum(dim=1)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        graph_embedding = graph_sum / mask_sum

        return self.final_head(graph_embedding)

def main():
    set_seed(RANDOM_SEED)

    if os.path.exists(OUTPUT_PKL):
        print(f"Loading existing output file from {OUTPUT_PKL}...")
        with open(OUTPUT_PKL, 'rb') as f:
            id_to_emb = pickle.load(f)
    else:
        id_to_emb = {}

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    unique_receptors = df.groupby('main_receptors_id')['mutated_sequence'].first().to_dict()

    remaining_receptors = {k: v for k, v in unique_receptors.items() if k not in id_to_emb}
    print(f"Receptors to process: {len(remaining_receptors)} (Total: {len(unique_receptors)})")

    if os.path.exists(COORD_CACHE):
        with open(COORD_CACHE, 'rb') as f: coord_cache = pickle.load(f)
    else: coord_cache = {}

    print("Fetching/Loading 3D Coordinates...")
    id_to_coords = {}
    cache_updated = False

    for rid, seq in tqdm(remaining_receptors.items()):
        coords = get_ca_coords(seq, coord_cache)
        if coords is not None and len(coords) == len(seq):
            id_to_coords[rid] = coords
            if seq not in coord_cache:
                coord_cache[seq] = coords
                cache_updated = True
        else:
            id_to_coords[rid] = None

    if cache_updated:
        with open(COORD_CACHE, 'wb') as f: pickle.dump(coord_cache, f)

    print(f"Initializing StructureEncoder (Input: 1, Internal: {INTERNAL_DIM}, Output: {OUTPUT_DIM})...")
    model = StructureEncoder(
        input_feature_dim=1,
        dim=INTERNAL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)
    model.eval()

    print("Generating SE(3) Embeddings (Structure Only)...")

    error_count = 0

    for rid, seq in tqdm(remaining_receptors.items()):
        coords = id_to_coords.get(rid)

        if coords is None:
            id_to_emb[rid] = np.zeros(OUTPUT_DIM, dtype=np.float32)
            continue

        feats = get_residue_features(seq)

        feats_tensor = feats.unsqueeze(0).to(DEVICE)
        coords_tensor = torch.from_numpy(coords).float().unsqueeze(0).to(DEVICE)

        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    emb = model(feats_tensor, coords_tensor)

                id_to_emb[rid] = emb.squeeze(0).float().cpu().numpy()

        except RuntimeError as e:
            print(f"\nError processing {rid}: {e}")
            id_to_emb[rid] = np.zeros(OUTPUT_DIM, dtype=np.float32)
            error_count += 1

        del feats_tensor, coords_tensor
        torch.cuda.empty_cache()

    print(f"Saving to {OUTPUT_PKL} (Errors: {error_count})...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(id_to_emb, f)

    print("Done!")

if __name__ == "__main__":
    main()
