import pandas as pd
import numpy as np
import torch
from torch import nn
from egnn_pytorch import EGNN
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_CSV = "../data/pairs_validated.csv"
OUTPUT_PKL = "../data/ligand_3d_embeddings.pkl"
RANDOM_SEED = 42

INTERNAL_DIM = 64
OUTPUT_DIM = 512
DEPTH = 1
NEIGHBORS = 6

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed to {seed}")

def get_molecule_node_features(mol):
    num_atoms = mol.GetNumAtoms()
    return np.ones((num_atoms, 1), dtype=np.float32)

def get_coords_and_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, randomSeed=RANDOM_SEED)
        if res == -1: return None, None
        features = get_molecule_node_features(mol)
        conf = mol.GetConformer()
        coords = conf.GetPositions()

        return features, coords
    except Exception as e:
        return None, None

class LigandEncoder(nn.Module):
    def __init__(self, input_feature_dim=1, dim=64, depth=1, num_nearest_neighbors=6, output_dim=512):
        super(LigandEncoder, self).__init__()

        self.feature_projector = nn.Linear(input_feature_dim, dim)

        self.egnn_layers = nn.ModuleList([
            EGNN(dim=dim, num_nearest_neighbors=num_nearest_neighbors) for _ in range(depth)
        ])

        self.final_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, output_dim)
        )

    def forward(self, feats: torch.Tensor, coors: torch.Tensor, mask: torch.Tensor = None):
        batch_size, num_atoms = coors.shape[0], coors.shape[1]

        if mask is None:
            mask = torch.ones(batch_size, num_atoms, device=coors.device).bool()

        projected_feats = self.feature_projector(feats)

        current_feats = projected_feats
        for egnn_layer in self.egnn_layers:
            current_feats, _ = egnn_layer(current_feats, coors, mask=mask)

        node_embeddings = current_feats

        masked_embeddings = node_embeddings.masked_fill(~mask[..., None], 0.)
        graph_sum = masked_embeddings.sum(dim=1)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        graph_embedding = graph_sum / mask_sum

        return self.final_head(graph_embedding)

def main():
    set_seed(RANDOM_SEED)

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    unique_compounds = df.groupby('main_compounds_id')['smiles'].first().to_dict()
    print(f"Unique Ligands to process: {len(unique_compounds)}")

    print(f"Initializing LigandEncoder (Output Dim: {OUTPUT_DIM}, Input Dim: 1)...")
    model = LigandEncoder(
        input_feature_dim=1,
        dim=INTERNAL_DIM,
        depth=DEPTH,
        num_nearest_neighbors=NEIGHBORS,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)
    model.eval()

    id_to_emb = {}

    print("Generating 3D Embeddings (Structure Only)...")
    for cid, smiles in tqdm(unique_compounds.items()):
        feats, coords = get_coords_and_features(smiles)

        if feats is None or coords is None:
            id_to_emb[cid] = np.zeros(OUTPUT_DIM, dtype=np.float32)
            continue

        feats_tensor = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)
        coords_tensor = torch.from_numpy(coords).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(feats_tensor, coords_tensor)
            id_to_emb[cid] = emb.squeeze(0).cpu().numpy()

    print(f"Saving to {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(id_to_emb, f)

    print("Done!")

if __name__ == "__main__":
    main()
