import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
N_SPLITS = 10
BATCH_SIZE = 128
SCENARIO = "Scenario5"

DATA_DIR = "../data"
PKL_PATH = os.path.join(DATA_DIR, "final_embeddings_all.pkl")
CSV_PATH = os.path.join(DATA_DIR, "final_dataset_weighted.csv")

MODEL_DIR = "../models/student_models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_CANDIDATES = np.linspace(0.01, 0.99, 100)

OUTPUT_DIR = "../results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AttentionModule(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, q, k, v):
        q_in, k_in, v_in = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        out, _ = self.attn(q_in, k_in, v_in)
        return self.ln(q + out.squeeze(1))

class GatedCrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate_fc = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, query, key_value):
        query_in, key_value_in = query.unsqueeze(1), key_value.unsqueeze(1)
        attn_output, _ = self.attention(query=query_in, key=key_value_in, value=key_value_in)
        attn_output = attn_output.squeeze(1)
        gate = self.sigmoid(self.gate_fc(query))
        return self.layer_norm(query + (gate * attn_output))

class StudentModelSemantic(nn.Module):
    def __init__(self, l_dim=768, p_dim=1024, proj_dim=512, num_heads=4):
        super().__init__()
        self.ligand_proj = nn.Linear(l_dim, proj_dim)
        self.protein_proj = nn.Linear(p_dim, proj_dim)
        dim = proj_dim
        self.l_self = AttentionModule(dim, num_heads)
        self.p_self = AttentionModule(dim, num_heads)
        self.l_cross = GatedCrossAttentionModule(dim, num_heads)
        self.p_cross = GatedCrossAttentionModule(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim, 1)
        )
    def forward(self, l_vec, p_vec):
        l_proj = self.ligand_proj(l_vec)
        p_proj = self.protein_proj(p_vec)
        l_ref = self.l_self(l_proj, l_proj, l_proj)
        p_ref = self.p_self(p_proj, p_proj, p_proj)
        l_ctx = self.l_cross(query=l_ref, key_value=p_ref)
        p_ctx = self.p_cross(query=p_ref, key_value=l_ref)
        final = torch.cat([l_ctx, p_ctx], dim=1)
        return self.mlp(final), None, None

if __name__ == "__main__":
    print("=" * 70)
    print("  Threshold Reconstruction from .pt Checkpoints")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    print("\n  Loading data...")
    df_map = pd.read_csv(CSV_PATH)
    with open(PKL_PATH, 'rb') as f:
        vault = pickle.load(f)

    l_seq = np.array([vault['ligand_embeddings'].get(i, np.zeros(768))
                       for i in df_map['main_compounds_id']], dtype=np.float32)
    p_seq = np.array([vault['protein_embeddings'].get(i, np.zeros(1024))
                       for i in df_map['main_receptors_id']], dtype=np.float32)
    labels = df_map['responsive'].values.flatten()

    l_seq_t = torch.tensor(l_seq)
    p_seq_t = torch.tensor(p_seq)
    labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    X_dummy = np.concatenate([l_seq, p_seq], axis=1)

    results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_dummy, labels)):
            fold_num = fold + 1
            ckpt_path = os.path.join(
                MODEL_DIR, f"seed_{seed}", f"{SCENARIO}_seed{seed}_fold{fold_num}.pt"
            )

            if not os.path.exists(ckpt_path):
                print(f"    Fold {fold_num}: checkpoint not found, skipping")
                continue

            model = StudentModelSemantic(proj_dim=512).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.eval()

            train_ds = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], labels_t[train_idx])
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

            train_preds, train_targets = [], []
            with torch.no_grad():
                for l_batch, p_batch, y_batch in train_loader:
                    l_batch = l_batch.to(DEVICE)
                    p_batch = p_batch.to(DEVICE)
                    out, _, _ = model(l_batch, p_batch)
                    train_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
                    train_targets.extend(y_batch.cpu().numpy().flatten())

            train_preds_np = np.array(train_preds)
            train_targets_np = np.array(train_targets)

            mccs = [matthews_corrcoef(train_targets_np, train_preds_np > thr)
                    for thr in THRESHOLD_CANDIDATES]
            optimal_thresh = THRESHOLD_CANDIDATES[np.argmax(mccs)]

            results.append({
                'Seed': seed,
                'Fold': fold_num,
                'Threshold': round(float(optimal_thresh), 6)
            })

            print(f"    Fold {fold_num}: threshold = {optimal_thresh:.4f}")

    df_out = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, f"thresholds_{SCENARIO}.csv")
    df_out.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print(f"  Saved {len(df_out)} thresholds to {output_path}")
    print(f"  Threshold range: [{df_out['Threshold'].min():.4f}, {df_out['Threshold'].max():.4f}]")
    print(f"  Threshold mean:  {df_out['Threshold'].mean():.4f} +/- {df_out['Threshold'].std():.4f}")
    print(f"{'='*70}")