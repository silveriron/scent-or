import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import warnings

warnings.filterwarnings("ignore")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
N_FOLDS = 10
SCENARIO = "Scenario5"
DATA_DIR = "../data"
MODEL_DIR = "../models/student_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_CSV = "../results/thresholds_Scenario5.csv"

COMPOUND_ID = 56
RECEPTOR_ID = 168
GROUND_TRUTH = 1

RAW_DATA_CSV = os.path.join(DATA_DIR, "final_dataset_weighted.csv")

OR51E2_SEQUENCE = "MSSCNFTHATFVLIGIPGLEKAHFWVGFPLLSMYVVAMFGNCIVVFIVRTERSLHAPMYLFLCMLAAIDLALSTSTMPKILALFWFDSREISFEACLTQMFFIHALSAIESTILLAMAFDRYVAICHPLRHAAVLNNTVTAQIGIVAVVRGSLFFFPLPLLIKRLAFCHSNVLSHSYCVHQDVMKLAYADTLPNVVYGLTAILLVMGVDVMFISLSYFLIIRTVLQLPSKSERAKAFGTCVSHIGVVLAFYVPLIGLSVVHRFGNSLHPIVRVVMGDIYLLLPPVINPIIYGAKTKQIRTRVLAMFKISCDKDLQAVGGK"
PROPIONIC_ACID_SMILES = "CCC(=O)O"

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

def generate_embeddings(smiles, sequence):
    print("  Loading MoLFormer and ProtT5...")
    mol_tok = AutoTokenizer.from_pretrained(
        "ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    mol_model = AutoModel.from_pretrained(
        "ibm-research/MoLFormer-XL-both-10pct",
        deterministic_eval=True, trust_remote_code=True
    ).to(DEVICE)
    prot_tok = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False
    )
    prot_model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc"
    ).to(DEVICE)

    mol_model.eval()
    prot_model.eval()

    proc_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    prot_inputs = prot_tok(proc_seq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        prot_out = prot_model(
            input_ids=prot_inputs['input_ids'],
            attention_mask=prot_inputs['attention_mask']
        )
        p_emb = prot_out.last_hidden_state.mean(dim=1)

    mol_inputs = mol_tok(smiles, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        mol_out = mol_model(
            input_ids=mol_inputs['input_ids'],
            attention_mask=mol_inputs['attention_mask']
        )
        l_emb = mol_out.last_hidden_state.mean(dim=1)

    print("  Embeddings generated successfully.")
    return l_emb, p_emb

def load_thresholds(csv_path=THRESHOLD_CSV):
    """
    Load pre-computed thresholds from xai_threshold.py output.
    Returns dict: {(seed, fold): threshold}
    """
    if not os.path.exists(csv_path):
        print(f"  [Error] Threshold CSV not found: {csv_path}")
        print(f"          Run xai_threshold.py first to generate it.")
        return {}
    df = pd.read_csv(csv_path)
    return {(int(row['Seed']), int(row['Fold'])): float(row['Threshold'])
            for _, row in df.iterrows()}

def get_oof_fold(target_index, labels, n_splits=N_FOLDS, seed=42):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(labels), 1))
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_dummy, labels)):
        if target_index in val_idx:
            return fold + 1
    raise ValueError(f"Index {target_index} not found in any fold (seed={seed})")

if __name__ == "__main__":
    print("=" * 70)
    print("  OR51E2–Propionic Acid: OOF Analysis")
    print("=" * 70)

    if not OR51E2_SEQUENCE:
        print("\n  [Error] Please fill in OR51E2_SEQUENCE.")
        exit(1)

    print(f"\n  Device: {DEVICE}")
    print(f"  SMILES: {PROPIONIC_ACID_SMILES}")

    df_raw = pd.read_csv(RAW_DATA_CSV)
    labels = df_raw['responsive'].values.flatten()

    target_mask = (df_raw['main_compounds_id'] == COMPOUND_ID) & \
                  (df_raw['main_receptors_id'] == RECEPTOR_ID)
    if not target_mask.any():
        print(f"\n  [Error] Pair (compound={COMPOUND_ID}, receptor={RECEPTOR_ID}) not found in dataset.")
        exit(1)
    target_index = df_raw.index[target_mask][0]
    print(f"  Target index in dataset: {target_index}")

    threshold_map = load_thresholds()
    if not threshold_map:
        print("\n  [Warning] No thresholds loaded. Run xai_threshold.py first.")

    l_emb, p_emb = generate_embeddings(PROPIONIC_ACID_SMILES, OR51E2_SEQUENCE)

    results = []

    print(f"\n  {'Seed':>5} | {'Fold':>4} | {'Prob':>7} | {'Threshold':>9} | {'Class':>5}")
    print(f"  {'-'*5} | {'-'*4} | {'-'*7} | {'-'*9} | {'-'*5}")

    for seed in SEEDS:
        seed_dir = os.path.join(MODEL_DIR, f"seed_{seed}")

        fold_num = get_oof_fold(target_index, labels, n_splits=N_FOLDS, seed=seed)

        ckpt_path = os.path.join(seed_dir, f"{SCENARIO}_seed{seed}_fold{fold_num}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [Warning] Checkpoint not found: {ckpt_path}")
            continue

        model = StudentModelSemantic(proj_dim=512).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            logit, _, _ = model(l_emb, p_emb)
            prob = torch.sigmoid(logit).item()

        threshold = threshold_map.get((seed, fold_num), None)

        if threshold is not None:
            prediction = 1 if prob >= threshold else 0
        else:
            prediction = None

        if prediction is not None:
            if GROUND_TRUTH == 1 and prediction == 1:
                category = 'TP'
            elif GROUND_TRUTH == 0 and prediction == 0:
                category = 'TN'
            elif GROUND_TRUTH == 0 and prediction == 1:
                category = 'FP'
            else:
                category = 'FN'
        else:
            category = 'N/A'

        thr_str = f"{threshold:.4f}" if threshold is not None else "N/A"
        print(f"  {seed:>5} | {fold_num:>4} | {prob:>7.4f} | {thr_str:>9} | {category:>5}")

        results.append({
            'Seed': seed,
            'Fold': fold_num,
            'Probability': prob,
            'Threshold': round(threshold, 6) if threshold is not None else None,
            'Classification': category,
        })

    df_results = pd.DataFrame(results)
    probs = df_results['Probability'].values
    n_tp = (df_results['Classification'] == 'TP').sum()
    n_total = len(df_results)

    print(f"\n{'='*70}")
    print(f"  OR51E2 – Propionic Acid")
    print(f"  Ground Truth: {GROUND_TRUTH} (Active)")
    print(f"{'='*70}")
    print(f"  Mean probability:    {np.mean(probs):.4f} +/- {np.std(probs, ddof=1):.4f}")
    print(f"  Median:              {np.median(probs):.4f}")
    print(f"  Range:               [{np.min(probs):.4f}, {np.max(probs):.4f}]")
    print(f"  TP:                  {n_tp}/{n_total}")

    if n_tp > 0:
        print(f"\n  TP seeds:")
        for _, r in df_results[df_results['Classification'] == 'TP'].iterrows():
            thr_val = f"{r['Threshold']:.4f}" if r['Threshold'] is not None else "N/A"
            print(f"    Seed {r['Seed']:>5} | Fold {r['Fold']:>2} | "
                  f"P = {r['Probability']:.4f} | Thr = {thr_val}")

    output_path = "../results/oof/oof_or51e2_propionic_acid.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")