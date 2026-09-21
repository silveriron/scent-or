import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import re
import os
import warnings
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../data"
RAW_DATA_CSV = os.path.join(DATA_DIR, "final_dataset_weighted.csv")

MODEL_DIR = "../models/student_models"

SCENARIO = "Scenario5"

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729, 1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
PAIRS = {
                                        
                                 
    "TN_Plus_Menthol": 18633,
    "TP_Minus_Menthol": 19073
}

OUTPUT_DIR = "./muta"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_SPLITS = 10

def get_oof_fold(target_index, labels, n_splits=N_SPLITS, seed=42):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(labels), 1))
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_dummy, labels)):
        if target_index in val_idx:
            return fold + 1
    raise ValueError(f"Index {target_index} not found in any fold (seed={seed})")

class AttentionModule(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, q, k, v):
        out, _ = self.attn(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
        return self.ln(q + out.squeeze(1))

class GatedCrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate_fc = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, query, key_value):
        attn_output, _ = self.attention(query.unsqueeze(1), key_value.unsqueeze(1), key_value.unsqueeze(1))
        gate = self.sigmoid(self.gate_fc(query))
        return self.layer_norm(query + (gate * attn_output.squeeze(1)))

class StudentModelSemantic(nn.Module):
    def __init__(self, l_dim=768, p_dim=1024, proj_dim=512, num_heads=4):
        super().__init__()
        self.ligand_proj = nn.Linear(l_dim, proj_dim)
        self.protein_proj = nn.Linear(p_dim, proj_dim)
        self.l_self = AttentionModule(proj_dim, num_heads)
        self.p_self = AttentionModule(proj_dim, num_heads)
        self.l_cross = GatedCrossAttentionModule(proj_dim, num_heads)
        self.p_cross = GatedCrossAttentionModule(proj_dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(proj_dim, 1))
        
    def forward(self, l_vec, p_vec):
        l_ref = self.l_self(self.ligand_proj(l_vec), self.ligand_proj(l_vec), self.ligand_proj(l_vec))
        p_ref = self.p_self(self.protein_proj(p_vec), self.protein_proj(p_vec), self.protein_proj(p_vec))
        final_latent = torch.cat([self.l_cross(l_ref, p_ref), self.p_cross(p_ref, l_ref)], dim=1)
        return self.mlp(final_latent), final_latent

def run_mutagenesis_for_all_seeds():
    print("Loading HuggingFace Models for Mutagenesis...")
    mol_tok = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
    mol_model = AutoModel.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(DEVICE)
    prot_tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    prot_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(DEVICE)
    
    mol_model.eval()
    prot_model.eval()

    df_raw = pd.read_csv(RAW_DATA_CSV)
    labels = df_raw['responsive'].values.flatten()

    for pair_name, target_index in PAIRS.items():
        print(f"\n=============================================")
        print(f"Processing Mutagenesis for: {pair_name} (Index: {target_index})")
        
        raw_row = df_raw.iloc[target_index]
        mid = int(raw_row['main_compounds_id'])
        pid = int(raw_row['main_receptors_id'])
        smiles = raw_row['smiles']
        sequence = raw_row['mutated_sequence']
        
        mol_inputs = mol_tok(smiles, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            mol_out = mol_model(input_ids=mol_inputs['input_ids'], attention_mask=mol_inputs['attention_mask'])
            l_emb = mol_out.last_hidden_state.mean(dim=1)

        for seed in SEEDS:
            print(f"\n   [Seed {seed}] Calculating Alanine Mutagenesis...")
            
            fold_num = get_oof_fold(target_index, labels, n_splits=N_SPLITS, seed=seed)
            print(f"      OOF Fold: {fold_num}")
            
            student_model = StudentModelSemantic(proj_dim=512).to(DEVICE)
            ckpt_path = os.path.join(MODEL_DIR, f"seed_{seed}", f"{SCENARIO}_seed{seed}_fold{fold_num}.pt")
            
            if not os.path.exists(ckpt_path):
                print(f"   [Warning] Model checkpoint not found: {ckpt_path}")
                continue

            student_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            student_model.eval()

            proc_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            prot_inputs = prot_tok(proc_seq, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                prot_out = prot_model(input_ids=prot_inputs['input_ids'], attention_mask=prot_inputs['attention_mask'])
                p_emb = prot_out.last_hidden_state.mean(dim=1)
                
                base_logits, _ = student_model(l_emb, p_emb)
                base_prob = torch.sigmoid(base_logits).item()

            muta_scores = []
            for i in range(len(sequence)):
                mut_seq = list(sequence)
                mut_seq[i] = 'A' if mut_seq[i] != 'A' else 'G' 
                mut_seq_str = "".join(mut_seq)
                
                proc_mut_seq = " ".join(list(re.sub(r"[UZOB]", "X", mut_seq_str)))
                mut_prot_inputs = prot_tok(proc_mut_seq, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    mut_prot_out = prot_model(input_ids=mut_prot_inputs['input_ids'], attention_mask=mut_prot_inputs['attention_mask'])
                    mut_p_emb = mut_prot_out.last_hidden_state.mean(dim=1)
                    mut_logits, _ = student_model(l_emb, mut_p_emb)
                    mut_prob = torch.sigmoid(mut_logits).item()
                
                score = base_prob - mut_prob
                muta_scores.append(score)

            muta_score_vec = np.array(muta_scores)
            np.save(os.path.join(OUTPUT_DIR, f"Muta_score_seed_{seed}_idx_{target_index}.npy"), muta_score_vec)

            abs_muta_scores = np.abs(muta_score_vec)
            top10_indices = np.argsort(abs_muta_scores)[-10:][::-1]
            print(f"      Top 10 Mutagenesis Residues:")
            for rank, idx in enumerate(top10_indices, 1):
                if idx < len(sequence):
                    res = sequence[idx]
                    print(f"         {rank}. {res}{idx+1}: {abs_muta_scores[idx]:.4f}")

if __name__ == "__main__":
    run_mutagenesis_for_all_seeds()
