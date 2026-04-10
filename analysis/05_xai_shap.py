import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import shap
import warnings

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../data"
CSV_PATH = os.path.join(DATA_DIR, "final_dataset_weighted.csv")
PKL_PATH = os.path.join(DATA_DIR, "final_embeddings_all.pkl")
STUDENT_SAVE_DIR = "../models/student_models"
OUTPUT_DIR = "../results/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
DIM_L_SEQ, DIM_P_SEQ = 768, 1024
DIM_L_STR, DIM_P_STR = 512, 512
PROJ_DIM = 512

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729, 1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
SCENARIOS = ['Scenario5', 'Scenario6', 'Scenario7', 'Scenario8']

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

class StudentModelStruct(nn.Module):
    def __init__(self, l_seq_dim=768, p_seq_dim=1024, s_dim=512, proj_dim=512, num_heads=4):
        super().__init__()
        self.l_seq_proj = nn.Linear(l_seq_dim, proj_dim)
        self.p_seq_proj = nn.Linear(p_seq_dim, proj_dim)
        self.l_str_proj = nn.Linear(s_dim, proj_dim) 
        self.p_str_proj = nn.Linear(s_dim, proj_dim)
        self.l_self = AttentionModule(proj_dim, num_heads)
        self.p_self = AttentionModule(proj_dim, num_heads)
        self.cross = GatedCrossAttentionModule(proj_dim, num_heads) 
        self.mlp = nn.Sequential(nn.Linear(proj_dim * 4, proj_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(proj_dim, 1))
        
    def forward(self, l_seq, p_seq, l_str, p_str):
        l_ref = self.l_self(self.l_seq_proj(l_seq), self.l_seq_proj(l_seq), self.l_seq_proj(l_seq))
        p_ref = self.p_self(self.p_seq_proj(p_seq), self.p_seq_proj(p_seq), self.p_seq_proj(p_seq))
        final_latent = torch.cat([self.cross(l_ref, p_ref), self.cross(p_ref, l_ref), self.l_str_proj(l_str), self.p_str_proj(p_str)], dim=1)
        return self.mlp(final_latent), final_latent

class ShapWrapperSem(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, c_in): return self.model(c_in[:, :DIM_L_SEQ], c_in[:, DIM_L_SEQ:])[0]

class ShapWrapperStr(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, c_in): return self.model(c_in[:, :768], c_in[:, 768:1792], c_in[:, 1792:2304], c_in[:, 2304:])[0]

if __name__ == "__main__":
    print("Loading Data...")
    df_map = pd.read_csv(CSV_PATH)
    with open(PKL_PATH, 'rb') as f: vault = pickle.load(f)
    
    l_seq = np.array([vault['ligand_embeddings'].get(i, np.zeros(DIM_L_SEQ)) for i in df_map['main_compounds_id']], dtype=np.float32)
    p_seq = np.array([vault['protein_embeddings'].get(i, np.zeros(DIM_P_SEQ)) for i in df_map['main_receptors_id']], dtype=np.float32)
    l_str = np.array([vault.get('ligand_structure_embeddings', {}).get(i, np.zeros(DIM_L_STR)) for i in df_map['main_compounds_id']], dtype=np.float32)
    p_str = np.array([vault.get('protein_structure_embeddings', {}).get(i, np.zeros(DIM_P_STR)) for i in df_map['main_receptors_id']], dtype=np.float32)
    labels = df_map['responsive'].values.flatten()
    
    l_seq_t, p_seq_t = torch.tensor(l_seq), torch.tensor(p_seq)
    l_str_t, p_str_t = torch.tensor(l_str), torch.tensor(p_str)

    global_seed_stats = {
        'Ligand_Seq': [], 'Protein_Seq': [], 
        'Ligand_Str': [], 'Protein_Str': []
    }
    
    for scenario in SCENARIOS:
        print(f"\n=============================================")
        print(f"Processing XAI for {scenario}")
        print(f"=============================================")
        is_semantic_only = scenario in ['Scenario5', 'Scenario6']
        
        global_seed_stats = {k: [] for k in global_seed_stats}

        for seed in SEEDS:
            print(f"\n   [Seed {seed}] Calculating 10-Fold SHAP Importance...")
            
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            fold_pcts = {'Ligand_Seq': [], 'Protein_Seq': [], 'Ligand_Str': [], 'Protein_Str': []}
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
                ckpt_path = os.path.join(STUDENT_SAVE_DIR, f"seed_{seed}", f"{scenario}_seed{seed}_fold{fold+1}.pt")
                
                if not os.path.exists(ckpt_path): continue
                    
                if is_semantic_only:
                    model = StudentModelSemantic(proj_dim=PROJ_DIM).to(DEVICE)
                else:
                    model = StudentModelStruct(proj_dim=PROJ_DIM).to(DEVICE)
                    
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                model.eval()

                np.random.seed(seed + fold)
                bg_idx = np.random.choice(train_idx, min(100, len(train_idx)), replace=False)
                test_idx = np.random.choice(val_idx, min(50, len(val_idx)), replace=False)
                
                if is_semantic_only:
                    wrapper = ShapWrapperSem(model).to(DEVICE)
                    bg_concat = torch.cat([l_seq_t[bg_idx], p_seq_t[bg_idx]], dim=1).to(DEVICE)
                    test_concat = torch.cat([l_seq_t[test_idx], p_seq_t[test_idx]], dim=1).to(DEVICE)
                else:
                    wrapper = ShapWrapperStr(model).to(DEVICE)
                    bg_concat = torch.cat([l_seq_t[bg_idx], p_seq_t[bg_idx], l_str_t[bg_idx], p_str_t[bg_idx]], dim=1).to(DEVICE)
                    test_concat = torch.cat([l_seq_t[test_idx], p_seq_t[test_idx], l_str_t[test_idx], p_str_t[test_idx]], dim=1).to(DEVICE)
                    
                explainer = shap.GradientExplainer(wrapper, bg_concat)
                shap_vals = explainer.shap_values(test_concat)
                if isinstance(shap_vals, list): shap_vals = shap_vals[0]
                
                v_l_seq = np.sum(np.abs(shap_vals[:, :DIM_L_SEQ]))
                v_p_seq = np.sum(np.abs(shap_vals[:, DIM_L_SEQ:DIM_L_SEQ+DIM_P_SEQ]))
                
                if is_semantic_only:
                    total = v_l_seq + v_p_seq
                    fold_pcts['Ligand_Seq'].append((v_l_seq / total) * 100)
                    fold_pcts['Protein_Seq'].append((v_p_seq / total) * 100)
                else:
                    v_l_str = np.sum(np.abs(shap_vals[:, DIM_L_SEQ+DIM_P_SEQ:DIM_L_SEQ+DIM_P_SEQ+DIM_L_STR]))
                    v_p_str = np.sum(np.abs(shap_vals[:, DIM_L_SEQ+DIM_P_SEQ+DIM_L_STR:]))
                    total = v_l_seq + v_p_seq + v_l_str + v_p_str
                    
                    fold_pcts['Ligand_Seq'].append((v_l_seq / total) * 100)
                    fold_pcts['Protein_Seq'].append((v_p_seq / total) * 100)
                    fold_pcts['Ligand_Str'].append((v_l_str / total) * 100)
                    fold_pcts['Protein_Str'].append((v_p_str / total) * 100)

            seed_res = f"--- 10-Fold CV SHAP Importance ({scenario}, Seed {seed}) ---\n"
            
            l_seq_mean, l_seq_std = np.mean(fold_pcts['Ligand_Seq']), np.std(fold_pcts['Ligand_Seq'])
            p_seq_mean, p_seq_std = np.mean(fold_pcts['Protein_Seq']), np.std(fold_pcts['Protein_Seq'])
            
            global_seed_stats['Ligand_Seq'].append(l_seq_mean)
            global_seed_stats['Protein_Seq'].append(p_seq_mean)
            
            seed_res += f"Ligand Sequence: {l_seq_mean:.2f}% ± {l_seq_std:.2f}%\n"
            seed_res += f"Protein Sequence: {p_seq_mean:.2f}% ± {p_seq_std:.2f}%\n"
            
            if not is_semantic_only:
                l_str_mean, l_str_std = np.mean(fold_pcts['Ligand_Str']), np.std(fold_pcts['Ligand_Str'])
                p_str_mean, p_str_std = np.mean(fold_pcts['Protein_Str']), np.std(fold_pcts['Protein_Str'])
                
                global_seed_stats['Ligand_Str'].append(l_str_mean)
                global_seed_stats['Protein_Str'].append(p_str_mean)
                
                seed_res += f"Ligand Structure: {l_str_mean:.2f}% ± {l_str_std:.2f}%\n"
                seed_res += f"Protein Structure: {p_str_mean:.2f}% ± {p_str_std:.2f}%\n"
                
            txt_path = os.path.join(OUTPUT_DIR, f"SHAP_Seed_{seed}_{scenario}.txt")
            with open(txt_path, "w") as f: f.write(seed_res)
        
        final_res = f"=== GLOBAL SHAP IMPORTANCE ACROSS {len(SEEDS)} SEEDS ({scenario}) ===\n"
        
        g_l_seq_mean, g_l_seq_std = np.mean(global_seed_stats['Ligand_Seq']), np.std(global_seed_stats['Ligand_Seq'])
        g_p_seq_mean, g_p_seq_std = np.mean(global_seed_stats['Protein_Seq']), np.std(global_seed_stats['Protein_Seq'])
        
        final_res += f"Ligand Sequence: {g_l_seq_mean:.2f}% ± {g_l_seq_std:.2f}%\n"
        final_res += f"Protein Sequence: {g_p_seq_mean:.2f}% ± {g_p_seq_std:.2f}%\n"
        
        if not is_semantic_only:
            g_l_str_mean, g_l_str_std = np.mean(global_seed_stats['Ligand_Str']), np.std(global_seed_stats['Ligand_Str'])
            g_p_str_mean, g_p_str_std = np.mean(global_seed_stats['Protein_Str']), np.std(global_seed_stats['Protein_Str'])
            
            final_res += f"Ligand Structure: {g_l_str_mean:.2f}% ± {g_l_str_std:.2f}%\n"
            final_res += f"Protein Structure: {g_p_str_mean:.2f}% ± {g_p_str_std:.2f}%\n"
        
        global_path = os.path.join(OUTPUT_DIR, f"SHAP_GLOBAL_{scenario}.txt")
        with open(global_path, "w") as f: f.write(final_res)
        print(f"Global SHAP stats for {scenario} saved to {global_path}\n")