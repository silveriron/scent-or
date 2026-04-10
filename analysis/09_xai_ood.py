import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import warnings

warnings.filterwarnings("ignore")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
N_FOLDS = 10
SCENARIO = "Scenario5"

MODEL_DIR = "../models/student_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD_CSV = "../results/thresholds_Scenario5.csv"

OR8D1_SEQUENCE = "MTMENYSMAAQFVLDGLTQQAELQLPLFLLFLGIYVVTVVGNLGMILLIAVSPLLHTPMYYFLSSLSFVDFCYSSVITPKMLVNFLGKKNTILYSECMVQLFFFVVFVVAEGYLLTAMAYDRYVAICSPLLYNAIMSSWVCSLLVLAAFFLGFLSALTHTSAMMKLSFCKSHIINHYFCDVLPLLNLSCSNTHLNELLLFIIAGFNTLVPTLAVAVSYAFILYSILHIRSSEGRSKAFGTCSSHLMAVVIFFGSITFMYFKPPSSNSLDQEKVSSVFYTTVIPMLNPLIYSLRNKDVKKALRKVLVGK"

S_SOTOLON_SMILES = "CC1=C(O)C(=O)O[C@H]1C"
R_SOTOLON_SMILES = "CC1=C(O)C(=O)O[C@@H]1C"

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

def load_thresholds(csv_path=THRESHOLD_CSV):
    if not os.path.exists(csv_path):
        print(f"  [Error] Threshold CSV not found: {csv_path}")
        print(f"          Run xai_threshold.py first to generate it.")
        return {}
    df = pd.read_csv(csv_path)
    return {(int(row['Seed']), int(row['Fold'])): float(row['Threshold'])
            for _, row in df.iterrows()}

def generate_embeddings(smiles_list, sequence):
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

    l_embs = {}
    for name, smi in smiles_list.items():
        mol_inputs = mol_tok(smi, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            mol_out = mol_model(
                input_ids=mol_inputs['input_ids'],
                attention_mask=mol_inputs['attention_mask']
            )
            l_embs[name] = mol_out.last_hidden_state.mean(dim=1)

    print("  Embeddings generated successfully.")
    return l_embs, p_emb

def run_inference(l_embs, p_emb):
    threshold_map = load_thresholds()
    if not threshold_map:
        print("  [Warning] No thresholds loaded. Run xai_threshold.py first.")

    results = []

    for seed in SEEDS:
        seed_dir = os.path.join(MODEL_DIR, f"seed_{seed}")

        for fold in range(1, N_FOLDS + 1):
            ckpt_path = os.path.join(seed_dir, f"{SCENARIO}_seed{seed}_fold{fold}.pt")

            if not os.path.exists(ckpt_path):
                print(f"  [Warning] Checkpoint not found: {ckpt_path}")
                continue

            model = StudentModelSemantic(proj_dim=512).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.eval()

            threshold = threshold_map.get((seed, fold), None)

            for lig_name, l_emb in l_embs.items():
                with torch.no_grad():
                    logit, _, _ = model(l_emb, p_emb)
                    prob = torch.sigmoid(logit).item()

                results.append({
                    'Seed': seed,
                    'Fold': fold,
                    'Ligand': lig_name,
                    'Probability': prob,
                    'Threshold': round(threshold, 6) if threshold is not None else None,
                })

        print(f"  Seed {seed} done ({N_FOLDS} folds x {len(l_embs)} ligands)")

    return pd.DataFrame(results)

def analyze_results(df):
    for lig_name in df['Ligand'].unique():
        df_lig = df[df['Ligand'] == lig_name]

        print(f"\n{'='*70}")
        print(f"  {lig_name}")
        print(f"{'='*70}")

        print(f"\n  Per-Seed Summary (mean across {N_FOLDS} folds):")
        print(f"  {'Seed':>6}  {'Mean Prob':>10}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

        seed_means = []
        for seed in SEEDS:
            df_seed = df_lig[df_lig['Seed'] == seed]
            probs = df_seed['Probability'].values
            mean_p = np.mean(probs)
            seed_means.append(mean_p)
            print(f"  {seed:>6}  {mean_p:>10.4f}  {np.std(probs):>8.4f}  "
                  f"{np.min(probs):>8.4f}  {np.max(probs):>8.4f}")

        print(f"\n  Per-Fold Summary (mean across {len(SEEDS)} seeds):")
        print(f"  {'Fold':>6}  {'Mean Prob':>10}  {'Std':>8}  {'Threshold':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*10}")

        for fold in range(1, N_FOLDS + 1):
            df_fold = df_lig[df_lig['Fold'] == fold]
            probs = df_fold['Probability'].values
            thresholds = df_fold['Threshold'].dropna().values
            thr_str = f"{np.mean(thresholds):.4f}" if len(thresholds) > 0 else "N/A"
            print(f"  {fold:>6}  {np.mean(probs):>10.4f}  {np.std(probs):>8.4f}  {thr_str:>10}")

        all_probs = df_lig['Probability'].values
        seed_level_mean = np.mean(seed_means)
        seed_level_std = np.std(seed_means)

        print(f"\n  Ensemble Summary ({len(all_probs)} predictions):")
        print(f"    Seed-level mean:  {seed_level_mean:.4f} ± {seed_level_std:.4f}")
        print(f"    Range: [{np.min(all_probs):.4f}, {np.max(all_probs):.4f}]")

        all_thr = df_lig['Threshold'].dropna().values
        if len(all_thr) > 0:
            print(f"    Threshold range:  [{np.min(all_thr):.4f}, {np.max(all_thr):.4f}]")
            print(f"    Min threshold:    {np.min(all_thr):.4f}")
            print(f"    All probs below min threshold: {np.all(all_probs < np.min(all_thr))}")

    ligands = df['Ligand'].unique()
    if len(ligands) == 2:
        print(f"\n{'='*70}")
        print(f"  Enantiomer Comparison")
        print(f"{'='*70}")

        probs_a = df[df['Ligand'] == ligands[0]]['Probability'].values
        probs_b = df[df['Ligand'] == ligands[1]]['Probability'].values

        print(f"  {ligands[0]}: {np.mean(probs_a):.4f} ± {np.std(probs_a):.4f}")
        print(f"  {ligands[1]}: {np.mean(probs_b):.4f} ± {np.std(probs_b):.4f}")
        print(f"  Difference: {np.mean(probs_a) - np.mean(probs_b):.4f}")

        print(f"\n  Per-Seed Paired Comparison:")
        print(f"  {'Seed':>6}  {ligands[0]:>12}  {ligands[1]:>12}  {'Diff':>8}  {'Correct?':>8}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")

        for seed in SEEDS:
            mean_a = df[(df['Ligand'] == ligands[0]) & (df['Seed'] == seed)]['Probability'].mean()
            mean_b = df[(df['Ligand'] == ligands[1]) & (df['Seed'] == seed)]['Probability'].mean()
            diff = mean_a - mean_b
            correct = "Yes" if diff > 0 else "No"
            print(f"  {seed:>6}  {mean_a:>12.4f}  {mean_b:>12.4f}  {diff:>+8.4f}  {correct:>8}")

if __name__ == "__main__":
    print("=" * 70)
    print("  OOD Test: OR8D1 - (R/S)-Sotolon Enantiomers")
    print("=" * 70)

    if not OR8D1_SEQUENCE:
        print("\n  [Error] Please fill in OR8D1_SEQUENCE.")
        exit(1)
    if not S_SOTOLON_SMILES or not R_SOTOLON_SMILES:
        print("\n  [Error] Please fill in S_SOTOLON_SMILES and R_SOTOLON_SMILES.")
        exit(1)

    print(f"\n  Device: {DEVICE}")
    print(f"  OR8D1 sequence length: {len(OR8D1_SEQUENCE)}")
    print(f"  (S)-Sotolon SMILES: {S_SOTOLON_SMILES}")
    print(f"  (R)-Sotolon SMILES: {R_SOTOLON_SMILES}")
    print(f"  Models: {len(SEEDS)} seeds × {N_FOLDS} folds = {len(SEEDS) * N_FOLDS} checkpoints")

    smiles_dict = {
        "(S)-Sotolon": S_SOTOLON_SMILES,
        "(R)-Sotolon": R_SOTOLON_SMILES
    }
    l_embs, p_emb = generate_embeddings(smiles_dict, OR8D1_SEQUENCE)

    print(f"\n  Running inference...")
    df_results = run_inference(l_embs, p_emb)

    analyze_results(df_results)

    output_path = "../results/ood/ood_sotolon_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n  Raw results saved to: {output_path}")

    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}")