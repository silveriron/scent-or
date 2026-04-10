import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
SCENARIO = "Scenario5"

DATA_DIR = "../data"
RESULTS_DIR = "../results"
OOF_DIR = RESULTS_DIR
RAW_DATA_CSV = os.path.join(DATA_DIR, "final_dataset_weighted.csv")

MIN_SEED_COUNT = 20
MIN_SIMILARITY = 0.80
MIN_TP_RATIO = 0.80

OUTPUT_DIR = os.path.join(RESULTS_DIR, "screening")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  STEP 1: Loading OOF predictions across all seeds")
print("=" * 70)

all_preds = []
for seed in SEEDS:
    file_path = os.path.join(OOF_DIR, f"OOF_CM_{SCENARIO}_seed{seed}.csv")
    if not os.path.exists(file_path):
        print(f"  [Warning] {file_path} not found.")
        continue
    df = pd.read_csv(file_path)
    df['Seed'] = seed
    all_preds.append(df)

full_df = pd.concat(all_preds, ignore_index=True)
n_seeds_loaded = len(all_preds)
print(f"  Loaded {n_seeds_loaded} seeds, {len(full_df)} total rows")

full_df['Is_TP'] = ((full_df['responsive'] == 1) & (full_df['Prediction'] == 1)).astype(int)
full_df['Is_TN'] = ((full_df['responsive'] == 0) & (full_df['Prediction'] == 0)).astype(int)
full_df['Is_FN'] = ((full_df['responsive'] == 1) & (full_df['Prediction'] == 0)).astype(int)
full_df['Is_FP'] = ((full_df['responsive'] == 0) & (full_df['Prediction'] == 1)).astype(int)

agg_df = full_df.groupby(['main_receptors_id', 'main_compounds_id']).agg(
    TP_Count=('Is_TP', 'sum'),
    TN_Count=('Is_TN', 'sum'),
    FN_Count=('Is_FN', 'sum'),
    FP_Count=('Is_FP', 'sum'),
    Mean_Prob=('Probability', 'mean'),
    Std_Prob=('Probability', 'std'),
    GroundTruth=('responsive', 'first')
).reset_index()

raw_df = pd.read_csv(RAW_DATA_CSV)

print("\n" + "=" * 70)
print("  PART 1: Screening Robust TP Pairs")
print("=" * 70)

active_pairs = agg_df[agg_df['GroundTruth'] == 1].copy()
active_pairs['TP_Ratio'] = active_pairs['TP_Count'] / n_seeds_loaded
robust_tp_all = active_pairs[active_pairs['TP_Ratio'] >= MIN_TP_RATIO].copy()
robust_tp_all = robust_tp_all.sort_values('TP_Count', ascending=False)

raw_info = raw_df[['main_receptors_id', 'main_compounds_id', 'smiles', 'mutated_sequence']].drop_duplicates(
    subset=['main_receptors_id', 'main_compounds_id']
)
raw_df_with_idx = raw_df.reset_index().rename(columns={'index': 'raw_dataset_index'})
raw_idx_map = raw_df_with_idx[['raw_dataset_index', 'main_receptors_id', 'main_compounds_id']].drop_duplicates(
    subset=['main_receptors_id', 'main_compounds_id']
)

robust_tp_all = pd.merge(robust_tp_all, raw_info, on=['main_receptors_id', 'main_compounds_id'], how='left')
robust_tp_all = pd.merge(robust_tp_all, raw_idx_map, on=['main_receptors_id', 'main_compounds_id'], how='left')

print(f"\n  Active pairs (GT=1): {len(active_pairs)}")
print(f"  Robust TP (>= {MIN_TP_RATIO*100:.0f}% TP rate): {len(robust_tp_all)}")

print(f"\n  Top 20 Robust TP Pairs (sorted by TP count, then mean probability):")
robust_tp_all_sorted = robust_tp_all.sort_values(['TP_Count', 'Mean_Prob'], ascending=[False, False])
display_cols = ['raw_dataset_index', 'main_receptors_id', 'main_compounds_id',
                'TP_Count', 'FN_Count', 'Mean_Prob', 'Std_Prob', 'smiles']
print(robust_tp_all_sorted[display_cols].head(20).to_string(index=False))

perfect_tp = robust_tp_all[robust_tp_all['TP_Count'] == n_seeds_loaded]
print(f"\n  Perfect 20/20 TP pairs: {len(perfect_tp)}")
if len(perfect_tp) > 0:
    print(perfect_tp[display_cols].head(20).to_string(index=False))

robust_tp_path = os.path.join(OUTPUT_DIR, "robust_tp_pairs.csv")
robust_tp_all_sorted.to_csv(robust_tp_path, index=False)
print(f"\n  Saved: {robust_tp_path}")

print("\n" + "=" * 70)
print("  PART 2: Finding Minimal Pairs")
print("=" * 70)

robust_tp = agg_df[agg_df['TP_Count'] >= MIN_SEED_COUNT].copy()
robust_tn = agg_df[agg_df['TN_Count'] >= MIN_SEED_COUNT].copy()

print(f"  Robust TP (>= {MIN_SEED_COUNT}/{n_seeds_loaded}): {len(robust_tp)}")
print(f"  Robust TN (>= {MIN_SEED_COUNT}/{n_seeds_loaded}): {len(robust_tn)}")

if len(robust_tp) == 0 or len(robust_tn) == 0:
    print("  [Error] Not enough robust candidates. Consider lowering MIN_SEED_COUNT.")
else:
    smiles_map = raw_df[['main_receptors_id', 'main_compounds_id', 'smiles']].drop_duplicates(
        subset=['main_receptors_id', 'main_compounds_id']
    )
    robust_tp = pd.merge(robust_tp, smiles_map, on=['main_receptors_id', 'main_compounds_id'], how='left')
    robust_tn = pd.merge(robust_tn, smiles_map, on=['main_receptors_id', 'main_compounds_id'], how='left')

    def get_dual_fps(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        rdkit_fp = Chem.RDKFingerprint(mol, maxPath=15)
        return morgan_fp, rdkit_fp

    def are_enantiomers(smi1, smi2):
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return False
        Chem.RemoveStereochemistry(mol1)
        Chem.RemoveStereochemistry(mol2)
        return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)

    print("  Calculating fingerprints...")
    tp_fps = {}
    for _, row in robust_tp.iterrows():
        if pd.notna(row['smiles']):
            tp_fps[row['main_compounds_id']] = get_dual_fps(row['smiles'])

    tn_fps = {}
    for _, row in robust_tn.iterrows():
        if pd.notna(row['smiles']):
            tn_fps[row['main_compounds_id']] = get_dual_fps(row['smiles'])

    minimal_pairs = []
    common_receptors = set(robust_tp['main_receptors_id']).intersection(
        set(robust_tn['main_receptors_id'])
    )
    print(f"  Common receptors with both TP and TN: {len(common_receptors)}")

    for rec_id in tqdm(common_receptors, desc="  Scanning Receptors"):
        tp_subset = robust_tp[robust_tp['main_receptors_id'] == rec_id]
        tn_subset = robust_tn[robust_tn['main_receptors_id'] == rec_id]

        for _, tp_row in tp_subset.iterrows():
            for _, tn_row in tn_subset.iterrows():
                fp_tp = tp_fps.get(tp_row['main_compounds_id'])
                fp_tn = tn_fps.get(tn_row['main_compounds_id'])

                if fp_tp is None or fp_tn is None:
                    continue
                if fp_tp[0] is None or fp_tn[0] is None:
                    continue

                morgan_sim = DataStructs.TanimotoSimilarity(fp_tp[0], fp_tn[0])
                rdkit_sim = DataStructs.TanimotoSimilarity(fp_tp[1], fp_tn[1])

                if morgan_sim >= MIN_SIMILARITY and rdkit_sim >= MIN_SIMILARITY:
                    is_enant = are_enantiomers(tp_row['smiles'], tn_row['smiles'])

                    tp_idx_match = raw_df_with_idx[
                        (raw_df_with_idx['main_receptors_id'] == rec_id) &
                        (raw_df_with_idx['main_compounds_id'] == tp_row['main_compounds_id'])
                    ]
                    tn_idx_match = raw_df_with_idx[
                        (raw_df_with_idx['main_receptors_id'] == rec_id) &
                        (raw_df_with_idx['main_compounds_id'] == tn_row['main_compounds_id'])
                    ]

                    tp_raw_idx = int(tp_idx_match['raw_dataset_index'].iloc[0]) if len(tp_idx_match) > 0 else -1
                    tn_raw_idx = int(tn_idx_match['raw_dataset_index'].iloc[0]) if len(tn_idx_match) > 0 else -1

                    minimal_pairs.append({
                        'Receptor_ID': rec_id,
                        'Morgan_Similarity': round(morgan_sim, 4),
                        'RDKit_Similarity': round(rdkit_sim, 4),
                        'Avg_Similarity': round((morgan_sim + rdkit_sim) / 2.0, 4),
                        'Is_Enantiomer': is_enant,
                        'TP_Compound_ID': tp_row['main_compounds_id'],
                        'TP_SMILES': tp_row['smiles'],
                        'TP_Mean_Prob': round(tp_row['Mean_Prob'], 4),
                        'TP_Raw_Index': tp_raw_idx,
                        'TN_Compound_ID': tn_row['main_compounds_id'],
                        'TN_SMILES': tn_row['smiles'],
                        'TN_Mean_Prob': round(tn_row['Mean_Prob'], 4),
                        'TN_Raw_Index': tn_raw_idx,
                    })

    if minimal_pairs:
        results_df = pd.DataFrame(minimal_pairs)
        results_df['Prob_Diff'] = results_df['TP_Mean_Prob'] - results_df['TN_Mean_Prob']
        results_df = results_df.sort_values(
            by=['Is_Enantiomer', 'Avg_Similarity', 'Prob_Diff'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        output_path = os.path.join(OUTPUT_DIR, "strict_robust_minimal_pairs.csv")
        results_df.to_csv(output_path, index=False)

        print(f"\n  Found {len(results_df)} minimal pairs!")
        enantiomer_count = results_df['Is_Enantiomer'].sum()
        print(f"  Of which {enantiomer_count} are enantiomer pairs")
        print(f"  Saved: {output_path}")

        if enantiomer_count > 0:
            print(f"\n  Enantiomer Minimal Pairs:")
            enant_df = results_df[results_df['Is_Enantiomer']]
            display_cols = ['Receptor_ID', 'TP_Compound_ID', 'TN_Compound_ID',
                            'Morgan_Similarity', 'RDKit_Similarity',
                            'TP_Mean_Prob', 'TN_Mean_Prob', 'Prob_Diff',
                            'TP_Raw_Index', 'TN_Raw_Index']
            print(enant_df[display_cols].to_string(index=False))

        print(f"\n  Top 10 Minimal Pairs (all):")
        display_cols = ['Receptor_ID', 'Is_Enantiomer',
                        'TP_Compound_ID', 'TN_Compound_ID',
                        'Avg_Similarity', 'TP_Mean_Prob', 'TN_Mean_Prob',
                        'TP_Raw_Index', 'TN_Raw_Index']
        print(results_df[display_cols].head(10).to_string(index=False))
    else:
        print("\n  No minimal pairs found. Consider lowering MIN_SIMILARITY or MIN_SEED_COUNT.")

print("\n" + "=" * 70)
print("  Done!")
print("=" * 70)