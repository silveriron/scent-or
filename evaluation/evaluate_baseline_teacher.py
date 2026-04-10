import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, confusion_matrix, average_precision_score

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729, 1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
N_SPLITS = 10
TEACHERS = ['xgb', 'lgb', 'cat']
MODALITIES = ['sem', 'str']
STRUCT_DIM = 512

DATA_DIR = "../data"
TEACHER_MODEL_BASE = "../models/teacher_models"
RESULTS_DIR = "../results"
CSV_PATH = os.path.join(DATA_DIR, "final_dataset_weighted.csv")
PKL_PATH = os.path.join(DATA_DIR, "final_embeddings_all.pkl")

print("Loading Data...")
df_map = pd.read_csv(CSV_PATH)
labels = df_map['responsive'].values.flatten()

import pickle
with open(PKL_PATH, 'rb') as f: vault = pickle.load(f)

l_seq = np.array([vault['ligand_embeddings'].get(i, np.zeros(768)) for i in df_map['main_compounds_id']], dtype=np.float32)
p_seq = np.array([vault['protein_embeddings'].get(i, np.zeros(1024)) for i in df_map['main_receptors_id']], dtype=np.float32)
l_str = np.array([vault.get('ligand_structure_embeddings', {}).get(i, np.zeros(STRUCT_DIM)) for i in df_map['main_compounds_id']], dtype=np.float32)
p_str = np.array([vault.get('protein_structure_embeddings', {}).get(i, np.zeros(STRUCT_DIM)) for i in df_map['main_receptors_id']], dtype=np.float32)

X_sem = np.concatenate([l_seq, p_seq], axis=1)
X_str = np.concatenate([l_seq, p_seq, l_str, p_str], axis=1)

results_summary_si = []
detailed_results = []

print("\n" + "="*60)
print("STARTING BASELINE TEACHER EVALUATION")
print("="*60)

for mod in MODALITIES:
    X_target = X_sem if mod == 'sem' else X_str

    for teacher in TEACHERS:
        for seed in SEEDS:
            print(f"\nEvaluating: {teacher.upper()} | Modality: {mod.upper()} | Seed: {seed}")
            kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_target, labels)):
                model_path = os.path.join(TEACHER_MODEL_BASE, f"seed_{seed}", f"{teacher}_seed{seed}_{mod}_fold{fold+1}.model")
                if not os.path.exists(model_path):
                    print(f"  [Warning] Missing model file: {model_path}")
                    continue

                m = joblib.load(model_path)

                if teacher in ['xgb', 'lgb', 'cat']:
                    train_preds = m.predict_proba(X_target[train_idx])[:, 1]
                    val_preds = m.predict_proba(X_target[val_idx])[:, 1]

                train_targets, val_targets = labels[train_idx], labels[val_idx]

                thresholds = np.linspace(0.01, 0.99, 100)
                train_mccs = [matthews_corrcoef(train_targets, train_preds > thr) for thr in thresholds]
                opt_thresh = thresholds[np.argmax(train_mccs)]

                val_binary = (val_preds >= opt_thresh).astype(int)

                auroc = roc_auc_score(val_targets, val_preds)
                auprc = average_precision_score(val_targets, val_preds)
                f1 = f1_score(val_targets, val_binary)
                mcc = matthews_corrcoef(val_targets, val_binary)
                tn, fp, fn, tp = confusion_matrix(val_targets, val_binary).ravel()

                fold_metrics.append({'auroc': auroc, 'auprc': auprc, 'f1': f1, 'mcc': mcc, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})

                detailed_results.append({
                    'Seed': seed,
                    'Model': f"Teacher_{teacher.upper()}_{mod}",
                    'Fold': fold + 1,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'AUROC': auroc, 'AUPRC': auprc, 'F1': f1, 'MCC': mcc
                })

            if fold_metrics:
                results_summary_si.append({
                    'Model': f"Teacher_{teacher.upper()}_{mod}",
                    'Seed': seed,
                    'AUROC_Mean': np.mean([m['auroc'] for m in fold_metrics]), 'AUROC_Std': np.std([m['auroc'] for m in fold_metrics]),
                    'AUPRC_Mean': np.mean([m['auprc'] for m in fold_metrics]), 'AUPRC_Std': np.std([m['auprc'] for m in fold_metrics]),
                    'F1_Mean': np.mean([m['f1'] for m in fold_metrics]),       'F1_Std': np.std([m['f1'] for m in fold_metrics]),
                    'MCC_Mean': np.mean([m['mcc'] for m in fold_metrics]),      'MCC_Std': np.std([m['mcc'] for m in fold_metrics]),
                    'Total_TP': np.sum([m['tp'] for m in fold_metrics]),        'Total_TN': np.sum([m['tn'] for m in fold_metrics]),
                    'Total_FP': np.sum([m['fp'] for m in fold_metrics]),        'Total_FN': np.sum([m['fn'] for m in fold_metrics])
                })

os.makedirs(RESULTS_DIR, exist_ok=True)

df_detail = pd.DataFrame(detailed_results)
df_detail.to_csv(os.path.join(RESULTS_DIR, "benchmark_baseline_teachers_Detailed_Folds.csv"), index=False)
print(f"\n Saved Detailed Folds to {RESULTS_DIR}/benchmark_baseline_teachers_Detailed_Folds.csv")
print(f"   Shape: {df_detail.shape} (expected: {len(SEEDS) * len(TEACHERS) * len(MODALITIES) * N_SPLITS} rows)")

df_si = pd.DataFrame(results_summary_si)
df_si.to_csv(os.path.join(RESULTS_DIR, "benchmark_baseline_teachers_SI.csv"), index=False)
print(f" Saved SI Table (Seed-wise 10-Fold stats) to {RESULTS_DIR}/benchmark_baseline_teachers_SI.csv")

main_summary = []
for model_name, group in df_si.groupby('Model'):
    main_summary.append({
        'Model': model_name,
        'AUROC (Mean ± Std)': f"{group['AUROC_Mean'].mean():.4f} ± {group['AUROC_Mean'].std():.4f}",
        'AUPRC (Mean ± Std)': f"{group['AUPRC_Mean'].mean():.4f} ± {group['AUPRC_Mean'].std():.4f}",
        'F1 (Mean ± Std)': f"{group['F1_Mean'].mean():.4f} ± {group['F1_Mean'].std():.4f}",
        'MCC (Mean ± Std)': f"{group['MCC_Mean'].mean():.4f} ± {group['MCC_Mean'].std():.4f}"
    })

df_main = pd.DataFrame(main_summary)
df_main.to_csv(os.path.join(RESULTS_DIR, "benchmark_baseline_teachers_Main.csv"), index=False)
print(f" Saved Main Table (Inter-seed stats) to {RESULTS_DIR}/benchmark_baseline_teachers_Main.csv")

print("\n ALL PIPELINES COMPLETED!")