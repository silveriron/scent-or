import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import pickle
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import random
import os
import joblib
from imblearn.combine import SMOTETomek

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
N_SPLITS = 10
STRUCT_DIM = 512

PKL_PATH = "../../data/final_embeddings_all.pkl"
CSV_PATH = "../../data/final_dataset_weighted.csv"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

def generate_teacher_oof_with_smote(X, y, seed, suffix=""):
    oof_preds = np.zeros(len(y))
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    print(f"   -> [Teacher] Training LGB{suffix} with SMOTE-Tomek...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X, y), total=N_SPLITS, desc=f"   lgb{suffix}")):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        try:
            smt = SMOTETomek(random_state=seed)
            X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"      [Warning] SMOTE-Tomek failed. Error: {e}")
            X_train_res, y_train_res = X_train, y_train

        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            random_state=seed, n_jobs=1, verbose=-1,
            deterministic=True, force_col_wise=True
        )
        model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], eval_metric='binary_logloss', callbacks=[lgb.early_stopping(20, verbose=False)])

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        save_path = os.path.join(TEACHER_SAVE_DIR, f"lgb_seed{seed}{suffix}_fold{fold+1}.model")
        joblib.dump(model, save_path)

    return oof_preds

print("Loading Data...")
df_map = pd.read_csv(CSV_PATH)
with open(PKL_PATH, 'rb') as f: vault = pickle.load(f)

l_seq = np.array([vault['ligand_embeddings'].get(i, np.zeros(768)) for i in df_map['main_compounds_id']], dtype=np.float32)
p_seq = np.array([vault['protein_embeddings'].get(i, np.zeros(1024)) for i in df_map['main_receptors_id']], dtype=np.float32)
l_str = np.array([vault.get('ligand_structure_embeddings', {}).get(i, np.zeros(STRUCT_DIM)) for i in df_map['main_compounds_id']], dtype=np.float32)
p_str = np.array([vault.get('protein_structure_embeddings', {}).get(i, np.zeros(STRUCT_DIM)) for i in df_map['main_receptors_id']], dtype=np.float32)

labels = df_map['responsive'].values.flatten()
weights = df_map['sample_weight'].values.flatten()

X_semantic_only = np.concatenate([l_seq, p_seq], axis=1)
X_structure_full = np.concatenate([l_seq, p_seq, l_str, p_str], axis=1)

for SEED in SEEDS:
    print(f"\n============================================================")
    print(f"  Seed {SEED}")
    print(f"============================================================")

    TEACHER_SAVE_DIR = f"../../models/teacher_models/seed_{SEED}"
    os.makedirs(TEACHER_SAVE_DIR, exist_ok=True)

    set_seed(SEED)

    print(f"\n[PHASE 1] Training Semantic-Only Teacher (LGB)...")
    oof_probs_sem = generate_teacher_oof_with_smote(X_semantic_only, labels, SEED, suffix="_sem")

    print(f"\n[PHASE 2] Training Hybrid Modality Teacher (LGB)...")
    oof_probs_str = generate_teacher_oof_with_smote(X_structure_full, labels, SEED, suffix="_str")

    print(f"Seed {SEED} complete.")
