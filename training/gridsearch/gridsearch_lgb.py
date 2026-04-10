import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
import pickle
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import random
import os
import itertools
import joblib
from imblearn.combine import SMOTETomek
import warnings

warnings.filterwarnings("ignore")

TARGET_TEACHER = 'lgb'

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

SEED = 42
N_SPLITS = 10
BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15
STRUCT_DIM = 512

TEACHER_SAVE_DIR = f"../../models/teacher_models/seed_{SEED}"
os.makedirs(TEACHER_SAVE_DIR, exist_ok=True)

SEARCH_SPACE = {
    'TEACHERS': [TARGET_TEACHER],
    'GAMMAS': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'ALPHAS': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

PKL_PATH = "../../data/final_embeddings_all.pkl"
CSV_PATH = "../../data/final_dataset_weighted.csv"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss

class EarlyStopping:
    def __init__(self, patience=15):
        self.patience, self.counter, self.best_score, self.early_stop = patience, 0, None, False
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

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
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim, 1))
    def forward(self, l_vec, p_vec):
        l_proj = self.ligand_proj(l_vec)
        p_proj = self.protein_proj(p_vec)
        l_ref = self.l_self(l_proj, l_proj, l_proj)
        p_ref = self.p_self(p_proj, p_proj, p_proj)
        l_ctx = self.l_cross(query=l_ref, key_value=p_ref)
        p_ctx = self.p_cross(query=p_ref, key_value=l_ref)
        final = torch.cat([l_ctx, p_ctx], dim=1)
        return self.mlp(final), None, None

class StudentModelStruct(nn.Module):
    def __init__(self, l_seq_dim=768, p_seq_dim=1024, s_dim=512, proj_dim=512, num_heads=4):
        super().__init__()
        self.l_seq_proj = nn.Linear(l_seq_dim, proj_dim)
        self.p_seq_proj = nn.Linear(p_seq_dim, proj_dim)
        self.l_str_proj = nn.Linear(s_dim, proj_dim)
        self.p_str_proj = nn.Linear(s_dim, proj_dim)
        dim = proj_dim
        self.l_self = AttentionModule(dim, num_heads)
        self.p_self = AttentionModule(dim, num_heads)
        self.cross = GatedCrossAttentionModule(dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(dim, 1))
    def forward(self, l_seq, p_seq, l_str, p_str):
        l_s = self.l_seq_proj(l_seq)
        p_s = self.p_seq_proj(p_seq)
        l_geo = self.l_str_proj(l_str)
        p_geo = self.p_str_proj(p_str)
        l_ref = self.l_self(l_s, l_s, l_s)
        p_ref = self.p_self(p_s, p_s, p_s)
        l_ctx = self.cross(query=l_ref, key_value=p_ref)
        p_ctx = self.cross(query=p_ref, key_value=l_ref)
        final_vec = torch.cat([l_ctx, p_ctx, l_geo, p_geo], dim=1)
        return self.mlp(final_vec), None, None

def generate_teacher_oof_with_smote(teacher_type, X, y, seed, suffix=""):
    oof_preds = np.zeros(len(y))
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    print(f"   -> [Teacher] Training {teacher_type.upper()}{suffix} with SMOTE-Tomek...")

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(X, y), total=N_SPLITS, desc=f"   {teacher_type}{suffix}")):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        try:
            smt = SMOTETomek(random_state=seed)
            X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"      [Warning] SMOTE-Tomek failed. Error: {e}")
            X_train_res, y_train_res = X_train, y_train

        if teacher_type == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                random_state=seed, n_jobs=1, verbose=-1,
                deterministic=True, force_col_wise=True
            )
            model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], eval_metric='binary_logloss', callbacks=[lgb.early_stopping(20, verbose=False)])
        elif teacher_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                eval_metric='logloss', early_stopping_rounds=20,
                random_state=seed, n_jobs=1, verbosity=0, tree_method='hist'
            )
            model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
        elif teacher_type == 'cat':
            model = cat.CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=5,
                random_seed=seed, task_type="CPU", thread_count=1,
                allow_writing_files=False, verbose=0
            )
            model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], early_stopping_rounds=20)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        save_path = os.path.join(TEACHER_SAVE_DIR, f"{teacher_type}_seed{seed}{suffix}_fold{fold+1}.model")
        joblib.dump(model, save_path)

    return oof_preds

def run_student_grid_search(student_type, train_ds, val_ds, results_list, teacher_type, case_name):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    configs = list(itertools.product(SEARCH_SPACE['GAMMAS'], SEARCH_SPACE['ALPHAS']))

    for gamma, alpha in tqdm(configs, desc=f"   Search ({case_name})", leave=False):
        set_seed(SEED)

        if student_type == 'sem':
            student = StudentModelSemantic().to(DEVICE)
        else:
            student = StudentModelStruct(s_dim=STRUCT_DIM).to(DEVICE)

        optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        criterion_hard = FocalLoss(gamma=gamma, reduction='none')
        criterion_soft = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

        best_f1, best_auroc, best_auprc = 0, 0, 0
        best_mcc = -1.0
        best_tp, best_tn, best_fp, best_fn = 0, 0, 0, 0
        best_thresh = 0.0

        for epoch in range(EPOCHS):
            student.train()
            train_preds, train_targets = [], []
            for b in train_loader:
                if student_type == 'sem':
                    inputs = [x.to(DEVICE) for x in b[:2]]
                    pl, ol, w = b[2].to(DEVICE), b[3].to(DEVICE), b[4].to(DEVICE)
                else:
                    inputs = [x.to(DEVICE) for x in b[:4]]
                    pl, ol, w = b[4].to(DEVICE), b[5].to(DEVICE), b[6].to(DEVICE)

                optimizer.zero_grad()
                out, _, _ = student(*inputs)
                loss = alpha * (criterion_hard(out, ol) * w).mean() + (1 - alpha) * criterion_soft(out, pl)
                loss.backward()
                optimizer.step()

                train_preds.extend(torch.sigmoid(out).detach().cpu().numpy().flatten())
                train_targets.extend(ol.cpu().numpy().flatten())

            student.eval()
            val_loss = 0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for b in val_loader:
                    if student_type == 'sem':
                        inputs = [x.to(DEVICE) for x in b[:2]]
                        ol, pl, w = b[2].to(DEVICE), b[3].to(DEVICE), b[4].to(DEVICE)
                    else:
                        inputs = [x.to(DEVICE) for x in b[:4]]
                        ol, pl, w = b[4].to(DEVICE), b[5].to(DEVICE), b[6].to(DEVICE)

                    out, _, _ = student(*inputs)
                    val_loss_hard = (criterion_hard(out, ol) * w).mean()
                    val_loss_soft = criterion_soft(out, pl)
                    val_loss += (alpha * val_loss_hard + (1 - alpha) * val_loss_soft).item()
                    val_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
                    val_targets.extend(ol.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(val_loader)
            early_stopping(avg_val_loss)

            train_preds_np = np.array(train_preds)
            train_targets_np = np.array(train_targets)
            thresholds = np.linspace(0.01, 0.99, 100)

            train_mccs = [matthews_corrcoef(train_targets_np, train_preds_np > thr) for thr in thresholds]
            optimal_train_thresh = thresholds[np.argmax(train_mccs)]

            val_preds_np = np.array(val_preds)
            val_targets_np = np.array(val_targets)

            auroc = roc_auc_score(val_targets_np, val_preds_np)
            auprc = average_precision_score(val_targets_np, val_preds_np)

            val_preds_binary = (val_preds_np >= optimal_train_thresh).astype(int)
            curr_f1 = f1_score(val_targets_np, val_preds_binary)
            curr_mcc = matthews_corrcoef(val_targets_np, val_preds_binary)
            tn, fp, fn, tp = confusion_matrix(val_targets_np, val_preds_binary).ravel()

            if auprc > best_auprc:
                best_f1 = curr_f1
                best_auroc = auroc
                best_auprc = auprc
                best_mcc = curr_mcc
                best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn
                best_thresh = optimal_train_thresh

            if early_stopping.early_stop: break
            scheduler.step()

        results_list.append({
            'teacher': teacher_type,
            'gamma': gamma,
            'alpha': alpha,
            'f1': best_f1,
            'mcc': best_mcc,
            'auroc': best_auroc,
            'auprc': best_auprc,
            'threshold': best_thresh,
            'tp': best_tp,
            'tn': best_tn,
            'fp': best_fp,
            'fn': best_fn
        })
        del student, optimizer
        torch.cuda.empty_cache()

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

l_seq_t = torch.tensor(l_seq)
p_seq_t = torch.tensor(p_seq)
l_str_t = torch.tensor(l_str)
p_str_t = torch.tensor(p_str)
labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
weights_t = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

print("="*60)
print(f" STARTING GRID SEARCH FOR [{TARGET_TEACHER.upper()}] - Seed {SEED}")
print("="*60)

set_seed(SEED)

kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_idx, val_idx = next(iter(kfold.split(X_semantic_only, labels)))

res_case5 = []
res_case6 = []
res_case7 = []
res_case8 = []

for teacher_type in SEARCH_SPACE['TEACHERS']:

    print(f"\n[PHASE 1] Training Semantic-Only Teacher ({teacher_type.upper()})...")
    oof_probs_sem = generate_teacher_oof_with_smote(teacher_type, X_semantic_only, labels, SEED, suffix="_sem")
    pseudo_labels_sem = torch.tensor(oof_probs_sem, dtype=torch.float32).view(-1, 1)

    ds_train_c5 = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], pseudo_labels_sem[train_idx], labels_t[train_idx], weights_t[train_idx])
    ds_val_c5 = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], labels_t[val_idx], pseudo_labels_sem[val_idx], weights_t[val_idx])
    run_student_grid_search('sem', ds_train_c5, ds_val_c5, res_case5, teacher_type, "Scenario V")

    ds_train_c7 = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], l_str_t[train_idx], p_str_t[train_idx], pseudo_labels_sem[train_idx], labels_t[train_idx], weights_t[train_idx])
    ds_val_c7 = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], l_str_t[val_idx], p_str_t[val_idx], labels_t[val_idx], pseudo_labels_sem[val_idx], weights_t[val_idx])
    run_student_grid_search('str', ds_train_c7, ds_val_c7, res_case7, teacher_type, "Scenario VII")

    print(f"\n[PHASE 2] Training Hybrid Modality Teacher ({teacher_type.upper()})...")
    oof_probs_str = generate_teacher_oof_with_smote(teacher_type, X_structure_full, labels, SEED, suffix="_str")
    pseudo_labels_str = torch.tensor(oof_probs_str, dtype=torch.float32).view(-1, 1)

    ds_train_c6 = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], pseudo_labels_str[train_idx], labels_t[train_idx], weights_t[train_idx])
    ds_val_c6 = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], labels_t[val_idx], pseudo_labels_str[val_idx], weights_t[val_idx])
    run_student_grid_search('sem', ds_train_c6, ds_val_c6, res_case6, teacher_type, "Scenario VI")

    ds_train_c8 = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], l_str_t[train_idx], p_str_t[train_idx], pseudo_labels_str[train_idx], labels_t[train_idx], weights_t[train_idx])
    ds_val_c8 = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], l_str_t[val_idx], p_str_t[val_idx], labels_t[val_idx], pseudo_labels_str[val_idx], weights_t[val_idx])
    run_student_grid_search('str', ds_train_c8, ds_val_c8, res_case8, teacher_type, "Scenario VIII")

print("\n" + "="*60)
print(f" GRID SEARCH COMPLETED FOR {TARGET_TEACHER.upper()}")
print("="*60)

pd.DataFrame(res_case5).sort_values('auprc', ascending=False).to_csv(f"Scenario5_{TARGET_TEACHER}.csv", index=False)
pd.DataFrame(res_case6).sort_values('auprc', ascending=False).to_csv(f"Scenario6_{TARGET_TEACHER}.csv", index=False)
pd.DataFrame(res_case7).sort_values('auprc', ascending=False).to_csv(f"Scenario7_{TARGET_TEACHER}.csv", index=False)
pd.DataFrame(res_case8).sort_values('auprc', ascending=False).to_csv(f"Scenario8_{TARGET_TEACHER}.csv", index=False)
