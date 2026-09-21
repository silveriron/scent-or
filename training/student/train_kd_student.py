import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, confusion_matrix, average_precision_score
from tqdm import tqdm
import pickle
import joblib
import copy

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]

def seed_everything(seed):
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = "../../data"
PKL_PATH = os.path.join(DATA_DIR, "final_embeddings_all.pkl")
CSV_PATH = os.path.join(DATA_DIR, "final_dataset_weighted.csv")

N_SPLITS = 10
BATCH_SIZE = 128
EPOCHS = 150
STRUCT_DIM = 512

BEST_CONFIGS = {
    "Scenario5": {"teacher": "lgb", "gamma": 1.0, "alpha": 0.5, "student": "sem", "name": "Scenario5"},
    "Scenario6": {"teacher": "xgb", "gamma": 1.0, "alpha": 0.5, "student": "sem", "name": "Scenario6"},
    "Scenario7": {"teacher": "lgb", "gamma": 1.0, "alpha": 0.4, "student": "str", "name": "Scenario7"},
    "Scenario8": {"teacher": "xgb", "gamma": 1.0, "alpha": 0.5, "student": "str", "name": "Scenario8"}
}

print("Loading Data...")
df_map = pd.read_csv(CSV_PATH)
with open(PKL_PATH, 'rb') as f:
    vault = pickle.load(f)

l_seq = np.array([vault['ligand_embeddings'].get(i, np.zeros(768)) for i in df_map['main_compounds_id']], dtype=np.float32)
p_seq = np.array([vault['protein_embeddings'].get(i, np.zeros(1024)) for i in df_map['main_receptors_id']], dtype=np.float32)
l_str = np.array([vault.get('ligand_structure_embeddings', {}).get(i, np.zeros(STRUCT_DIM)) for i in df_map['main_compounds_id']], dtype=np.float32)
p_str = np.array([vault.get('protein_structure_embeddings', {}).get(i, np.zeros(STRUCT_DIM)) for i in df_map['main_receptors_id']], dtype=np.float32)

labels = df_map['responsive'].values.flatten()
weights = df_map['sample_weight'].values.flatten()

l_seq_t = torch.tensor(l_seq)
p_seq_t = torch.tensor(p_seq)
l_str_t = torch.tensor(l_str)
p_str_t = torch.tensor(p_str)
labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
weights_t = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

X_sem = np.concatenate([l_seq, p_seq], axis=1)
X_str = np.concatenate([l_seq, p_seq, l_str, p_str], axis=1)

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

class StudentModelStructural(nn.Module):
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

def get_teacher_probs(scenario_key, cfg, SEED):
    models = []
    print(f"Loading Teacher models for {scenario_key} ({cfg['teacher']})...")

    teacher_dir = os.path.join("../../models/teacher_models", f"seed_{SEED}")
    teacher_suffix = "_sem" if scenario_key in ["Scenario5", "Scenario7"] else "_str"

    for fold in range(1, N_SPLITS+1):
        filename = f"{cfg['teacher']}_seed{SEED}{teacher_suffix}_fold{fold}.model"
        filepath = os.path.join(teacher_dir, filename)

        m = joblib.load(filepath)
        models.append(m)

    probs = np.zeros(len(df_map))
    X_target = X_sem if scenario_key in ["Scenario5", "Scenario7"] else X_str

    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_target, labels)):
        if cfg['teacher'] in ['xgb', 'lgb', 'cat']:
            p = models[fold].predict_proba(X_target[val_idx])[:, 1]
        probs[val_idx] = p
    return probs

def train_student(fold, model, train_loader, val_loader, scenario_key, cfg, save_dir, SEED):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=15)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    criterion_hard = FocalLoss(gamma=cfg['gamma'], reduction='none')
    criterion_soft = nn.BCEWithLogitsLoss(reduction='mean')
    alpha = cfg['alpha']

    epoch_pbar = tqdm(range(EPOCHS), desc=f"      Fold {fold+1} Training", leave=False)
    for epoch in epoch_pbar:
        model.train()
        for b in train_loader:
            inputs = [x.to(DEVICE) for x in b[:-3]]
            ol = b[-3].to(DEVICE)
            sl = b[-2].to(DEVICE)
            wt = b[-1].to(DEVICE)

            optimizer.zero_grad()
            out, _, _ = model(*inputs)
            loss = alpha * (criterion_hard(out, ol) * wt).mean() + (1 - alpha) * criterion_soft(out, sl)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                inputs = [x.to(DEVICE) for x in b[:-3]]
                ol = b[-3].to(DEVICE)
                sl = b[-2].to(DEVICE)
                wt = b[-1].to(DEVICE)
                out, _, _ = model(*inputs)
                loss = alpha * (criterion_hard(out, ol) * wt).mean() + (1 - alpha) * criterion_soft(out, sl)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        early_stopping(avg_val_loss)
        epoch_pbar.set_postfix(val_loss=f"{avg_val_loss:.4f}")

        if early_stopping.early_stop:
            break
        scheduler.step()

    model.load_state_dict(best_model_wts)
    model.eval()

    train_preds, train_targets = [], []
    with torch.no_grad():
        for b in train_loader:
            inputs = [x.to(DEVICE) for x in b[:-3]]
            ol = b[-3].to(DEVICE)
            wt = b[-1].to(DEVICE)
            out, _, _ = model(*inputs)
            train_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
            train_targets.extend(ol.cpu().numpy().flatten())

    train_preds_np = np.array(train_preds)
    train_targets_np = np.array(train_targets)

    thresholds = np.linspace(0.01, 0.99, 100)
    train_mccs = [matthews_corrcoef(train_targets_np, train_preds_np > thr) for thr in thresholds]
    optimal_thresh = thresholds[np.argmax(train_mccs)]

    val_preds, val_targets = [], []
    with torch.no_grad():
        for b in val_loader:
            inputs = [x.to(DEVICE) for x in b[:-3]]
            ol = b[-3].to(DEVICE)

            out, _, _ = model(*inputs)
            val_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
            val_targets.extend(ol.cpu().numpy().flatten())

    val_preds_np = np.array(val_preds)
    val_targets_np = np.array(val_targets)

    final_preds_binary = (val_preds_np >= optimal_thresh).astype(int)

    final_auroc = roc_auc_score(val_targets_np, val_preds_np)
    final_auprc = average_precision_score(val_targets_np, val_preds_np)
    final_f1 = f1_score(val_targets_np, final_preds_binary)
    final_mcc = matthews_corrcoef(val_targets_np, final_preds_binary)

    tn, fp, fn, tp = confusion_matrix(val_targets_np, final_preds_binary).ravel()

    tqdm.write(f"      Fold {fold+1} Done | AUROC: {final_auroc:.4f} | AUPRC: {final_auprc:.4f} | F1: {final_f1:.4f} | MCC: {final_mcc:.4f} | Thresh: {optimal_thresh:.2f}")

    save_filename = f"{scenario_key}_seed{SEED}_fold{fold+1}.pt"
    save_path = os.path.join(save_dir, save_filename)
    torch.save(best_model_wts, save_path)

    return {
        'auroc': final_auroc, 'auprc': final_auprc, 'f1': final_f1, 'mcc': final_mcc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'val_preds': val_preds_np,
        'val_binary': final_preds_binary
    }

def main():
    for SEED in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Running all KD scenarios for Seed {SEED}")
        print(f"{'='*60}")

        seed_everything(SEED)

        STUDENT_SAVE_DIR_SEED = os.path.join("../../models/student_models", f"seed_{SEED}")
        os.makedirs(STUDENT_SAVE_DIR_SEED, exist_ok=True)

        RESULTS_DIR = "../../results"
        os.makedirs(RESULTS_DIR, exist_ok=True)

        final_results = []
        detailed_results = []

        for scenario_key, cfg in BEST_CONFIGS.items():
            print(f"\n==================================================")
            print(f"  Starting {cfg['name']} (Seed: {SEED})")
            print(f"  Teacher: {cfg['teacher']} | alpha: {cfg['alpha']} | gamma: {cfg['gamma']}")
            print(f"==================================================")

            teacher_probs = get_teacher_probs(scenario_key, cfg, SEED)
            teacher_probs_t = torch.tensor(teacher_probs, dtype=torch.float32).view(-1, 1)

            kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            fold_metrics = []
            oof_dfs = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_sem, labels)):
                if cfg['student'] == 'sem':
                    train_ds = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], labels_t[train_idx], teacher_probs_t[train_idx], weights_t[train_idx])
                    val_ds = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], labels_t[val_idx], teacher_probs_t[val_idx], weights_t[val_idx])
                    model = StudentModelSemantic().to(DEVICE)
                else:
                    train_ds = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], l_str_t[train_idx], p_str_t[train_idx], labels_t[train_idx], teacher_probs_t[train_idx], weights_t[train_idx])
                    val_ds = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], l_str_t[val_idx], p_str_t[val_idx], labels_t[val_idx], teacher_probs_t[val_idx], weights_t[val_idx])
                    model = StudentModelStructural().to(DEVICE)

                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

                metrics = train_student(fold, model, train_loader, val_loader, scenario_key, cfg, STUDENT_SAVE_DIR_SEED, SEED)
                fold_metrics.append(metrics)

                fold_df = df_map.iloc[val_idx].copy()
                fold_df['Probability'] = metrics['val_preds']
                fold_df['Prediction'] = metrics['val_binary']

                def get_category(row):
                    if row['responsive'] == 1 and row['Prediction'] == 1: return 'TP'
                    elif row['responsive'] == 0 and row['Prediction'] == 0: return 'TN'
                    elif row['responsive'] == 0 and row['Prediction'] == 1: return 'FP'
                    else: return 'FN'

                fold_df['Category'] = fold_df.apply(get_category, axis=1)
                oof_dfs.append(fold_df)

            aurocs = [m['auroc'] for m in fold_metrics]
            auprcs = [m['auprc'] for m in fold_metrics]
            f1s = [m['f1'] for m in fold_metrics]
            mccs = [m['mcc'] for m in fold_metrics]

            for i, m in enumerate(fold_metrics):
                detailed_results.append({
                    'Seed': SEED,
                    'Scenario': cfg['name'],
                    'Fold': i + 1,
                    'TP': m['tp'], 'TN': m['tn'], 'FP': m['fp'], 'FN': m['fn'],
                    'AUROC': m['auroc'], 'AUPRC': m['auprc'], 'F1': m['f1'], 'MCC': m['mcc']
                })

            total_tp = sum([m['tp'] for m in fold_metrics])
            total_tn = sum([m['tn'] for m in fold_metrics])
            total_fp = sum([m['fp'] for m in fold_metrics])
            total_fn = sum([m['fn'] for m in fold_metrics])

            final_results.append({
                'Seed': SEED,
                'Scenario': cfg['name'],
                'AUROC_Mean': np.mean(aurocs), 'AUROC_Std': np.std(aurocs),
                'AUPRC_Mean': np.mean(auprcs), 'AUPRC_Std': np.std(auprcs),
                'F1_Mean': np.mean(f1s), 'F1_Std': np.std(f1s),
                'MCC_Mean': np.mean(mccs), 'MCC_Std': np.std(mccs),
                'Total_TP': total_tp, 'Total_TN': total_tn,
                'Total_FP': total_fp, 'Total_FN': total_fn
            })

            case_oof_df = pd.concat(oof_dfs, ignore_index=True)
            case_oof_df.to_csv(os.path.join(RESULTS_DIR, f"OOF_CM_{scenario_key}_seed{SEED}.csv"), index=False)

        print(f"\n  Seed {SEED} complete.")

        df_res = pd.DataFrame(final_results)
        df_res.to_csv(os.path.join(RESULTS_DIR, f"10fold_cv_seed{SEED}.csv"), index=False)

        df_detail = pd.DataFrame(detailed_results)
        df_detail.to_csv(os.path.join(RESULTS_DIR, f"10fold_cv_cm_seed{SEED}.csv"), index=False)

if __name__ == "__main__":
    main()