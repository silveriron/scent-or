import argparse
import os
import random
import copy
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, f1_score,
                             confusion_matrix, average_precision_score)


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

N_SPLITS = 10
BATCH_SIZE = 128
EPOCHS = 150
STRUCT_DIM = 512

BEST_CONFIGS = {
    "Scenario5": {"teacher": "lgb", "gamma": 1.0, "alpha": 0.5, "student": "sem", "name": "Scenario5"},
    "Scenario6": {"teacher": "xgb", "gamma": 1.0, "alpha": 0.5, "student": "sem", "name": "Scenario6"},
    "Scenario7": {"teacher": "lgb", "gamma": 1.0, "alpha": 0.4, "student": "str", "name": "Scenario7"},
    "Scenario8": {"teacher": "xgb", "gamma": 1.0, "alpha": 0.5, "student": "str", "name": "Scenario8"},
}


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
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                               batch_first=True)
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
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(dim, 1))

    def forward(self, l_vec, p_vec):
        l_proj = self.ligand_proj(l_vec)
        p_proj = self.protein_proj(p_vec)
        l_ref = self.l_self(l_proj, l_proj, l_proj)
        p_ref = self.p_self(p_proj, p_proj, p_proj)
        l_ctx = self.l_cross(query=l_ref, key_value=p_ref)
        p_ctx = self.p_cross(query=p_ref, key_value=l_ref)
        final = torch.cat([l_ctx, p_ctx], dim=1)
        return self.mlp(final), None, None


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience, self.counter, self.best_score, self.early_stop = patience, 0, None, False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_student(fold, model, train_loader, val_loader, cfg, save_dir, scenario_key, seed):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=15)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    criterion_hard = FocalLoss(gamma=cfg['gamma'], reduction='none')
    criterion_soft = nn.BCEWithLogitsLoss(reduction='mean')
    alpha = cfg['alpha']
    n_epochs_run = 0

    for epoch in range(EPOCHS):
        n_epochs_run = epoch + 1
        model.train()
        for b in train_loader:
            inputs = [x.to(DEVICE) for x in b[:-3]]
            ol, sl, wt = b[-3].to(DEVICE), b[-2].to(DEVICE), b[-1].to(DEVICE)
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
                ol, sl, wt = b[-3].to(DEVICE), b[-2].to(DEVICE), b[-1].to(DEVICE)
                out, _, _ = model(*inputs)
                loss = alpha * (criterion_hard(out, ol) * wt).mean() + (1 - alpha) * criterion_soft(out, sl)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        early_stopping(avg_val_loss)
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
            out, _, _ = model(*inputs)
            train_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
            train_targets.extend(ol.cpu().numpy().flatten())

    thresholds = np.linspace(0.01, 0.99, 100)
    train_mccs = [matthews_corrcoef(np.array(train_targets), np.array(train_preds) > t)
                  for t in thresholds]
    optimal_thresh = thresholds[int(np.argmax(train_mccs))]

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
    final_binary = (val_preds_np >= optimal_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(val_targets_np, final_binary).ravel()

    print(f"      Fold {fold + 1} | AUROC {roc_auc_score(val_targets_np, val_preds_np):.4f} "
          f"| MCC {matthews_corrcoef(val_targets_np, final_binary):.4f} "
          f"| thresh {optimal_thresh:.4f} | epochs {n_epochs_run}", flush=True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_model_wts,
                   os.path.join(save_dir, f"{scenario_key}_seed{seed}_fold{fold + 1}.pt"))

    return {
        'auroc': roc_auc_score(val_targets_np, val_preds_np),
        'auprc': average_precision_score(val_targets_np, val_preds_np),
        'f1': f1_score(val_targets_np, final_binary),
        'mcc': matthews_corrcoef(val_targets_np, final_binary),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'threshold': optimal_thresh, 'epochs': n_epochs_run,
        'val_preds': val_preds_np, 'val_binary': final_binary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="Scenario5")
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--save-checkpoints", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")
    out_dir = os.path.join(repo_root, "results", "revision", "gpu_check", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  T0e — diagnostic retraining for cross-GPU reproducibility")
    print("=" * 78)
    print(f"  host        : {os.uname().nodename}")
    print(f"  device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"  torch       : {torch.__version__}  cuda {torch.version.cuda}  "
          f"cudnn {torch.backends.cudnn.version()}")
    print(f"  numpy       : {np.__version__}")
    print(f"  seed        : {args.seed}   scenario: {args.scenario}   tag: {args.tag}")
    print(f"  output      : {out_dir}", flush=True)

    df_map = pd.read_csv(os.path.join(data_dir, "final_dataset_weighted.csv"))
    with open(os.path.join(data_dir, "final_embeddings_all.pkl"), "rb") as fh:
        vault = pickle.load(fh)

    l_seq = np.array([vault['ligand_embeddings'].get(i, np.zeros(768))
                      for i in df_map['main_compounds_id']], dtype=np.float32)
    p_seq = np.array([vault['protein_embeddings'].get(i, np.zeros(1024))
                      for i in df_map['main_receptors_id']], dtype=np.float32)
    labels = df_map['responsive'].values.flatten()
    weights = df_map['sample_weight'].values.flatten()

    l_seq_t = torch.tensor(l_seq)
    p_seq_t = torch.tensor(p_seq)
    labels_t = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    weights_t = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    X_sem = np.concatenate([l_seq, p_seq], axis=1)

    cfg = BEST_CONFIGS[args.scenario]
    seed_everything(args.seed)

    teacher_dir = os.path.join(repo_root, "models", "teacher_models", f"seed_{args.seed}")
    suffix = "_sem" if args.scenario in ("Scenario5", "Scenario7") else "_str"
    teacher_models = [joblib.load(os.path.join(
        teacher_dir, f"{cfg['teacher']}_seed{args.seed}{suffix}_fold{f}.model"))
        for f in range(1, N_SPLITS + 1)]

    teacher_probs = np.zeros(len(df_map))
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    for fold, (_, val_idx) in enumerate(kf.split(X_sem, labels)):
        teacher_probs[val_idx] = teacher_models[fold].predict_proba(X_sem[val_idx])[:, 1]
    teacher_probs_t = torch.tensor(teacher_probs, dtype=torch.float32).view(-1, 1)
    print(f"  teacher probs: mean {teacher_probs.mean():.6f}  "
          f"sha-ish {float(teacher_probs.sum()):.6f}", flush=True)

    save_dir = os.path.join(out_dir, "checkpoints") if args.save_checkpoints else None
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    fold_metrics, oof_dfs = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sem, labels)):
        train_ds = TensorDataset(l_seq_t[train_idx], p_seq_t[train_idx], labels_t[train_idx],
                                 teacher_probs_t[train_idx], weights_t[train_idx])
        val_ds = TensorDataset(l_seq_t[val_idx], p_seq_t[val_idx], labels_t[val_idx],
                               teacher_probs_t[val_idx], weights_t[val_idx])
        model = StudentModelSemantic().to(DEVICE)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        m = train_student(fold, model, train_loader, val_loader, cfg, save_dir,
                          args.scenario, args.seed)
        fold_metrics.append(m)

        fold_df = df_map.iloc[val_idx].copy()
        fold_df['Probability'] = m['val_preds']
        fold_df['Prediction'] = m['val_binary']
        fold_df['Category'] = np.where(
            fold_df['responsive'] == 1,
            np.where(fold_df['Prediction'] == 1, 'TP', 'FN'),
            np.where(fold_df['Prediction'] == 1, 'FP', 'TN'))
        fold_df['Fold'] = fold + 1
        oof_dfs.append(fold_df)

    pd.concat(oof_dfs, ignore_index=True).to_csv(
        os.path.join(out_dir, f"OOF_CM_{args.scenario}_seed{args.seed}.csv"), index=False)
    pd.DataFrame([{
        'Seed': args.seed, 'Scenario': args.scenario, 'Fold': i + 1,
        'tp': m['tp'], 'tn': m['tn'], 'fp': m['fp'], 'fn': m['fn'],
        'auroc': m['auroc'], 'auprc': m['auprc'], 'f1': m['f1'], 'mcc': m['mcc'],
        'threshold': m['threshold'], 'epochs': m['epochs'],
    } for i, m in enumerate(fold_metrics)]).to_csv(
        os.path.join(out_dir, f"fold_metrics_{args.scenario}_seed{args.seed}.csv"), index=False)

    print(f"\n  seed mean AUROC {np.mean([m['auroc'] for m in fold_metrics]):.6f}  "
          f"MCC {np.mean([m['mcc'] for m in fold_metrics]):.6f}")
    print(f"  totals TP {sum(m['tp'] for m in fold_metrics)} "
          f"TN {sum(m['tn'] for m in fold_metrics)} "
          f"FP {sum(m['fp'] for m in fold_metrics)} "
          f"FN {sum(m['fn'] for m in fold_metrics)}")
    print(f"  saved: {out_dir}")


if __name__ == "__main__":
    main()
