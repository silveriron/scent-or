import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, matthews_corrcoef, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
SCENARIO = "Scenario5"
N_SPLITS = 10
ALPHA = 0.05
METRICS = ["AUROC", "AUPRC", "F1", "MCC"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
os.makedirs(OUT_DIR, exist_ok=True)

CURATED_CSV = os.path.join(DATA_DIR, "final_dataset_weighted.csv")
ASSIGN_CSV = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")
OUT_FOLD = os.path.join(OUT_DIR, "T2_subset_metrics_fold.csv")
OUT_SEED = os.path.join(OUT_DIR, "T2_subset_metrics.csv")
OUT_STATS = os.path.join(OUT_DIR, "T2_statistical_tests.csv")
OUT_MD = os.path.join(OUT_DIR, "T2_summary.md")


def safe(fn, y, p, default=np.nan):
    if len(np.unique(y)) < 2:
        return default
    return fn(y, p)


print("=" * 78)
print("  T2 — human-only subset metrics from stored OOF predictions")
print("=" * 78)

curated = pd.read_csv(CURATED_CSV)
assign = pd.read_csv(ASSIGN_CSV)
labels = curated["responsive"].values
print(f"  curated : {CURATED_CSV}  ({len(curated)} rows)")
print(f"  assign  : {ASSIGN_CSV}  ({len(assign)} receptors)")

group_of = {}
fine_of = {}
for _, r in assign.iterrows():
    rid = int(r["main_receptors_id"])
    org = r["organism"]
    if pd.isna(org):
        group_of[rid] = "unassigned"
        fine_of[rid] = "unassigned"
    elif bool(r["is_human"]):
        group_of[rid] = "human"
        fine_of[rid] = "human"
    else:
        group_of[rid] = "non_human"
        fine_of[rid] = "mouse" if str(org).startswith("Mus ") else "other_non_human"

curated_group = curated["main_receptors_id"].map(group_of).values
curated_fine = curated["main_receptors_id"].map(fine_of).values
print("\n  receptor group -> pair counts")
for g, n in pd.Series(curated_group).value_counts().items():
    n_act = int((labels[curated_group == g] == 1).sum())
    print(f"    {g:12s}: pairs={n:6d}  active={n_act:5d}  active_frac={n_act / n:.4f}")

print("\n  NOTE: thresholds are NOT re-optimised. The stored per-fold Prediction column")
print("        (fold-wise MCC-optimal threshold fitted on the training split) is reused")
print("        verbatim; subsetting only selects rows.")

fold_order = {}
for seed in SEEDS:
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    assign_arr = np.empty(len(curated), dtype=int)
    order = []
    for fold, (_, val_idx) in enumerate(kf.split(np.zeros(len(labels)), labels), start=1):
        assign_arr[val_idx] = fold
        order.append(val_idx)
    order = np.concatenate(order)
    fold_order[seed] = (assign_arr[order], curated_group[order], curated_fine[order])

rows = []
for seed in SEEDS:
    oof = pd.read_csv(os.path.join(RESULTS_DIR, f"OOF_CM_{SCENARIO}_seed{seed}.csv"))
    fold_seq, group_seq, fine_seq = fold_order[seed]
    y_all = oof["responsive"].values
    p_all = oof["Probability"].values
    yh_all = oof["Prediction"].values
    for fold in range(1, N_SPLITS + 1):
        fold_mask = fold_seq == fold
        for cond, cond_mask in [
            ("all", np.ones(len(oof), dtype=bool)),
            ("human", group_seq == "human"),
            ("non_human", group_seq == "non_human"),
            ("unassigned", group_seq == "unassigned"),
            ("mouse", fine_seq == "mouse"),
            ("other_non_human", fine_seq == "other_non_human"),
        ]:
            m = fold_mask & cond_mask
            y, p, yh = y_all[m], p_all[m], yh_all[m]
            n_pos = int((y == 1).sum())
            entry = {"Seed": seed, "Fold": fold, "Condition": cond,
                     "n_pairs": int(m.sum()), "n_active": n_pos,
                     "n_inactive": int((y == 0).sum())}
            entry["AUROC"] = safe(roc_auc_score, y, p)
            entry["AUPRC"] = safe(average_precision_score, y, p)
            if len(np.unique(y)) < 2:
                entry["F1"] = np.nan
                entry["MCC"] = np.nan
                entry.update(dict(tp=np.nan, tn=np.nan, fp=np.nan, fn=np.nan))
            else:
                entry["F1"] = f1_score(y, yh, zero_division=0)
                entry["MCC"] = matthews_corrcoef(y, yh)
                tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
                entry.update(dict(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn)))
            rows.append(entry)

fold_df = pd.DataFrame(rows)
fold_df.to_csv(OUT_FOLD, index=False)
print(f"\n  fold-level rows: {len(fold_df)}  -> {OUT_FOLD}")

seed_df = (fold_df.groupby(["Seed", "Condition"])[METRICS + ["n_pairs", "n_active"]]
           .mean().reset_index())

summary_rows = []
print("\n[1] 20-seed summary (per-seed mean over 10 folds, then mean +/- std ddof=1)")
for cond in ["all", "human", "non_human", "mouse", "other_non_human", "unassigned"]:
    sub = seed_df[seed_df["Condition"] == cond]
    entry = {"Condition": cond,
             "mean_pairs_per_fold": sub["n_pairs"].mean(),
             "mean_active_per_fold": sub["n_active"].mean()}
    line = f"  {cond:11s} (~{sub['n_pairs'].mean():7.1f} pairs/fold)"
    for met in METRICS:
        mu, sd = sub[met].mean(), sub[met].std(ddof=1)
        entry[f"{met}_mean"] = mu
        entry[f"{met}_std_ddof1"] = sd
        line += f"  {met} {mu:.4f}+/-{sd:.4f}"
    summary_rows.append(entry)
    print(line)

summary = pd.DataFrame(summary_rows)
summary["prevalence"] = summary["mean_active_per_fold"] / summary["mean_pairs_per_fold"]
summary["AUPRC_normalised"] = ((summary["AUPRC_mean"] - summary["prevalence"])
                               / (1.0 - summary["prevalence"]))
summary.to_csv(OUT_SEED, index=False)

print("\n[1b] Prevalence correction (AUPRC baseline equals the positive rate)")
for _, r in summary.iterrows():
    print(f"  {r['Condition']:11s} prevalence {r['prevalence']:.4f}  "
          f"AUPRC {r['AUPRC_mean']:.4f}  normalised {r['AUPRC_normalised']:.4f}")

print("\n[2] Statistical comparison (seed-level, n=20 paired observations)")
conds = ["all", "human", "non_human"]
stat_rows = []
for met in METRICS:
    pivot = seed_df[seed_df["Condition"].isin(conds)].pivot(
        index="Seed", columns="Condition", values=met)[conds]
    arrays = [pivot[c].values for c in conds]

    norm_p = {c: stats.shapiro(pivot[c].values)[1] for c in conds}
    normal = all(p > ALPHA for p in norm_p.values())
    levene_p = stats.levene(*arrays)[1]
    homosced = levene_p > ALPHA
    parametric = normal and homosced

    print(f"\n  [{met}]")
    print(f"    Shapiro-Wilk (normality): {'Pass' if normal else 'Fail'}  "
          + ", ".join(f"{c}={norm_p[c]:.4f}" for c in conds))
    print(f"    Levene (homoscedasticity): {'Pass' if homosced else 'Fail'} (p={levene_p:.4f})")

    long = pivot.reset_index().melt(id_vars="Seed", value_vars=conds,
                                    var_name="Condition", value_name=met)
    if parametric:
        global_name = "RM-ANOVA"
        try:
            global_p = AnovaRM(data=long, depvar=met, subject="Seed",
                               within=["Condition"]).fit().anova_table["Pr > F"].iloc[0]
        except Exception:
            global_name = "Friedman"
            global_p = stats.friedmanchisquare(*arrays)[1]
    else:
        global_name = "Friedman"
        global_p = stats.friedmanchisquare(*arrays)[1]
    print(f"    Global [{global_name}]: p = {global_p:.4e}")

    posthoc = []
    for a, b in itertools.combinations(conds, 2):
        sa, sb = pivot[a].values, pivot[b].values
        if np.allclose(sa, sb):
            raw_p = 1.0
        elif parametric:
            raw_p = stats.ttest_rel(sa, sb)[1]
        else:
            raw_p = stats.wilcoxon(sa, sb)[1]
        posthoc.append({"metric": met, "comparison": f"{a} vs {b}",
                        "mean_diff": sa.mean() - sb.mean(), "raw_p": raw_p})
    ph = pd.DataFrame(posthoc)
    if global_p < ALPHA:
        reject, adj, _, _ = multipletests(ph["raw_p"], alpha=ALPHA, method="fdr_bh")
        ph["adj_p"] = adj
        ph["significant"] = reject
    else:
        ph["adj_p"] = np.nan
        ph["significant"] = False
        print("    => global test not significant; post-hoc not interpreted")
    for _, r in ph.iterrows():
        star = ("***" if r["adj_p"] < 0.001 else "**" if r["adj_p"] < 0.01
                else "*" if r["adj_p"] < 0.05 else "ns") if pd.notna(r["adj_p"]) else "-"
        print(f"      {r['comparison']:24s} diff {r['mean_diff']:+.4f}  "
              f"adj_p {r['adj_p'] if pd.notna(r['adj_p']) else float('nan'):.4e} ({star})")
    ph["normality_pass"] = normal
    ph["levene_p"] = levene_p
    ph["global_test"] = global_name
    ph["global_p"] = global_p
    ph["posthoc_test"] = "paired t-test" if parametric else "Wilcoxon signed-rank"
    stat_rows.append(ph)

stats_df = pd.concat(stat_rows, ignore_index=True)
stats_df.to_csv(OUT_STATS, index=False)

n_nan = int(fold_df[fold_df["Condition"] == "unassigned"]["AUROC"].isna().sum())

with open(OUT_MD, "w") as fh:
    fh.write("# T2 — Human-only subset performance\n\n")
    fh.write(f"Scenario {SCENARIO}, 20 seeds x {N_SPLITS} folds, recomputed from stored OOF "
             "predictions. **No retraining and no threshold re-optimisation.**\n\n")
    fh.write("## Method\n\n")
    fh.write("- Receptor species from `T1_receptor_assignment.csv`.\n")
    fh.write("- AUROC/AUPRC come from the stored `Probability`; F1/MCC come from the stored "
             "`Prediction`, which already encodes each fold's MCC-optimal threshold fitted on "
             "that fold's *training* split. Subsetting selects rows only, so no information "
             "from the subset enters the threshold.\n")
    fh.write("- Folds reconstructed with `StratifiedKFold(n_splits=10, shuffle=True, "
             "random_state=SEED)` on the curated row order.\n\n")
    fh.write("## Results\n\n")
    fh.write("| condition | pairs/fold | active/fold | AUROC | AUPRC | F1 | MCC |\n")
    fh.write("|---|---|---|---|---|---|---|\n")
    for _, r in summary.iterrows():
        fh.write(f"| {r['Condition']} | {r['mean_pairs_per_fold']:.1f} | "
                 f"{r['mean_active_per_fold']:.1f} | "
                 + " | ".join(f"{r[m + '_mean']:.4f} ± {r[m + '_std_ddof1']:.4f}"
                              for m in METRICS) + " |\n")
    fh.write("\n## Prevalence correction\n\n")
    fh.write("AUPRC, F1 and MCC all depend on the positive base rate, which differs between "
             "the subsets; AUROC does not. The random-classifier AUPRC baseline equals the "
             "prevalence, so normalised AUPRC = (AUPRC - prevalence) / (1 - prevalence).\n\n")
    fh.write("| condition | prevalence | AUPRC | normalised AUPRC |\n|---|---|---|---|\n")
    for _, r in summary.iterrows():
        fh.write(f"| {r['Condition']} | {r['prevalence']:.4f} | {r['AUPRC_mean']:.4f} | "
                 f"{r['AUPRC_normalised']:.4f} |\n")
    fh.write("\n## Statistical tests (seed-level, n = 20 paired)\n\n")
    fh.write("| metric | comparison | mean diff | global test | global p | adj p | significant |\n")
    fh.write("|---|---|---|---|---|---|---|\n")
    for _, r in stats_df.iterrows():
        fh.write(f"| {r['metric']} | {r['comparison']} | {r['mean_diff']:+.4f} | "
                 f"{r['global_test']} | {r['global_p']:.3e} | "
                 f"{r['adj_p']:.3e} | {'yes' if r['significant'] else 'no'} |\n")
    fh.write("\n## Standard deviation convention\n\n")
    fh.write("- Per-seed value = mean over its 10 folds; reported spread = sample standard "
             "deviation (`ddof=1`) across the 20 seeds.\n")
    fh.write(f"\n- Folds where the `unassigned` subset had a single class present: {n_nan}/200 "
             "(metrics undefined, excluded from that condition's mean).\n")

print(f"\n  saved: {OUT_FOLD}")
print(f"  saved: {OUT_SEED}")
print(f"  saved: {OUT_STATS}")
print(f"  saved: {OUT_MD}")
print("=" * 78)
