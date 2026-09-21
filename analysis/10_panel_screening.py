import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
DATASET_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted.csv")
ASSIGNMENT_PATH = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")
LEGACY_PANEL_PATH = os.path.join(OUT_DIR, "T4_combinatorial.csv")
OOF_TEMPLATE = os.path.join(RESULTS_DIR, "OOF_CM_Scenario5_seed{seed}.csv")
TABLE_PATH = os.path.join(OUT_DIR, "table_S11.csv")
SEED_PATH = os.path.join(OUT_DIR, "table_S11_seed_level.csv")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
TOP_FRACTION = 0.20
HUMAN_ORGANISM = "Homo sapiens"
HUMAN_DIFFERENCE_BOUND = 0.003
EXPECTED = {
    "(E)-Isoeugenol": {"panel": 457, "K": 116, "positive_rate": 0.2538,
                       "auroc": (0.9001, 0.0059), "precision": (0.6978, 0.0159),
                       "enrichment": 2.75, "recall": (0.581, 0.020), "top20": 91},
    "Androstenone": {"panel": 637, "K": 95, "positive_rate": 0.1491,
                     "auroc": (0.8123, 0.0074), "precision": (0.3868, 0.0229),
                     "enrichment": 2.59, "recall": (0.493, 0.024), "top20": 127},
}
TOLERANCE = 0.0005

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:44s} expected {expected}   observed {observed}")


print("=" * 78)
print("  T2 - Table S10, panel-level metrics")
print("=" * 78)

dataset = pd.read_csv(DATASET_PATH)
assignment = pd.read_csv(ASSIGNMENT_PATH)
organism = dict(zip(assignment.main_receptors_id, assignment.organism))
print(f"\n[1] inputs")
print(f"  {DATASET_PATH}  {len(dataset)} rows")
print(f"  {ASSIGNMENT_PATH}  {len(assignment)} receptors")

responsive_counts = (dataset[dataset.responsive == 1].groupby("main_compounds_id").size()
                     .sort_values(ascending=False))
selected = responsive_counts.index[:2].tolist()
print(f"\n[2] ligand selection by number of responsive ORs, no further filter")
for compound in selected:
    print(f"  compound {compound}: {responsive_counts[compound]} responsive ORs, "
          f"{int((dataset.main_compounds_id == compound).sum())} ORs in the panel")
names = {selected[0]: "(E)-Isoeugenol", selected[1]: "Androstenone"}\
    if (dataset.main_compounds_id == selected[0]).sum() == 457\
    else {selected[1]: "(E)-Isoeugenol", selected[0]: "Androstenone"}

legacy = pd.read_csv(LEGACY_PANEL_PATH)
legacy_human = {compound: set(legacy[(legacy.main_compounds_id == compound)
                                     & legacy.is_human].main_receptors_id)
                for compound in selected}
print(f"  {LEGACY_PANEL_PATH}: human ORs per panel "
      f"{ {names[c]: len(v) for c, v in legacy_human.items()} }")

probabilities = {}
for seed in SEEDS:
    frame = pd.read_csv(OOF_TEMPLATE.format(seed=seed))
    assert len(frame) == len(dataset)
    probabilities[seed] = dict(zip(zip(frame.main_receptors_id, frame.main_compounds_id),
                                   frame.Probability.values))
print(f"  {len(probabilities)} out-of-fold prediction files loaded")


def panel_metrics(receptors, labels, scores, k, top_n):
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    prevalence = labels.mean()
    return {
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "precision_at_K": ranked[:k].mean(),
        "enrichment": ranked[:k].mean() / prevalence,
        "recall_top20": ranked[:top_n].sum() / labels.sum(),
    }


seed_rows = []
summary_rows = []
print("\n[3] per-panel computation, one ranking per seed over the complete panel")
for compound in selected:
    name = names[compound]
    panel = dataset[dataset.main_compounds_id == compound]
    receptors = panel.main_receptors_id.values
    labels = panel.responsive.values.astype(int)
    k = int(labels.sum())
    top_n = int(round(TOP_FRACTION * len(panel)))
    human_mask = np.array([organism[r] == HUMAN_ORGANISM for r in receptors])
    legacy_mask = np.array([r in legacy_human[compound] for r in receptors])
    print(f"\n  {name}: panel {len(panel)}, responsive {k}, "
          f"positive rate {labels.mean():.4f}, K {k}, top 20% rank {top_n}")
    print(f"    human by T1 assignment {int(human_mask.sum())} ORs "
          f"({int(labels[human_mask].sum())} responsive); "
          f"human by T4 is_human {int(legacy_mask.sum())} ORs "
          f"({int(labels[legacy_mask].sum())} responsive)")
    for seed in SEEDS:
        scores = np.array([probabilities[seed][(r, compound)] for r in receptors])
        metrics = panel_metrics(receptors, labels, scores, k, top_n)
        for mask, tag in ((human_mask, "human_t1"), (legacy_mask, "human_t4")):
            subset_labels = labels[mask]
            metrics[f"auroc_{tag}"] = roc_auc_score(subset_labels, scores[mask])
        metrics.update({"odorant_ligand": name, "compound_id": compound, "Seed": seed})
        seed_rows.append(metrics)
    frame = pd.DataFrame([row for row in seed_rows if row["compound_id"] == compound])
    summary = {"odorant_ligand": name, "compound_id": compound,
               "panel_size": len(panel), "n_responsive_K": k,
               "positive_rate": labels.mean(),
               "top20_rank": top_n,
               "human_t1_panel_size": int(human_mask.sum()),
               "human_t4_panel_size": int(legacy_mask.sum())}
    for metric in ("auroc", "auprc", "precision_at_K", "enrichment", "recall_top20",
                   "auroc_human_t1", "auroc_human_t4"):
        summary[f"{metric}_mean"] = frame[metric].mean()
        summary[f"{metric}_sd"] = frame[metric].std(ddof=1)
    summary_rows.append(summary)

summary_frame = pd.DataFrame(summary_rows)
print("\n[4] summary over 20 seeds (ddof=1)")
for _, row in summary_frame.iterrows():
    print(f"  {row.odorant_ligand}")
    print(f"    AUROC          {row.auroc_mean:.4f} +/- {row.auroc_sd:.4f}")
    print(f"    AUPRC          {row.auprc_mean:.4f} +/- {row.auprc_sd:.4f}")
    print(f"    precision@K    {row.precision_at_K_mean:.4f} +/- {row.precision_at_K_sd:.4f}")
    print(f"    enrichment     {row.enrichment_mean:.4f} +/- {row.enrichment_sd:.4f}")
    print(f"    recall@top20%  {row.recall_top20_mean:.4f} +/- {row.recall_top20_sd:.4f}")
    print(f"    human AUROC, T1 assignment  {row.auroc_human_t1_mean:.4f} "
          f"+/- {row.auroc_human_t1_sd:.4f}  ({row.human_t1_panel_size} ORs)")
    print(f"    human AUROC, T4 is_human    {row.auroc_human_t4_mean:.4f} "
          f"+/- {row.auroc_human_t4_sd:.4f}  ({row.human_t4_panel_size} ORs)")

print("\n[5] verification against the printed values")
indexed = summary_frame.set_index("odorant_ligand")
for name, expected in EXPECTED.items():
    row = indexed.loc[name]
    record(f"{name} panel size", expected["panel"], int(row.panel_size),
           int(row.panel_size) == expected["panel"])
    record(f"{name} responsive K", expected["K"], int(row.n_responsive_K),
           int(row.n_responsive_K) == expected["K"])
    record(f"{name} positive rate", expected["positive_rate"], round(row.positive_rate, 4),
           abs(row.positive_rate - expected["positive_rate"]) <= TOLERANCE)
    record(f"{name} top 20% rank", expected["top20"], int(row.top20_rank),
           int(row.top20_rank) == expected["top20"])
    record(f"{name} AUROC", f"{expected['auroc'][0]:.4f} +/- {expected['auroc'][1]:.4f}",
           f"{row.auroc_mean:.4f} +/- {row.auroc_sd:.4f}",
           abs(row.auroc_mean - expected["auroc"][0]) <= TOLERANCE
           and abs(row.auroc_sd - expected["auroc"][1]) <= TOLERANCE)
    record(f"{name} precision@K",
           f"{expected['precision'][0]:.4f} +/- {expected['precision'][1]:.4f}",
           f"{row.precision_at_K_mean:.4f} +/- {row.precision_at_K_sd:.4f}",
           abs(row.precision_at_K_mean - expected["precision"][0]) <= TOLERANCE
           and abs(row.precision_at_K_sd - expected["precision"][1]) <= TOLERANCE)
    record(f"{name} enrichment", expected["enrichment"], round(row.enrichment_mean, 2),
           abs(round(row.enrichment_mean, 2) - expected["enrichment"]) <= 0.005)
    record(f"{name} recall@top20%",
           f"{expected['recall'][0]:.3f} +/- {expected['recall'][1]:.3f}",
           f"{row.recall_top20_mean:.3f} +/- {row.recall_top20_sd:.3f}",
           abs(row.recall_top20_mean - expected["recall"][0]) <= 0.001
           and abs(row.recall_top20_sd - expected["recall"][1]) <= 0.001)
    record(f"{name} enrichment equals precision over prevalence",
           round(expected["precision"][0] / expected["positive_rate"], 3),
           round(row.precision_at_K_mean / row.positive_rate, 3),
           abs(row.precision_at_K_mean / row.positive_rate - row.enrichment_mean) < 1e-9)

print("\n[6] the human-only claim of Section S11.3")
for name in EXPECTED:
    row = indexed.loc[name]
    for tag, label in (("t1", "T1 species assignment"), ("t4", "T4 is_human column")):
        difference = abs(row[f"auroc_human_{tag}_mean"] - row.auroc_mean)
        record(f"{name} human-only shift under the {label}",
               f"< {HUMAN_DIFFERENCE_BOUND}", round(difference, 4),
               difference < HUMAN_DIFFERENCE_BOUND)

os.makedirs(OUT_DIR, exist_ok=True)
summary_frame.to_csv(TABLE_PATH, index=False)
pd.DataFrame(seed_rows).to_csv(SEED_PATH, index=False)
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[7] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table is written as computed and has not been adjusted")
    for _, row in summary[summary.Verdict == "FAIL"].iterrows():
        print(f"    {row.Check}: expected {row.Expected}, observed {row.Observed}")
summary.to_csv(os.path.join(OUT_DIR, "table_S11_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}")
print(f"  wrote {SEED_PATH}")

if emit_latex:
    latex_path = TABLE_PATH.replace(".csv", ".tex")
    with open(latex_path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write("\\caption{Panel-level metrics for the two odorant ligands with the "
                     "largest number of annotated responsive ORs. Values are the mean "
                     "$\\pm$ standard deviation over 20 seeds, each computed on that seed's "
                     "out-of-fold ranking over the complete panel (Section~S6.10). "
                     "Precision@$K$ uses $K$ equal to the number of annotated responsive ORs "
                     "and enrichment is precision@$K$ divided by the panel positive rate; "
                     "recall is measured in the top 20\\% of the ranking.}\n")
        handle.write("\\label{tab:panel-metrics}\n")
        handle.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        handle.write("Odorant Ligand & Panel Size & Responsive ($K$) & Positive Rate & "
                     "AUROC & AUPRC & Precision@$K$ & Enrichment \\\\\n\\midrule\n")
        for _, row in summary_frame.iterrows():
            handle.write(f"{row.odorant_ligand} & {row.panel_size} & {row.n_responsive_K} & "
                         f"{row.positive_rate:.3f} & "
                         f"{row.auroc_mean:.4f} $\\pm$ {row.auroc_sd:.4f} & "
                         f"{row.auprc_mean:.4f} $\\pm$ {row.auprc_sd:.4f} & "
                         f"{row.precision_at_K_mean:.4f} $\\pm$ {row.precision_at_K_sd:.4f} & "
                         f"{row.enrichment_mean:.2f} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
        handle.write("\\begin{tablenotes}\\footnotesize\n")
        for _, row in summary_frame.iterrows():
            handle.write(f"\\item {row.odorant_ligand}: recall in the top 20\\% "
                         f"({row.top20_rank} ORs) is "
                         f"{row.recall_top20_mean:.3f} $\\pm$ {row.recall_top20_sd:.3f}; "
                         f"restricting the panel to human ORs gives AUROC "
                         f"{row.auroc_human_t4_mean:.4f} $\\pm$ "
                         f"{row.auroc_human_t4_sd:.4f} "
                         f"over {row.human_t4_panel_size} ORs.\n")
        handle.write("\\end{tablenotes}\n\\end{table}\n")
    print(f"  wrote {latex_path}")
print("=" * 78)
