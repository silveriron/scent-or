import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_utils

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
TEACHER_PATH = os.path.join(RESULTS_DIR, "baseline", "benchmark_baseline_teachers_SI.csv")
STUDENT_PATH = os.path.join(RESULTS_DIR, "baseline", "benchmark_baseline_students_SI.csv")
TABLE_PATH = os.path.join(OUT_DIR, "table_S10.csv")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
METRICS = ["AUROC", "AUPRC", "F1", "MCC"]
ALPHA = 0.05
COMPARISONS = [
    ("XGBoost", "I vs. II", "Teacher_XGB_sem", "Teacher_XGB_str"),
    ("LightGBM", "I vs. II", "Teacher_LGB_sem", "Teacher_LGB_str"),
    ("CatBoost", "I vs. II", "Teacher_CAT_sem", "Teacher_CAT_str"),
    ("Student", "III vs. IV", "Baseline_(III)Semantic_Only", "Baseline_(IV)Hybrid_Modality"),
]
EXPECTED_MCC = {
    "XGBoost": (0.477, 0.004, 0.489, 0.004, 0.0115),
    "LightGBM": (0.477, 0.003, 0.484, 0.004, 0.0070),
    "CatBoost": (0.453, 0.003, 0.465, 0.003, 0.0112),
    "Student": (0.470, 0.006, 0.480, 0.005, 0.0104),
}
TOLERANCE = 0.0005

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:46s} expected {expected}   observed {observed}")


print("=" * 78)
print("  T1 - Table S9, structural modality effect in the non-distilled baselines")
print("=" * 78)

teachers = pd.read_csv(TEACHER_PATH)
students = pd.read_csv(STUDENT_PATH)
print(f"\n[1] inputs")
print(f"  {TEACHER_PATH}  {len(teachers)} rows, models {sorted(teachers.Model.unique())}")
print(f"  {STUDENT_PATH}  {len(students)} rows, models {sorted(students.Model.unique())}")

frame = pd.concat([teachers[["Model", "Seed"] + [f"{m}_Mean" for m in METRICS]],
                   students[["Model", "Seed"] + [f"{m}_Mean" for m in METRICS]]],
                  ignore_index=True)

rows = []
print("\n[2] paired comparisons, seed-level metrics")
for metric in METRICS:
    column = f"{metric}_Mean"
    table = frame.pivot(index="Seed", columns="Model", values=column).loc[SEEDS]
    arms = []
    for name in {model for _, _, *models in COMPARISONS for model in models}:
        arms.append(table[name].values)
    normal = all(stats.shapiro(values)[1] > ALPHA for values in arms)
    equal_variance = stats.levene(*arms)[1] > ALPHA
    parametric = normal and equal_variance
    print(f"\n  [{metric}] Shapiro-Wilk {'pass' if normal else 'fail'}, "
          f"Levene {'pass' if equal_variance else 'fail'} -> "
          f"{'paired t-test' if parametric else 'Wilcoxon signed-rank'}")
    metric_rows = []
    for model, scenarios, semantic_name, hybrid_name in COMPARISONS:
        semantic = table[semantic_name].values
        hybrid = table[hybrid_name].values
        differences = hybrid - semantic
        if parametric:
            test = "paired t-test"
            raw_p = float(stats.ttest_rel(hybrid, semantic)[1])
            effect_name = "Cohen's dz"
            effect = stats_utils.cohens_dz(hybrid, semantic)
            _, ci_low, ci_high = stats_utils.paired_mean_ci(hybrid, semantic)
        else:
            test = "Wilcoxon signed-rank"
            raw_p = float(stats.wilcoxon(hybrid, semantic)[1])
            effect_name = "Hodges-Lehmann"
            effect = stats_utils.hodges_lehmann(differences)
            ci_low, ci_high = stats_utils.hl_ci(differences)
        metric_rows.append({
            "model": model, "scenarios": scenarios, "metric": metric,
            "mean_semantic": semantic.mean(), "sd_semantic": semantic.std(ddof=1),
            "mean_hybrid": hybrid.mean(), "sd_hybrid": hybrid.std(ddof=1),
            "delta": hybrid.mean() - semantic.mean(), "test": test, "raw_p": raw_p,
            "effect_size_name": effect_name, "effect_size": effect,
            "ci_low": ci_low, "ci_high": ci_high, "n_seeds": len(SEEDS)})
    adjusted = stats_utils.bh_adjust([row["raw_p"] for row in metric_rows])
    for row, value in zip(metric_rows, adjusted):
        row["adj_p"] = float(value)
        print(f"    {row['model']:9s} {row['mean_semantic']:.4f} +/- {row['sd_semantic']:.4f}"
              f"  ->  {row['mean_hybrid']:.4f} +/- {row['sd_hybrid']:.4f}"
              f"   delta {row['delta']:+.4f}   {row['effect_size_name']} {row['effect_size']:+.4f}"
              f" [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
              f"   raw {row['raw_p']:.3e}   BH {row['adj_p']:.3e}")
    rows.extend(metric_rows)

table_frame = pd.DataFrame(rows)[[
    "model", "scenarios", "metric", "mean_semantic", "sd_semantic", "mean_hybrid", "sd_hybrid",
    "delta", "test", "raw_p", "adj_p", "effect_size_name", "effect_size", "ci_low", "ci_high",
    "n_seeds"]]

print("\n[3] verification against the printed MCC column")
mcc = table_frame[table_frame.metric == "MCC"].set_index("model")
for model, (semantic, semantic_sd, hybrid, hybrid_sd, delta) in EXPECTED_MCC.items():
    row = mcc.loc[model]
    record(f"{model} MCC semantic-only", f"{semantic:.3f} +/- {semantic_sd:.3f}",
           f"{row.mean_semantic:.3f} +/- {row.sd_semantic:.3f}",
           abs(row.mean_semantic - semantic) <= TOLERANCE
           and abs(row.sd_semantic - semantic_sd) <= TOLERANCE)
    record(f"{model} MCC hybrid", f"{hybrid:.3f} +/- {hybrid_sd:.3f}",
           f"{row.mean_hybrid:.3f} +/- {row.sd_hybrid:.3f}",
           abs(row.mean_hybrid - hybrid) <= TOLERANCE
           and abs(row.sd_hybrid - hybrid_sd) <= TOLERANCE)
    record(f"{model} delta MCC", f"{delta:+.4f}", f"{row.delta:+.4f}",
           abs(row.delta - delta) <= TOLERANCE)

significant = mcc.adj_p < ALPHA
record("all four MCC comparisons significant after BH", True, bool(significant.all()),
       bool(significant.all()))

print("\n[4] the manuscript range statement")
teacher_deltas = mcc.loc[["XGBoost", "LightGBM", "CatBoost"], "delta"]
print(f"  teacher delta MCC spans {teacher_deltas.min():.4f} to {teacher_deltas.max():.4f}   "
      f"manuscript 'from 0.007 to 0.012'")
print(f"  student delta MCC {mcc.loc['Student', 'delta']:.4f}   manuscript '0.010'")

os.makedirs(OUT_DIR, exist_ok=True)
table_frame.to_csv(TABLE_PATH, index=False)
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[5] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table is written as computed and has not been adjusted")
summary.to_csv(os.path.join(OUT_DIR, "table_S10_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}")

if emit_latex:
    latex_path = TABLE_PATH.replace(".csv", ".tex")
    metric_labels = {"AUROC": "AUROC", "AUPRC": "AUPRC", "F1": "$F_1$ score", "MCC": "MCC"}
    with open(latex_path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write("\\caption{Paired comparison between the semantic-only and hybrid "
                     "modality configurations of the same model. Scenarios I and II for each "
                     "GB teacher and scenarios III and IV for the student. The test follows "
                     "the assumption checks of Section~S6.1 and adjusted $p$-values reflect "
                     "Benjamini--Hochberg FDR correction applied jointly to the four "
                     "comparisons within each metric ($N = 20$ seeds). Effect sizes follow "
                     "Section~S6.6: Cohen's $d_z$ with the 95\\% interval of the mean paired "
                     "difference for $t$-tests, and the Hodges--Lehmann median difference "
                     "with its distribution-free interval for Wilcoxon tests.}\n")
        handle.write("\\label{tab:modality-effect}\n")
        handle.write("\\begin{tabular}{llrrrlrrlr}\n\\toprule\n")
        handle.write("Model & Scenarios & Semantic-Only & Hybrid Modality & $\\Delta$ & "
                     "Test & Raw $p$ & Adj. $p$ & Effect Size & 95\\% CI \\\\\n")
        for metric in METRICS:
            handle.write("\\midrule\n\\multicolumn{10}{l}{\\textit{"
                         f"{metric_labels[metric]}" "}} \\\\\n")
            for _, row in table_frame[table_frame.metric == metric].iterrows():
                test_label = "$t$" if row.test.startswith("paired") else "W"
                effect_label = ("$d_z = " + f"{row.effect_size:.2f}$")\
                    if row.effect_size_name.startswith("Cohen")\
                    else ("HL $= " + f"{row.effect_size:+.4f}$")
                handle.write(
                    f"{row.model} & {row.scenarios} & "
                    f"{row.mean_semantic:.3f} $\\pm$ {row.sd_semantic:.3f} & "
                    f"{row.mean_hybrid:.3f} $\\pm$ {row.sd_hybrid:.3f} & "
                    f"{row.delta:+.3f} & {test_label} & "
                    f"{row.raw_p:.2e} & {row.adj_p:.2e} & {effect_label} & "
                    f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {latex_path}")
print("=" * 78)
