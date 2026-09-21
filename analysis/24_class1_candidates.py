import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
DATASET_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted.csv")
ASSIGNMENT_PATH = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")
STRINGS_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted_with_strings.csv")
OOF_TEMPLATE = os.path.join(RESULTS_DIR, "OOF_CM_Scenario5_seed{seed}.csv")
TABLE_PATH = os.path.join(OUT_DIR, "table_S16.csv")
SHORT_PATH = os.path.join(OUT_DIR, "table_S16_top20.csv")
STORED_PATH = os.path.join(OUT_DIR, "T7_classI_candidates.csv")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
CLASS_I_PATTERN = re.compile(r"^OR(51|52|55|56)")
N_SPLITS = 10
TP_SEED_MINIMUM = 16
HUMAN_ORGANISM = "Homo sapiens"
ARCHIVED_SOURCES = {"uniparc_exact", "uniparc_exact+human_preferred"}
SHORT_LIST_LENGTH = 20
EXPECTED_PAIRS = 111
EXPECTED_RECEPTORS = 37
EXPECTED_TOP = ("OR56A5", "CCCCCCCC(=O)O", 20, 0.918, 0.027)

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:44s} expected {expected}   observed {observed}")


print("=" * 78)
print("  T7 - Table S15, Class I candidate pairs")
print("=" * 78)

dataset = pd.read_csv(DATASET_PATH)
assignment = pd.read_csv(ASSIGNMENT_PATH)
strings = pd.read_csv(STRINGS_PATH, usecols=["main_compounds_id", "smiles"]).drop_duplicates()
print(f"\n[1] inputs")
print(f"  {DATASET_PATH}  {len(dataset)} rows")
print(f"  {ASSIGNMENT_PATH}  {len(assignment)} receptors")
print(f"  {OOF_TEMPLATE.format(seed='*')}  {len(SEEDS)} files")

labels = dataset.responsive.values.flatten()
dummy = np.zeros((len(labels), 1))
predictions = np.zeros((len(SEEDS), len(dataset)), dtype=np.int8)
probabilities = np.zeros((len(SEEDS), len(dataset)))
for index, seed in enumerate(SEEDS):
    frame = pd.read_csv(OOF_TEMPLATE.format(seed=seed))
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    order = np.concatenate([validation for _, validation in splitter.split(dummy, labels)])
    assert (frame.responsive.values == labels[order]).all(), f"fold order mismatch {seed}"
    assert (frame.main_receptors_id.values
            == dataset.main_receptors_id.values[order]).all(), f"row order mismatch {seed}"
    predictions[index, order] = frame.Prediction.values
    probabilities[index, order] = frame.Probability.values

positive = dataset.responsive.values == 1
tp_seeds = (predictions == 1).sum(axis=0) * positive
candidates = pd.DataFrame({
    "main_receptors_id": dataset.main_receptors_id.values,
    "main_compounds_id": dataset.main_compounds_id.values,
    "raw_dataset_index": np.arange(len(dataset)),
    "responsive": dataset.responsive.values,
    "tp_seeds": tp_seeds,
    "mean_probability": probabilities.mean(axis=0),
    "sd_probability": probabilities.std(axis=0, ddof=1),
})
candidates = candidates.merge(
    assignment[["main_receptors_id", "gene_name", "organism", "assignment_source",
                "uniparc_upi"]], on="main_receptors_id", how="left")
candidates = candidates.merge(strings, on="main_compounds_id", how="left")

print("\n[2] filter cascade, filters fixed before any attribution was inspected")
stages = []
current = candidates[candidates.responsive == 1]
stages.append(("annotated active pairs", current))
current = current[current.gene_name.fillna("").str.match(CLASS_I_PATTERN)]
stages.append(("C1 Class I gene symbol (OR51/52/55/56)", current))
current = current[current.tp_seeds >= TP_SEED_MINIMUM]
stages.append((f"C2 TP in at least {TP_SEED_MINIMUM} of {len(SEEDS)} seeds", current))
current = current[current.organism == HUMAN_ORGANISM]
stages.append(("C3 human", current))
current = current[current.assignment_source.isin(ARCHIVED_SOURCES)]
stages.append(("C4 UniParc-archived sequence", current))
for label, frame in stages:
    print(f"  {label:44s} {len(frame):6d} pairs  "
          f"{frame.main_receptors_id.nunique():4d} receptors")

ranked = current.sort_values(["tp_seeds", "mean_probability", "sd_probability"],
                             ascending=[False, False, True]).reset_index(drop=True)
ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
ranked["family"] = ranked.gene_name.str.extract(r"^(OR5[1256])")
ranked["ligand_name"] = "NOT DERIVABLE"
ranked["selected"] = (ranked["rank"] == 1)

print("\n[3] top of the ranking")
print(f"  {'rank':>4s} {'gene':10s} {'uniprot':12s} {'smiles':26s} {'TP':>3s} "
      f"{'mean prob':>10s} {'sd':>8s}")
for _, row in ranked.head(5).iterrows():
    print(f"  {row['rank']:4d} {row.gene_name:10s} {str(row.uniparc_upi):12s} "
          f"{str(row.smiles)[:26]:26s} {row.tp_seeds:3d} {row.mean_probability:10.4f} "
          f"{row.sd_probability:8.4f}")

print("\n[4] verification")
record("candidate pairs", EXPECTED_PAIRS, len(ranked), len(ranked) == EXPECTED_PAIRS)
record("candidate receptors", EXPECTED_RECEPTORS, ranked.main_receptors_id.nunique(),
       ranked.main_receptors_id.nunique() == EXPECTED_RECEPTORS)
top = ranked.iloc[0]
record("rank 1 gene", EXPECTED_TOP[0], top.gene_name, top.gene_name == EXPECTED_TOP[0])
record("rank 1 SMILES", EXPECTED_TOP[1], top.smiles, top.smiles == EXPECTED_TOP[1])
record("rank 1 TP seeds", EXPECTED_TOP[2], int(top.tp_seeds), int(top.tp_seeds) == EXPECTED_TOP[2])
record("rank 1 mean probability",
       f"{EXPECTED_TOP[3]:.3f} +/- {EXPECTED_TOP[4]:.3f}",
       f"{top.mean_probability:.3f} +/- {top.sd_probability:.3f}",
       abs(top.mean_probability - EXPECTED_TOP[3]) <= 0.0005
       and abs(top.sd_probability - EXPECTED_TOP[4]) <= 0.0005)

if os.path.exists(STORED_PATH):
    stored = pd.read_csv(STORED_PATH)
    stored_keys = set(zip(stored.main_receptors_id, stored.main_compounds_id))
    new_keys = set(zip(ranked.main_receptors_id, ranked.main_compounds_id))
    record("agreement with the stored candidate list", len(stored_keys),
           len(stored_keys & new_keys), stored_keys == new_keys)

print("\n[5] families represented")
print(f"  {dict(ranked.drop_duplicates('main_receptors_id').family.value_counts())} receptors")
print(f"  {dict(ranked.family.value_counts())} pairs")
print(f"  ligand names: no name field exists in data/pairs.csv or any curated table; "
       f"ligands are identified by compound id and SMILES (NOT DERIVABLE)")

columns = ["rank", "main_receptors_id", "gene_name", "family", "uniparc_upi",
           "main_compounds_id", "ligand_name", "smiles", "tp_seeds", "mean_probability",
           "sd_probability", "selected"]
os.makedirs(OUT_DIR, exist_ok=True)
ranked[columns].to_csv(TABLE_PATH, index=False)
ranked[columns].head(SHORT_LIST_LENGTH).to_csv(SHORT_PATH, index=False)
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[6] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table is written as computed and has not been adjusted")
summary.to_csv(os.path.join(OUT_DIR, "table_S16_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}  ({len(ranked)} rows)")
print(f"  wrote {SHORT_PATH}  ({min(SHORT_LIST_LENGTH, len(ranked))} rows)")

if emit_latex:
    for path, frame, style in ((TABLE_PATH, ranked[columns], "longtable"),
                               (SHORT_PATH, ranked[columns].head(SHORT_LIST_LENGTH), "table")):
        latex_path = path.replace(".csv", ".tex")
        caption = ("All {n} Class I candidate pairs passing the case-selection filters of "
                   "Section~S13.10, in the ranking order used for selection. Filters were "
                   "fixed before any attribution was inspected: Class I family "
                   "(OR51/52/55/56), true positive in at least 80\\% of the 20 seeds, human, "
                   "and UniParc-archived sequence. Ranking is by number of true-positive "
                   "seeds, then mean predicted probability, then low variance. Ligands are "
                   "identified by SMILES because the curated dataset carries no trivial "
                   "names.").format(n=len(frame))
        if style == "table":
            caption = caption.replace("All {} Class I candidate pairs".format(len(ranked)),
                                      f"The {SHORT_LIST_LENGTH} highest-ranked of the "
                                      f"{len(ranked)} Class I candidate pairs")
        with open(latex_path, "w") as handle:
            if style == "longtable":
                handle.write("\\begin{longtable}{rlllrrr}\n")
                handle.write(f"\\caption{{{caption}}}\\label{{tab:class1-candidates}}\\\\\n")
                handle.write("\\toprule\n# & OR & UniParc & SMILES & TP Seeds & "
                             "Mean Probability & SD \\\\\n\\midrule\n\\endfirsthead\n")
                handle.write("\\toprule\n# & OR & UniParc & SMILES & TP Seeds & "
                             "Mean Probability & SD \\\\\n\\midrule\n\\endhead\n")
            else:
                handle.write("\\begin{table}[htbp]\n\\centering\n")
                handle.write(f"\\caption{{{caption}}}\n\\label{{tab:class1-candidates-short}}\n")
                handle.write("\\begin{tabular}{rlllrrr}\n\\toprule\n")
                handle.write("# & OR & UniParc & SMILES & TP Seeds & Mean Probability & "
                             "SD \\\\\n\\midrule\n")
            for _, row in frame.iterrows():
                marker = "$^{\\ast}$" if row.selected else ""
                smiles = str(row.smiles).replace("#", "\\#")
                handle.write(f"{row['rank']}{marker} & {row.gene_name} & {row.uniparc_upi} & "
                             f"\\texttt{{{smiles}}} & {row.tp_seeds} & "
                             f"{row.mean_probability:.4f} & {row.sd_probability:.4f} \\\\\n")
            if style == "longtable":
                handle.write("\\bottomrule\n\\end{longtable}\n")
            else:
                handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        print(f"  wrote {latex_path}")
print("=" * 78)
