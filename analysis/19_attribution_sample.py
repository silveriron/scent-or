import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
SCREEN_PATH = os.path.join(RESULTS_DIR, "screening", "robust_tp_pairs.csv")
ASSIGNMENT_PATH = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")
STRINGS_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted_with_strings.csv")
STEREO_PATH = os.path.join(OUT_DIR, "T5_stereoisomer_pairs.csv")
STORED_SELECTED_PATH = os.path.join(OUT_DIR, "T9b_selected.csv")
TABLE_PATH = os.path.join(OUT_DIR, "table_S13.csv")

MIN_LIGANDS = 3
MAX_LIGANDS_PER_RECEPTOR = 4
TARGET_RECEPTORS = 18
CLASS_I_PATTERN = re.compile(r"^OR(51|52|55|56)")
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
ARCHIVED_PREFIX = "uniparc_exact"
EXPECTED_SELECTED = 18
EXPECTED_PAIRS = 72
EXPECTED_WITHIN = 108
EXPECTED_BETWEEN = 2448
EXPECTED_TANIMOTO_MEAN = 0.112
EXPECTED_TANIMOTO_RANGE = (0.029, 0.478)
SI_CANDIDATE_CLAIM = 43

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:48s} expected {expected}   observed {observed}")


print("=" * 78)
print("  T4 - Table S12, candidate ORs for the attribution sample")
print("=" * 78)

RDLogger.DisableLog("rdApp.*")
screen = pd.read_csv(SCREEN_PATH)
assignment = pd.read_csv(ASSIGNMENT_PATH).set_index("main_receptors_id")
strings = pd.read_csv(STRINGS_PATH, usecols=["main_compounds_id", "smiles"]).drop_duplicates()
stereo = pd.read_csv(STEREO_PATH)
print(f"\n[1] inputs")
print(f"  {SCREEN_PATH}  {len(screen)} pairs")
print(f"  {ASSIGNMENT_PATH}  {len(assignment)} receptors")
print(f"  {STEREO_PATH}  {len(stereo)} stereoisomer pairs")

smiles_of = dict(zip(strings.main_compounds_id, strings.smiles))
screen["gene_name"] = screen.main_receptors_id.map(assignment.gene_name)
screen["is_human"] = screen.main_receptors_id.map(assignment.is_human).fillna(False)
screen["assignment_source"] = screen.main_receptors_id.map(assignment.assignment_source)
screen["uniparc_upi"] = screen.main_receptors_id.map(assignment.uniparc_upi)
screen["receptor_class"] = np.where(
    screen.gene_name.astype(str).str.upper().str.match(CLASS_I_PATTERN), "I", "II")

print("\n[2] filter cascade, filters fixed before any attribution was inspected")
stages = [("robust TP pairs on file", screen)]
current = screen[screen.TP_Count == 20]
stages.append(("C1 true positive in all 20 seeds", current))
current = current[current.is_human]
stages.append(("C2 human receptor", current))
current = current[current.assignment_source.astype(str).str.startswith(ARCHIVED_PREFIX)]
stages.append(("C3 sequence archived verbatim in UniParc", current))
for label, frame in stages:
    print(f"  {label:44s} {len(frame):6d} pairs  "
          f"{frame.main_receptors_id.nunique():4d} receptors")

generator = rdFingerprintGenerator.GetMorganGenerator(radius=FINGERPRINT_RADIUS,
                                                      fpSize=FINGERPRINT_BITS)
fingerprints = {}
for compound_id, smiles in smiles_of.items():
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is not None:
        fingerprints[compound_id] = generator.GetFingerprint(molecule)

stereo_groups = {}
for group_id, row in stereo.iterrows():
    stereo_groups.setdefault(int(row.compound_a), set()).add(group_id)
    stereo_groups.setdefault(int(row.compound_b), set()).add(group_id)


def tanimoto(first, second):
    if first not in fingerprints or second not in fingerprints:
        return 1.0
    return DataStructs.TanimotoSimilarity(fingerprints[first], fingerprints[second])


def pick_diverse(candidates, count):
    if len(candidates) <= count:
        return list(candidates)
    start = max(candidates,
                key=lambda c: sum(1 - tanimoto(c, other) for other in candidates if other != c))
    chosen = [start]
    while len(chosen) < count:
        chosen.append(max((c for c in candidates if c not in chosen),
                          key=lambda c: min(1 - tanimoto(c, s) for s in chosen)))
    return chosen


print("\n[3] per-receptor ligand selection")
records = []
for receptor_id, group in current.groupby("main_receptors_id"):
    available = sorted(group.main_compounds_id.unique())
    kept, used_groups = [], set()
    ordering = sorted(available,
                      key=lambda c: -float(group[group.main_compounds_id == c].Mean_Prob.iloc[0]))
    for compound in ordering:
        groups = stereo_groups.get(compound, set())
        if groups & used_groups:
            continue
        used_groups |= groups
        kept.append(compound)
    if len(kept) < MIN_LIGANDS:
        continue
    chosen = pick_diverse(kept, MAX_LIGANDS_PER_RECEPTOR)
    similarities = [tanimoto(a, b) for a, b in itertools.combinations(chosen, 2)]
    records.append({
        "main_receptors_id": int(receptor_id),
        "or_name": group.gene_name.iloc[0],
        "uniparc_upi": group.uniparc_upi.iloc[0],
        "or_class": group.receptor_class.iloc[0],
        "n_distinct_ligands": len(kept),
        "n_robust_tp_pairs": len(group),
        "n_ligands_selected": len(chosen),
        "selected_compounds": ";".join(str(c) for c in chosen),
        "mean_within_or_tanimoto": float(np.mean(similarities)) if similarities else np.nan,
        "n_excluded_as_stereoisomer": len(available) - len(kept)})

candidates = (pd.DataFrame(records)
              .sort_values(["n_distinct_ligands", "mean_within_or_tanimoto"],
                           ascending=[False, True])
              .reset_index(drop=True))
print(f"  receptors with at least {MIN_LIGANDS} non-stereoisomeric ligands: {len(candidates)}")
print(f"  class composition {dict(candidates.or_class.value_counts())}")
print(f"  ligands dropped as stereoisomer duplicates "
      f"{int(candidates.n_excluded_as_stereoisomer.sum())}")

selected_ids = list(candidates.head(TARGET_RECEPTORS).main_receptors_id)
for receptor_class in ("I", "II"):
    if receptor_class not in set(candidates.head(TARGET_RECEPTORS).or_class):
        extra = candidates[candidates.or_class == receptor_class].head(1)
        selected_ids.append(int(extra.main_receptors_id.iloc[0]))
        print(f"  class {receptor_class} absent from the top {TARGET_RECEPTORS}; "
              f"added receptor {int(extra.main_receptors_id.iloc[0])}")
candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
candidates["selected"] = candidates.main_receptors_id.isin(selected_ids)

selected = candidates[candidates.selected]
pair_count = int(selected.n_ligands_selected.sum())
within = int(sum(n * (n - 1) // 2 for n in selected.n_ligands_selected))
between = pair_count * (pair_count - 1) // 2 - within

print("\n[4] verification")
record("SI claim on the candidate OR count", SI_CANDIDATE_CLAIM, len(candidates),
       len(candidates) == SI_CANDIDATE_CLAIM)
record("selected receptors", EXPECTED_SELECTED, len(selected), len(selected) == EXPECTED_SELECTED)
record("every selected receptor carries exactly four ligands", MAX_LIGANDS_PER_RECEPTOR,
       sorted(selected.n_ligands_selected.unique()),
       set(selected.n_ligands_selected) == {MAX_LIGANDS_PER_RECEPTOR})
record("analysed pairs", EXPECTED_PAIRS, pair_count, pair_count == EXPECTED_PAIRS)
record("within-OR comparisons", EXPECTED_WITHIN, within, within == EXPECTED_WITHIN)
record("between-OR comparisons", EXPECTED_BETWEEN, between, between == EXPECTED_BETWEEN)
mean_tanimoto = float(selected.mean_within_or_tanimoto.mean())
record("mean within-OR Tanimoto over the selected ORs", EXPECTED_TANIMOTO_MEAN,
       round(mean_tanimoto, 3), abs(mean_tanimoto - EXPECTED_TANIMOTO_MEAN) <= 0.0005)
low = float(selected.mean_within_or_tanimoto.min())
high = float(selected.mean_within_or_tanimoto.max())
record("within-OR Tanimoto range", list(EXPECTED_TANIMOTO_RANGE), [round(low, 3), round(high, 3)],
       abs(low - EXPECTED_TANIMOTO_RANGE[0]) <= 0.0005
       and abs(high - EXPECTED_TANIMOTO_RANGE[1]) <= 0.0005)
median_tanimoto = float(selected.mean_within_or_tanimoto.median())
print(f"  for reference the median of the same quantity is {median_tanimoto:.3f}")

if os.path.exists(STORED_SELECTED_PATH):
    stored = pd.read_csv(STORED_SELECTED_PATH)
    stored_pairs = set(zip(stored.main_receptors_id, stored.main_compounds_id))
    new_pairs = set()
    for _, row in selected.iterrows():
        for compound in row.selected_compounds.split(";"):
            new_pairs.add((row.main_receptors_id, int(compound)))
    record("agreement with the stored 72-pair sample", len(stored_pairs),
           len(stored_pairs & new_pairs), stored_pairs == new_pairs)

print("\n[5] candidate table, ranking order")
print(f"  {'#':>3s} {'OR':10s} {'UniParc':14s} {'Cls':3s} {'Lig':>4s} {'TP pairs':>9s} "
      f"{'Tanimoto':>9s} {'sel':>4s}")
for _, row in candidates.iterrows():
    print(f"  {row['rank']:3d} {str(row.or_name):10s} {str(row.uniparc_upi):14s} "
          f"{row.or_class:3s} {row.n_distinct_ligands:4d} {row.n_robust_tp_pairs:9d} "
          f"{row.mean_within_or_tanimoto:9.3f} {'yes' if row.selected else '':>4s}")

columns = ["rank", "main_receptors_id", "or_name", "uniparc_upi", "or_class",
           "n_distinct_ligands", "n_robust_tp_pairs", "n_ligands_selected",
           "mean_within_or_tanimoto", "selected_compounds", "selected"]
os.makedirs(OUT_DIR, exist_ok=True)
candidates[columns].to_csv(TABLE_PATH, index=False)
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[6] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table is written as computed and has not been adjusted")
    for _, row in summary[summary.Verdict == "FAIL"].iterrows():
        print(f"    {row.Check}: expected {row.Expected}, observed {row.Observed}")
summary.to_csv(os.path.join(OUT_DIR, "table_S13_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}")

if emit_latex:
    latex_path = TABLE_PATH.replace(".csv", ".tex")
    with open(latex_path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\\footnotesize\n")
        handle.write(
            "\\caption{All " + str(len(candidates)) + " ORs passing the sample selection "
            "filters of Section~S13.1, in the ranking order used for selection. Filters were "
            "fixed before any attribution was inspected: true positive in all 20 seeds, human "
            "receptor, sequence archived verbatim in UniParc, and at least three chemically "
            "distinct ligands after stereoisomeric duplicates are removed. Receptors are "
            "ranked by number of qualifying ligands and then by increasing mean pairwise "
            "Tanimoto similarity; the " + str(len(selected)) + " highest-ranked, marked in "
            "the last column, contribute four ligands each to the " + str(pair_count) +
            " analysed pairs. One receptor has no resolved gene symbol and is listed by its "
            "UniParc identifier alone.}\n")
        handle.write("\\begin{tabular}{rllcrrrc}\n\\toprule\n")
        handle.write("\\# & OR & UniParc & Class & Distinct Ligands & "
                     "Robust TP Pairs (20/20 Seeds) & Mean Within-OR Tanimoto & Selected "
                     "\\\\\n\\midrule\n")
        for _, row in candidates.iterrows():
            handle.write(f"{row['rank']} & {row.or_name} & {row.uniparc_upi} & "
                         f"{row.or_class} & {row.n_distinct_ligands} & "
                         f"{row.n_robust_tp_pairs} & "
                         f"{row.mean_within_or_tanimoto:.3f} & "
                         f"{'$\\checkmark$' if row.selected else ''} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {latex_path}")
print("=" * 78)
