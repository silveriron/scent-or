import itertools
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy import stats
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_utils

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
DATA_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted_with_strings.csv")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "thresholds_Scenario5.csv")
TABLE_PATH = os.path.join(OUT_DIR, "table_S6.csv")
PAIR_PATH = os.path.join(OUT_DIR, "stereo_pair_summary.csv")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
N_SPLITS = 10
EXPECTED_PAIRS = 27
EXPECTED_CLASSES = {"enantiomer": 17, "double_bond_isomer": 9, "diastereomer": 1}
EXPECTED_INSTANCES = 129
EXPECTED_RECEPTORS = 126
EXPECTED_UNORDERED = 12
EXPECTED_ORDERED = 15
EXPECTED_MARGINALS = {"ordering_correct": 0.822, "correct_and_split": 0.252,
                      "both_above": 0.216, "correct_and_both_below": 0.395}
MARGINAL_TOLERANCE = 0.001
NAMED_PAIRS = {"menthol": 77, "carvone": 39, "fenchone": 2}

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} expected {expected}   observed {observed}")


print("=" * 78)
print("  T8 - Table S16, stereoisomer ordering by threshold configuration")
print("=" * 78)

RDLogger.DisableLog("rdApp.*")
dataset = pd.read_csv(DATA_PATH)
print(f"\n[1] {DATA_PATH}  {len(dataset)} rows")
ligands = dataset.drop_duplicates("main_compounds_id")[["main_compounds_id", "smiles"]]
print(f"  distinct ligands {len(ligands)}")

def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def mirrored(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(molecule)


def bond_stereo(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return set()
    return {(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetStereo()))
            for bond in molecule.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE}


def skeleton_of(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


skeletons = {}
for compound_id, smiles in zip(ligands.main_compounds_id, ligands.smiles):
    skeleton = skeleton_of(smiles)
    if skeleton is None:
        continue
    skeletons.setdefault(skeleton, []).append((compound_id, smiles))

pairs = []
for skeleton, members in skeletons.items():
    if len(members) < 2:
        continue
    for (first_id, first_smiles), (second_id, second_smiles) in itertools.combinations(members, 2):
        if canonical(first_smiles) == canonical(second_smiles):
            continue
        if mirrored(first_smiles) == canonical(second_smiles):
            stereo_class = "enantiomer"
        elif bond_stereo(first_smiles) == bond_stereo(second_smiles):
            stereo_class = "diastereomer"
        else:
            stereo_class = "double_bond_isomer"
        pairs.append({"compound_a": first_id, "compound_b": second_id,
                      "skeleton": skeleton, "smiles_a": first_smiles,
                      "smiles_b": second_smiles, "stereo_class": stereo_class})
pair_frame = pd.DataFrame(pairs)
print(f"\n[2] stereoisomer enumeration")
print(f"  pairs {len(pair_frame)}   classes {dict(pair_frame.stereo_class.value_counts())}")

print("\n[3] enumeration checks")
record("stereoisomer pairs", EXPECTED_PAIRS, len(pair_frame), len(pair_frame) == EXPECTED_PAIRS)
observed_classes = dict(pair_frame.stereo_class.value_counts())
for name, expected in EXPECTED_CLASSES.items():
    record(f"class {name}", expected, observed_classes.get(name, 0),
           observed_classes.get(name, 0) == expected)

label_lookup = {(receptor, compound): responsive for receptor, compound, responsive
                in zip(dataset.main_receptors_id, dataset.main_compounds_id, dataset.responsive)}
row_lookup = {(receptor, compound): index for index, (receptor, compound)
              in enumerate(zip(dataset.main_receptors_id, dataset.main_compounds_id))}

instances = []
for _, pair in pair_frame.iterrows():
    for receptor in dataset.main_receptors_id.unique():
        first = label_lookup.get((receptor, pair.compound_a))
        second = label_lookup.get((receptor, pair.compound_b))
        if first is None or second is None or first == second:
            continue
        active = pair.compound_a if first == 1 else pair.compound_b
        inactive = pair.compound_b if first == 1 else pair.compound_a
        instances.append({"receptor_id": receptor, "compound_active": active,
                          "compound_inactive": inactive, "stereo_class": pair.stereo_class,
                          "pair_key": tuple(sorted((pair.compound_a, pair.compound_b))),
                          "row_active": row_lookup[(receptor, active)],
                          "row_inactive": row_lookup[(receptor, inactive)]})
instance_frame = pd.DataFrame(instances)
print(f"\n[4] opposite-label instances {len(instance_frame)}")
record("instances", EXPECTED_INSTANCES, len(instance_frame),
       len(instance_frame) == EXPECTED_INSTANCES)
record("receptors", EXPECTED_RECEPTORS, instance_frame.receptor_id.nunique(),
       instance_frame.receptor_id.nunique() == EXPECTED_RECEPTORS)
record("unordered pairs", EXPECTED_UNORDERED, instance_frame.pair_key.nunique(),
       instance_frame.pair_key.nunique() == EXPECTED_UNORDERED)
ordered = set(zip(instance_frame.compound_active, instance_frame.compound_inactive))
record("ordered combinations", EXPECTED_ORDERED, len(ordered), len(ordered) == EXPECTED_ORDERED)

pair_counts = instance_frame.pair_key.value_counts()
print(f"  instances per unordered pair: {sorted(pair_counts.tolist(), reverse=True)}")
top_three = sorted(pair_counts.tolist(), reverse=True)[:3]
record("three largest pairs (menthol, carvone, fenchone)",
       sorted(NAMED_PAIRS.values(), reverse=True), top_three,
       top_three == sorted(NAMED_PAIRS.values(), reverse=True))
singletons = int((pair_counts == 1).sum())
record("unordered pairs contributing one instance", 7, singletons, singletons == 7)

labels = pd.read_csv(os.path.join(BASE_DIR, "data", "final_dataset_weighted.csv"))
label_values = labels["responsive"].values.flatten()
dummy = np.zeros((len(label_values), 1))
thresholds = pd.read_csv(THRESHOLD_PATH)

print("\n[5] seed-instance construction")
records = []
split_fold_instances = 0
for seed in SEEDS:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_of_row = np.empty(len(label_values), dtype=int)
    order = []
    for index, (_, validation) in enumerate(splitter.split(dummy, label_values)):
        fold_of_row[validation] = index + 1
        order.append(validation)
    order = np.concatenate(order)
    oof = pd.read_csv(os.path.join(RESULTS_DIR, f"OOF_CM_Scenario5_seed{seed}.csv"))
    assert (oof.responsive.values == label_values[order]).all(), f"fold order mismatch {seed}"
    probability = np.empty(len(label_values))
    probability[order] = oof.Probability.values
    lookup = thresholds[thresholds.Seed == seed].set_index("Fold").Threshold
    threshold_of_row = lookup.loc[fold_of_row].values
    for _, instance in instance_frame.iterrows():
        active_row, inactive_row = instance.row_active, instance.row_inactive
        if fold_of_row[active_row] != fold_of_row[inactive_row]:
            split_fold_instances += 1
        records.append({
            "Seed": seed, "receptor_id": instance.receptor_id,
            "compound_active": instance.compound_active,
            "compound_inactive": instance.compound_inactive,
            "pair_key": instance.pair_key, "stereo_class": instance.stereo_class,
            "p_active": probability[active_row], "p_inactive": probability[inactive_row],
            "t_active": threshold_of_row[active_row],
            "t_inactive": threshold_of_row[inactive_row],
            "fold_active": fold_of_row[active_row], "fold_inactive": fold_of_row[inactive_row]})
seed_instances = pd.DataFrame(records)
print(f"  seed-instances {len(seed_instances)}")
record("seed-instances", EXPECTED_INSTANCES * len(SEEDS), len(seed_instances),
       len(seed_instances) == EXPECTED_INSTANCES * len(SEEDS))
print(f"  seed-instances whose two rows fall in different folds: {split_fold_instances} "
      f"({split_fold_instances / len(seed_instances):.1%}); each row keeps its own fold threshold")

seed_instances["ordering"] = np.where(
    seed_instances.p_active > seed_instances.p_inactive, "correct",
    np.where(seed_instances.p_active < seed_instances.p_inactive, "inverted", "tie"))
call_active = seed_instances.p_active >= seed_instances.t_active
call_inactive = seed_instances.p_inactive >= seed_instances.t_inactive
seed_instances["threshold_config"] = np.select(
    [~call_active & ~call_inactive, call_active & ~call_inactive,
     ~call_active & call_inactive, call_active & call_inactive],
    ["both_below", "split", "split_inverted", "both_above"], default="undefined")
ties = int((seed_instances.ordering == "tie").sum())
print(f"\n[6] ties in probability: {ties}")

contingency = (seed_instances.groupby(["ordering", "threshold_config"]).size()
               .rename("n_seed_instances").reset_index())
contingency["share"] = contingency.n_seed_instances / len(seed_instances)
print("\n[7] contingency")
print(f"  {'ordering':10s} {'threshold_config':16s} {'n':>6s} {'share':>8s}")
for _, row in contingency.sort_values(["ordering", "threshold_config"]).iterrows():
    print(f"  {row.ordering:10s} {row.threshold_config:16s} {row.n_seed_instances:6d} "
          f"{row.share:8.4f}")
non_zero = set(zip(contingency.ordering, contingency.threshold_config))
impossible = [cell for cell in non_zero
              if (cell[0] == "correct" and cell[1] == "split_inverted")
              or (cell[0] == "inverted" and cell[1] == "split")]
print(f"  logically impossible cells that are non-zero: {impossible if impossible else 'none'}")
print(f"  occupied cells: {len(non_zero)}")


def share(ordering=None, configuration=None):
    mask = np.ones(len(seed_instances), dtype=bool)
    if ordering is not None:
        mask &= (seed_instances.ordering == ordering).values
    if configuration is not None:
        mask &= (seed_instances.threshold_config == configuration).values
    return mask.sum() / len(seed_instances)


print("\n[8] marginal checks")
observed_marginals = {
    "ordering_correct": share(ordering="correct"),
    "correct_and_split": share(ordering="correct", configuration="split"),
    "both_above": share(configuration="both_above"),
    "correct_and_both_below": share(ordering="correct", configuration="both_below"),
}
for name, expected in EXPECTED_MARGINALS.items():
    observed = observed_marginals[name]
    record(name, expected, round(observed, 4), abs(observed - expected) <= MARGINAL_TOLERANCE)
total = contingency.share.sum()
record("shares sum to one", 1.0, round(total, 6), abs(total - 1.0) < 1e-9)

inverted_split = share(ordering="inverted", configuration="split_inverted")
print(f"\n[9] where the 6.9 percent of Section S4.6 sits")
print(f"  inverted ordering, inverted split call            {inverted_split:.4f}")
print(f"  inverted ordering, any threshold configuration    {share(ordering='inverted'):.4f}")
print(f"  correct ordering inside both_above               "
      f"{share(ordering='correct', configuration='both_above'):.4f}")
print(f"  inverted ordering inside both_above               "
      f"{share(ordering='inverted', configuration='both_above'):.4f}")
print(f"  inverted ordering inside both_below               "
      f"{share(ordering='inverted', configuration='both_below'):.4f}")

print("\n[10] pair-level summary")
seed_instances["instance_key"] = list(zip(seed_instances.receptor_id,
                                          seed_instances.compound_active,
                                          seed_instances.compound_inactive))
instance_scores = (seed_instances.groupby(["instance_key", "pair_key", "compound_active",
                                           "compound_inactive", "stereo_class"], sort=False)
                   [["p_active", "p_inactive"]].mean().reset_index())
print(f"  instances aggregated over seeds: {len(instance_scores)}")

pair_rows = []
for pair_key, group in seed_instances.groupby("pair_key"):
    first, second = pair_key
    directions = {}
    for active_id in (first, second):
        subset = group[group.compound_active == active_id]
        directions[active_id] = (len(subset) // len(SEEDS),
                                 float((subset.p_active > subset.p_inactive).mean())
                                 if len(subset) else float("nan"))
    a1 = directions[first][1] if not np.isnan(directions[first][1]) else 0.0
    a2 = directions[second][1] if not np.isnan(directions[second][1]) else 0.0
    instances_here = instance_scores[instance_scores.pair_key == pair_key]
    first_higher = 0
    for _, instance in instances_here.iterrows():
        score_first = (instance.p_active if instance.compound_active == first
                       else instance.p_inactive)
        score_second = (instance.p_active if instance.compound_active == second
                        else instance.p_inactive)
        first_higher += int(score_first > score_second)
    total_instances = len(instances_here)
    consistency = max(first_higher, total_instances - first_higher) / total_instances
    pair_rows.append({
        "pair_id": f"{first}|{second}", "ligand_A": first, "ligand_B": second,
        "n_instances": total_instances,
        "n_dir_A_active": directions[first][0], "n_dir_B_active": directions[second][0],
        "a1": a1, "a2": a2, "sigma": a1 + a2, "ordering_consistency": consistency,
        "n_instances_favouring_A": first_higher,
        "stereo_class": group.stereo_class.iloc[0],
        "both_directions": directions[first][0] > 0 and directions[second][0] > 0})
pair_summary = pd.DataFrame(pair_rows).sort_values("n_instances", ascending=False)
print(f"  {'pair':14s} {'n':>4s} {'dirA':>5s} {'dirB':>5s} {'a1':>7s} {'a2':>7s} "
      f"{'sigma':>7s} {'consistency':>12s}")
for _, row in pair_summary.iterrows():
    print(f"  {row.pair_id:14s} {row.n_instances:4d} {row.n_dir_A_active:5d} "
          f"{row.n_dir_B_active:5d} {row.a1:7.3f} {row.a2:7.3f} {row.sigma:7.3f} "
          f"{row.ordering_consistency:12.3f}")

doubly = pair_summary[pair_summary.both_directions]
print(f"\n[11] pair-level checks")
sigmas = sorted(doubly.sigma.round(3).tolist(), reverse=True)
record("sigma of the doubly observed pairs", [1.001, 0.857, 0.15], sigmas,
       sigmas == [1.001, 0.857, 0.15])
median_consistency = float(pair_summary.ordering_consistency.median())
record("median ordering consistency (instance level)", 1.0,
       round(median_consistency, 3), abs(median_consistency - 1.0) < 5e-4)
above_095 = int((pair_summary.ordering_consistency >= 0.95).sum())
record("pairs with ordering consistency at or above 0.95", 10, above_095, above_095 == 10)
label_balanced = float((doubly.sigma / 2).mean())
record("label-balanced accuracy, unweighted", 0.335, round(label_balanced, 3),
       abs(label_balanced - 0.335) <= 0.001)
weights = doubly.n_instances.values
weighted = float(np.average((doubly.sigma / 2).values, weights=weights))
record("label-balanced accuracy, instance weighted", 0.47, round(weighted, 3),
       abs(weighted - 0.470) <= 0.001)

combination_accuracy = (seed_instances.groupby(["compound_active", "compound_inactive"])
                        .apply(lambda g: float((g.p_active > g.p_inactive).mean()),
                               include_groups=False))
combination_median = float(combination_accuracy.median())
q1, q3 = np.percentile(combination_accuracy.values, [25, 75])
wilcoxon_p = stats.wilcoxon(combination_accuracy.values - 0.5)[1]
record("ordered combination median", 0.2, round(combination_median, 3),
       abs(combination_median - 0.200) <= 0.001)
record("ordered combination IQR", [0.05, 0.878], [round(q1, 3), round(q3, 3)],
       abs(q1 - 0.050) <= 0.001 and abs(q3 - 0.878) <= 0.001)
record("ordered combination Wilcoxon vs 0.5", 0.649, round(float(wilcoxon_p), 3),
       abs(wilcoxon_p - 0.649) <= 0.001)

unordered_accuracy = (seed_instances.groupby("pair_key")
                      .apply(lambda g: float((g.p_active > g.p_inactive).mean()),
                             include_groups=False))
unordered_median = float(unordered_accuracy.median())
uq1, uq3 = np.percentile(unordered_accuracy.values, [25, 75])
record("unordered pair median", 0.4, round(unordered_median, 3),
       abs(unordered_median - 0.400) <= 0.001)
record("unordered pair IQR", [0.169, 0.904], [round(uq1, 3), round(uq3, 3)],
       abs(uq1 - 0.169) <= 0.001 and abs(uq3 - 0.904) <= 0.001)

per_seed_correct = seed_instances.groupby("Seed").apply(
    lambda g: float((g.ordering == "correct").mean()), include_groups=False)
per_seed_strict = seed_instances.groupby("Seed").apply(
    lambda g: float((g.threshold_config == "split").mean()), include_groups=False)
record("case-weighted ranking accuracy", "0.822 +/- 0.018",
       f"{per_seed_correct.mean():.3f} +/- {per_seed_correct.std(ddof=1):.3f}",
       abs(per_seed_correct.mean() - 0.822) <= 0.001
       and abs(per_seed_correct.std(ddof=1) - 0.018) <= 0.001)
record("strict separation", "0.252 +/- 0.029",
       f"{per_seed_strict.mean():.3f} +/- {per_seed_strict.std(ddof=1):.3f}",
       abs(per_seed_strict.mean() - 0.252) <= 0.001
       and abs(per_seed_strict.std(ddof=1) - 0.029) <= 0.001)

print("\n[12] instance count ratio quoted as a factor of 70")
counts = doubly.n_instances.values
directional = []
for _, row in doubly.iterrows():
    directional.extend([row.n_dir_A_active, row.n_dir_B_active])
directional = [value for value in directional if value > 0]
print(f"  pair level   max/min = {counts.max()}/{counts.min()} = {counts.max() / counts.min():.1f}")
print(f"  direction level max/min = {max(directional)}/{min(directional)} = "
      f"{max(directional) / min(directional):.1f}")

os.makedirs(OUT_DIR, exist_ok=True)
contingency.to_csv(TABLE_PATH, index=False)
pair_summary.to_csv(PAIR_PATH, index=False)
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[13] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table below is written as computed and has not been adjusted")
    for _, row in summary[summary.Verdict == "FAIL"].iterrows():
        print(f"    {row.Check}: expected {row.Expected}, observed {row.Observed}")
summary.to_csv(os.path.join(OUT_DIR, "table_S6_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}")
print(f"  wrote {PAIR_PATH}")
if emit_latex:
    latex_path = TABLE_PATH.replace(".csv", ".tex")
    order_names = {"correct": "Correct", "inverted": "Inverted", "tie": "Tie"}
    config_names = {"both_below": "Both below threshold", "split": "Split, correct direction",
                    "split_inverted": "Split, inverted direction",
                    "both_above": "Both above threshold"}
    layout = ["both_below", "split", "split_inverted", "both_above"]
    counts = {(row.ordering, row.threshold_config): (row.n_seed_instances, row.share)
              for _, row in contingency.iterrows()}
    with open(latex_path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write("\\caption{Exhaustive cross-classification of the "
                     f"{len(seed_instances):,} stereoisomer seed-instances "
                     f"({EXPECTED_INSTANCES} opposite-label instances $\\times$ "
                     f"{len(SEEDS)} seeds) by probability ordering and by the binary calls "
                     "at the fold-specific thresholds. Ordering is threshold-independent; "
                     "the threshold configuration records where the two probabilities fall "
                     "relative to their own fold thresholds. Cells that are logically "
                     "excluded are marked with a dash. Each row of a seed-instance keeps the "
                     f"threshold of its own fold; {split_fold_instances / len(seed_instances):.1%} "
                     "of seed-instances have their two rows in different folds.}\n")
        handle.write("\\label{tab:stereo-contingency}\n")
        handle.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        handle.write("Threshold configuration & \\multicolumn{2}{c}{Correct ordering} & "
                     "\\multicolumn{2}{c}{Inverted ordering} \\\\\n")
        handle.write("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n")
        handle.write(" & $n$ & Share & $n$ & Share \\\\\n\\midrule\n")
        for configuration in layout:
            cells = []
            for ordering in ("correct", "inverted"):
                value = counts.get((ordering, configuration))
                cells.append("-- & --" if value is None
                             else f"{value[0]:,} & {value[1]:.4f}")
            handle.write(f"{config_names[configuration]} & {cells[0]} & {cells[1]} \\\\\n")
        handle.write("\\midrule\n")
        correct_total = int((seed_instances.ordering == "correct").sum())
        inverted_total = int((seed_instances.ordering == "inverted").sum())
        handle.write(f"Total & {correct_total:,} & {correct_total / len(seed_instances):.4f} & "
                     f"{inverted_total:,} & {inverted_total / len(seed_instances):.4f} "
                     "\\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {latex_path}")
print("=" * 78)
