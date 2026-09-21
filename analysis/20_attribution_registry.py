import itertools
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_utils

OUT_DIR = os.path.join(BASE_DIR, "results", "revision")
STAGE_DIR = os.path.join(OUT_DIR, "attribution_registry")
SELECTED_PATH = os.path.join(OUT_DIR, "T9b_selected.csv")
DATASET_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted_with_strings.csv")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
ALPHA = 0.05
EQUIVALENCE_BOUND = 0.05
SKEW_THRESHOLD = 1.0
UNTRAINED_CORRELATION = 0.153
CONDITIONS = [
    {"key": "ig_or", "label": "IG / OR side", "directory": "T9b_ig",
     "template": "IG_protein_seed_{seed}_idx_{index}.npy", "cluster": "receptor",
     "truncate": True, "published_floor": 0.571, "published_convention": "median"},
    {"key": "muta_or", "label": "dP / OR side", "directory": "T10b_muta",
     "template": "Muta_score_seed_{seed}_idx_{index}.npy", "cluster": "receptor",
     "truncate": True, "published_floor": 0.640, "published_convention": "median"},
    {"key": "ig_ligand", "label": "IG / Ligand side", "directory": "T9b_ig",
     "template": "IG_ligand_seed_{seed}_idx_{index}.npy", "cluster": "ligand",
     "truncate": False, "published_floor": 0.876, "published_convention": "median"},
]

print("=" * 78)
print("  Stage 2 - across-seed floor unified on the median")
print("=" * 78)
print(f"\n  skewness criterion fixed before inspecting results: |skew| > {SKEW_THRESHOLD}")
print("  or a sign disagreement between median(d) and the Hodges-Lehmann estimate")

selected = pd.read_csv(SELECTED_PATH)
dataset = pd.read_csv(DATASET_PATH)
indices = selected.raw_dataset_index.values
receptor_of = dict(zip(selected.raw_dataset_index, selected.main_receptors_id))
ligand_of = dict(zip(selected.raw_dataset_index, selected.main_compounds_id))
symbol_of = dict(zip(selected.main_receptors_id, selected.gene_name))
sequence_length = {index: len(dataset.iloc[index].mutated_sequence) for index in indices}

RDLogger.DisableLog("rdApp.*")
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fingerprints = {}
for compound_id, smiles in zip(dataset.main_compounds_id, dataset.smiles):
    if compound_id in fingerprints:
        continue
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is not None:
        fingerprints[compound_id] = generator.GetFingerprint(molecule)


def spearman(first, second):
    length = min(len(first), len(second))
    return float(stats.spearmanr(first[:length], second[:length]).statistic)


def hl_ci_indexed(differences, alpha=ALPHA, mode="normal", indexing="one_based"):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    lower, upper = np.triu_indices(n)
    walsh = np.sort((values[lower] + values[upper]) / 2.0)
    total = len(walsh)
    if mode == "normal":
        centre = n * (n + 1) / 4.0
        spread = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        k = int(np.floor(centre - stats.norm.ppf(1 - alpha / 2) * spread))
    else:
        k = exact_critical(n, alpha)
    if k < 1:
        return float(walsh[0]), float(walsh[-1]), k
    if indexing == "one_based":
        return float(walsh[k - 1]), float(walsh[total - k]), k
    return float(walsh[k]), float(walsh[total - k - 1]), k


def exact_critical(n, alpha=ALPHA):
    distribution = np.zeros(n * (n + 1) // 2 + 1)
    distribution[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(distribution)
        shifted[rank:] = distribution[:-rank]
        distribution = distribution + shifted
    distribution /= distribution.sum()
    cumulative = np.cumsum(distribution)
    candidates = np.nonzero(cumulative <= alpha / 2.0)[0]
    return int(candidates[-1]) + 1 if len(candidates) else 0


def sign_test(differences):
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    positive = int((values > 0).sum())
    return positive, len(values), float(stats.binomtest(positive, len(values), 0.5).pvalue)


print("\n[T1] floor definition, recomputed from the stored profiles")
print("  per pair: the C(20,2) = 190 Spearman correlations between the 20 per-seed")
print("  absolute profiles, reduced to one number; per within-pair comparison the")
print("  paired floor is the arithmetic mean of the two members' reduced values,")
print("  matching revision_10b_analysis.py:183 and revision_11b_ligand_specificity.py:152")

raw_rows = []
for condition in CONDITIONS:
    directory = os.path.join(OUT_DIR, condition["directory"])
    per_seed, mean_profile = {}, {}
    for index in indices:
        limit = sequence_length[index] if condition["truncate"] else None
        stack = np.array([np.load(os.path.join(
            directory, condition["template"].format(seed=seed, index=index)))[:limit]
            for seed in SEEDS])
        per_seed[index] = np.abs(stack)
        mean_profile[index] = np.abs(stack.mean(axis=0))
    counts = set()
    across_seed = {}
    for index in indices:
        values = [spearman(per_seed[index][i], per_seed[index][j])
                  for i, j in itertools.combinations(range(len(SEEDS)), 2)]
        counts.add(len(values))
        across_seed[index] = values
    floor_mean = {index: float(np.mean(v)) for index, v in across_seed.items()}
    floor_median = {index: float(np.median(v)) for index, v in across_seed.items()}
    print(f"\n  {condition['label']}: seed-pair comparisons per pair {sorted(counts)}")

    grouping = receptor_of if condition["cluster"] == "receptor" else ligand_of
    for first, second in itertools.combinations(indices, 2):
        if grouping[first] != grouping[second]:
            continue
        value = spearman(mean_profile[first], mean_profile[second])
        raw_rows.append({
            "condition": condition["label"], "cluster_unit": condition["cluster"],
            "cluster_id": grouping[first],
            "or_symbol": symbol_of.get(receptor_of[first], ""),
            "ligand_a": ligand_of[first], "ligand_b": ligand_of[second],
            "pair_a_idx": first, "pair_b_idx": second, "rho_within": value,
            "rho_floor_median": (floor_median[first] + floor_median[second]) / 2,
            "rho_floor_mean": (floor_mean[first] + floor_mean[second]) / 2,
            "tanimoto": DataStructs.TanimotoSimilarity(fingerprints[ligand_of[first]],
                                                       fingerprints[ligand_of[second]])})
raw = pd.DataFrame(raw_rows)
raw["d_median_floor"] = raw.rho_within - raw.rho_floor_median
raw["d_mean_floor"] = raw.rho_within - raw.rho_floor_mean

print(f"\n[T1] floor marginal values and the preservation check")
print(f"  {'condition':18s} {'published':>10s} {'conv':>7s} {'mean floor':>11s} "
      f"{'median floor':>13s} {'IQR (median)':>22s} verdict")
stop = False
for condition in CONDITIONS:
    frame = raw[raw.condition == condition["label"]]
    mean_marginal = float(frame.rho_floor_mean.median())
    median_marginal = float(frame.rho_floor_median.median())
    low = float(frame.rho_floor_median.quantile(0.25))
    high = float(frame.rho_floor_median.quantile(0.75))
    reproduced = mean_marginal if condition["published_convention"] == "mean" else median_marginal
    agrees = abs(reproduced - condition["published_floor"]) <= 5e-4
    preserved = (condition["published_convention"] == "mean"
                 or abs(median_marginal - condition["published_floor"]) <= 5e-4)
    if not agrees or not preserved:
        stop = True
    print(f"  {condition['label']:18s} {condition['published_floor']:10.3f} "
          f"{condition['published_convention']:>7s} {mean_marginal:11.4f} "
          f"{median_marginal:13.4f} "
          f"{'[' + format(low, '.4f') + ', ' + format(high, '.4f') + ']':>22s} "
          f"{'ok' if agrees and preserved else 'STOP'}")
if stop:
    raise SystemExit("floor preservation check failed, see the stop conditions of the brief")
print("  the two conditions already on the median are preserved exactly; "
      "only IG / OR side moves")

print("\n[T2] does the within-OR correlation still exceed its paired floor everywhere")
violation_rows = []
for condition in CONDITIONS:
    frame = raw[raw.condition == condition["label"]]
    for floor_name, column in (("published", "rho_floor_mean" if
                                condition["published_convention"] == "mean"
                                else "rho_floor_median"),
                               ("unified median", "rho_floor_median")):
        exceeding = int((frame.rho_within > frame[column]).sum())
        print(f"  {condition['label']:18s} floor={floor_name:15s} "
              f"{exceeding}/{len(frame)} comparisons exceed their floor")
    losers = frame[frame.rho_within <= frame.rho_floor_median]
    for _, row in losers.iterrows():
        violation_rows.append({
            "condition": row.condition, "or_symbol": row.or_symbol,
            "cluster_id": row.cluster_id, "ligand_a": row.ligand_a,
            "ligand_b": row.ligand_b, "rho_within": row.rho_within,
            "rho_floor": row.rho_floor_median, "d": row.d_median_floor})
violations = pd.DataFrame(violation_rows)
if len(violations):
    print(f"\n  comparisons that no longer exceed the unified floor: {len(violations)}")
    print(violations.to_string(index=False))
else:
    print("\n  every within-cluster comparison still exceeds its paired floor")

print("\n[T3] paired differences under the unified median floor")
summary_rows = []
for condition in CONDITIONS:
    frame = raw[raw.condition == condition["label"]]
    blocks = [("comparison", "profile pair", frame.d_median_floor.values,
               frame.rho_within, frame.rho_floor_median, len(frame))]
    grouped = frame.groupby("cluster_id")
    reduced_within = grouped.rho_within.median()
    reduced_floor = grouped.rho_floor_median.median()
    blocks.append((f"cluster:{condition['cluster']}", condition["cluster"],
                   (reduced_within - reduced_floor).values,
                   reduced_within, reduced_floor, len(reduced_within)))
    blocks.append((f"cluster_median_of_d:{condition['cluster']}", condition["cluster"],
                   grouped.d_median_floor.median().values,
                   reduced_within, reduced_floor, len(reduced_within)))
    for level, unit, differences, within_values, floor_values, count in blocks:
        normal_low, normal_high, normal_k = hl_ci_indexed(differences, mode="normal")
        exact_low, exact_high, exact_k = hl_ci_indexed(differences, mode="exact",
                                                       indexing="zero_based")
        positive, tested, sign_p = sign_test(differences)
        skew = float(stats.skew(differences))
        hodges = stats_utils.hodges_lehmann(differences)
        median_value = float(np.median(differences))
        summary_rows.append({
            "condition": condition["label"], "level": level, "unit": unit, "n": count,
            "marginal_median_within": float(np.median(within_values)),
            "iqr_within_low": float(np.percentile(within_values, 25)),
            "iqr_within_high": float(np.percentile(within_values, 75)),
            "marginal_median_floor": float(np.median(floor_values)),
            "iqr_floor_low": float(np.percentile(floor_values, 25)),
            "iqr_floor_high": float(np.percentile(floor_values, 75)),
            "median_d": median_value, "hodges_lehmann": hodges,
            "ci_low_normal": normal_low, "ci_high_normal": normal_high,
            "critical_k_normal": normal_k,
            "ci_low_exact": exact_low, "ci_high_exact": exact_high,
            "critical_k_exact": exact_k,
            "wilcoxon_p": float(stats.wilcoxon(differences)[1]),
            "skewness": skew,
            "sign_positive": positive, "sign_n": tested, "sign_p": sign_p,
            "markedly_skewed": bool(abs(skew) > SKEW_THRESHOLD
                                    or np.sign(median_value) != np.sign(hodges)),
            "n_exceeding_floor": int((differences > 0).sum()),
            "spearman_vs_tanimoto": (float("nan") if frame.tanimoto.nunique() <= 1
                                     or level != "comparison"
                                     else float(stats.spearmanr(frame.rho_within,
                                                                frame.tanimoto).statistic))})
        row = summary_rows[-1]
        print(f"  {condition['label']:18s} {level:28s} n {count:4d}  "
              f"within {row['marginal_median_within']:.4f}  floor {row['marginal_median_floor']:.4f}  "
              f"median(d) {median_value:+.4f}  HL {hodges:+.4f}  "
              f"[{normal_low:+.4f}, {normal_high:+.4f}]  p {row['wilcoxon_p']:.3e}")
summary = pd.DataFrame(summary_rows)

print("\n[T3] Hodges-Lehmann confidence interval index check")
print(f"  {'n':>4s} {'k normal':>9s} {'k exact':>8s} {'one-based CI (normal k)':>34s} "
      f"{'zero-based CI (normal k)':>34s}")
for count in sorted({int(value) for value in summary.n}):
    frame = summary[summary.n == count].iloc[0]
    differences_source = raw[raw.condition == frame.condition]
    if frame.level == "comparison":
        differences = differences_source.d_median_floor.values
    else:
        grouped = differences_source.groupby("cluster_id")
        differences = (grouped.rho_within.median() - grouped.rho_floor_median.median()).values
    one_low, one_high, k_normal = hl_ci_indexed(differences, mode="normal",
                                                indexing="one_based")
    zero_low, zero_high, _ = hl_ci_indexed(differences, mode="normal",
                                           indexing="zero_based")
    k_exact = exact_critical(count)
    print(f"  {count:4d} {k_normal:9d} {k_exact:8d} "
          f"{'[' + format(one_low, '+.4f') + ', ' + format(one_high, '+.4f') + ']':>34s} "
          f"{'[' + format(zero_low, '+.4f') + ', ' + format(zero_high, '+.4f') + ']':>34s}")
print("  revision_12c_effect_sizes.py uses the one-based form and reproduces the")
print("  published intervals; the snippet in the first brief uses the zero-based form")

print("\n[T4] skewness and the sign test")
print(f"  {'condition':18s} {'level':28s} {'skew':>8s} {'median':>9s} {'HL':>9s} "
      f"{'flagged':>8s} {'sign test':>22s}")
for _, row in summary.iterrows():
    print(f"  {row.condition:18s} {row.level:28s} {row.skewness:8.3f} "
          f"{row.median_d:+9.4f} {row.hodges_lehmann:+9.4f} "
          f"{str(row.markedly_skewed):>8s} "
          f"{format(row.sign_positive, 'd') + '/' + format(row.sign_n, 'd') + ' p=' + format(row.sign_p, '.3f'):>22s}")

print("\n[T5] the equivalence bound and the untrained-control comparison")
for _, row in summary[summary.condition == "dP / OR side"].iterrows():
    worst = max(row.ci_high_normal, row.ci_high_exact)
    print(f"  {row.level:28s} CI upper normal {row.ci_high_normal:+.4f}  "
          f"exact {row.ci_high_exact:+.4f}  "
          f"{'within 0.05' if worst < EQUIVALENCE_BOUND else 'EXCEEDS 0.05'}")
new_floor = float(summary[(summary.condition == "IG / OR side")
                          & (summary.level == "comparison")].marginal_median_floor.iloc[0])
print(f"  SI S13.5 compares the trained-to-untrained correlation {UNTRAINED_CORRELATION} "
      f"with the IG floor")
print(f"    across-seed floor {new_floor:.4f}; "
      f"{UNTRAINED_CORRELATION} is {'still well below' if UNTRAINED_CORRELATION < new_floor * 0.5 else 'NOT clearly below'} it")

os.makedirs(STAGE_DIR, exist_ok=True)
raw.to_csv(os.path.join(STAGE_DIR, "pairwise_raw_unified.csv"), index=False)
summary.to_csv(os.path.join(STAGE_DIR, "summary_unified.csv"), index=False)
violations.to_csv(os.path.join(STAGE_DIR, "floor_violations.csv"), index=False)
print(f"\n[out] wrote {STAGE_DIR}/pairwise_raw_unified.csv ({len(raw)} rows)")
print(f"      wrote {STAGE_DIR}/summary_unified.csv ({len(summary)} rows)")
print(f"      wrote {STAGE_DIR}/floor_violations.csv ({len(violations)} rows)")
print("=" * 78)
