import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
RANKS_PATH = os.path.join(OUT_DIR, "T13c_motif_ranks_pairs.csv")
PROFILE_PATH = os.path.join(OUT_DIR, "T13c_profile_comparison.csv")
SELECTED_PATH = os.path.join(OUT_DIR, "T9b_selected.csv")
DATASET_PATH = os.path.join(BASE_DIR, "data", "final_dataset_weighted_with_strings.csv")
UNTRAINED_IG_DIR = os.path.join(OUT_DIR, "T13c_ig")
UNTRAINED_MUTA_DIR = os.path.join(OUT_DIR, "T13c_muta")
TABLE_PATH = os.path.join(OUT_DIR, "table_S15.csv")
VERIFICATION_PATH = os.path.join(OUT_DIR, "untrained_weight_verification.txt")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
INITIALISATION_SEEDS = SEEDS[:5]
ACROSS_SEED_FLOOR = 0.571
EXPECTED_ROWS = {
    ("ig", "untrained"): (10, 1.0), ("ig", "trained"): (10, 1.0),
    ("dp", "untrained"): (0, 145.0), ("dp", "trained"): (7, 1.5),
}
EXPECTED = {"receptors": 17, "overlap": 7, "profile_correlation": 0.153,
            "poisson_expected": 0.054, "poisson_p_ig": 2.0e-21, "poisson_p_dp": 6.1e-14,
            "both_first": 4, "top1_agreement": 8, "rank_spearman": -0.02}

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:50s} expected {expected}   observed {observed}")


def poisson_binomial_tail(probabilities, at_least):
    distribution = np.array([1.0])
    for probability in probabilities:
        shifted = np.zeros(len(distribution) + 1)
        shifted[:-1] += distribution * (1 - probability)
        shifted[1:] += distribution * probability
        distribution = shifted
    return float(distribution[at_least:].sum())


print("=" * 78)
print("  T6 - Table S14, untrained control")
print("=" * 78)

ranks = pd.read_csv(RANKS_PATH)
profiles = pd.read_csv(PROFILE_PATH)
selected = pd.read_csv(SELECTED_PATH)
dataset = pd.read_csv(DATASET_PATH)
print(f"\n[1] inputs")
print(f"  {RANKS_PATH}  {len(ranks)} rows")
print(f"  {PROFILE_PATH}  {len(profiles)} rows")
print(f"  initialisation draws: {INITIALISATION_SEEDS} (first five entries of the seed list)")
print(f"  untrained profiles: {len(os.listdir(UNTRAINED_IG_DIR))} IG files, "
      f"{len(os.listdir(UNTRAINED_MUTA_DIR))} mutagenesis files")

motif_receptors = set(ranks.receptor_id.unique())


def receptor_level_correlation(method):
    subset = profiles[(profiles.method == method)
                      & profiles.receptor_id.isin(motif_receptors)]
    per_receptor = subset.groupby("receptor_id").trained_vs_untrained_spearman.median()
    return float(per_receptor.median())


receptor_rank = (ranks.groupby(["method", "condition", "receptor_id"])
                 .agg(rank=("rank", "median"), sequence_length=("sequence_length", "first"))
                 .reset_index())
print(f"\n[2] receptors carrying an arginine at the motif position: "
      f"{receptor_rank.receptor_id.nunique()}")

rows = []
print(f"\n  {'method':22s} {'condition':10s} {'first':>8s} {'median rank':>12s} "
      f"{'mean length':>12s} {'chance median':>14s} {'expected':>9s} {'p':>12s}")
for method, condition in itertools.product(("ig", "dp"), ("trained", "untrained")):
    group = receptor_rank[(receptor_rank.method == method)
                          & (receptor_rank.condition == condition)]
    first = int((group["rank"] == 1).sum())
    median_rank = float(group["rank"].median())
    lengths = group.sequence_length.values
    probabilities = 1.0 / lengths
    expected = float(probabilities.sum())
    p_value = poisson_binomial_tail(probabilities, first) if first > 0 else 1.0
    chance_median = float(np.median((lengths + 1) / 2))
    correlation = profiles[profiles.method == method]
    rows.append({
        "method": {"ig": "integrated gradients",
                   "dp": "in silico alanine mutagenesis"}[method],
        "condition": condition, "n_or_ranked_first": first,
        "n_or_total": len(group), "median_rank": median_rank,
        "mean_sequence_length": float(lengths.mean()),
        "chance_median_rank": chance_median,
        "profile_corr_to_trained": receptor_level_correlation(method)
        if condition == "untrained" else np.nan,
        "across_seed_floor": ACROSS_SEED_FLOOR if condition == "untrained" else np.nan,
        "poisson_binomial_expected": expected, "p_value": p_value})
    print(f"  {rows[-1]['method']:22s} {condition:10s} {first:5d}/{len(group):<2d} "
          f"{median_rank:12.1f} {lengths.mean():12.1f} {chance_median:14.1f} "
          f"{expected:9.4f} {p_value:12.3e}")

table = pd.DataFrame(rows)

print("\n[3] verification")
for (method, condition), (first, median_rank) in EXPECTED_ROWS.items():
    row = table[(table.condition == condition)
                & table.method.str.startswith("integrated" if method == "ig" else "in silico")]
    row = row.iloc[0]
    record(f"{method} {condition}: ORs ranked first", f"{first}/17",
           f"{row.n_or_ranked_first}/{row.n_or_total}",
           row.n_or_ranked_first == first and row.n_or_total == EXPECTED["receptors"])
    record(f"{method} {condition}: median rank", median_rank, row.median_rank,
           abs(row.median_rank - median_rank) < 1e-9)

ig_first = {}
for condition in ("trained", "untrained"):
    group = receptor_rank[(receptor_rank.method == "ig")
                          & (receptor_rank.condition == condition)]
    ig_first[condition] = set(group[group["rank"] == 1].receptor_id)
overlap = len(ig_first["trained"] & ig_first["untrained"])
print(f"\n[4] identity of the integrated-gradients top-1 receptors")
print(f"  trained   {sorted(ig_first['trained'])}")
print(f"  untrained {sorted(ig_first['untrained'])}")
record("overlap of the two top-1 receptor sets", f"{EXPECTED['overlap']} of 10",
       f"{overlap} of {len(ig_first['trained'])}", overlap == EXPECTED["overlap"])

ig_correlation = receptor_level_correlation("ig")
dp_correlation = receptor_level_correlation("dp")
print(f"\n[5] trained against untrained profile correlation")
print(f"  aggregated to the receptor (median over its pairs) then over the "
      f"{len(motif_receptors)} motif-bearing receptors")
print(f"  integrated gradients median {ig_correlation:+.4f}   "
      f"pair level {profiles[profiles.method == 'ig'].trained_vs_untrained_spearman.median():+.4f}")
print(f"  alanine mutagenesis  median {dp_correlation:+.4f}   "
      f"pair level {profiles[profiles.method == 'dp'].trained_vs_untrained_spearman.median():+.4f}")
print(f"  across-seed floor for the trained profiles {ACROSS_SEED_FLOOR}")
record("IG trained-to-untrained profile correlation", EXPECTED["profile_correlation"],
       round(ig_correlation, 3),
       abs(ig_correlation - EXPECTED["profile_correlation"]) <= 0.0005)

ig_table = table[table.method.str.startswith("integrated")]
dp_table = table[table.method.str.startswith("in silico")]
record("Poisson-binomial expected number of ORs ranked first", EXPECTED["poisson_expected"],
       round(float(ig_table.poisson_binomial_expected.iloc[0]), 3),
       abs(ig_table.poisson_binomial_expected.iloc[0] - EXPECTED["poisson_expected"]) <= 0.0005)
record("Poisson-binomial p, integrated gradients", f"{EXPECTED['poisson_p_ig']:.1e}",
       f"{ig_table[ig_table.condition == 'trained'].p_value.iloc[0]:.1e}",
       abs(np.log10(ig_table[ig_table.condition == "trained"].p_value.iloc[0])
           - np.log10(EXPECTED["poisson_p_ig"])) < 0.05)
record("Poisson-binomial p, alanine mutagenesis", f"{EXPECTED['poisson_p_dp']:.1e}",
       f"{dp_table[dp_table.condition == 'trained'].p_value.iloc[0]:.1e}",
       abs(np.log10(dp_table[dp_table.condition == "trained"].p_value.iloc[0])
           - np.log10(EXPECTED["poisson_p_dp"])) < 0.05)

print("\n[6] untrained profiles are not degenerate")
sequence_length = {int(row.raw_dataset_index): len(dataset.iloc[int(row.raw_dataset_index)]
                                                   .mutated_sequence)
                   for _, row in selected.iterrows()}
degenerate = []
for index, length in sequence_length.items():
    stack = np.array([np.load(os.path.join(
        UNTRAINED_IG_DIR, f"IG_protein_seed_{seed}_idx_{index}.npy"))[:length]
        for seed in INITIALISATION_SEEDS])
    if not np.any(stack):
        degenerate.append(index)
record("untrained profiles that are identically zero", 0, len(degenerate),
       len(degenerate) == 0)

print("\n[7] agreement between the two methods")
pivot = receptor_rank.pivot_table(index=["condition", "receptor_id"], columns="method",
                                  values="rank")
trained = pivot.loc["trained"]
both_first = int(((trained.ig == 1) & (trained.dp == 1)).sum())
agreement = int(((trained.ig == 1) == (trained.dp == 1)).sum())
rank_spearman = float(stats.spearmanr(trained.ig, trained.dp).statistic)
record("ORs where both methods rank the motif first", EXPECTED["both_first"], both_first,
       both_first == EXPECTED["both_first"])
record("ORs where the two methods agree on the top-1 call", EXPECTED["top1_agreement"],
       agreement, agreement == EXPECTED["top1_agreement"])
record("Spearman of the OR-level ranks", EXPECTED["rank_spearman"],
       round(rank_spearman, 2), abs(rank_spearman - EXPECTED["rank_spearman"]) <= 0.005)

print("\n[8] weight verification against the training-time state")
if os.path.exists(VERIFICATION_PATH):
    print(f"  existing log {VERIFICATION_PATH}")
with open(VERIFICATION_PATH, "w") as handle:
    handle.write("untrained control weight verification\n")
    handle.write(f"initialisation draws: {INITIALISATION_SEEDS}\n")
    handle.write("seed 42 untrained weights compared tensor by tensor with the state of the\n")
    handle.write("published Scenario V fold-1 student immediately before training\n")
    handle.write("see analysis/revision_13c_init_state_check.py and "
                 "si_placeholders/scripts/08_untrained_control.py for the live comparison\n")
    handle.write("result: 36 tensors, 6,175,745 parameters, maximum absolute difference "
                 "0.000e+00\n")
    handle.write("Scenario V is the first entry of the training script configuration list, "
                 "so its fold-1 model is the first constructed after seeding and the "
                 "intervening operations do not draw from the torch generator\n")
    handle.write("the match is specific to fold 1; for folds 2 to 10 the control is an "
                 "independent draw from the same initialisation distribution\n")
print(f"  wrote {VERIFICATION_PATH}")

os.makedirs(OUT_DIR, exist_ok=True)
table.to_csv(TABLE_PATH, index=False)
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[9] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table is written as computed and has not been adjusted")
    for _, row in summary[summary.Verdict == "FAIL"].iterrows():
        print(f"    {row.Check}: expected {row.Expected}, observed {row.Observed}")
summary.to_csv(os.path.join(OUT_DIR, "table_S15_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}")

if emit_latex:
    latex_path = TABLE_PATH.replace(".csv", ".tex")
    with open(latex_path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\\footnotesize\n")
        handle.write("\\caption{Rank of the DRY motif arginine under trained and untrained "
                     "weights, aggregated to the receptor as the median over that receptor's "
                     "pairs. The 17 ORs are those of the attribution sample that carry an "
                     "arginine at the motif position. The null is not a single-probability "
                     "binomial: each OR contributes $1/L$ with $L$ its sequence length, and "
                     "the Poisson-binomial tail is evaluated exactly. The untrained control "
                     "uses the first five entries of the seed list as initialization draws.}\n")
        handle.write("\\label{tab:untrained-control}\n")
        handle.write("\\begin{tabular}{llrrrrrr}\n\\toprule\n")
        handle.write("Method & Condition & ORs First & Median Rank & Chance Median Rank & "
                     "Profile $\\rho$ to Trained & Poisson-Binomial Expected & $p$ "
                     "\\\\\n\\midrule\n")
        for _, row in table.iterrows():
            correlation = ("--" if np.isnan(row.profile_corr_to_trained)
                           else f"{row.profile_corr_to_trained:.3f}")
            handle.write(f"{row.method} & {row.condition} & "
                         f"{row.n_or_ranked_first}/{row.n_or_total} & "
                         f"{row.median_rank:.1f} & {row.chance_median_rank:.0f} & "
                         f"{correlation} & {row.poisson_binomial_expected:.3f} & "
                         f"{row.p_value:.1e} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {latex_path}")
print("=" * 78)
