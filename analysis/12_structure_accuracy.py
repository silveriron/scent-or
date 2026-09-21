import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_utils

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
STRUCTURE_DIR = os.path.join(OUT_DIR, "structure")
CONTACT_PATH = os.path.join(OUT_DIR, "T6_8f76_contacts.csv")
DEVIATION_PATH = os.path.join(OUT_DIR, "T8_or51e2_deviation.csv")
TRANSFORM_PATH = os.path.join(STRUCTURE_DIR, "superposition_transform.txt")
SUPERPOSED_PATH = os.path.join(STRUCTURE_DIR, "esmfold_or51e2_superposed.pdb")
IG_DIR = os.path.join(OUT_DIR, "T6_ig")
MUTA_DIR = os.path.join(OUT_DIR, "T6_muta")
TABLE_PATH = os.path.join(OUT_DIR, "table_S12.csv")
NOTES_PATH = os.path.join(OUT_DIR, "table_S12_notes.txt")
EXPORT_PDB = os.path.join(OUT_DIR, "or51e2_superposed.pdb")
EXPORT_TRANSFORM = os.path.join(OUT_DIR, "or51e2_transform.txt")

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
OR51E2_RAW_INDEX = 4336
CONTACT_CUTOFF = 4.5
TOP_N = 30
EXPECTED_CONTACTS = {"R262", "H180", "Q181", "F155", "L158", "S258", "G198", "H104",
                     "I202", "L199"}
EXPECTED = {"rmsd": 4.30, "within_2": 0.427, "within_3": 0.599, "contact_mean": 3.05,
            "R262": (2.66, 5.36), "H180": (2.40, 5.01),
            "median_rank_ig": 134, "median_rank_muta": 129,
            "cles_ig": (0.531, 0.365, 0.696), "cles_muta": (0.509, 0.326, 0.685),
            "ks_ig": 0.92, "ks_muta": 0.95, "expected_top30": 0.99}

emit_latex = "--latex" in sys.argv
checks = []


def record(label, expected, observed, ok):
    checks.append({"Check": label, "Expected": expected, "Observed": observed,
                   "Verdict": "PASS" if ok else "FAIL"})
    print(f"  {'PASS' if ok else 'FAIL'}  {label:46s} expected {expected}   observed {observed}")


print("=" * 78)
print("  T3 - Table S11, OR51E2 propionate contacts")
print("=" * 78)

contacts = pd.read_csv(CONTACT_PATH)
deviation = pd.read_csv(DEVIATION_PATH)
print(f"\n[1] inputs")
print(f"  {CONTACT_PATH}  {len(contacts)} resolved residues")
print(f"  {DEVIATION_PATH}  {len(deviation)} resolved residues")
print(f"  {IG_DIR}  {len(os.listdir(IG_DIR))} files")
print(f"  {MUTA_DIR}  {len(os.listdir(MUTA_DIR))} files")
print(f"  sequence alignment: all {len(contacts)} resolved 8F76 residues map with "
      f"identical numbering, mismatches "
      f"{int((~contacts.residue_matches).sum())}")

ig_profiles = np.array([np.abs(np.load(os.path.join(
    IG_DIR, f"IG_protein_seed_{seed}_idx_{OR51E2_RAW_INDEX}.npy"))) for seed in SEEDS])
muta_profiles = np.array([np.abs(np.load(os.path.join(
    MUTA_DIR, f"Muta_score_seed_{seed}_idx_{OR51E2_RAW_INDEX}.npy"))) for seed in SEEDS])
print(f"  attribution profiles {ig_profiles.shape} (IG) and {muta_profiles.shape} "
      f"(alanine mutagenesis), seed-averaged before ranking")
ig_mean = ig_profiles.mean(axis=0)
muta_mean = muta_profiles.mean(axis=0)

resolved = deviation.m2or_position.values
resolved_index = resolved - 1
ig_resolved = ig_mean[resolved_index]
muta_resolved = muta_mean[resolved_index]


def ranks_of(values):
    order = np.argsort(-values, kind="stable")
    rank = np.empty(len(values), dtype=int)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


ig_rank = ranks_of(ig_resolved)
muta_rank = ranks_of(muta_resolved)
rank_lookup = {position: (int(ig_rank[i]), int(muta_rank[i]))
               for i, position in enumerate(resolved)}
deviation_lookup = dict(zip(deviation.m2or_position, deviation.deviation_angstrom))

contact_rows = []
for _, row in contacts[contacts.min_distance_to_ligand <= CONTACT_CUTOFF].iterrows():
    position = int(row.m2or_position_1based)
    ig_value, muta_value = rank_lookup[position]
    contact_rows.append({
        "residue": f"{row.m2or_residue}{position}",
        "residue_number": position,
        "min_distance_to_ligand_A": float(row.min_distance_to_ligand),
        "ca_deviation_A": float(deviation_lookup[position]),
        "ig_rank": ig_value, "delta_p_rank": muta_value})
table = pd.DataFrame(contact_rows).sort_values("min_distance_to_ligand_A").reset_index(drop=True)

print("\n[2] contact table")
print(f"  {'residue':8s} {'distance':>9s} {'deviation':>10s} {'IG rank':>8s} {'dP rank':>8s}")
for _, row in table.iterrows():
    print(f"  {row.residue:8s} {row.min_distance_to_ligand_A:9.3f} "
          f"{row.ca_deviation_A:10.3f} {row.ig_rank:8d} {row.delta_p_rank:8d}")

print("\n[3] verification")
record("number of contacts within 4.5 A", 10, len(table), len(table) == 10)
record("contact identities", sorted(EXPECTED_CONTACTS), sorted(table.residue),
       set(table.residue) == EXPECTED_CONTACTS)
rmsd = float(np.sqrt((deviation.deviation_angstrom.values ** 2).mean()))
record("global Ca RMSD", EXPECTED["rmsd"], round(rmsd, 2), abs(rmsd - EXPECTED["rmsd"]) <= 0.005)
within_2 = float((deviation.deviation_angstrom < 2).mean())
within_3 = float((deviation.deviation_angstrom < 3).mean())
record("fraction within 2 A", EXPECTED["within_2"], round(within_2, 3),
       abs(within_2 - EXPECTED["within_2"]) <= 0.0005)
record("fraction within 3 A", EXPECTED["within_3"], round(within_3, 3),
       abs(within_3 - EXPECTED["within_3"]) <= 0.0005)
contact_mean = float(table.ca_deviation_A.mean())
record("mean deviation over the ten contacts", EXPECTED["contact_mean"], round(contact_mean, 2),
       abs(contact_mean - EXPECTED["contact_mean"]) <= 0.005)
indexed = table.set_index("residue")
for residue, (distance, shift) in ((k, v) for k, v in EXPECTED.items()
                                   if k in ("R262", "H180")):
    row = indexed.loc[residue]
    record(f"{residue} distance and deviation", [distance, shift],
           [round(row.min_distance_to_ligand_A, 2), round(row.ca_deviation_A, 2)],
           abs(row.min_distance_to_ligand_A - distance) <= 0.005
           and abs(row.ca_deviation_A - shift) <= 0.005)
sub_angstrom = set(indexed[indexed.ca_deviation_A < 1.0].index)
record("contacts with deviation below 1 A", {"I202", "L199"}, sub_angstrom,
       sub_angstrom == {"I202", "L199"})

contact_positions = set(table.residue_number)
background_mask = np.array([position not in contact_positions for position in resolved])
for method, rank_array, key in (("integrated gradients", ig_rank, "ig"),
                                ("alanine mutagenesis", muta_rank, "muta")):
    contact_ranks = rank_array[~background_mask]
    background_ranks = rank_array[background_mask]
    in_top = int((contact_ranks <= TOP_N).sum())
    expected_top = len(contact_ranks) * TOP_N / len(resolved)
    median_rank = float(np.median(contact_ranks))
    point, low, high = stats_utils.cles_ci(background_ranks, contact_ranks)
    ks_p = float(stats.ks_2samp(contact_ranks, background_ranks)[1])
    print(f"\n[4] {method}")
    print(f"  contacts inside the top {TOP_N}: {in_top}   expected by chance "
          f"{expected_top:.2f}")
    print(f"  median contact rank {median_rank:.0f} of {len(resolved)}")
    print(f"  common-language effect size {point:.3f} [{low:.3f}, {high:.3f}]")
    print(f"  Kolmogorov-Smirnov shape check p = {ks_p:.3f}")
    record(f"{method}: contacts in the top {TOP_N}", 0, in_top, in_top == 0)
    record(f"{method}: expected contacts in the top {TOP_N}", EXPECTED["expected_top30"],
           round(expected_top, 2), abs(expected_top - EXPECTED["expected_top30"]) <= 0.005)
    record(f"{method}: median contact rank", EXPECTED[f"median_rank_{key}"],
           int(median_rank), int(median_rank) == EXPECTED[f"median_rank_{key}"])
    expected_cles = EXPECTED[f"cles_{key}"]
    record(f"{method}: common-language effect size", expected_cles[0], round(point, 3),
           abs(point - expected_cles[0]) <= 0.0005)
    record(f"{method}: KS shape check p", EXPECTED[f"ks_{key}"], round(ks_p, 2),
           abs(ks_p - EXPECTED[f"ks_{key}"]) <= 0.005)

print("\n[5] superposition artefacts")
with open(TRANSFORM_PATH) as handle:
    transform_text = handle.read()
numbers = [float(value) for line in transform_text.splitlines()
           for value in line.split()
           if value.replace("-", "").replace(".", "").isdigit()]
rotation_lines = transform_text.split("rotation matrix R (3x3, row major)")[1]
rotation = np.array([[float(value) for value in line.split()]
                     for line in rotation_lines.strip().splitlines()[:3]])
translation_lines = transform_text.split("translation vector t (Angstrom)")[1]
translation = np.array([float(value) for value in translation_lines.strip().split()[:3]])
homogeneous = np.eye(4)
homogeneous[:3, :3] = rotation
homogeneous[:3, 3] = translation
with open(EXPORT_TRANSFORM, "w") as handle:
    handle.write("ESMFold OR51E2 -> PDB 8F76 chain A\n")
    handle.write("Kabsch/SVD least-squares fit, unweighted, 302 Ca correspondence pairs\n")
    handle.write("8F76 is the reference and is not transformed\n")
    handle.write("homogeneous 4x4 matrix M, applied as [x' y' z' 1]^T = M [x y z 1]^T\n")
    handle.write("coordinates in Angstrom\n\n")
    for row in homogeneous:
        handle.write("   " + "".join(f"{value:22.12f}" for value in row) + "\n")
    handle.write(f"\ndet(R) = {np.linalg.det(rotation):.12f}\n")
    handle.write(f"max |R R^T - I| = {np.abs(rotation @ rotation.T - np.eye(3)).max():.3e}\n")
    handle.write(f"global Ca RMSD over the 302 pairs = {rmsd:.4f} Angstrom\n")
shutil.copyfile(SUPERPOSED_PATH, EXPORT_PDB)
print(f"  wrote {EXPORT_TRANSFORM}")
print(f"  wrote {EXPORT_PDB}")
record("rotation is orthonormal with unit determinant", 1.0,
       round(float(np.linalg.det(rotation)), 6),
       abs(np.linalg.det(rotation) - 1.0) < 1e-9)

os.makedirs(OUT_DIR, exist_ok=True)
table.to_csv(TABLE_PATH, index=False)
with open(NOTES_PATH, "w") as handle:
    handle.write(f"global Ca RMSD over {len(deviation)} aligned positions: {rmsd:.4f} A\n")
    handle.write(f"positions within 2 A: {within_2:.4f}; within 3 A: {within_3:.4f}\n")
    handle.write(f"mean deviation over the ten contacts: {contact_mean:.4f} A\n")
    for method, rank_array, key in (("integrated gradients", ig_rank, "ig"),
                                    ("alanine mutagenesis", muta_rank, "muta")):
        contact_ranks = rank_array[~background_mask]
        background_ranks = rank_array[background_mask]
        point, low, high = stats_utils.cles_ci(background_ranks, contact_ranks)
        handle.write(f"{method}: contacts in the top {TOP_N} "
                     f"{int((contact_ranks <= TOP_N).sum())} of 10, expected "
                     f"{len(contact_ranks) * TOP_N / len(resolved):.2f}; median contact rank "
                     f"{np.median(contact_ranks):.0f} of {len(resolved)}; CLES {point:.3f} "
                     f"[{low:.3f}, {high:.3f}]; KS shape check p "
                     f"{stats.ks_2samp(contact_ranks, background_ranks)[1]:.3f}\n")
summary = pd.DataFrame(checks)
failures = int((summary.Verdict == "FAIL").sum())
print(f"\n[6] verification summary: {len(summary) - failures} PASS / {failures} FAIL")
if failures:
    print("  WARNING: the table is written as computed and has not been adjusted")
    for _, row in summary[summary.Verdict == "FAIL"].iterrows():
        print(f"    {row.Check}: expected {row.Expected}, observed {row.Observed}")
summary.to_csv(os.path.join(OUT_DIR, "table_S12_checks.csv"), index=False)
print(f"  wrote {TABLE_PATH}")
print(f"  wrote {NOTES_PATH}")

if emit_latex:
    latex_path = TABLE_PATH.replace(".csv", ".tex")
    with open(latex_path, "w") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write("\\caption{The ten residues within 4.5~\\AA{} of the propionate bound in "
                     "PDB ID 8F76, their minimum heavy-atom distance to the ligand, the "
                     "C$\\alpha$ deviation of the ESMFold model after least-squares "
                     "superposition, and their rank within each attribution profile over the "
                     "302 resolved residues. Attribution profiles are seed-averaged absolute "
                     "values over the 20 seeds and ranked in decreasing order.}\n")
        handle.write("\\label{tab:or51e2-contacts}\n")
        handle.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        handle.write("Residue & Min. Distance to Ligand (\\AA) & C$\\alpha$ Deviation (\\AA) & "
                     "IG Rank (of 302) & $\\Delta P$ Rank (of 302) \\\\\n\\midrule\n")
        for _, row in table.iterrows():
            handle.write(f"{row.residue} & {row.min_distance_to_ligand_A:.2f} & "
                         f"{row.ca_deviation_A:.2f} & {row.ig_rank} & "
                         f"{row.delta_p_rank} \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {latex_path}")
print("=" * 78)
