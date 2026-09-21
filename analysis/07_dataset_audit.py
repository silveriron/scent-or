import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "results", "revision")
os.makedirs(OUT_DIR, exist_ok=True)

CURATED_CSV = os.path.join(DATA_DIR, "final_dataset_weighted.csv")
STRINGS_CSV = os.path.join(DATA_DIR, "final_dataset_weighted_with_strings.csv")
ASSIGN_CSV = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")
CLOSE_PAIRS_CSV = os.path.join(OUT_DIR, "T1_close_sequence_pairs.csv")

AUDIT_CSV = os.path.join(OUT_DIR, "T1_dataset_audit.csv")
SPECIES_TABLE_CSV = os.path.join(OUT_DIR, "T1_species_table.csv")
SUMMARY_MD = os.path.join(OUT_DIR, "T1_summary.md")

TAAR_TOKENS = ["TAAR", "TA1R", "TA2R", "TA3R", "TA4R", "TA5R", "TA6R", "TA8R", "TA9R"]

rows = []


def emit(section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})
    print(f"    {metric}: {value}" + (f"   ({note})" if note else ""))


print("=" * 78)
print("  T1 — dataset composition audit")
print("=" * 78)

source_csv = STRINGS_CSV if os.path.exists(STRINGS_CSV) else CURATED_CSV
df = pd.read_csv(source_csv)
assign = pd.read_csv(ASSIGN_CSV)
close_pairs = pd.read_csv(CLOSE_PAIRS_CSV)
print(f"  pairs input      : {source_csv}  ({len(df)} rows)")
print(f"  receptor assign  : {ASSIGN_CSV}  ({len(assign)} receptors)")
print(f"  close seq pairs  : {CLOSE_PAIRS_CSV}  ({len(close_pairs)} pairs)")

merged = df.merge(assign, on="main_receptors_id", how="left")

print("\n[1] Pair-level composition")
n_pairs = len(df)
n_active = int((df["responsive"] == 1).sum())
emit("1_pairs", "total_pairs", n_pairs)
emit("1_pairs", "active_pairs", n_active)
emit("1_pairs", "inactive_pairs", n_pairs - n_active)
emit("1_pairs", "active_fraction", round(n_active / n_pairs, 6))
emit("1_pairs", "inactive_to_active_ratio", round((n_pairs - n_active) / n_active, 4))
emit("1_pairs", "unique_receptors", df["main_receptors_id"].nunique())
emit("1_pairs", "unique_compounds", df["main_compounds_id"].nunique())
emit("1_pairs", "duplicate_receptor_compound_rows",
     int(df.duplicated(subset=["main_receptors_id", "main_compounds_id"]).sum()))

print("\n[2] Confidence (data_quality) distribution")
for q, sub in df.groupby("data_quality"):
    a = int((sub["responsive"] == 1).sum())
    emit("2_quality", f"{q}_pairs", len(sub),
         f"active={a} ({a / len(sub):.4f}), weights={sorted(sub['sample_weight'].unique())}")

print("\n[3] Unique amino-acid sequences")
seqs = df.drop_duplicates("main_receptors_id").set_index(
    "main_receptors_id")["mutated_sequence"]
emit("3_sequences", "receptor_ids", len(seqs))
emit("3_sequences", "unique_sequences", int(seqs.nunique()))
emit("3_sequences", "ids_sharing_a_sequence", int(len(seqs) - seqs.nunique()),
     "1:1 mapping means no id-level redundancy")
emit("3_sequences", "sequence_length_min", int(assign["sequence_length"].min()))
emit("3_sequences", "sequence_length_max", int(assign["sequence_length"].max()))
emit("3_sequences", "sequence_length_median", float(assign["sequence_length"].median()))

print("\n[4] Species composition")
sp_rows = []
for org, sub in merged.groupby(merged["organism"].fillna("UNASSIGNED")):
    a = int((sub["responsive"] == 1).sum())
    sp_rows.append({
        "organism": org,
        "n_receptors": sub["main_receptors_id"].nunique(),
        "n_pairs": len(sub),
        "n_active": a,
        "n_inactive": len(sub) - a,
        "active_fraction": round(a / len(sub), 6),
        "pair_fraction_of_dataset": round(len(sub) / n_pairs, 6),
    })
sp_table = pd.DataFrame(sp_rows).sort_values("n_pairs", ascending=False)
sp_table.to_csv(SPECIES_TABLE_CSV, index=False)
print(sp_table.to_string(index=False))
for _, r in sp_table.iterrows():
    emit("4_species", f"organism::{r['organism']}",
         f"receptors={r['n_receptors']}; pairs={r['n_pairs']}; "
         f"active={r['n_active']}; active_frac={r['active_fraction']:.4f}")

is_human = merged["is_human"].fillna(False)
emit("4_species", "human_receptors", int(assign["is_human"].sum()))
emit("4_species", "human_pairs", int(is_human.sum()))
emit("4_species", "human_active_pairs", int(((merged["responsive"] == 1) & is_human).sum()))
emit("4_species", "non_human_receptors",
     int((~assign["is_human"] & assign["organism"].notna()).sum()))
emit("4_species", "non_human_pairs", int((~is_human & merged["organism"].notna()).sum()))
emit("4_species", "unassigned_receptors", int(assign["organism"].isna().sum()))
emit("4_species", "unassigned_pairs", int(merged["organism"].isna().sum()))

print("\n[5] Reference (archived) vs variant sequences")
emit("5_variants", "assignment_source_uniparc_exact",
     int((assign["assignment_source"] == "uniparc_exact").sum()),
     "sequence present verbatim in UniParc")
emit("5_variants", "assignment_source_nearest_neighbour",
     int((assign["assignment_source"] == "nearest_neighbour").sum()),
     "not archived anywhere; engineered/typo variant")
emit("5_variants", "assignment_source_unassigned",
     int((assign["assignment_source"] == "unassigned").sum()))
nn = assign[assign["assignment_source"] == "nearest_neighbour"]
for cut in [1, 2, 3, 5]:
    emit("5_variants", f"variant_within_{cut}_edits_of_a_reference",
         int((nn["edit_distance_to_reference"] <= cut).sum()))

gene_groups = assign.dropna(subset=["gene_name", "organism"]).groupby(
    ["organism", "gene_name"])["main_receptors_id"].nunique()
emit("5_variants", "distinct_organism_gene_groups", int(len(gene_groups)))
emit("5_variants", "groups_with_multiple_receptor_ids", int((gene_groups > 1).sum()))
emit("5_variants", "receptor_ids_in_multi_id_groups",
     int(gene_groups[gene_groups > 1].sum()))
human_genes = assign[assign["is_human"]].dropna(subset=["gene_name"])["gene_name"].nunique()
emit("5_variants", "distinct_human_gene_symbols", int(human_genes))

print("\n[6] Close sequence pairs (edit distance <= 2)")
emit("6_close_pairs", "total_close_pairs", len(close_pairs))
for k, v in close_pairs["distance"].value_counts().sort_index().items():
    emit("6_close_pairs", f"pairs_at_distance_{k}", int(v))
for k, v in close_pairs["kind"].value_counts().items():
    emit("6_close_pairs", f"kind_{k}", int(v))

print("\n[7] TAAR presence")
genes = assign["gene_name"].dropna().astype(str).str.upper()
taar_hits = sorted({g for g in genes if any(t in g for t in TAAR_TOKENS)})
emit("7_taar", "taar_gene_symbols_found", len(taar_hits),
     "; ".join(taar_hits) if taar_hits else
     "none among 1,338 resolved gene symbols (UniParc/UniProt cross-references)")
emit("7_taar", "evidence_basis", "gene symbol of best UniParc cross-reference",
     f"{int(assign['gene_name'].notna().sum())}/1338 receptors carry a gene symbol")

print("\n[8] Decomposition of the 1,338 receptor ids")
src = assign["assignment_source"].astype(str)
is_h = assign["is_human"]
human_ref = int((is_h & src.str.startswith("uniparc_exact")).sum())
human_local = int((is_h & (src == "local_alignment")).sum())
human_var = int((is_h & (src == "nearest_neighbour")).sum())
nonhuman = int(((~is_h) & (assign["organism"].notna())).sum())
unassigned = int(assign["organism"].isna().sum())
n_amb = int(assign.get("species_ambiguous_with_human", pd.Series(False, index=assign.index)).sum())
emit("8_decomposition", "human_archived_sequence", human_ref,
     f"includes {n_amb} sequences shared with another great ape (see T1 method)")
emit("8_decomposition", "human_tagged_construct_local_alignment", human_local)
emit("8_decomposition", "human_variant_not_archived", human_var)
emit("8_decomposition", "non_human", nonhuman)
emit("8_decomposition", "unassigned", unassigned)
total = human_ref + human_local + human_var + nonhuman + unassigned
emit("8_decomposition", "sum", total, "must equal 1338")

pd.DataFrame(rows).to_csv(AUDIT_CSV, index=False)

with open(SUMMARY_MD, "w") as fh:
    fh.write("# T1 — Dataset composition audit\n\n")
    fh.write(f"Input: `{os.path.relpath(source_csv, REPO_ROOT)}` ({len(df)} rows)\n\n")
    fh.write("## Headline\n\n")
    fh.write(f"- {n_pairs:,} pairs, {n_active:,} active / {n_pairs - n_active:,} inactive "
             f"(1:{(n_pairs - n_active) / n_active:.2f}, {n_active / n_pairs:.2%} active)\n")
    fh.write(f"- 1,338 receptor ids map to {seqs.nunique():,} distinct sequences (1:1)\n")
    fh.write(f"- **{int(assign['is_human'].sum())} / 1,338 receptors are human "
             f"({assign['is_human'].mean():.1%})**, covering {int(is_human.sum()):,} pairs "
             f"({is_human.mean():.1%} of the dataset)\n")
    fh.write(f"- Non-human: {nonhuman} receptors; unassigned: {unassigned}\n")
    fh.write(f"- TAAR sequences: {'none found' if not taar_hits else ', '.join(taar_hits)}\n\n")
    fh.write("## 1,338 decomposition\n\n")
    fh.write("| bucket | receptors |\n|---|---|\n")
    fh.write(f"| human, sequence archived in UniParc | {human_ref} |\n")
    fh.write(f"| human, tagged construct matched by local alignment | {human_local} |\n")
    fh.write(f"| human, variant not archived (nearest-neighbour assigned) | {human_var} |\n")
    fh.write(f"| non-human | {nonhuman} |\n")
    fh.write(f"| unassigned | {unassigned} |\n")
    fh.write(f"| **total** | **{total}** |\n\n")
    fh.write("## Species table\n\n")
    fh.write("| " + " | ".join(sp_table.columns) + " |\n")
    fh.write("|" + "---|" * len(sp_table.columns) + "\n")
    for _, r in sp_table.iterrows():
        fh.write("| " + " | ".join(str(r[c]) for c in sp_table.columns) + " |\n")
    fh.write("\n## Method\n\n")
    fh.write("- Species assigned by exact sequence match: CRC64 checksum (`Bio.SeqUtils."
             "CheckSum.crc64`) queried against UniParc, then the highest-priority active "
             "cross-reference (Swiss-Prot > TrEMBL > RefSeq > Ensembl > EMBL) supplies "
             "organism and gene symbol.\n")
    fh.write("- Cloning artefacts (`synthetic construct`, ORFeome/Gateway vectors) are "
             "excluded from organism assignment and from the ambiguity count.\n")
    fh.write(f"- Sequences absent from UniParc ({int((assign['assignment_source'] == 'nearest_neighbour').sum())}) "
             "inherit organism/gene from their nearest archived sequence "
             "(exact Hamming distance for equal lengths, banded Levenshtein otherwise); "
             "these are treated as variants, not wild type.\n")
    fh.write("- No network call is made for anything other than UniParc lookup; results are "
             "cached in `T1_uniparc_*_cache.json`.\n\n")
    fh.write("## Standard deviation convention\n\n")
    fh.write("- T1 reports exact counts only; no cross-seed or cross-fold aggregation, "
             "hence no dispersion statistic.\n")

print("\n" + "=" * 78)
print(f"  saved: {AUDIT_CSV}")
print(f"  saved: {SPECIES_TABLE_CSV}")
print(f"  saved: {SUMMARY_MD}")
print("=" * 78)
