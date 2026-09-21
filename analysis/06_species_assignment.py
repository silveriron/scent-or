import os
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "results", "revision")
os.makedirs(OUT_DIR, exist_ok=True)

STRINGS_CSV = os.path.join(DATA_DIR, "final_dataset_weighted_with_strings.csv")
SPECIES_CSV = os.path.join(OUT_DIR, "T1_receptor_species.csv")
OUT_ASSIGN = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")
OUT_PAIRS = os.path.join(OUT_DIR, "T1_close_sequence_pairs.csv")
ENTRY_CACHE = os.path.join(OUT_DIR, "T1_uniparc_entry_cache.json")

MAX_INHERIT_DISTANCE = 15
CLOSE_PAIR_MAX = 2
KMER = 5
TOP_CANDIDATES = 25


def levenshtein(a, b, cap):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = np.arange(len(b) + 1, dtype=np.int32)
    a_arr = np.frombuffer(a.encode(), dtype=np.uint8)
    b_arr = np.frombuffer(b.encode(), dtype=np.uint8)
    for i in range(1, len(a_arr) + 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + (a_arr[i - 1] != b_arr)
        cur[1:] = sub
        np.minimum(cur[1:], prev[1:] + 1, out=cur[1:])
        for j in range(1, len(b_arr) + 1):
            if cur[j - 1] + 1 < cur[j]:
                cur[j] = cur[j - 1] + 1
        if cur.min() > cap:
            return cap + 1
        prev = cur
    return int(prev[-1])


def kmer_set(seq):
    return {seq[i:i + KMER] for i in range(len(seq) - KMER + 1)}


print("=" * 78)
print("  T1b — sequence distance, WT/variant grouping, species backfill")
print("=" * 78)

df = pd.read_csv(STRINGS_CSV)
seqs = df.drop_duplicates("main_receptors_id").set_index(
    "main_receptors_id")["mutated_sequence"].sort_index()
species = pd.read_csv(SPECIES_CSV).set_index("main_receptors_id")
print(f"  receptors: {len(seqs)}   resolved by UniParc: "
      f"{int(species['organism'].notna().sum())}")

ids = np.array(seqs.index)
seq_list = [seqs[i] for i in ids]
lengths = np.array([len(s) for s in seq_list])
pos_of = {int(r): k for k, r in enumerate(ids)}
resolved_mask = np.array([pd.notna(species.loc[int(r), "organism"]) for r in ids])
print(f"  length range: {lengths.min()}-{lengths.max()}, "
      f"distinct lengths: {len(set(lengths.tolist()))}")

print("\n[1] Exhaustive same-length Hamming distances")
close_pairs = []
by_len = defaultdict(list)
for k, L in enumerate(lengths):
    by_len[int(L)].append(k)
for L, idxs in sorted(by_len.items()):
    if len(idxs) < 2:
        continue
    mat = np.frombuffer("".join(seq_list[k] for k in idxs).encode(),
                        dtype=np.uint8).reshape(len(idxs), L)
    for a in range(len(idxs)):
        d = (mat[a + 1:] != mat[a]).sum(axis=1)
        hits = np.nonzero(d <= CLOSE_PAIR_MAX)[0]
        for h in hits:
            b = a + 1 + h
            close_pairs.append({"id_a": int(ids[idxs[a]]), "id_b": int(ids[idxs[b]]),
                                "distance": int(d[h]), "kind": "substitution_only"})
print(f"  same-length pairs with distance <= {CLOSE_PAIR_MAX}: {len(close_pairs)}")

print("\n[2] Cross-length Levenshtein (indel candidates)")
kmers = [kmer_set(s) for s in seq_list]
cross = 0
for a in range(len(ids)):
    cand = np.nonzero((np.abs(lengths - lengths[a]) >= 1)
                      & (np.abs(lengths - lengths[a]) <= CLOSE_PAIR_MAX))[0]
    cand = cand[cand > a]
    for b in cand:
        inter = len(kmers[a] & kmers[b])
        union = len(kmers[a] | kmers[b])
        if union and inter / union < 0.90:
            continue
        d = levenshtein(seq_list[a], seq_list[b], CLOSE_PAIR_MAX)
        if d <= CLOSE_PAIR_MAX:
            close_pairs.append({"id_a": int(ids[a]), "id_b": int(ids[b]),
                                "distance": int(d), "kind": "with_indel"})
            cross += 1
print(f"  cross-length pairs with distance <= {CLOSE_PAIR_MAX}: {cross}")

pairs_df = pd.DataFrame(close_pairs).sort_values(["distance", "id_a", "id_b"])
pairs_df.to_csv(OUT_PAIRS, index=False)
print(f"  total close pairs: {len(pairs_df)}  -> {OUT_PAIRS}")

print("\n[3] Nearest resolved neighbour for unresolved sequences")
unresolved = np.nonzero(~resolved_mask)[0]
resolved_idx = np.nonzero(resolved_mask)[0]
backfill = {}
for a in unresolved:
    best_d, best_b = None, None
    same_len = [b for b in resolved_idx if lengths[b] == lengths[a]]
    if same_len:
        mat = np.frombuffer("".join(seq_list[b] for b in same_len).encode(),
                            dtype=np.uint8).reshape(len(same_len), lengths[a])
        ref = np.frombuffer(seq_list[a].encode(), dtype=np.uint8)
        d = (mat != ref).sum(axis=1)
        j = int(np.argmin(d))
        best_d, best_b = int(d[j]), same_len[j]
    if best_d is None or best_d > CLOSE_PAIR_MAX:
        scored = sorted(((len(kmers[a] & kmers[b]) / max(1, len(kmers[a] | kmers[b])), b)
                         for b in resolved_idx), reverse=True)[:TOP_CANDIDATES]
        for _, b in scored:
            d = levenshtein(seq_list[a], seq_list[b], MAX_INHERIT_DISTANCE)
            if best_d is None or d < best_d:
                best_d, best_b = d, b
    backfill[int(ids[a])] = (best_d, int(ids[best_b]) if best_b is not None else None)

n_backfilled = sum(1 for d, _ in backfill.values() if d is not None and d <= MAX_INHERIT_DISTANCE)
print(f"  unresolved: {len(unresolved)}")
print(f"  with a resolved neighbour within {MAX_INHERIT_DISTANCE} edits: {n_backfilled}")
dist_vals = [d for d, _ in backfill.values() if d is not None]
if dist_vals:
    for cut in [0, 1, 2, 3, 5, 10, MAX_INHERIT_DISTANCE]:
        print(f"    distance <= {cut}: {sum(1 for d in dist_vals if d <= cut)}")

print("\n[3b] Local-alignment rescue for sequences still unassigned")
from Bio.Align import PairwiseAligner

LOCAL_MIN_IDENTITY = 0.95
LOCAL_MIN_SPAN = 250
local_aligner = PairwiseAligner(mode="local", match_score=2, mismatch_score=-1,
                                open_gap_score=-11, extend_gap_score=-1)
rescued = {}
still_unassigned = [a for a in unresolved
                    if not (backfill[int(ids[a])][0] is not None
                            and backfill[int(ids[a])][0] <= MAX_INHERIT_DISTANCE)]
for a in still_unassigned:
    q = seq_list[a]
    best = None
    for b in resolved_idx:
        s = local_aligner.score(q, seq_list[b])
        if best is None or s > best[0]:
            best = (s, b)
    aln = local_aligner.align(q, seq_list[best[1]])[0]
    top, bottom = str(aln[0]), str(aln[1])
    span = len(top)
    ident = sum(1 for x, y in zip(top, bottom) if x == y and x != "-")
    if span >= LOCAL_MIN_SPAN and ident / span >= LOCAL_MIN_IDENTITY:
        rescued[int(ids[a])] = (int(ids[best[1]]), ident / span, span)
        print(f"  receptor {int(ids[a])} -> {int(ids[best[1]])} "
              f"({species.loc[int(ids[best[1]]), 'organism']}, "
              f"{species.loc[int(ids[best[1]]), 'gene_name']}): "
              f"identity {ident / span:.3f} over {span} residues")
print(f"  rescued {len(rescued)}/{len(still_unassigned)} by local alignment "
      f"(N-terminal expression tags leave the core sequence intact)")

print("\n[4] Final receptor assignment")
rows = []
for r in ids:
    r = int(r)
    rec = species.loc[r]
    org, gene, acc = rec["organism"], rec["gene_name"], rec["accession"]
    src, dist_wt, ref_id = "uniparc_exact", 0, r
    if pd.isna(org):
        d, nb = backfill.get(r, (None, None))
        if nb is not None and d is not None and d <= MAX_INHERIT_DISTANCE:
            nrec = species.loc[nb]
            org, gene, acc = nrec["organism"], nrec["gene_name"], nrec["accession"]
            src, dist_wt, ref_id = "nearest_neighbour", int(d), nb
        elif r in rescued:
            nb, ident, span = rescued[r]
            nrec = species.loc[nb]
            org, gene, acc = nrec["organism"], nrec["gene_name"], nrec["accession"]
            src, dist_wt, ref_id = "local_alignment", None, nb
        else:
            src, dist_wt, ref_id = "unassigned", None, None
    rows.append({
        "main_receptors_id": r,
        "sequence_length": int(lengths[pos_of[r]]),
        "organism": org,
        "gene_name": gene,
        "accession": acc,
        "assignment_source": src,
        "reference_receptor_id": ref_id,
        "edit_distance_to_reference": dist_wt,
        "is_reference_sequence": bool(src == "uniparc_exact"),
        "organism_ambiguous": bool(rec["organism_ambiguous"]),
        "uniparc_upi": rec["uniparc_upi"],
    })

assign = pd.DataFrame(rows)

GREAT_APES = ("Pan ", "Gorilla ", "Pongo ")
PATENT_DBS = {"EPO", "JPO", "KIPO", "USPTO", "PDB"}
import json as _json
_entry_cache = _json.load(open(ENTRY_CACHE))
amb_human, amb_alt = [], {}
for i, r in assign.iterrows():
    org = str(r["organism"])
    if not org.startswith(GREAT_APES):
        continue
    upi = species.loc[r["main_receptors_id"], "uniparc_upi"]
    refs = _entry_cache.get(str(upi), []) if pd.notna(upi) else []
    human_ok = [x for x in refs if x["organism"] == "Homo sapiens"
                and x["active"] and x["database"] not in PATENT_DBS]
    if human_ok:
        amb_human.append(i)
        amb_alt[i] = org
assign["species_ambiguous_with_human"] = False
if amb_human:
    assign.loc[amb_human, "species_ambiguous_with_human"] = True
    assign.loc[amb_human, "alternative_organism"] = [amb_alt[i] for i in amb_human]
    assign.loc[amb_human, "organism"] = "Homo sapiens"
    assign.loc[amb_human, "assignment_source"] = (
        assign.loc[amb_human, "assignment_source"] + "+human_preferred")
print(f"\n[4b] Great-ape sequences also carried by an active non-patent human entry: "
      f"{len(amb_human)}")
print("     reassigned to Homo sapiens (identical sequence cannot distinguish the species;")
print("     the assayed receptor in a human-focused screening database is taken as human).")
print("     alternatives retained in `alternative_organism`; flagged by "
      "`species_ambiguous_with_human`.")
for i in amb_human:
    print(f"       receptor {int(assign.loc[i, 'main_receptors_id'])} "
          f"({assign.loc[i, 'gene_name']}): was {amb_alt[i]}")

MUS_CANONICAL = "Mus musculus"
mus_mask = assign["organism"].astype(str).str.startswith("Mus ") & (
    assign["organism"] != MUS_CANONICAL)
if mus_mask.any():
    assign.loc[mus_mask, "uniparc_organism"] = assign.loc[mus_mask, "organism"]
    assign.loc[mus_mask, "organism"] = MUS_CANONICAL
    print(f"\n[4c] Mus entries normalised to {MUS_CANONICAL}: {int(mus_mask.sum())}")
    print("     UniParc top cross-reference named a sibling taxon, but the M2OR source")
    print("     records lists every Mus receptor as Mus musculus; the database annotation")
    print("     is authoritative for which animal was assayed.")
    for _, r in assign[mus_mask].iterrows():
        print(f"       receptor {int(r['main_receptors_id'])} "
              f"({r['gene_name']}): was {r['uniparc_organism']}")

assign["species_confirmed_against_m2or"] = assign["organism"].isin(
    [MUS_CANONICAL, "Homo sapiens"]) & (
    assign.get("species_ambiguous_with_human", False) | mus_mask)

assign["is_human"] = assign["organism"].astype(str).str.contains("Homo sapiens", na=False)
assign.to_csv(OUT_ASSIGN, index=False)

print(f"  assignment_source:\n{assign['assignment_source'].value_counts().to_string()}")
print(f"\n  organism (top 12):\n"
      f"{assign['organism'].fillna('UNASSIGNED').value_counts().head(12).to_string()}")
print(f"\n  human receptors: {int(assign['is_human'].sum())} / {len(assign)}")
print(f"\n  saved: {OUT_ASSIGN}")
print("=" * 78)
