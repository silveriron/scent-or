import os
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from Bio.SeqUtils.CheckSum import crc64

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "results", "revision")
os.makedirs(OUT_DIR, exist_ok=True)

STRINGS_CSV = os.path.join(DATA_DIR, "final_dataset_weighted_with_strings.csv")
UPI_CACHE = os.path.join(OUT_DIR, "T1_uniparc_upi_cache.json")
ENTRY_CACHE = os.path.join(OUT_DIR, "T1_uniparc_entry_cache.json")
OUT_CSV = os.path.join(OUT_DIR, "T1_receptor_species.csv")

BATCH = 40
WORKERS = 8
TIMEOUT = 60
RETRIES = 3
DB_PRIORITY = ["UniProtKB/Swiss-Prot", "UniProtKB/TrEMBL", "RefSeq", "Ensembl",
               "EnsemblRapid", "EMBL"]
NON_SPECIES_TOKENS = ["synthetic construct", "vector", "orfeome", "cloning",
                      "expression construct", "unidentified", "plasmid"]


def is_real_species(name):
    if not name:
        return False
    low = name.lower()
    return not any(tok in low for tok in NON_SPECIES_TOKENS)


def http_json(url):
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
                return json.load(fh)
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def load_cache(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_cache(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


print("=" * 78)
print("  T1a — receptor species resolution via UniParc exact-sequence lookup")
print("=" * 78)

df = pd.read_csv(STRINGS_CSV)
print(f"  input: {STRINGS_CSV}  ({len(df)} rows)")
seqs = df.drop_duplicates("main_receptors_id").set_index("main_receptors_id")["mutated_sequence"]
seqs = seqs.sort_index()
print(f"  receptor ids: {len(seqs)}  unique sequences: {seqs.nunique()}")

checksum_of = {int(rid): crc64(seq).replace("CRC-", "") for rid, seq in seqs.items()}
unique_ck = sorted(set(checksum_of.values()))
print(f"  unique CRC64 checksums: {len(unique_ck)}")

upi_cache = load_cache(UPI_CACHE)
todo = [c for c in unique_ck if c not in upi_cache]
print(f"\n  UniParc checksum -> UPI: {len(unique_ck) - len(todo)} cached, {len(todo)} to fetch")

for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    q = "checksum:(" + " OR ".join(chunk) + ")"
    url = ("https://rest.uniprot.org/uniparc/stream?query="
           + urllib.parse.quote(q) + "&format=json")
    data = http_json(url)
    got = {}
    for e in data.get("results", []):
        ck = (e.get("sequence") or {}).get("crc64")
        if ck:
            got[ck] = e.get("uniParcId")
    for c in chunk:
        upi_cache[c] = got.get(c)
    save_cache(UPI_CACHE, upi_cache)
    print(f"    {min(i + BATCH, len(todo))}/{len(todo)}  (+{len(got)} hits)")

n_upi = sum(1 for c in unique_ck if upi_cache.get(c))
print(f"  resolved to UPI: {n_upi}/{len(unique_ck)}")

entry_cache = load_cache(ENTRY_CACHE)
upis = sorted({upi_cache[c] for c in unique_ck if upi_cache.get(c)})
todo_e = [u for u in upis if u not in entry_cache]
print(f"\n  UPI -> cross-references: {len(upis) - len(todo_e)} cached, {len(todo_e)} to fetch")


def fetch_entry(upi):
    url = f"https://rest.uniprot.org/uniparc/{upi}?format=json"
    data = http_json(url)
    out = []
    for x in data.get("uniParcCrossReferences", []):
        org = x.get("organism") or {}
        out.append({
            "database": x.get("database"),
            "id": x.get("id"),
            "gene": x.get("geneName"),
            "organism": org.get("scientificName"),
            "taxon": org.get("taxonId"),
            "active": bool(x.get("active")),
        })
    return upi, out


if todo_e:
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for upi, refs in ex.map(fetch_entry, todo_e):
            entry_cache[upi] = refs
            done += 1
            if done % 100 == 0 or done == len(todo_e):
                save_cache(ENTRY_CACHE, entry_cache)
                print(f"    {done}/{len(todo_e)}")
    save_cache(ENTRY_CACHE, entry_cache)


def pick(refs):
    organisms = Counter()
    genes = Counter()
    construct_only = Counter()
    for r in refs:
        if r["organism"]:
            if is_real_species(r["organism"]):
                organisms[r["organism"]] += 1
            else:
                construct_only[r["organism"]] += 1
        if r["gene"]:
            genes[r["gene"]] += 1
    for active_only in (True, False):
        for db in DB_PRIORITY:
            for r in refs:
                if (r["database"] == db and is_real_species(r["organism"])
                        and (r["active"] or not active_only)):
                    return r, organisms, genes, construct_only
    return None, organisms, genes, construct_only


rows = []
for rid in seqs.index:
    ck = checksum_of[int(rid)]
    upi = upi_cache.get(ck)
    refs = entry_cache.get(upi, []) if upi else []
    best, organisms, genes, constructs = pick(refs)
    rows.append({
        "main_receptors_id": int(rid),
        "sequence_length": len(seqs[rid]),
        "crc64": ck,
        "uniparc_upi": upi,
        "source_db": best["database"] if best else None,
        "accession": best["id"] if best else None,
        "gene_name": best["gene"] if best else (genes.most_common(1)[0][0] if genes else None),
        "organism": best["organism"] if best else None,
        "taxon_id": best["taxon"] if best else None,
        "n_crossrefs": len(refs),
        "n_distinct_organisms": len(organisms),
        "organism_ambiguous": len(organisms) > 1,
        "all_organisms": "|".join(sorted(organisms)) if organisms else None,
        "construct_labels": "|".join(sorted(constructs)) if constructs else None,
        "resolution": ("uniparc_exact" if best else
                       ("upi_no_real_species" if upi else "no_uniparc_hit")),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

print("\n" + "=" * 78)
print("  resolution breakdown")
for k, v in out["resolution"].value_counts().items():
    print(f"    {k}: {v}")
print("\n  organism breakdown (receptor ids)")
for k, v in out["organism"].fillna("UNRESOLVED").value_counts().items():
    print(f"    {k}: {v}")
print(f"\n  organism-ambiguous UPIs: {int(out['organism_ambiguous'].sum())}")
print(f"  saved: {OUT_CSV}")
print("=" * 78)
