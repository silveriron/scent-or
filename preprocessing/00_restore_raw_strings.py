import argparse
import pandas as pd


def _canonicalize(smiles):
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol) if mol is not None else smiles
    except Exception:
        return smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2or", default="data/pairs.csv")
    ap.add_argument("--curated", default="data/final_dataset_weighted.csv")
    ap.add_argument("--out", default="data/final_dataset_weighted_with_strings.csv")
    ap.add_argument("--canonicalize", action="store_true")
    args = ap.parse_args()

    curated = pd.read_csv(args.curated)
    m2or = pd.read_csv(args.m2or, sep=";")

    lig = (m2or[["main_compounds_id", "smiles"]]
           .dropna(subset=["main_compounds_id"])
           .drop_duplicates(subset=["main_compounds_id"]))
    rec = (m2or[["main_receptors_id", "mutated_sequence"]]
           .dropna(subset=["main_receptors_id"])
           .drop_duplicates(subset=["main_receptors_id"]))

    merged = curated.merge(lig, on="main_compounds_id", how="left")
    merged = merged.merge(rec, on="main_receptors_id", how="left")

    if args.canonicalize:
        unique_smiles = merged["smiles"].dropna().unique()
        canon_lookup = {s: _canonicalize(s) for s in unique_smiles}
        merged["smiles"] = merged["smiles"].map(
            lambda s: canon_lookup.get(s, s) if pd.notna(s) else s)

    n_missing_smiles = merged["smiles"].isna().sum()
    n_missing_seq = merged["mutated_sequence"].isna().sum()
    if n_missing_smiles or n_missing_seq:
        print(f"[warning] unmatched rows -- smiles: {n_missing_smiles}, "
              f"mutated_sequence: {n_missing_seq}. "
              f"Check that the M2OR export version matches the curated identifiers.")

    merged.to_csv(args.out, index=False)
    print(f"[done] wrote {len(merged)} rows with restored strings to {args.out}")
    print("Point the XAI scripts (RAW_DATA_CSV) at this file.")


if __name__ == "__main__":
    main()
