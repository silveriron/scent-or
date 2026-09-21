import importlib.util
import os
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_SCRIPT = os.path.join(BASE_DIR, "01_threshold_fitting.py")
OOD_SCRIPT = os.path.join(BASE_DIR, "23_ood_enantiomer.py")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "fn_margin")
CSV_PATH = os.path.join(DATA_DIR, "final_dataset_weighted.csv")
STRINGS_PATH = os.path.join(DATA_DIR, "final_dataset_weighted_with_strings.csv")
PKL_PATH = os.path.join(DATA_DIR, "final_embeddings_all.pkl")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "thresholds_Scenario5.csv")
OOD_PATH = os.path.join(RESULTS_DIR, "ood", "ood_sotolon_results.csv")
OOF_TEMPLATE = os.path.join(RESULTS_DIR, "OOF_CM_Scenario5_seed{seed}.csv")
RECEPTOR_TABLE_PATH = os.path.join(RESULTS_DIR, "revision", "T1_receptor_master_table.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "oof_predictions.parquet")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "fn_margin_summary.csv")
CANDIDATES_PATH = os.path.join(OUTPUT_DIR, "sotolon_candidates.csv")
DIAGNOSTICS_PATH = os.path.join(OUTPUT_DIR, "sotolon_diagnostics.csv")

PROPIONIC_COMPOUND_ID = 56
OR51E2_RECEPTOR_ID = 168
MAJORITY_SEEDS = 10
MANUSCRIPT_TOTAL = 794460
MANUSCRIPT_FN = 20248
MANUSCRIPT_NEAR_FRACTION = 0.277
MANUSCRIPT_FAR_FRACTION = 0.224
MANUSCRIPT_SOTOLON_MARGIN = -0.327
MANUSCRIPT_SOTOLON_PERCENTILE = 6.0
MANUSCRIPT_OR51E2_BEST = -0.0075
MANUSCRIPT_OR51E2_PERCENTILE = 65.0
NEAR_BAND = 0.10
FAR_BAND = -0.30


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(smiles, isomeric):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=isomeric)


def percentile_below(distribution, value):
    return 100.0 * float(np.mean(distribution < value))


def percentile_by_distance(distribution, value):
    return 100.0 * float(np.mean(np.abs(distribution) < abs(value)))


def task_one(threshold_module, ood_module):
    print("\n" + "=" * 78)
    print("  Task 1 — is the sotolon / OR8D1 pair present in the training data")
    print("=" * 78)
    strings = pd.read_csv(STRINGS_PATH)
    mapping = pd.read_csv(CSV_PATH)
    print(f"  {STRINGS_PATH}  {len(strings)} rows")
    print(f"  {CSV_PATH}  {len(mapping)} rows")
    assert len(strings) == len(mapping)
    assert (strings.main_compounds_id.values == mapping.main_compounds_id.values).all()
    assert (strings.main_receptors_id.values == mapping.main_receptors_id.values).all()

    sequence = ood_module.OR8D1_SEQUENCE
    exact = strings[strings.mutated_sequence == sequence]
    receptor_ids = sorted(exact.main_receptors_id.unique().tolist())
    print(f"\n  OR8D1 sequence length {len(sequence)}")
    print(f"  exact sequence matches in the dataset: {len(exact)} rows, "
          f"receptor ids {receptor_ids}")
    if os.path.exists(RECEPTOR_TABLE_PATH):
        table = pd.read_csv(RECEPTOR_TABLE_PATH)
        gene_column = next((c for c in table.columns if "gene" in c.lower()), None)
        if gene_column is not None:
            hits = table[table[gene_column].astype(str).str.upper() == "OR8D1"]
            print(f"  {RECEPTOR_TABLE_PATH}: gene column '{gene_column}', "
                  f"OR8D1 entries {hits.iloc[:, 0].tolist() if len(hits) else 'none'}")

    RDLogger.DisableLog("rdApp.*")
    targets = {"(S)-Sotolon": ood_module.S_SOTOLON_SMILES,
               "(R)-Sotolon": ood_module.R_SOTOLON_SMILES}
    dataset_smiles = strings.drop_duplicates("main_compounds_id")[
        ["main_compounds_id", "smiles"]]
    isomeric_map, skeleton_map = {}, {}
    for compound_id, smiles in zip(dataset_smiles.main_compounds_id,
                                   dataset_smiles.smiles):
        isomeric_map.setdefault(canonical(smiles, True), []).append(int(compound_id))
        skeleton_map.setdefault(canonical(smiles, False), []).append(int(compound_id))
    print(f"\n  unique compounds in the dataset: {len(dataset_smiles)}")
    compound_hits = {}
    for name, smiles in targets.items():
        isomeric = canonical(smiles, True)
        skeleton = canonical(smiles, False)
        isomeric_hit = isomeric_map.get(isomeric, [])
        skeleton_hit = skeleton_map.get(skeleton, [])
        compound_hits[name] = {"isomeric": isomeric_hit, "skeleton": skeleton_hit}
        print(f"  {name:12s} canonical {isomeric}")
        print(f"  {'':12s} isomeric match {isomeric_hit if isomeric_hit else 'none'}   "
              f"skeleton match {skeleton_hit if skeleton_hit else 'none'}")

    pair_rows = []
    for name, hit in compound_hits.items():
        for compound_id in hit["isomeric"] + hit["skeleton"]:
            for receptor_id in receptor_ids:
                found = mapping[(mapping.main_compounds_id == compound_id)
                                & (mapping.main_receptors_id == receptor_id)]
                if len(found):
                    pair_rows.append((name, compound_id, receptor_id, len(found)))
    print(f"\n  sotolon / OR8D1 pairs in the dataset: "
          f"{pair_rows if pair_rows else 'none'}")
    ligand_present = any(hit["isomeric"] or hit["skeleton"]
                         for hit in compound_hits.values())
    if pair_rows:
        verdict = "pair present in the training data"
    elif not ligand_present:
        verdict = ("sotolon absent in any form; the manuscript claim holds "
                   "(the OR8D1 receptor itself does appear with other ligands)")
    else:
        verdict = "individual entities present, pair absent"
    print(f"  receptor present: {bool(receptor_ids)}   "
          f"ligand present: {ligand_present}   pair present: {bool(pair_rows)}")
    print(f"  classification: {verdict}")
    return {"receptor_ids": receptor_ids, "compound_hits": compound_hits,
            "pair_rows": pair_rows, "verdict": verdict}


def task_two():
    print("\n" + "=" * 78)
    print("  Task 2 — sotolon margins over 400 predictions")
    print("=" * 78)
    ood = pd.read_csv(OOD_PATH)
    print(f"  {OOD_PATH}  {len(ood)} rows")
    if len(ood) != 400:
        missing = 400 - len(ood)
        raise SystemExit(f"expected 400 rows, found {len(ood)} ({missing} missing)")
    ood["Margin"] = ood.Probability - ood.Threshold
    ood["Active"] = ood.Probability > ood.Threshold
    print(f"  predictions above threshold: {int(ood.Active.sum())} of {len(ood)}")

    candidates = {}
    for ligand in sorted(ood.Ligand.unique()):
        subset = ood[ood.Ligand == ligand]
        seed_means = subset.groupby("Seed").Margin.mean()
        key = "S" if "(S)" in ligand else "R"
        candidates[f"M1_{key}"] = (float(subset.Margin.mean()),
                                   float(seed_means.std(ddof=1)))
        candidates[f"M2_{key}"] = (float(seed_means.mean()),
                                   float(seed_means.std(ddof=1)))
        print(f"  {ligand:14s} n {len(subset)}  pooled mean {subset.Margin.mean():.4f}  "
              f"seed-level mean {seed_means.mean():.4f}  "
              f"seed SD {seed_means.std(ddof=1):.4f}")
    pair_seed_means = ood.groupby(["Seed", "Ligand"]).Margin.mean().unstack()
    combined = pair_seed_means.mean(axis=1)
    candidates["M3_mean"] = ((candidates["M2_S"][0] + candidates["M2_R"][0]) / 2.0,
                             float(combined.std(ddof=1)))
    pooled_seed_means = ood.groupby("Seed").Margin.mean()
    candidates["M4_pooled"] = (float(ood.Margin.mean()),
                               float(pooled_seed_means.std(ddof=1)))

    s_column = [c for c in pair_seed_means.columns if "(S)" in c][0]
    r_column = [c for c in pair_seed_means.columns if "(R)" in c][0]
    s_higher = int((pair_seed_means[s_column] > pair_seed_means[r_column]).sum())
    print(f"  (S) above (R) in {s_higher} of {len(pair_seed_means)} seeds "
          f"(fold-mean margin)")
    fold_level = ood.pivot_table(index=["Seed", "Fold"], columns="Ligand",
                                 values="Probability")
    s_higher_folds = int((fold_level[s_column] > fold_level[r_column]).sum())
    print(f"  (S) above (R) in {s_higher_folds} of {len(fold_level)} "
          f"(seed, fold) checkpoints")
    print("\n  candidate statistics")
    print(f"  {'id':10s} {'margin':>10s} {'seed SD':>10s}")
    for key in ("M1_S", "M1_R", "M2_S", "M2_R", "M3_mean", "M4_pooled"):
        print(f"  {key:10s} {candidates[key][0]:10.4f} {candidates[key][1]:10.4f}")
    return candidates, {"above_threshold": int(ood.Active.sum()),
                        "s_higher_seeds": s_higher,
                        "s_higher_folds": s_higher_folds,
                        "n_seeds": len(pair_seed_means)}


def run_inference(threshold_module):
    print("\n  running out-of-fold inference on the validation partitions")
    mapping = pd.read_csv(CSV_PATH)
    with open(PKL_PATH, "rb") as handle:
        vault = pickle.load(handle)
    ligand = np.array([vault["ligand_embeddings"].get(i, np.zeros(768))
                       for i in mapping["main_compounds_id"]], dtype=np.float32)
    protein = np.array([vault["protein_embeddings"].get(i, np.zeros(1024))
                        for i in mapping["main_receptors_id"]], dtype=np.float32)
    labels = mapping["responsive"].values.flatten()
    ligand_t = torch.tensor(ligand)
    protein_t = torch.tensor(protein)
    features = np.concatenate([ligand, protein], axis=1)
    thresholds = pd.read_csv(THRESHOLD_PATH)
    threshold_map = {(int(r.Seed), int(r.Fold)): float(r.Threshold)
                     for r in thresholds.itertuples()}

    missing = []
    for seed in threshold_module.SEEDS:
        for fold_num in range(1, threshold_module.N_SPLITS + 1):
            path = os.path.join(threshold_module.MODEL_DIR, f"seed_{seed}",
                                f"{threshold_module.SCENARIO}_seed{seed}_fold{fold_num}.pt")
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise SystemExit(f"missing {len(missing)} checkpoints, first: {missing[0]}")
    print(f"  checkpoints present: "
          f"{len(threshold_module.SEEDS) * threshold_module.N_SPLITS}")

    frames = []
    for seed in threshold_module.SEEDS:
        kfold = StratifiedKFold(n_splits=threshold_module.N_SPLITS, shuffle=True,
                                random_state=seed)
        for fold, (_, val_idx) in enumerate(kfold.split(features, labels)):
            fold_num = fold + 1
            path = os.path.join(threshold_module.MODEL_DIR, f"seed_{seed}",
                                f"{threshold_module.SCENARIO}_seed{seed}_fold{fold_num}.pt")
            model = threshold_module.StudentModelSemantic(proj_dim=512).to(
                threshold_module.DEVICE)
            model.load_state_dict(torch.load(path,
                                             map_location=threshold_module.DEVICE))
            model.eval()
            dataset = TensorDataset(ligand_t[val_idx], protein_t[val_idx])
            loader = DataLoader(dataset, batch_size=threshold_module.BATCH_SIZE,
                                shuffle=False)
            probabilities = []
            with torch.no_grad():
                for ligand_batch, protein_batch in loader:
                    output, _, _ = model(ligand_batch.to(threshold_module.DEVICE),
                                         protein_batch.to(threshold_module.DEVICE))
                    probabilities.extend(
                        torch.sigmoid(output).cpu().numpy().flatten())
            probabilities = np.asarray(probabilities, dtype=np.float64)
            threshold = threshold_map[(seed, fold_num)]
            predicted = probabilities > threshold
            observed = labels[val_idx].astype(int)
            classes = np.where(observed == 1,
                               np.where(predicted, "TP", "FN"),
                               np.where(predicted, "FP", "TN"))
            frames.append(pd.DataFrame({
                "Seed": seed, "Fold": fold_num, "RowIndex": val_idx,
                "Label": observed, "Probability": probabilities,
                "Threshold": threshold, "Margin": probabilities - threshold,
                "Class": classes}))
        print(f"  seed {seed} done")
    return pd.concat(frames, ignore_index=True)


def cross_check_stored_oof(predictions, threshold_module):
    print("\n  cross-check against the stored OOF files")
    deltas = []
    for seed in threshold_module.SEEDS:
        path = OOF_TEMPLATE.format(seed=seed)
        if not os.path.exists(path):
            print(f"  {path} not found")
            return None
        stored = pd.read_csv(path)
        current = predictions[predictions.Seed == seed]
        if len(stored) != len(current):
            print(f"  seed {seed}: stored {len(stored)} rows, recomputed {len(current)}")
            return None
        deltas.append(float(np.abs(stored.Probability.values
                                   - current.Probability.values).max()))
    print(f"  max absolute probability difference over {len(deltas)} seeds: "
          f"{max(deltas):.3e}")
    return max(deltas)


def task_three(threshold_module):
    print("\n" + "=" * 78)
    print("  Task 3 — out-of-fold FN margin distribution")
    print("=" * 78)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(PREDICTIONS_PATH):
        predictions = pd.read_parquet(PREDICTIONS_PATH)
        print(f"  reusing {PREDICTIONS_PATH}  {len(predictions)} rows")
    else:
        predictions = run_inference(threshold_module)
        predictions.to_parquet(PREDICTIONS_PATH, index=False)
        print(f"  saved {PREDICTIONS_PATH}  {len(predictions)} rows")
    delta = cross_check_stored_oof(predictions, threshold_module)

    false_negatives = predictions[predictions.Class == "FN"]
    near = float(np.mean(np.abs(false_negatives.Margin) <= NEAR_BAND))
    far = float(np.mean(false_negatives.Margin < FAR_BAND))
    rows = [
        ("total OOF predictions", MANUSCRIPT_TOTAL, len(predictions)),
        ("false negatives", MANUSCRIPT_FN, len(false_negatives)),
        ("FN within 0.10 of threshold", MANUSCRIPT_NEAR_FRACTION, round(near, 4)),
        ("FN more than 0.30 below", MANUSCRIPT_FAR_FRACTION, round(far, 4)),
    ]
    summary = pd.DataFrame(rows, columns=["item", "manuscript", "recomputed"])
    summary["match"] = [
        summary.manuscript[0] == summary.recomputed[0],
        summary.manuscript[1] == summary.recomputed[1],
        abs(near - MANUSCRIPT_NEAR_FRACTION) < 5e-4,
        abs(far - MANUSCRIPT_FAR_FRACTION) < 5e-4,
    ]
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\n  {'item':32s} {'manuscript':>12s} {'recomputed':>12s}  match")
    for row in summary.itertuples():
        print(f"  {row.item:32s} {str(row.manuscript):>12s} "
              f"{str(row.recomputed):>12s}  {'yes' if row.match else 'NO'}")
    print(f"  saved {SUMMARY_PATH}")

    mapping = pd.read_csv(CSV_PATH)
    positives = predictions[predictions.Label == 1].copy()
    pair_frame = positives.groupby("RowIndex").agg(
        mean_margin=("Margin", "mean"),
        fn_seeds=("Class", lambda values: int((values == "FN").sum())),
        n_seeds=("Class", "size")).reset_index()
    pair_frame["compound"] = mapping.main_compounds_id.values[pair_frame.RowIndex]
    pair_frame["receptor"] = mapping.main_receptors_id.values[pair_frame.RowIndex]
    fn_pair = pair_frame[pair_frame.fn_seeds > MAJORITY_SEEDS]
    print(f"\n  FN_pred distribution: {len(false_negatives)} individual predictions")
    print(f"  FN_pair distribution: {len(fn_pair)} active pairs that are FN in more "
          f"than {MAJORITY_SEEDS} of {int(pair_frame.n_seeds.max())} seeds")
    return predictions, false_negatives.Margin.values, fn_pair, pair_frame, delta


def task_four_diagnostics(fn_pred, fn_pair):
    print("\n  diagnostics for the quoted margin")
    ood = pd.read_csv(OOD_PATH)
    ood["Margin"] = ood.Probability - ood.Threshold
    pair_values = fn_pair.mean_margin.values
    rows = []
    for ligand in sorted(ood.Ligand.unique()):
        subset = ood[ood.Ligand == ligand]
        seed_means = subset.groupby("Seed").Margin.mean()
        rows.append((f"{ligand} best checkpoint", float(subset.Margin.max())))
        rows.append((f"{ligand} best seed", float(seed_means.max())))
        rows.append((f"{ligand} worst checkpoint", float(subset.Margin.min())))
    rows.append(("both best checkpoint", float(ood.Margin.max())))
    print(f"  {'statistic':34s} {'margin':>9s} {'pct(FN_pred)':>13s} "
          f"{'pct(FN_pair)':>13s}")
    for label, value in rows:
        print(f"  {label:34s} {value:9.4f} "
              f"{percentile_below(fn_pred, value):13.2f} "
              f"{percentile_below(pair_values, value):13.2f}")
    quoted = percentile_below(fn_pred, MANUSCRIPT_SOTOLON_MARGIN)
    print(f"  {'manuscript -0.327':34s} {MANUSCRIPT_SOTOLON_MARGIN:9.4f} "
          f"{quoted:13.2f} "
          f"{percentile_below(pair_values, MANUSCRIPT_SOTOLON_MARGIN):13.2f}")
    print(f"  margin at the 6th percentile: FN_pred "
          f"{float(np.percentile(fn_pred, MANUSCRIPT_SOTOLON_PERCENTILE)):.4f}  "
          f"FN_pair {float(np.percentile(pair_values, MANUSCRIPT_SOTOLON_PERCENTILE)):.4f}")
    rows.append(("manuscript quoted value", MANUSCRIPT_SOTOLON_MARGIN))
    frame = pd.DataFrame([
        {"statistic": label, "margin": round(value, 4),
         "pct_fn_pred_ascending": round(percentile_below(fn_pred, value), 2),
         "pct_fn_pair_ascending": round(percentile_below(pair_values, value), 2)}
        for label, value in rows])
    frame.to_csv(DIAGNOSTICS_PATH, index=False)
    print(f"  saved {DIAGNOSTICS_PATH}")
    return rows, quoted


def task_four(candidates, fn_pred, fn_pair, pair_frame, predictions):
    print("\n" + "=" * 78)
    print("  Task 4 — percentile matching")
    print("=" * 78)
    mapping = pd.read_csv(CSV_PATH)
    target = mapping[(mapping.main_compounds_id == PROPIONIC_COMPOUND_ID)
                     & (mapping.main_receptors_id == OR51E2_RECEPTOR_ID)]
    if len(target) != 1:
        raise SystemExit(f"OR51E2 / propionic acid rows found: {len(target)}")
    row_index = int(target.index[0])
    subset = predictions[predictions.RowIndex == row_index]
    seed_margins = subset.set_index("Seed").Margin.sort_index()
    best = float(seed_margins.max())
    averaged = float(seed_margins.mean())
    fn_seeds = int((subset.Class == "FN").sum())
    pair_values = fn_pair.mean_margin.values
    print(f"  OR51E2 / propionic acid: row {row_index}, label "
          f"{int(target.responsive.iloc[0])}, FN in {fn_seeds} of {len(subset)} seeds")
    print(f"  best seed margin {best:.4f} (manuscript {MANUSCRIPT_OR51E2_BEST})")
    print(f"  seed-averaged margin {averaged:.4f}")
    print(f"  percentile ascending  FN_pred {percentile_below(fn_pred, averaged):.1f}  "
          f"FN_pair {percentile_below(pair_values, averaged):.1f}")
    print(f"  percentile by distance FN_pred "
          f"{percentile_by_distance(fn_pred, averaged):.1f}  "
          f"FN_pair {percentile_by_distance(pair_values, averaged):.1f}   "
          f"(manuscript {MANUSCRIPT_OR51E2_PERCENTILE})")

    records = []
    for key in ("M1_S", "M1_R", "M2_S", "M2_R", "M3_mean", "M4_pooled"):
        margin, sd = candidates[key]
        records.append({
            "candidate": key,
            "margin": round(margin, 4),
            "seed_sd": round(sd, 4),
            "pct_fn_pred_ascending": round(percentile_below(fn_pred, margin), 2),
            "pct_fn_pair_ascending": round(percentile_below(pair_values, margin), 2),
            "pct_fn_pred_distance": round(percentile_by_distance(fn_pred, margin), 2),
            "pct_fn_pair_distance": round(percentile_by_distance(pair_values, margin), 2),
            "abs_margin_error": round(abs(margin - MANUSCRIPT_SOTOLON_MARGIN), 4),
        })
    table = pd.DataFrame(records)
    table["pct_error_vs_6"] = (table.pct_fn_pred_ascending
                               - MANUSCRIPT_SOTOLON_PERCENTILE).round(2)
    table.to_csv(CANDIDATES_PATH, index=False)
    print(f"\n  {'candidate':10s} {'margin':>9s} {'pct(FN_pred)':>13s} "
          f"{'pct(FN_pair)':>13s} {'|m+0.327|':>10s} {'pct err':>9s}")
    for row in table.itertuples():
        print(f"  {row.candidate:10s} {row.margin:9.4f} "
              f"{row.pct_fn_pred_ascending:13.2f} {row.pct_fn_pair_ascending:13.2f} "
              f"{row.abs_margin_error:10.4f} {row.pct_error_vs_6:9.2f}")
    print(f"  saved {CANDIDATES_PATH}")
    return table, {"row_index": row_index, "best": best, "averaged": averaged,
                   "fn_seeds": fn_seeds,
                   "pct_pred_asc": percentile_below(fn_pred, averaged),
                   "pct_pair_asc": percentile_below(pair_values, averaged),
                   "pct_pred_dist": percentile_by_distance(fn_pred, averaged),
                   "pct_pair_dist": percentile_by_distance(pair_values, averaged)}


if __name__ == "__main__":
    print("=" * 78)
    print("  FN margin verification for the sotolon limitation statement")
    print("=" * 78)
    threshold_module = load_module(THRESHOLD_SCRIPT, "threshold_fitting_module")
    ood_module = load_module(OOD_SCRIPT, "ood_enantiomer_module")
    print(f"  reusing {THRESHOLD_SCRIPT}")
    print(f"  reusing {OOD_SCRIPT}")
    print(f"  device {threshold_module.DEVICE}")
    print(f"  seeds {threshold_module.SEEDS}")
    print(f"  splits StratifiedKFold(n_splits={threshold_module.N_SPLITS}, "
          f"shuffle=True, random_state=seed)")
    presence = task_one(threshold_module, ood_module)
    candidates, ordering = task_two()
    predictions, fn_pred, fn_pair, pair_frame, delta = task_three(threshold_module)
    table, or51e2 = task_four(candidates, fn_pred, fn_pair, pair_frame, predictions)
    task_four_diagnostics(fn_pred, fn_pair)
    print("\n" + "=" * 78)
    print("  done")
    print("=" * 78)
