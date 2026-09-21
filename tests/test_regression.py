import math
import os
import re

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "results")
REVISION = os.path.join(RESULTS, "revision")
CANONICAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "canonical_values.yaml")

with open(CANONICAL_PATH) as handle:
    CANONICAL = yaml.safe_load(handle)

UNVERIFIED = {
    ("25_determinism_check", "archive_members_matched"):
        "the archive comparison runs against the author's private source tree and is "
        "not deposited",
    ("16_stereoisomer_exhaustive", "case_weighted_accuracy"):
        "the per-seed accuracies behind the standard deviation are not written to a "
        "stored artefact",
    ("16_stereoisomer_exhaustive", "strict_separation"):
        "the per-seed accuracies behind the standard deviation are not written to a "
        "stored artefact",
}
BOOTSTRAP_INTERVALS = {("12_structure_accuracy", "cles_ig"),
                       ("12_structure_accuracy", "cles_dp")}


def digits(value):
    text = repr(float(value))
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def agrees(observed, expected):
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(round(observed)) == expected
    if expected == 0:
        return abs(observed) < 1e-12
    if abs(expected) < 1e-3:
        exponent = math.floor(math.log10(abs(expected)))
        return (math.floor(math.log10(abs(observed))) == exponent
                and round(observed / 10 ** exponent, 1) == round(expected / 10 ** exponent, 1))
    return round(observed, digits(expected)) == pytest.approx(expected, abs=1e-12)


def read(path):
    full = path if os.path.isabs(path) else os.path.join(REVISION, path)
    if not os.path.exists(full):
        pytest.skip(f"missing artefact {os.path.relpath(full, REPO_ROOT)}")
    return pd.read_csv(full)


def significance_modality():
    table = read("table_S10.csv")
    mcc = table[table.metric == "MCC"].set_index("model")
    keys = {"xgboost": "XGBoost", "lightgbm": "LightGBM",
            "catboost": "CatBoost", "student": "Student"}
    return {key: {"semantic": [float(mcc.loc[label].mean_semantic),
                               float(mcc.loc[label].sd_semantic)],
                  "hybrid": [float(mcc.loc[label].mean_hybrid),
                             float(mcc.loc[label].sd_hybrid)],
                  "delta": float(mcc.loc[label].delta),
                  "cohens_dz": float(mcc.loc[label].effect_size),
                  "adjusted_p": float(mcc.loc[label].adj_p)}
            for key, label in keys.items()}


def dataset_audit():
    species = read("T1_species_table.csv")
    assignment = read("T1_receptor_assignment.csv")
    source = assignment.assignment_source.value_counts()
    human = int(species[species.organism == "Homo sapiens"].n_receptors.iloc[0])
    return {
        "human_ors": human,
        "nonhuman_ors": int(species.n_receptors.sum()) - human,
        "organisms": int(len(species)),
        "mus_musculus": int(species[species.organism == "Mus musculus"].n_receptors.iloc[0]),
        "uniparc_archived": int(sum(count for name, count in source.items()
                                    if str(name).startswith("uniparc_exact"))),
        "unarchived_variants": int(source.get("nearest_neighbour", 0)),
        "tagged_constructs": int(source.get("local_alignment", 0)),
    }


def species_restricted_eval():
    metrics = read("T2_subset_metrics.csv").set_index("Condition")
    return {
        "auroc_human_only": [float(metrics.loc["human"].AUROC_mean),
                             float(metrics.loc["human"].AUROC_std_ddof1)],
        "auroc_full": [float(metrics.loc["all"].AUROC_mean),
                       float(metrics.loc["all"].AUROC_std_ddof1)],
        "auroc_mus": float(metrics.loc["mouse"].AUROC_mean),
    }


def near_identical_variants():
    subgroups = read("T3_subgroups.csv").set_index("subgroup")
    pairs = read("T3_variant_pairs.csv")
    seeds = read("T3_seed_level.csv")
    close = read("T1_close_sequence_pairs.csv")
    distinct = pairs.assign(
        key=list(zip(pairs.receptor_active, pairs.receptor_inactive)))
    first = pairs[pairs.edit_distance == 1]
    return {
        "near_identical_pairs": int(len(close)),
        "opposite_label_triples": int(subgroups.loc["all"].n_triples),
        "distance_one_triples": int(subgroups.loc["edit_distance_1"].n_triples),
        "distance_one_pairs": int(distinct[distinct.edit_distance == 1].key.nunique()),
        "distance_two_pairs": int(distinct[distinct.edit_distance == 2].key.nunique()),
        "ranking_accuracy_overall": [float(subgroups.loc["all"].ranking_rate),
                                     float(seeds.prob_ordering_rate.std(ddof=1))],
        "ranking_accuracy_d1": float(subgroups.loc["edit_distance_1"].ranking_rate),
        "ranking_accuracy_d2": float(subgroups.loc["edit_distance_2"].ranking_rate),
        "ranking_accuracy_ec50": [
            float(subgroups.loc["edit_distance_1_both_ec50"].ranking_rate), None],
        "strict_separation": [float(subgroups.loc["all"].strict_rate),
                              float(seeds.strict_separation_rate.std(ddof=1))],
        "cosine_mean": float(first.prot_cosine.mean()),
        "cosine_median": float(first.prot_cosine.median()),
    }


def panel_screening():
    summary = read("table_S11.csv").set_index("odorant_ligand")
    result = {}
    for key, label in (("isoeugenol", "(E)-Isoeugenol"),
                       ("androstenone", "Androstenone")):
        row = summary.loc[label]
        result[key] = {
            "N": int(row.panel_size), "K": int(row.n_responsive_K),
            "K_prime": int(row.top20_rank),
            "auroc": [float(row.auroc_mean), float(row.auroc_sd)],
            "auprc": [float(row.auprc_mean), float(row.auprc_sd)],
            "precision_at_K": [float(row.precision_at_K_mean),
                               float(row.precision_at_K_sd)],
            "enrichment": float(row.enrichment_mean),
            "recall_at_Kprime": [float(row.recall_top20_mean), float(row.recall_top20_sd)],
        }
    return result


def additivity_decomposition():
    frame = read("T13d_variance_decomposition.csv")
    logit = frame[frame.scale == "logit"]
    probability = frame[frame.scale == "probability"]
    return {
        "additive_r2": [float(logit.additive_r2.mean()),
                        float(logit.additive_r2.std(ddof=1))],
        "interaction_share": [float(logit.share_interaction.mean()),
                              float(logit.share_interaction.std(ddof=1))],
        "ligand_marginal": float(logit.share_ligand_marginal.mean()),
        "or_marginal": float(logit.share_receptor_marginal.mean()),
        "probability_scale_interaction": float(probability.share_interaction.mean()),
        "median_iterations": int(logit.iterations.median()),
    }


def structure_accuracy():
    deviation = read("T8_or51e2_deviation.csv")
    contacts = read("table_S12.csv").set_index("residue")
    enrichment = read("T6d_enrichment.csv")
    overall = enrichment[enrichment.top_n.isna()].set_index("method")
    notes_path = os.path.join(REVISION, "table_S12_notes.txt")
    if not os.path.exists(notes_path):
        pytest.skip("missing artefact results/revision/table_S12_notes.txt")
    notes = open(notes_path).read()
    cles = {name: [float(value) for value in match]
            for name, match in zip(("ig", "dp"),
                                   re.findall(r"CLES ([\d.]+) \[([\d.]+), ([\d.]+)\]",
                                              notes))}
    return {
        "global_ca_rmsd": float(np.sqrt((deviation.deviation_angstrom ** 2).mean())),
        "within_2A_pct": float((deviation.deviation_angstrom < 2).mean() * 100),
        "within_3A_pct": float((deviation.deviation_angstrom < 3).mean() * 100),
        "contact_mean_deviation": float(contacts.ca_deviation_A.mean()),
        "r262_deviation": float(contacts.loc["R262"].ca_deviation_A),
        "h180_deviation": float(contacts.loc["H180"].ca_deviation_A),
        "median_rank_ig": int(overall.loc["integrated_gradients"].median_rank_of_contacts),
        "median_rank_dp": int(overall.loc["alanine_mutagenesis"].median_rank_of_contacts),
        "cles_ig": cles["ig"],
        "cles_dp": cles["dp"],
    }


def stereoisomer_exhaustive():
    contingency = read("table_S6.csv")
    pairs = read("T5_stereoisomer_pairs.csv")
    instances = read("T5_chirality_all.csv")
    summary = read("stereo_pair_summary.csv")
    doubly = summary[summary.both_directions]
    total = int(contingency.n_seed_instances.sum())
    correct = contingency[contingency.ordering == "correct"]
    return {
        "stereoisomeric_pairs": int(len(pairs)),
        "usable_pairs": int(len({tuple(sorted(item)) for item
                                 in zip(instances.compound_active,
                                        instances.compound_inactive)})),
        "instances": int(len(instances)),
        "ors": int(instances.receptor_id.nunique()),
        "ordered_combinations": int(len(set(zip(instances.compound_active,
                                                instances.compound_inactive)))),
        "seed_instances": total,
        "correct_ordering_pct": float(correct.share.sum() * 100),
        "correct_below_threshold_pct": float(
            correct[correct.threshold_config == "both_below"].share.sum() * 100),
        "correct_above_threshold_pct": float(
            correct[correct.threshold_config == "both_above"].share.sum() * 100),
        "below_threshold_any_ordering_pct": float(
            contingency[contingency.threshold_config == "both_below"].share.sum() * 100),
        "label_balanced_accuracy": float((doubly.sigma / 2).mean()),
    }


def attribution_registry():
    summary = read(os.path.join(REVISION, "attribution_registry", "summary_unified.csv"))
    summary = summary.set_index(["condition", "level"])
    blocks = {
        "ig_or_comparison": ("IG / OR side", "comparison"),
        "ig_or_cluster": ("IG / OR side", "cluster:receptor"),
        "dp_or_comparison": ("dP / OR side", "comparison"),
        "dp_or_cluster": ("dP / OR side", "cluster:receptor"),
        "ig_ligand_comparison": ("IG / Ligand side", "comparison"),
        "ig_ligand_cluster": ("IG / Ligand side", "cluster:ligand"),
    }
    result = {}
    for name, key in blocks.items():
        row = summary.loc[key]
        entry = {"n": int(row.n), "hl": float(row.hodges_lehmann),
                 "ci": [float(row.ci_low_normal), float(row.ci_high_normal)],
                 "within_median": float(row.marginal_median_within),
                 "floor_median": float(row.marginal_median_floor),
                 "within_iqr": [float(row.iqr_within_low), float(row.iqr_within_high)],
                 "floor_iqr": [float(row.iqr_floor_low), float(row.iqr_floor_high)],
                 "wilcoxon_p": float(row.wilcoxon_p),
                 "n_exceeding_floor": int(row.n_exceeding_floor),
                 "sign_test": [int(row.sign_positive), int(row.sign_n),
                               float(row.sign_p)]}
        result[name] = entry
    return result


def untrained_control():
    table = read("table_S15.csv").set_index(["method", "condition"])
    ranks = read("T13c_motif_ranks_pairs.csv")
    receptor = (ranks.groupby(["method", "condition", "receptor_id"])["rank"]
                .median().reset_index())
    trained = receptor[receptor.condition == "trained"].pivot(
        index="receptor_id", columns="method", values="rank")
    first = {condition: set(receptor[(receptor.method == "ig")
                                     & (receptor.condition == condition)
                                     & (receptor["rank"] == 1)].receptor_id)
             for condition in ("trained", "untrained")}
    from scipy import stats as scipy_stats
    return {
        "ig_first_untrained": int(table.loc[("integrated gradients",
                                             "untrained")].n_or_ranked_first),
        "ig_first_trained": int(table.loc[("integrated gradients",
                                           "trained")].n_or_ranked_first),
        "ig_overlap": int(len(first["trained"] & first["untrained"])),
        "dp_first_untrained": int(table.loc[("in silico alanine mutagenesis",
                                             "untrained")].n_or_ranked_first),
        "dp_first_trained": int(table.loc[("in silico alanine mutagenesis",
                                           "trained")].n_or_ranked_first),
        "ors": int(table.loc[("integrated gradients", "trained")].n_or_total),
        "poisson_binomial_expectation": float(
            table.loc[("integrated gradients", "trained")].poisson_binomial_expected),
        "p_ig": float(table.loc[("integrated gradients", "trained")].p_value),
        "p_dp": float(table.loc[("in silico alanine mutagenesis", "trained")].p_value),
        "both_first": int(((trained.ig == 1) & (trained.dp == 1)).sum()),
        "agree_top1": int(((trained.ig == 1) == (trained.dp == 1)).sum()),
        "rank_correlation": float(scipy_stats.spearmanr(trained.ig, trained.dp).statistic),
    }


def class1_candidates():
    table = read("table_S16.csv")
    return {
        "pairs": int(len(table)),
        "receptor_entries": int(table.main_receptors_id.nunique()),
        "gene_symbols": int(table.gene_name.nunique()),
    }


EXTRACTORS = {
    "04_significance_modality": significance_modality,
    "07_dataset_audit": dataset_audit,
    "08_species_restricted_eval": species_restricted_eval,
    "09_near_identical_variants": near_identical_variants,
    "10_panel_screening": panel_screening,
    "11_additivity_decomposition": additivity_decomposition,
    "12_structure_accuracy": structure_accuracy,
    "16_stereoisomer_exhaustive": stereoisomer_exhaustive,
    "20_attribution_registry": attribution_registry,
    "21_untrained_control": untrained_control,
    "24_class1_candidates": class1_candidates,
}


def flatten(prefix, value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(prefix + (key,), item)
    else:
        yield prefix, value


CASES = [(script, path, expected)
         for script, block in CANONICAL.items()
         for path, expected in flatten((), block)]


@pytest.mark.parametrize("script,path,expected",
                         [pytest.param(*case, id=f"{case[0]}::{'.'.join(case[1])}")
                          for case in CASES])
def test_canonical_value(script, path, expected):
    if (script, path[0]) in UNVERIFIED:
        pytest.skip(UNVERIFIED[(script, path[0])])
    if script not in EXTRACTORS:
        pytest.skip(f"no stored artefact is wired up for {script}")
    observed = EXTRACTORS[script]()
    for key in path:
        observed = observed[key]
    if isinstance(expected, list):
        assert isinstance(observed, list) and len(observed) == len(expected)
        limit = 1 if (script, path[0]) in BOOTSTRAP_INTERVALS else len(expected)
        for index, (value, target) in enumerate(zip(observed, expected)):
            if target is None or value is None or index >= limit:
                continue
            assert agrees(value, target), f"{path}: {value} vs {target}"
    else:
        assert agrees(observed, expected), f"{path}: {observed} vs {expected}"
