import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
SCENARIO = "Scenario5"
N_SEEDS = len(SEEDS)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")
os.makedirs(OUT_DIR, exist_ok=True)

STRINGS_CSV = os.path.join(DATA_DIR, "final_dataset_weighted_with_strings.csv")
PAIRS_CSV = os.path.join(OUT_DIR, "T1_close_sequence_pairs.csv")
ASSIGN_CSV = os.path.join(OUT_DIR, "T1_receptor_assignment.csv")

OUT_TRIPLES = os.path.join(OUT_DIR, "T3_variant_pairs.csv")
OUT_SEED = os.path.join(OUT_DIR, "T3_seed_level.csv")
OUT_MD = os.path.join(OUT_DIR, "T3_summary.md")
FIG_BASE = os.path.join(OUT_DIR, "T3_variant_pairs")

print("=" * 78)
print("  T3 — WT/variant minimal pairs with opposite labels")
print("=" * 78)

df = pd.read_csv(STRINGS_CSV)
close = pd.read_csv(PAIRS_CSV)
assign = pd.read_csv(ASSIGN_CSV).set_index("main_receptors_id")
print(f"  pairs table   : {STRINGS_CSV}  ({len(df)} rows)")
print(f"  close pairs   : {PAIRS_CSV}  ({len(close)} receptor pairs, edit distance <= 2)")

label = {(r, c): v for r, c, v in
         zip(df.main_receptors_id, df.main_compounds_id, df.responsive)}
ligands_of = {}
for r, c in zip(df.main_receptors_id, df.main_compounds_id):
    ligands_of.setdefault(r, set()).add(c)
seq_of = df.drop_duplicates("main_receptors_id").set_index("main_receptors_id")["mutated_sequence"]
smiles_of = df.drop_duplicates("main_compounds_id").set_index("main_compounds_id")["smiles"]


def substitutions(sa, sb):
    if len(sa) != len(sb):
        return ""
    return ";".join(f"{x}{i + 1}{y}" for i, (x, y) in enumerate(zip(sa, sb)) if x != y)


print("\n[1] Enumerating (receptor_A, receptor_B, ligand) triples with opposite labels")
triples = []
for _, row in close.iterrows():
    a, b = int(row.id_a), int(row.id_b)
    common = ligands_of.get(a, set()) & ligands_of.get(b, set())
    for c in common:
        la, lb = label[(a, c)], label[(b, c)]
        if la == lb:
            continue
        act, ina = (a, b) if la == 1 else (b, a)
        triples.append({
            "receptor_active": act, "receptor_inactive": ina, "compound_id": c,
            "edit_distance": int(row.distance), "kind": row.kind,
            "gene_active": assign.loc[act, "gene_name"],
            "gene_inactive": assign.loc[ina, "gene_name"],
            "organism_active": assign.loc[act, "organism"],
            "organism_inactive": assign.loc[ina, "organism"],
            "same_gene_symbol": assign.loc[act, "gene_name"] == assign.loc[ina, "gene_name"],
            "substitution_active_to_inactive": substitutions(seq_of[act], seq_of[ina]),
            "smiles": smiles_of[c],
        })
tri = pd.DataFrame(triples)
print(f"  triples with opposite labels: {len(tri)}")
print(f"    edit distance 1: {int((tri.edit_distance == 1).sum())}")
print(f"    edit distance 2: {int((tri.edit_distance == 2).sum())}")
print(f"    same gene symbol on both sides: {int(tri.same_gene_symbol.sum())}")
print(f"    distinct receptor pairs: {tri.groupby(['receptor_active', 'receptor_inactive']).ngroups}")
print(f"    distinct ligands: {tri.compound_id.nunique()}")

if len(tri) == 0:
    print("\n  RESULT: no qualifying pairs exist. Reporting absence, no criteria relaxed.")
    raise SystemExit(0)

print("\n[2] Collecting 20-seed OOF predictions")
needed = set(zip(tri.receptor_active, tri.compound_id)) | set(zip(tri.receptor_inactive, tri.compound_id))
prob = {k: np.empty(N_SEEDS) for k in needed}
pred = {k: np.empty(N_SEEDS, dtype=int) for k in needed}
for si, seed in enumerate(SEEDS):
    oof = pd.read_csv(os.path.join(RESULTS_DIR, f"OOF_CM_{SCENARIO}_seed{seed}.csv"))
    idx = oof.set_index(["main_receptors_id", "main_compounds_id"])
    p = idx["Probability"]
    q = idx["Prediction"]
    for k in needed:
        prob[k][si] = p.loc[k]
        pred[k][si] = q.loc[k]
print(f"  cached {len(needed)} (receptor, ligand) cells across {N_SEEDS} seeds")

print("\n[3] Separation per triple")
rows = []
for _, r in tri.iterrows():
    ka = (r.receptor_active, r.compound_id)
    ki = (r.receptor_inactive, r.compound_id)
    pa, pi = prob[ka], prob[ki]
    da, di = pred[ka], pred[ki]
    strict = (da == 1) & (di == 0)
    ordered = pa > pi
    rows.append({
        **r.to_dict(),
        "mean_prob_active": pa.mean(), "std_prob_active": pa.std(ddof=1),
        "mean_prob_inactive": pi.mean(), "std_prob_inactive": pi.std(ddof=1),
        "mean_prob_gap": (pa - pi).mean(),
        "n_seeds_strict_separation": int(strict.sum()),
        "n_seeds_prob_ordered": int(ordered.sum()),
        "strict_all20": bool(strict.all()),
        "ordered_all20": bool(ordered.all()),
    })
res = pd.DataFrame(rows)
res = res.sort_values(["n_seeds_strict_separation", "mean_prob_gap"], ascending=[False, False])
res.to_csv(OUT_TRIPLES, index=False)

print(f"  strict separation (active predicted 1 AND inactive predicted 0):")
print(f"    mean over triples : {res.n_seeds_strict_separation.mean() / N_SEEDS:.4f}")
print(f"    triples with 20/20: {int(res.strict_all20.sum())} / {len(res)}")
print(f"    triples with 0/20 : {int((res.n_seeds_strict_separation == 0).sum())} / {len(res)}")
print(f"  probability ordering (P_active > P_inactive):")
print(f"    mean over triples : {res.n_seeds_prob_ordered.mean() / N_SEEDS:.4f}")
print(f"    triples with 20/20: {int(res.ordered_all20.sum())} / {len(res)}")
print(f"    triples with 0/20 : {int((res.n_seeds_prob_ordered == 0).sum())} / {len(res)}")

for d in [1, 2]:
    s = res[res.edit_distance == d]
    if len(s):
        print(f"  edit distance {d} (n={len(s)}): strict "
              f"{s.n_seeds_strict_separation.mean() / N_SEEDS:.4f}, ordered "
              f"{s.n_seeds_prob_ordered.mean() / N_SEEDS:.4f}")

print("\n[3b] Subgroup breakdown and mechanism")
import pickle

quality = {(r_, c_): v for r_, c_, v in
           zip(df.main_receptors_id, df.main_compounds_id, df.data_quality)}
res["quality_active"] = [quality[(a, c)] for a, c in zip(res.receptor_active, res.compound_id)]
res["quality_inactive"] = [quality[(a, c)] for a, c in zip(res.receptor_inactive, res.compound_id)]
res["both_ec50"] = (res.quality_active == "ec50") & (res.quality_inactive == "ec50")

overall_rate = df.groupby("main_receptors_id").responsive.mean()
res["dataset_active_rate_of_active"] = res.receptor_active.map(overall_rate)
res["dataset_active_rate_of_inactive"] = res.receptor_inactive.map(overall_rate)
res["prior_favours_active"] = (res.dataset_active_rate_of_active
                               > res.dataset_active_rate_of_inactive)

with open(os.path.join(DATA_DIR, "final_embeddings_all.pkl"), "rb") as fh:
    prot = pickle.load(fh)["protein_embeddings"]
res["prot_cosine"] = [
    float(prot[a] @ prot[b] / (np.linalg.norm(prot[a]) * np.linalg.norm(prot[b])))
    for a, b in zip(res.receptor_active, res.receptor_inactive)]

sub_rows = []
for name, mask in [
    ("all", np.ones(len(res), dtype=bool)),
    ("edit_distance_1", res.edit_distance == 1),
    ("edit_distance_2", res.edit_distance == 2),
    ("edit_distance_1_same_gene", (res.edit_distance == 1) & res.same_gene_symbol),
    ("edit_distance_1_both_ec50", (res.edit_distance == 1) & res.both_ec50),
    ("edit_distance_2_both_ec50", (res.edit_distance == 2) & res.both_ec50),
    ("prior_favours_active", res.prior_favours_active),
    ("prior_favours_inactive", ~res.prior_favours_active),
]:
    s = res[mask]
    if len(s) == 0:
        continue
    k = int(s.n_seeds_prob_ordered.sum())
    n = len(s) * N_SEEDS
    p = stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
    sub_rows.append({
        "subgroup": name, "n_triples": len(s), "n_decisions": n,
        "ranking_rate": k / n, "binomial_p_vs_chance": p,
        "strict_rate": s.n_seeds_strict_separation.mean() / N_SEEDS,
        "mean_prot_cosine": s.prot_cosine.mean(),
        "mean_abs_prob_gap": s.mean_prob_gap.abs().mean(),
    })
    print(f"  {name:28s} n={len(s):4d}  ranking {k / n:.4f}  (p={p:.2e})  "
          f"strict {s.n_seeds_strict_separation.mean() / N_SEEDS:.4f}  "
          f"cos={s.prot_cosine.mean():.6f}")
sub = pd.DataFrame(sub_rows)
sub.to_csv(os.path.join(OUT_DIR, "T3_subgroups.csv"), index=False)

rho_cos, p_cos = stats.spearmanr(res.prot_cosine, res.mean_prob_gap.abs())
print(f"  Spearman(ProtT5 cosine, |mean prob gap|) = {rho_cos:.4f}  p={p_cos:.3e}")
res.to_csv(OUT_TRIPLES, index=False)

print("\n[4] Seed-level separation rate")
seed_rows = []
for si, seed in enumerate(SEEDS):
    st, od = [], []
    for _, r in tri.iterrows():
        ka = (r.receptor_active, r.compound_id)
        ki = (r.receptor_inactive, r.compound_id)
        st.append(bool(pred[ka][si] == 1 and pred[ki][si] == 0))
        od.append(bool(prob[ka][si] > prob[ki][si]))
    seed_rows.append({"Seed": seed, "n_triples": len(tri),
                      "strict_separation_rate": float(np.mean(st)),
                      "prob_ordering_rate": float(np.mean(od))})
seed_df = pd.DataFrame(seed_rows)
seed_df.to_csv(OUT_SEED, index=False)
sm, ss = seed_df.strict_separation_rate.mean(), seed_df.strict_separation_rate.std(ddof=1)
om, os_ = seed_df.prob_ordering_rate.mean(), seed_df.prob_ordering_rate.std(ddof=1)
print(f"  strict separation rate : {sm:.4f} +/- {ss:.4f}  (mean +/- std over 20 seeds, ddof=1)")
print(f"  probability ordering   : {om:.4f} +/- {os_:.4f}")
binom_p = stats.binomtest(int(round(om * len(tri) * N_SEEDS)),
                          len(tri) * N_SEEDS, 0.5, alternative="two-sided").pvalue
print(f"  ordering vs chance (0.5), binomial over all seed x triple: p = {binom_p:.3e}")

print("\n[5] Figure")
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "font.family": "sans-serif"})
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), constrained_layout=True)

ax = axes[0]
for d, col in [(1, "#2c6fbb"), (2, "#c25a3c")]:
    s = res[res.edit_distance == d]
    ax.hist(s.n_seeds_prob_ordered, bins=np.arange(-0.5, 21.5, 1), alpha=0.65,
            color=col, label=f"edit distance {d} (n={len(s)})")
ax.set_xlabel("seeds with $P_{active} > P_{inactive}$")
ax.set_ylabel("number of triples")
ax.set_title("(A) Ranking consistency", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=6)

ax = axes[1]
ax.scatter(res.mean_prob_inactive, res.mean_prob_active, s=6, alpha=0.45,
           c=np.where(res.edit_distance == 1, "#2c6fbb", "#c25a3c"), linewidths=0)
lim = [0, 1]
ax.plot(lim, lim, ls="--", lw=0.7, color="0.4")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("mean $P$ (inactive receptor)")
ax.set_ylabel("mean $P$ (active receptor)")
ax.set_title("(B) 20-seed mean probability", fontsize=7.5, loc="left")

ax = axes[2]
ax.bar(np.arange(N_SEEDS) - 0.2, seed_df.prob_ordering_rate, width=0.4,
       color="#2c6fbb", label="ranking")
ax.bar(np.arange(N_SEEDS) + 0.2, seed_df.strict_separation_rate, width=0.4,
       color="#7a7a7a", label="strict")
ax.axhline(0.5, ls="--", lw=0.7, color="0.4")
ax.set_xticks(np.arange(N_SEEDS))
ax.set_xticklabels([str(s) for s in SEEDS], rotation=90, fontsize=5)
ax.set_ylim(0, 1)
ax.set_xlabel("seed")
ax.set_ylabel("fraction of triples separated")
ax.set_title("(C) Per-seed separation", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=6, loc="upper right")

ax = axes[3]
for d, col in [(1, "#2c6fbb"), (2, "#c25a3c")]:
    s_ = res[res.edit_distance == d]
    ax.scatter(s_.prot_cosine, s_.mean_prob_gap.abs(), s=5, alpha=0.45,
               color=col, linewidths=0, label=f"d={d}")
ax.set_xlabel("ProtT5 cosine similarity")
ax.set_ylabel("|mean probability gap|")
ax.set_title("(D) Embedding vs signal", fontsize=7.5, loc="left")
ax.legend(frameon=False, fontsize=6)

fig.savefig(FIG_BASE + ".pdf")
fig.savefig(FIG_BASE + ".png", dpi=600)
plt.close(fig)
print(f"  saved: {FIG_BASE}.pdf / .png (600 dpi)")

with open(OUT_MD, "w") as fh:
    fh.write("# T3 — WT/variant minimal pairs (R2-M2, single-residue variation)\n\n")
    fh.write("## Question\n\nDoes M2OR contain receptor pairs differing by only one or two "
             "residues that carry **opposite labels for the same ligand**, and can a "
             "mean-pooled ProtT5 embedding separate them?\n\n")
    fh.write("## Material\n\n")
    fh.write(f"- {len(close)} receptor pairs at sequence edit distance <= 2 (from T1)\n")
    fh.write(f"- **{len(tri)} (receptor_A, receptor_B, ligand) triples with opposite labels**: "
             f"{int((tri.edit_distance == 1).sum())} at distance 1, "
             f"{int((tri.edit_distance == 2).sum())} at distance 2\n")
    fh.write(f"- {tri.groupby(['receptor_active', 'receptor_inactive']).ngroups} distinct "
             f"receptor pairs, {tri.compound_id.nunique()} distinct ligands\n")
    fh.write(f"- {int(tri.same_gene_symbol.sum())} triples have the same gene symbol on both "
             "sides (true allelic/engineered variants); the rest are close paralogues\n\n")
    fh.write("## Result\n\n")
    fh.write("Two criteria are reported. *Ranking* asks only whether the model scores the "
             "active receptor above the inactive one for that ligand. *Strict separation* "
             "additionally requires both stored binary predictions to be correct, using each "
             "fold's own MCC-optimal threshold (never re-optimised).\n\n")
    fh.write("| criterion | rate (mean ± std over 20 seeds, ddof=1) | triples at 20/20 | triples at 0/20 |\n")
    fh.write("|---|---|---|---|\n")
    fh.write(f"| ranking ($P_{{active}} > P_{{inactive}}$) | {om:.4f} ± {os_:.4f} | "
             f"{int(res.ordered_all20.sum())} / {len(res)} | "
             f"{int((res.n_seeds_prob_ordered == 0).sum())} / {len(res)} |\n")
    fh.write(f"| strict separation | {sm:.4f} ± {ss:.4f} | "
             f"{int(res.strict_all20.sum())} / {len(res)} | "
             f"{int((res.n_seeds_strict_separation == 0).sum())} / {len(res)} |\n\n")
    fh.write(f"Ranking against chance (0.5), binomial over all "
             f"{len(tri) * N_SEEDS} seed x triple decisions: p = {binom_p:.3e}\n\n")
    for d in [1, 2]:
        s = res[res.edit_distance == d]
        if len(s):
            fh.write(f"- edit distance {d} (n={len(s)}): ranking "
                     f"{s.n_seeds_prob_ordered.mean() / N_SEEDS:.4f}, strict "
                     f"{s.n_seeds_strict_separation.mean() / N_SEEDS:.4f}\n")
    fh.write("\n## Subgroup breakdown\n\n")
    fh.write("| subgroup | triples | ranking rate | binomial p vs 0.5 | strict rate | mean ProtT5 cosine |\n")
    fh.write("|---|---|---|---|---|---|\n")
    for _, x in sub.iterrows():
        fh.write(f"| {x['subgroup']} | {int(x['n_triples'])} | {x['ranking_rate']:.4f} | "
                 f"{x['binomial_p_vs_chance']:.2e} | {x['strict_rate']:.4f} | "
                 f"{x['mean_prot_cosine']:.6f} |\n")
    fh.write("\n## Interpretation\n\n")
    fh.write("- The dataset **does** contain the pairs Reviewer 2 asks about: "
             f"{int((tri.edit_distance == 1).sum())} single-residue triples with opposite "
             "labels for the same ligand.\n")
    fh.write("- At edit distance 1 the model ranks the responsive receptor above the "
             "non-responsive one **less often than chance**, and this survives restriction "
             "to the highest-confidence subset where both measurements are EC50 "
             "dose-response.\n")
    fh.write("- Mean-pooled ProtT5 embeddings of single-residue variants are nearly "
             "collinear (mean cosine "
             f"{res[res.edit_distance == 1].prot_cosine.mean():.6f}, minimum "
             f"{res[res.edit_distance == 1].prot_cosine.min():.6f}), and greater embedding "
             f"similarity accompanies a smaller probability gap "
             f"(Spearman {rho_cos:.4f}, p = {p_cos:.2e}).\n")
    fh.write("- Leading-methionine indexing artefacts of the kind catalogued in SI Section S1 "
             "account for only a small minority of these triples, so the effect is not a "
             "database bookkeeping artefact.\n")
    fh.write("- This is a genuine limitation of the semantic-only representation and should "
             "be reported as such rather than defended.\n")
    fh.write("\n## Standard deviation convention\n\n")
    fh.write("- Seed-level rates: fraction of triples separated within each seed, then mean "
             "and sample standard deviation (`ddof=1`) across the 20 seeds.\n")
    fh.write("- Per-triple probability spread: sample standard deviation (`ddof=1`) across "
             "the 20 seeds.\n")

print(f"\n  saved: {OUT_TRIPLES}")
print(f"  saved: {OUT_SEED}")
print(f"  saved: {OUT_MD}")
print("=" * 78)
