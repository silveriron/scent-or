import os

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
SCENARIO = "Scenario5"
MAX_ITER = 500
TOLERANCE = 1e-10
CLIP = 1e-6

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "revision")

OUT_VAR = os.path.join(OUT_DIR, "T13d_variance_decomposition.csv")
OUT_MAIN = os.path.join(OUT_DIR, "T13d_main_effects.csv")
OUT_LINK = os.path.join(OUT_DIR, "T13d_chirality_link.csv")

print("=" * 78)
print("  T13d — how much of the prediction is additive in ligand and receptor")
print("=" * 78)

data = pd.read_csv(os.path.join(DATA_DIR, "final_dataset_weighted.csv"))
lig_codes, lig_uniq = pd.factorize(data.main_compounds_id)
rec_codes, rec_uniq = pd.factorize(data.main_receptors_id)
n_lig, n_rec = len(lig_uniq), len(rec_uniq)
print(f"\n  input: final_dataset_weighted.csv  rows {len(data)}")
print(f"  {n_lig} ligands x {n_rec} receptors = {n_lig * n_rec:,} cells, "
      f"{len(data):,} observed ({len(data) / (n_lig * n_rec):.2%})")


def absorb(y, lig, rec):
    mu = y.mean()
    alpha = np.zeros(n_lig)
    beta = np.zeros(n_rec)
    lig_count = np.bincount(lig, minlength=n_lig).astype(float)
    rec_count = np.bincount(rec, minlength=n_rec).astype(float)
    lig_count[lig_count == 0] = 1
    rec_count[rec_count == 0] = 1
    prev = np.inf
    for it in range(MAX_ITER):
        r = y - mu - beta[rec]
        alpha = np.bincount(lig, weights=r, minlength=n_lig) / lig_count
        r = y - mu - alpha[lig]
        beta = np.bincount(rec, weights=r, minlength=n_rec) / rec_count
        resid = y - mu - alpha[lig] - beta[rec]
        ss = float(resid @ resid)
        if abs(prev - ss) < TOLERANCE * max(1.0, abs(prev)):
            return mu, alpha, beta, resid, it + 1, True
        prev = ss
    return mu, alpha, beta, resid, MAX_ITER, False


def sequential_ss(y, first, n_first, second, n_second):
    mu = y.mean()
    c1 = np.bincount(first, minlength=n_first).astype(float)
    c1[c1 == 0] = 1
    e1 = np.bincount(first, weights=y - mu, minlength=n_first) / c1
    r1 = y - mu - e1[first]
    c2 = np.bincount(second, minlength=n_second).astype(float)
    c2[c2 == 0] = 1
    e2 = np.bincount(second, weights=r1, minlength=n_second) / c2
    r2 = r1 - e2[second]
    total = float(((y - mu) ** 2).sum())
    return (total - float((r1 ** 2).sum())) / total,\
        (float((r1 ** 2).sum()) - float((r2 ** 2).sum())) / total,\
        float((r2 ** 2).sum()) / total


print("\n[2] Two-way decomposition per seed")
print("  logit(p) = mu + alpha_ligand + beta_receptor + interaction")
print("  main effects fitted by alternating projection (iterative demeaning)")
rows, alpha_stack, beta_stack = [], [], []
for seed in SEEDS:
    oof = pd.read_csv(os.path.join(RESULTS_DIR, f"OOF_CM_{SCENARIO}_seed{seed}.csv"))
    p = np.clip(oof.Probability.values.astype(float), CLIP, 1 - CLIP)
    for scale, y in (("logit", np.log(p / (1 - p))), ("probability", p)):
        mu, alpha, beta, resid, iters, ok = absorb(y, lig_codes, rec_codes)
        var_y = float(y.var())
        fitted = mu + alpha[lig_codes] + beta[rec_codes]
        lig_first = sequential_ss(y, lig_codes, n_lig, rec_codes, n_rec)
        rec_first = sequential_ss(y, rec_codes, n_rec, lig_codes, n_lig)
        rows.append({
            "seed": seed, "scale": scale, "converged": ok, "iterations": iters,
            "variance_total": var_y,
            "variance_ligand_effect": float(alpha[lig_codes].var()),
            "variance_receptor_effect": float(beta[rec_codes].var()),
            "variance_interaction_residual": float(resid.var()),
            "share_ligand_marginal": float(alpha[lig_codes].var() / var_y),
            "share_receptor_marginal": float(beta[rec_codes].var() / var_y),
            "share_interaction": float(resid.var() / var_y),
            "additive_r2": float(1 - resid.var() / var_y),
            "sequential_ligand_first_ligand": lig_first[0],
            "sequential_ligand_first_receptor": lig_first[1],
            "sequential_ligand_first_residual": lig_first[2],
            "sequential_receptor_first_receptor": rec_first[0],
            "sequential_receptor_first_ligand": rec_first[1],
            "sequential_receptor_first_residual": rec_first[2],
            "corr_fitted_observed": float(np.corrcoef(fitted, y)[0, 1]),
        })
        if scale == "logit":
            alpha_stack.append(alpha)
            beta_stack.append(beta)

var = pd.DataFrame(rows)
var.to_csv(OUT_VAR, index=False)

for scale in ("logit", "probability"):
    s = var[var.scale == scale]
    print(f"\n  scale: {scale}   converged {int(s.converged.sum())}/{len(s)} seeds, "
          f"median {int(s.iterations.median())} iterations")
    print(f"    additive R2                 {s.additive_r2.mean():.4f} "
          f"+/- {s.additive_r2.std(ddof=1):.4f}")
    print(f"    interaction share           {s.share_interaction.mean():.4f} "
          f"+/- {s.share_interaction.std(ddof=1):.4f}")
    print(f"    ligand main effect share    {s.share_ligand_marginal.mean():.4f} "
          f"+/- {s.share_ligand_marginal.std(ddof=1):.4f}")
    print(f"    receptor main effect share  {s.share_receptor_marginal.mean():.4f} "
          f"+/- {s.share_receptor_marginal.std(ddof=1):.4f}")
    print(f"    sequential, ligand first    ligand {s.sequential_ligand_first_ligand.mean():.4f}"
          f"  receptor {s.sequential_ligand_first_receptor.mean():.4f}"
          f"  residual {s.sequential_ligand_first_residual.mean():.4f}")
    print(f"    sequential, receptor first  receptor {s.sequential_receptor_first_receptor.mean():.4f}"
          f"  ligand {s.sequential_receptor_first_ligand.mean():.4f}"
          f"  residual {s.sequential_receptor_first_residual.mean():.4f}")
print("\n  standard deviations across the 20 seeds use ddof=1")
print("  marginal shares do not sum to one because the unbalanced design leaves the")
print("  two main effects correlated; the sequential rows are the exact partition")

print("\n[3] Main effects, averaged over seeds")
alpha_mean = np.mean(alpha_stack, axis=0)
beta_mean = np.mean(beta_stack, axis=0)
marg = data.groupby("main_compounds_id").responsive.agg(["size", "mean"])
main = pd.DataFrame({
    "main_compounds_id": lig_uniq,
    "ligand_effect_logit": alpha_mean,
    "ligand_effect_sd_across_seeds": np.std(alpha_stack, axis=0, ddof=1),
})
main["n_pairs"] = marg.loc[main.main_compounds_id, "size"].values
main["marginal_active_rate"] = marg.loc[main.main_compounds_id, "mean"].values
main = main.sort_values("ligand_effect_logit", ascending=False)
rec_marg = data.groupby("main_receptors_id").responsive.agg(["size", "mean"])
rec_main = pd.DataFrame({
    "main_receptors_id": rec_uniq,
    "receptor_effect_logit": beta_mean,
    "receptor_effect_sd_across_seeds": np.std(beta_stack, axis=0, ddof=1),
    "n_pairs": rec_marg.loc[rec_uniq, "size"].values,
    "marginal_active_rate": rec_marg.loc[rec_uniq, "mean"].values,
})
main.to_csv(OUT_MAIN, index=False)
rec_main.to_csv(os.path.join(OUT_DIR, "T13d_receptor_effects.csv"), index=False)
sp_l = stats.spearmanr(main.ligand_effect_logit, main.marginal_active_rate)
sp_r = stats.spearmanr(rec_main.receptor_effect_logit, rec_main.marginal_active_rate)
print(f"  ligand effect vs marginal active rate   Spearman {sp_l.statistic:+.4f} "
      f"(p={sp_l.pvalue:.3e}, n={len(main)})")
print(f"  receptor effect vs marginal active rate Spearman {sp_r.statistic:+.4f} "
      f"(p={sp_r.pvalue:.3e}, n={len(rec_main)})")
print(f"  ligand effect spread   SD {main.ligand_effect_logit.std(ddof=1):.4f}, "
      f"range {main.ligand_effect_logit.min():.3f} to {main.ligand_effect_logit.max():.3f}")
print(f"  receptor effect spread SD {rec_main.receptor_effect_logit.std(ddof=1):.4f}, "
      f"range {rec_main.receptor_effect_logit.min():.3f} to "
      f"{rec_main.receptor_effect_logit.max():.3f}")

print("\n[4] Link to the T12f fixed compound preference")
ordering = pd.read_csv(os.path.join(OUT_DIR, "T12f_isomer_pair_ordering.csv"))
eff_of = dict(zip(main.main_compounds_id, main.ligand_effect_logit))
link_rows = []
for _, r in ordering.iterrows():
    lo, hi = int(r.compound_low), int(r.compound_high)
    if lo not in eff_of or hi not in eff_of:
        continue
    gap = eff_of[hi] - eff_of[lo]
    link_rows.append({
        "compound_low": lo, "compound_high": hi, "stereo_class": r.stereo_class,
        "n_cases": int(r.n_cases),
        "fraction_high_scored_above_low": r.fraction_high_scored_above_low,
        "ligand_effect_low": eff_of[lo], "ligand_effect_high": eff_of[hi],
        "ligand_effect_gap_high_minus_low": gap,
        "preference_matches_ligand_effect": bool(
            (r.fraction_high_scored_above_low > 0.5) == (gap > 0)),
    })
link = pd.DataFrame(link_rows)
link.to_csv(OUT_LINK, index=False)
print(link[["compound_low", "compound_high", "n_cases",
            "fraction_high_scored_above_low", "ligand_effect_gap_high_minus_low",
            "preference_matches_ligand_effect"]].round(4).to_string(index=False))
k = int(link.preference_matches_ligand_effect.sum())
bt = stats.binomtest(k, len(link), 0.5)
sp = stats.spearmanr(link.ligand_effect_gap_high_minus_low,
                     link.fraction_high_scored_above_low)
print(f"\n  the fixed preference agrees with the sign of the ligand main effect in "
      f"{k}/{len(link)} pairs (binomial p={bt.pvalue:.4f})")
print(f"  Spearman(ligand effect gap, fraction scored above) = {sp.statistic:+.4f} "
      f"(p={sp.pvalue:.4f})")
big = link[link.n_cases >= 5]
if len(big):
    print(f"  restricted to pairs with >= 5 cases: "
          f"{int(big.preference_matches_ligand_effect.sum())}/{len(big)}")
    for _, r in big.iterrows():
        print(f"    ({int(r.compound_low)}, {int(r.compound_high)}): "
              f"effect {r.ligand_effect_low:+.3f} vs {r.ligand_effect_high:+.3f}, "
              f"gap {r.ligand_effect_gap_high_minus_low:+.3f}, "
              f"observed preference {r.fraction_high_scored_above_low:.4f}")

print("\n[5] Consistency with the earlier attribution results")
s = var[var.scale == "logit"]
print(f"  interaction share on the logit scale: {s.share_interaction.mean():.4f}")
print(f"  additive R2 on the logit scale      : {s.additive_r2.mean():.4f}")
print("  the conditional additive-baseline comparison of the task specification is")
print("  gated on the interaction share; the threshold there is 0.30")
print(f"  observed {s.share_interaction.mean():.4f} -> "
      f"{'above' if s.share_interaction.mean() >= 0.30 else 'below'} the threshold")

print(f"\n  saved: {OUT_VAR}")
print(f"  saved: {OUT_MAIN}")
print(f"  saved: {os.path.join(OUT_DIR, 'T13d_receptor_effects.csv')}")
print(f"  saved: {OUT_LINK}")
print("=" * 78)
