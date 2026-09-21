import pandas as pd
import numpy as np
from scipy import stats
import itertools
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]

DATA_DIR = "../results"
ALPHA = 0.05

SCENARIOS = {
    "Scenario5": "(V)Semantic_Only_KD",
    "Scenario6": "(VI)LUPI",
    "Scenario7": "(VII)Blind_Guidance",
    "Scenario8": "(VIII)Hybrid_Modality_KD"
}

METRICS = ['AUROC_Mean', 'AUPRC_Mean', 'F1_Mean', 'MCC_Mean']

print("=" * 70)
print("  Loading per-seed 10-fold CV results")
print("=" * 70)

all_data = []
for seed in SEEDS:
    csv_path = f"{DATA_DIR}/10fold_cv_seed{seed}.csv"
    try:
        df = pd.read_csv(csv_path)
        df['Seed'] = seed
        all_data.append(df)
    except FileNotFoundError:
        print(f"  [Warning] Not found: {csv_path}")

df_all = pd.concat(all_data, ignore_index=True)

scenario_col = 'Scenario'
df_all['Scenario'] = df_all[scenario_col]

print(f"  Loaded {len(df_all)} rows from {len(all_data)} seeds")
print(f"  Scenarios found: {df_all['Scenario'].unique().tolist()}")

scenario_names = list(SCENARIOS.values())
df_kd = df_all[df_all['Scenario'].isin(scenario_names)].copy()

if len(df_kd) == 0:
    df_kd = df_all[df_all[scenario_col].isin(SCENARIOS.keys())].copy()
    df_kd['Scenario'] = df_kd[scenario_col].map(SCENARIOS)

print(f"  KD scenario rows: {len(df_kd)}")

for metric in METRICS:
    print(f"\n{'='*70}")
    print(f"  Statistical Testing: {metric}")
    print(f"{'='*70}")

    df_pivot = df_kd.pivot(index='Seed', columns='Scenario', values=metric)

    available = [s for s in scenario_names if s in df_pivot.columns]
    if len(available) < 2:
        print(f"  [Error] Not enough scenarios found. Available: {available}")
        continue

    df_pivot = df_pivot[available].dropna()
    models = df_pivot.columns.tolist()

    print(f"\n  Descriptive Statistics:")
    for m in models:
        vals = df_pivot[m].values
        print(f"    {m:<35}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print(f"\n  1. Assumption Checks:")

    normality_results = {}
    normality_all_pass = True
    for m in models:
        stat, p_val = stats.shapiro(df_pivot[m].dropna().values)
        normality_results[m] = p_val
        if p_val <= ALPHA:
            normality_all_pass = False
    print(f"     Shapiro-Wilk (Normality): {'Pass' if normality_all_pass else 'Fail'}")
    for m, p in normality_results.items():
        flag = "" if p > ALPHA else " <-- FAIL"
        print(f"       {m:<35}: p={p:.4f}{flag}")

    arrays = [df_pivot[m].dropna().values for m in models]
    _, levene_p = stats.levene(*arrays)
    levene_passed = levene_p > ALPHA
    print(f"     Levene's (Homoscedasticity): {'Pass' if levene_passed else 'Fail'} (p={levene_p:.4f})")

    is_parametric = normality_all_pass and levene_passed

    df_long = df_pivot.reset_index().melt(
        id_vars='Seed', value_vars=models,
        var_name='Scenario', value_name=metric
    )

    if is_parametric:
        global_test_name = "Repeated Measures ANOVA"
        try:
            res = AnovaRM(data=df_long, depvar=metric, subject='Seed', within=['Scenario']).fit()
            global_p = res.anova_table['Pr > F'].iloc[0]
        except Exception as e:
            print(f"     RM-ANOVA failed ({e}), falling back to Friedman")
            global_test_name = "Friedman Test"
            _, global_p = stats.friedmanchisquare(*arrays)
    else:
        global_test_name = "Friedman Test"
        _, global_p = stats.friedmanchisquare(*arrays)

    print(f"\n  2. Global Test [{global_test_name}]:")
    print(f"     p-value = {global_p:.4e}")

    if global_p >= ALPHA:
        print("     => No significant difference among scenarios. Stopping.")
        print("-" * 70)
        continue

    print("     => Significant difference found. Proceeding to post-hoc.")

    post_hoc_name = "Paired t-test" if is_parametric else "Wilcoxon signed-rank"
    print(f"\n  3. Post-hoc [{post_hoc_name} + Benjamini-Hochberg FDR]:")

    results = []
    for model_a, model_b in itertools.combinations(models, 2):
        scores_a = df_pivot[model_a].values
        scores_b = df_pivot[model_b].values

        if np.allclose(scores_a, scores_b):
            raw_p = 1.0
        else:
            if is_parametric:
                _, raw_p = stats.ttest_rel(scores_a, scores_b)
            else:
                _, raw_p = stats.wilcoxon(scores_a, scores_b)

        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        better = model_a if mean_diff > 0 else model_b

        results.append({
            'A': model_a, 'B': model_b,
            'Comparison': f"{model_a} vs {model_b}",
            'Mean_Diff': mean_diff,
            'Better': better,
            'Raw_p': raw_p
        })

    df_results = pd.DataFrame(results)
    reject, pvals_corrected, _, _ = multipletests(
        df_results['Raw_p'], alpha=ALPHA, method='fdr_bh'
    )
    df_results['Adj_p'] = pvals_corrected
    df_results['Significant'] = reject

    for _, row in df_results.iterrows():
        p_adj = row['Adj_p']
        if p_adj < 0.001: sig = "***"
        elif p_adj < 0.01: sig = "**"
        elif p_adj < 0.05: sig = "*"
        else: sig = "ns"

        print(f"     {row['Comparison']:<65}")
        print(f"       Diff: {row['Mean_Diff']:>+.4f}  |  p-adj: {p_adj:.4e} ({sig})  |  Winner: {row['Better']}")

    print(f"\n  Summary for ScentOR (Semantic-Only KD):")
    scentor_name = SCENARIOS["Scenario5"]
    for _, row in df_results.iterrows():
        if scentor_name in row['A'] or scentor_name in row['B']:
            opponent = row['B'] if scentor_name in row['A'] else row['A']
            p_adj = row['Adj_p']
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
            winner = "ScentOR" if row['Better'] == scentor_name else opponent
            print(f"     vs {opponent:<50}: p-adj={p_adj:.4e} ({sig}), Winner={winner}")

    print("-" * 70)

print(f"\n{'='*70}")
print("  Done!")
print(f"{'='*70}")
