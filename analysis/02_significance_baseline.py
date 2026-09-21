import pandas as pd
import numpy as np
from scipy import stats
import itertools
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM

DATA_PATH = "../results/benchmark_baseline_teachers_SI.csv"
ALPHA = 0.05
METRICS = ['AUROC_Mean', 'AUPRC_Mean', 'F1_Mean', 'MCC_Mean']

df_si = pd.read_csv(DATA_PATH)

modalities = ['sem', 'str']

for TARGET_METRIC in METRICS:
    print(f"\n{'='*70}")
    print(f"  Statistical Testing: {TARGET_METRIC}")
    print(f"{'='*70}")

    for mod in modalities:
        print(f"\n  [ Modality: {mod.upper()} ]")

        df_mod = df_si[df_si['Model'].str.endswith(mod)]
        df_pivot = df_mod.pivot(index='Seed', columns='Model', values=TARGET_METRIC)
        models = df_pivot.columns.tolist()

        print(f"\n  Descriptive Statistics:")
        for m in models:
            vals = df_pivot[m].dropna().values
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
            var_name='Model', value_name=TARGET_METRIC
        )

        if is_parametric:
            global_test_name = "Repeated Measures ANOVA"
            try:
                res = AnovaRM(data=df_long, depvar=TARGET_METRIC,
                              subject='Seed', within=['Model']).fit()
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
            print("     => No significant difference among models. Stopping.")
            print("  " + "-" * 66)
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

            print(f"     {row['Comparison']:<55}")
            print(f"       Diff: {row['Mean_Diff']:>+.4f}  |  p-adj: {p_adj:.4e} ({sig})  |  Winner: {row['Better']}")

        print("  " + "-" * 66)

print(f"\n{'='*70}")
print("  Done!")
print(f"{'='*70}")
