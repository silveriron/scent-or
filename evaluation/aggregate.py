import pandas as pd
import os

SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]

RESULTS_DIR = "../results"

SUMMARY_TEMPLATE = os.path.join(RESULTS_DIR, "10fold_cv_seed{seed}.csv")
FOLDRAW_TEMPLATE = os.path.join(RESULTS_DIR, "10fold_cv_cm_seed{seed}.csv")

OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "ensemble_results_Main_Table.csv")
OUTPUT_FOLDRAW_CSV = os.path.join(RESULTS_DIR, "ensemble_results_Fold_Level.csv")

SUMMARY_COLUMNS = [
    'Seed', 'Scenario', 'AUROC_Mean', 'AUROC_Std', 'AUPRC_Mean', 'AUPRC_Std',
    'F1_Mean', 'F1_Std', 'MCC_Mean', 'MCC_Std',
    'Total_TP', 'Total_TN', 'Total_FP', 'Total_FN'
]

FOLDRAW_COLUMNS = [
    'Seed', 'Scenario', 'Fold', 'TP', 'TN', 'FP', 'FN',
    'AUROC', 'AUPRC', 'F1', 'MCC'
]

def find_file(seed, template):
    primary = template.format(seed=seed)
    if os.path.exists(primary):
        return primary
    filename = os.path.basename(primary)
    fallback = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(fallback):
        return fallback
    return None

def collect_data(seeds, template, label):
    all_data = []
    missing = []
    for seed in seeds:
        path = find_file(seed, template)
        if path:
            df = pd.read_csv(path)
            if 'Seed' not in df.columns:
                df['Seed'] = seed
            all_data.append(df)
        else:
            missing.append(seed)
    if missing:
        print(f"  [Warning] {label}: missing seeds {missing}")
    if not all_data:
        print(f"  [Error] {label}: no data loaded.")
        return None
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"  {label}: loaded {len(df_all)} rows from {len(all_data)} seeds")
    return df_all

def filter_and_sort(df, target_columns):
    existing = [c for c in target_columns if c in df.columns]
    df_out = df[existing].copy()
    sort_cols = [c for c in ['Scenario', 'Seed', 'Fold'] if c in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(sort_cols).reset_index(drop=True)
    return df_out

def main():
    print("=" * 60)
    print("  Aggregating KD Results")
    print("=" * 60)

    df_summary = collect_data(SEEDS, SUMMARY_TEMPLATE, "Seed-level summary")
    if df_summary is not None:
        df_summary = filter_and_sort(df_summary, SUMMARY_COLUMNS)
        df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"  Saved: {OUTPUT_SUMMARY_CSV}")

        main_table = []
        for scenario_name, group in df_summary.groupby('Scenario'):
            main_table.append({
                'Scenario': scenario_name,
                'AUROC': f"{group['AUROC_Mean'].mean():.4f} +/- {group['AUROC_Mean'].std():.4f}",
                'AUPRC': f"{group['AUPRC_Mean'].mean():.4f} +/- {group['AUPRC_Mean'].std():.4f}",
                'F1': f"{group['F1_Mean'].mean():.4f} +/- {group['F1_Mean'].std():.4f}",
                'MCC': f"{group['MCC_Mean'].mean():.4f} +/- {group['MCC_Mean'].std():.4f}",
            })
        df_main = pd.DataFrame(main_table)
        print("\n  Main Table Summary:")
        print(df_main.to_string(index=False))

    df_foldraw = collect_data(SEEDS, FOLDRAW_TEMPLATE, "Fold-level raw data")
    if df_foldraw is not None:
        df_foldraw = filter_and_sort(df_foldraw, FOLDRAW_COLUMNS)
        df_foldraw.to_csv(OUTPUT_FOLDRAW_CSV, index=False)
        print(f"  Saved: {OUTPUT_FOLDRAW_CSV}")

    print("\nDone.")

if __name__ == "__main__":
    main()