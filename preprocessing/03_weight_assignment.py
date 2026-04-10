import pandas as pd

INPUT_CSV = "../data/final_dataset_map.csv"
OUTPUT_CSV = "../data/final_dataset_weighted.csv"

SOURCE_COL = 'data_quality'
LABEL_COL = 'responsive'
ID_COL = 'main_compounds_id'
SEQ_COL = 'mutated_sequence'

PAPER_WEIGHTS = {
    'ec50': {0: 1.0, 1: 1.0},
    'secondaryScreening': {0: 0.77, 1: 0.72},
    'primaryScreening': {0: 0.69, 1: 0.40}
}

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    original_len = len(df)

    print("\n[Step 1] Removing exact duplicates/conflicts...")

    cols = [ID_COL, SEQ_COL]
    conflict_indices = df.groupby(cols)[LABEL_COL].filter(lambda x: x.nunique() > 1).index

    if len(conflict_indices) > 0:
        df_clean = df.drop(conflict_indices)
        print(f"Dropped {len(conflict_indices)} exact conflicting rows.")
    else:
        df_clean = df.copy()
        print("No exact conflicts found.")

    print("\n[Step 2] Assigning weights...")

    def get_weight(row):
        source = row[SOURCE_COL]
        label = row[LABEL_COL]

        if source in PAPER_WEIGHTS:
            return PAPER_WEIGHTS[source].get(label, 1.0)
        return 1.0

    df_clean['sample_weight'] = df_clean.apply(get_weight, axis=1)

    print("\n[Step 3] Verification & Saving...")

    summary = df_clean.groupby([SOURCE_COL, LABEL_COL])['sample_weight'].mean()
    print("\n--- Assigned Weights Summary ---")
    print(summary)

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved processed dataset to {OUTPUT_CSV}")
    print(f"Final Data Count: {len(df_clean)}")

if __name__ == "__main__":
    main()
