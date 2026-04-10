import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from tqdm import tqdm

INPUT_FILE = '../data/pairs.csv'
OUTPUT_FILE = '../data/pairs_validated.csv'
RANDOM_SEED = 42

def validate_smiles_strict(smi):
    if pd.isna(smi): return None
    smi = str(smi).strip()

    if ' ' in smi or '.' in smi:
        return None

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None

        mol_h = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=RANDOM_SEED)

        if embed_status == -1:
            return None

        isomers = tuple(EnumerateStereoisomers(mol))
        if len(isomers) > 1:
            return None

        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    except:
        return None

def main():
    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, delimiter=';')
    print(f"Original Rows (All Data): {len(df)}")


    unique_smiles = df['smiles'].unique()
    print(f"Unique SMILES to validate: {len(unique_smiles)}")

    valid_map = {}
    print("Validating SMILES (3D Embed + EnumerateStereoisomers)...")
    for s in tqdm(unique_smiles):
        valid_map[s] = validate_smiles_strict(s)

    df['clean_smiles'] = df['smiles'].map(valid_map)

    df_clean = df.dropna(subset=['clean_smiles']).copy()

    df_clean['original_smiles'] = df_clean['smiles']
    df_clean['smiles'] = df_clean['clean_smiles']

    cols_to_keep = [
        'main_compounds_id',
        'main_receptors_id',
        'responsive',
        'smiles',
        'data_quality'
    ]

    if 'sequence' in df_clean.columns:
        cols_to_keep.append('sequence')
    elif 'mutated_sequence' in df_clean.columns:
        cols_to_keep.append('mutated_sequence')

    df_final = df_clean[cols_to_keep]

    print("-" * 30)
    print(f"Original Count : {len(df)}")
    print(f"Final Valid Count: {len(df_final)}")
    print("-" * 30)

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
