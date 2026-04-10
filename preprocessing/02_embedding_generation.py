import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
import re
from tqdm import tqdm
import pickle
import gc
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

INPUT_CSV = "../data/pairs_validated.csv"
OUTPUT_MAP_CSV = "../data/final_dataset_map.csv"
OUTPUT_PKL = "../data/final_embeddings.pkl"

BATCH_SIZE_MOL = 64
BATCH_SIZE_PROT = 8

def get_masked_mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask

def get_mol_embeddings(id_to_smiles):
    print(f"\n[MolFormer] Embedding {len(id_to_smiles)} unique ligands...")

    tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
    model = AutoModel.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(DEVICE)
    model.eval()

    ids = list(id_to_smiles.keys())
    smiles_list = list(id_to_smiles.values())
    id_to_emb = {}

    MAX_LEN_MOL = 256

    with torch.no_grad():
        for i in tqdm(range(0, len(ids), BATCH_SIZE_MOL)):
            batch_ids = ids[i : i + BATCH_SIZE_MOL]
            batch_smiles = smiles_list[i : i + BATCH_SIZE_MOL]

            try:
                inputs = tokenizer(
                    batch_smiles,
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_LEN_MOL,
                    return_tensors="pt"
                ).to(DEVICE)

                outputs = model(**inputs)

                embeddings = get_masked_mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = embeddings.cpu().numpy()

                for bid, emb in zip(batch_ids, embeddings):
                    id_to_emb[bid] = emb.astype(np.float32)

            except Exception as e:
                print(f"Error in Mol batch {i}: {e}")

    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return id_to_emb

def get_prot_embeddings(id_to_seq):
    print(f"\n[ProtT5] Embedding {len(id_to_seq)} unique proteins...")

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(DEVICE)
    model.eval()

    ids = list(id_to_seq.keys())
    seqs_list = list(id_to_seq.values())
    id_to_emb = {}

    MAX_LEN_PROT = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(ids), BATCH_SIZE_PROT)):
            batch_ids = ids[i : i + BATCH_SIZE_PROT]
            batch_seqs = seqs_list[i : i + BATCH_SIZE_PROT]

            try:
                processed_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in batch_seqs]

                inputs = tokenizer(
                    processed_seqs,
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_LEN_PROT,
                    return_tensors="pt"
                ).to(DEVICE)

                outputs = model(**inputs)

                embeddings = get_masked_mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = embeddings.cpu().numpy()

                for bid, emb in zip(batch_ids, embeddings):
                    id_to_emb[bid] = emb.astype(np.float32)

            except Exception as e:
                print(f"Error in Prot batch {i}: {e}")

    del model, tokenizer, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return id_to_emb

def main():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print("Extracting unique compounds and receptors...")
    unique_compounds = df.groupby('main_compounds_id')['smiles'].first().to_dict()
    unique_receptors = df.groupby('main_receptors_id')['mutated_sequence'].first().to_dict()

    print(f"Unique Compounds IDs: {len(unique_compounds)}")
    print(f"Unique Receptors IDs: {len(unique_receptors)}")

    mol_embeddings = get_mol_embeddings(unique_compounds)
    prot_embeddings = get_prot_embeddings(unique_receptors)


    cols_to_save = ['main_compounds_id', 'main_receptors_id', 'responsive', 'smiles', 'mutated_sequence', 'data_quality']
    existing_cols = [c for c in cols_to_save if c in df.columns]
    df[existing_cols].to_csv(OUTPUT_MAP_CSV, index=False)
    print(f"\n[Saved] Dataset Map -> {OUTPUT_MAP_CSV}")

    data_to_save = {
        'ligand_embeddings': mol_embeddings,
        'protein_embeddings': prot_embeddings,
        'metadata_ligands': unique_compounds,
        'metadata_proteins': unique_receptors
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"[Saved] Embeddings Lookup -> {OUTPUT_PKL}")
    print("\nProcess Complete! Ready for training.")

if __name__ == "__main__":
    main()
