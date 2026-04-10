import pickle

with open("../data/final_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

with open("../data/ligand_3d_embeddings.pkl", "rb") as f:
    ligand_3d = pickle.load(f)
with open("../data/protein_3d_embeddings.pkl", "rb") as f:
    protein_3d = pickle.load(f)

data['ligand_structure_embeddings'] = ligand_3d
data['protein_structure_embeddings'] = protein_3d

with open("../data/final_embeddings_all.pkl", "wb") as f:
    pickle.dump(data, f)

print("Merged all embeddings into final_embeddings_all.pkl")
