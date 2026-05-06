# ScentOR

**S**emantic **C**hemical **E**ncoder via **N**eural **T**ransfer for **O**lfactory **R**eceptors

A heterogeneous knowledge distillation (KD) framework for computational deorphanization of human olfactory receptors (ORs). ScentOR bridges gradient boosting (GB)-based teacher models with neural network (NN)-based student models using bidirectional gated cross-attention, achieving robust odorant ligand–OR protein interaction prediction from 1D sequence-based embeddings without explicit 3D structural input.

> **Paper**: Eun Cheol Kim and YounJoon Jung. ScentOR: Heterogeneous Knowledge Distillation for Human Olfactory Receptor Deorphanization. *In preparation for Journal of Chemical Information and Modeling*.

## Overview

ScentOR demonstrates that combining the implicit evolutionary knowledge of 1D sequence-based foundation models (MoLFormer for ligands, ProtT5 for proteins) with the non-linear physicochemical decision boundaries of a GB teacher yields superior generalization for OR deorphanization — without requiring any explicit 3D structural prior.

The framework systematically evaluates eight scenarios (I–VIII) spanning baseline teachers/students and four heterogeneous KD configurations, showing that **teacher modality is the dominant factor** in distillation efficacy.

## Repository Structure

```
ScentOR/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── pairs.csv                         # M2OR raw dataset (see Data Availability)
│   ├── final_dataset_weighted.csv        # Curated dataset with confidence weights (see Data Availability)
│   └── final_embeddings_all.pkl          # Pre-computed embeddings
│
├── preprocessing/
│   ├── 01_smiles_validation.py           # SMILES curation & stereoisomer filtering
│   ├── 02_embedding_generation.py        # MoLFormer + ProtT5 semantic embeddings
│   ├── 03_weight_assignment.py           # Confidence-based sample weighting
│   ├── 04_ligand_3d_embedding.py         # EGNN structural embeddings (control)
│   ├── 05_protein_3d_embedding.py        # SE(3)-Transformer structural embeddings (control)
│   └── 06_merge_embeddings.py            # Merge semantic + structural into single pkl
│
├── training/
│   ├── gridsearch/
│   │   ├── gridsearch_lgb.py             # α, γ grid search (LightGBM teacher)
│   │   ├── gridsearch_xgb.py             # α, γ grid search (XGBoost teacher)
│   │   └── gridsearch_cat.py             # α, γ grid search (CatBoost teacher)
│   ├── teacher/
│   │   ├── train_teacher_lgb.py          # LightGBM teacher (OOF soft labels)
│   │   ├── train_teacher_xgb.py          # XGBoost teacher (OOF soft labels)
│   │   └── train_teacher_cat.py          # CatBoost teacher (OOF soft labels)
│   └── student/
│       ├── train_kd_student.py           # KD student training (Scenarios V–VIII)
│       └── train_baseline_student.py     # Baseline student training (Scenarios III–IV)
│
├── evaluation/
│   ├── evaluate_baseline_teacher.py      # Baseline teacher evaluation (Scenarios I–II)
│   └── aggregate.py                      # Aggregate 20-seed results into summary tables
│
├── analysis/
│   ├── 01_p_value_baseline.py            # Statistical testing for baselines (I–IV)
│   ├── 02_p_value_kd.py                  # Statistical testing for KD scenarios (V–VIII)
│   ├── 03_xai_screening.py               # Robust TP screening & minimal pair finder
│   ├── 04_xai_ig.py                      # Integrated gradients analysis
│   ├── 05_xai_shap.py                    # SHAP-based feature importance analysis
│   ├── 06_xai_muta.py                    # In silico alanine mutagenesis
│   ├── 07_xai_threshold.py               # Threshold reconstruction from checkpoints
│   ├── 08_xai_oof.py                     # OOF validation for specific odorant ligand–OR protein pairs
│   └── 09_xai_ood.py                     # Out-of-distribution test
│
└── models/                               # See "Model Weights" in Data Availability
    ├── teacher_models/
    └── student_models/
```

## Reproduction Guide

### Prerequisites

**Hardware**: NVIDIA GPU with ≥ 10 GB VRAM (tested on RTX 3080). CPU execution is possible but significantly slower for embedding generation and student training.

**Software**: Python 3.9+, CUDA 11.x or 12.x

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `rdkit`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `captum`, `egnn-pytorch`, `se3-transformer-pytorch`, `biopython`, `shap`, `tqdm`

### Quick Start (Using Pre-computed Data)

If you want to skip the preprocessing steps (Phase 1–2) and jump directly to model training, use the provided `data/final_dataset_weighted.csv` and `data/final_embeddings_all.pkl`:

```bash
# Phase 3: Grid Search (single seed, first fold — for hyperparameter selection)
cd training/gridsearch/
python gridsearch_lgb.py
python gridsearch_xgb.py
python gridsearch_cat.py

# Phase 4: Teacher Training (repeat for each seed)
cd ../teacher/
python train_teacher_lgb.py
python train_teacher_xgb.py
python train_teacher_cat.py

# Phase 5: Student Training
cd ../student/
python train_kd_student.py
python train_baseline_student.py

# Phase 6: Evaluation
cd ../../evaluation/
python evaluate_baseline_teacher.py
python aggregate.py
```

### Full Reproduction (From Raw Data)

#### Phase 1: Dataset Curation

```bash
cd preprocessing/
python 01_smiles_validation.py
```

**Input**: `data/pairs.csv` (M2OR database export, semicolon-delimited)

**Output**: `pairs_validated.csv` — curated dataset (39,723 pairs from 53,444 initial entries)

This script performs multi-stage filtering: mixture removal (whitespace/dot in SMILES), RDKit parsing, 3D conformer generation via `AllChem.EmbedMolecule`, and strict stereoisomer enumeration. Molecules with ambiguous stereocenters or failed 3D embedding are excluded.

#### Phase 2: Feature Embedding

```bash
python 02_embedding_generation.py
```

**Output**: `final_dataset_map.csv` + `final_embeddings.pkl`

Generates 1D semantic embeddings using masked mean pooling:
- **Ligand**: MoLFormer-XL (`ibm-research/MoLFormer-XL-both-10pct`) → 768-dim
- **Protein**: ProtT5-XL (`Rostlab/prot_t5_xl_half_uniref50-enc`) → 1024-dim

> **Note**: The `mutated_sequence` column is used for protein embeddings. This column contains actual OR amino acid sequences from M2OR (including minor inconsistencies retained for robustness testing).

```bash
python 03_weight_assignment.py
```

**Output**: `final_dataset_weighted.csv`

Assigns confidence-based sample weights by Hladiš et al. and removes exact-duplicate conflicts (same odorant ligand–OR protein pair with contradictory labels).

```bash
python 04_ligand_3d_embedding.py
python 05_protein_3d_embedding.py
python 06_merge_embeddings.py
```

**Output**: `final_embeddings_all.pkl` — unified lookup table containing all four embedding types

- `04`: EGNN with dummy node features (structure-only control) → 512-dim per ligand
- `05`: SE(3)-Transformer with ESMFold-predicted Cα coordinates → 512-dim per protein
- `06`: Merges semantic and structural embeddings into a single pickle file

> **Important**: `05_protein_3d_embedding.py` calls the ESMFold API (`api.esmatlas.com`). This API may be unavailable or rate-limited. The provided `final_embeddings_all.pkl` includes pre-computed structural embeddings for all 1,338 OR proteins.

#### Phase 3: Hyperparameter Grid Search

```bash
cd training/gridsearch/
python gridsearch_lgb.py    # TARGET_TEACHER = 'lgb'
python gridsearch_xgb.py    # TARGET_TEACHER = 'xgb'
python gridsearch_cat.py    # TARGET_TEACHER = 'cat'
```

Grid search is performed on the first fold of a 10-fold stratified CV split (seed = 42) to maximize AUPRC. The search space covers α ∈ [0.0, 1.0] (step 0.1) and γ ∈ [1.0, 5.0] (step 0.5) across all four KD scenarios (Scenarios V–VIII).

**Output**: CSV files ranked by AUPRC for each scenario, e.g., `Scenario5_lgb.csv`

#### Phase 4: Teacher Training (20 Seeds)

```bash
cd training/teacher/
```

Each teacher script trains on a single seed. For the full 20-seed ensemble reported in the paper, modify the `SEED` variable and execute for each seed:

```python
SEEDS = [42, 137, 273, 314, 440, 1013, 1380, 1602, 1618, 1729,
         1953, 2017, 2718, 2997, 4184, 5291, 6022, 6626, 8314, 9648]
```

Each execution produces:
- SMOTE-Tomek resampled teacher models (`.model` files via joblib)
- OOF soft label probabilities for downstream KD

#### Phase 5: Student Training & Baseline Evaluation

```bash
cd training/student/

# KD student (Scenarios V–VIII) — modify SEED for each of the 20 seeds
python train_kd_student.py

# Baseline students (Scenarios III–IV) — already iterates over all 20 seeds
python train_baseline_student.py
```

```bash
cd ../../evaluation/

# Baseline teacher evaluation (Scenarios I–II) — iterates over all 20 seeds
python evaluate_baseline_teacher.py
```

The KD student script (`train_kd_student.py`) evaluates four configurations per seed:
- Scenario V: Semantic-Only KD (T_sem → S_sem) — **this is the core ScentOR architecture**
- Scenario VI: LUPI (T_sem,str → S_sem)
- Scenario VII: Blind Guidance (T_sem → S_sem,str)
- Scenario VIII: Hybrid Modality KD (T_sem,str → S_sem,str)

Key implementation details:
- Threshold moving is applied exclusively on the training set (no data leakage)
- Weighted focal loss with sample-specific confidence weights
- AdamW optimizer with cosine annealing (lr: 1e-4 → 1e-6)
- Early stopping (patience = 15 epochs)

#### Phase 6: Result Aggregation

```bash
cd evaluation/
python aggregate.py
```

**Output**:
- `ensemble_results_Main_Table.csv` — seed-level summary (Mean ± Std across 10 folds)
- `ensemble_results_Fold_Level.csv` — raw fold-level metrics for all seeds

#### Phase 7: Statistical Significance Testing

```bash
cd analysis/
python 01_p_value_baseline.py    # Scenarios I–IV (baseline comparisons)
python 02_p_value_kd.py          # Scenarios V–VIII (KD comparisons)
```

Statistical testing pipeline:
1. Shapiro-Wilk test (normality) + Levene's test (homoscedasticity)
2. If both pass → RM-ANOVA; otherwise → Friedman test
3. Post-hoc: paired t-test or Wilcoxon signed-rank test
4. Benjamini-Hochberg FDR correction for multiple comparisons

#### Phase 8: Explainable AI (XAI) Analysis

```bash
cd analysis/

# Step 1: Screen robust TP pairs and find minimal pairs
python 03_xai_screening.py

# Step 2: Integrated Gradients (macroscopic + microscopic)
python 04_xai_ig.py

# Step 3: SHAP-based feature importance analysis
python 05_xai_shap.py

# Step 4: In silico alanine mutagenesis
python 06_xai_muta.py

# Step 5: Threshold reconstruction from checkpoints
python 07_xai_threshold.py

# Step 6 (Optional): OOF validation for specific pairs
python 08_xai_oof.py

# Step 7 (Optional): Out-of-distribution enantiomer test
python 09_xai_ood.py
```

- `03_xai_screening.py`: Identifies robust TP pairs (≥ 80% TP rate across 20 seeds) for macroscopic XAI and minimal pairs (high Tanimoto similarity, opposite predictions) for microscopic XAI
- `04_xai_ig.py`: Computes per-residue and per-atom integrated gradients via Captum, saving attribution vectors as `.npy` files
- `05_xai_shap.py`: Calculates SHAP-based feature importance decomposition (ligand/protein × semantic/structural) across all seeds and folds
- `06_xai_muta.py`: Performs sequential alanine substitution (glycine for native alanine) to evaluate residue-level functional indispensability via ΔP
- `07_xai_threshold.py`: Reconstructs MCC-optimal decision thresholds from trained student model checkpoints; required by `08_xai_oof.py` and `09_xai_ood.py`
- `08_xai_oof.py`: Cross-validates OOF predictions for specific ligand–OR pairs (e.g., propionic acid with OR51E2)
- `09_xai_ood.py`: Tests model behavior on out-of-distribution enantiomers (e.g., (R/S)-sotolon with OR8D1)

## Random Seeds

The 20 random seeds used throughout this work encode scientifically meaningful numbers:

| Seed | Significance |
|------|-------------|
| 42 | Answer to the Ultimate Question of Life, the Universe, and Everything (*The Hitchhiker's Guide to the Galaxy*) |
| 137 | Fine-structure constant (~1/137), fundamental constant of quantum electrodynamics |
| 273 | Absolute zero rounded (0 K = -273.15 °C) |
| 314 | $\pi$ (3.14...) |
| 440 | Concert pitch A4 = 440 Hz |
| 1013 | Standard atmospheric pressure (1013 hPa) |
| 1380 | Boltzmann constant (1.380 × 10⁻²³ J/K) |
| 1602 | Elementary charge (1.602 × 10⁻¹⁹ C) |
| 1618 | Golden ratio (1.618...) |
| 1729 | Hardy–Ramanujan number (smallest taxicab number) |
| 1953 | Discovery of the DNA double helix (Watson & Crick, 1953) |
| 2017 | Transformer architecture published (*Attention Is All You Need*) |
| 2718 | $e$ (2.718...) |
| 2997 | Speed of light (2.997 × 10⁸ m/s) |
| 4184 | Specific heat capacity of water (4.184 J/(g·K)) |
| 5291 | Bohr radius (0.5291 Å) |
| 6022 | Avogadro's number (6.022 × 10²³ mol⁻¹) |
| 6626 | Planck's constant (6.626 × 10⁻³⁴ J·s) |
| 8314 | Universal gas constant (8.314 J/(mol·K)) |
| 9648 | Faraday constant (96485 C/mol) |

## Data Availability

**M2OR Dataset**: The M2OR database is available at [https://m2or.chemsensim.fr/](https://m2or.chemsensim.fr/). The raw `pairs.csv` file included in this repository was obtained from the M2OR database website. Please refer to the original publication for licensing terms:

> Lalis, M. et al. M2OR: a database of olfactory receptor–odorant pairs for understanding the molecular mechanisms of olfaction. *Nucleic Acids Res.* **2024**, *52*(D1), D1370-D1379.

**Curated Dataset**: `data/final_dataset_weighted.csv` is the preprocessed dataset derived from M2OR, produced by `01_smiles_validation.py` through `03_weight_assignment.py`. It contains curated odorant ligand–OR protein pairs with confidence-based sample weights and is the direct input to all model training scripts (Phase 3 onward).

**Pre-computed Embeddings**: `data/final_embeddings_all.pkl` contains all four embedding types (ligand semantic, protein semantic, ligand structural, protein structural) and is provided for convenience to enable reproduction from Phase 3 onward without requiring GPU-intensive embedding generation or external API access.

**Model Weights**: Trained teacher (`.model`) and student (`.pt`) weights for all 20 seeds × 10 folds are archived on Zenodo:

> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20019062.svg)](https://doi.org/10.5281/zenodo.20019062)
>
> Kim, E. C., & Jung, Y. (2026). ScentOR: Trained Model Weights [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20019062

The archive contains two compressed files:

| File | Contents | Size (approx.) |
|------|----------|----------------|
| `teacher_models.tar.gz` | LightGBM, XGBoost, CatBoost `.model` files (20 seeds × 10 folds × 2 modalities × 3 models) | 536.1 MB |
| `student_models.tar.gz` | PyTorch `.pt` checkpoints for Scenarios V–VIII (20 seeds × 10 folds × 4 scenarios) | 17.9 GB |

Baseline student weights (Scenarios III–IV) are not included, as they correspond to α = 0 (no distillation) and can be reproduced by running `train_kd_student.py` with α = 0. To use the pre-trained weights, download and extract into the `models/` directory:

```bash
# Download from Zenodo
wget https://zenodo.org/records/20019062/files/teacher_models.tar.gz
wget https://zenodo.org/records/20019062/files/student_models.tar.gz

# Extract into models/
tar -xzf teacher_models.tar.gz -C models/
tar -xzf student_models.tar.gz -C models/
```

## Computational Resources

All experiments were performed on a high-performance computing cluster:
- **GPU**: 4× NVIDIA GeForce RTX 3080 (10 GB VRAM)
- **CPU**: Intel Xeon Gold 6226R (2.90 GHz)
- **RAM**: 256 GB
- **OS**: Rocky Linux 8.10

## License

MIT License

## Citation

If you use ScentOR in your research, please cite:

```bibtex
@article{kim2026scentor,
  title={ScentOR: Heterogeneous Knowledge Distillation for Human Olfactory Receptor Deorphanization},
  author={Kim, Eun Cheol and Jung, YounJoon},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  note={In preparation}
}
```
