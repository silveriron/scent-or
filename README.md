# ScentOR

**S**emantic **C**hemical **E**ncoder via **N**eural **T**ransfer for **O**dorant **R**eceptors

A heterogeneous knowledge distillation (KD) framework for computational deorphanization of odorant receptors (ORs). ScentOR bridges gradient boosting (GB)-based teacher models with neural network (NN)-based student models using bidirectional gated cross-attention, achieving robust odorant ligand–OR protein interaction prediction from 1D sequence-based embeddings without explicit 3D structural input.

> **Paper**: *ScentOR: Semantic-Only Heterogeneous Knowledge Distillation for Odorant Receptor Deorphanization*
> Eun Cheol Kim and YounJoon Jung — Department of Chemistry, Seoul National University
> Under review

## Overview

ScentOR demonstrates that combining the implicit evolutionary knowledge of 1D sequence-based foundation models (MoLFormer for ligands, ProtT5 for proteins) with the non-linear physicochemical decision boundaries of a GB teacher yields superior generalization for OR deorphanization — without requiring any explicit 3D structural prior.

The framework systematically evaluates eight scenarios (I–VIII) spanning baseline teachers/students and four heterogeneous KD configurations, showing that **teacher modality is the dominant factor** in distillation efficacy.

### Dataset composition

The curated benchmark comprises 39,723 odorant ligand–OR protein pairs over 552 odorant ligands and 1,338 OR protein sequences spanning 16 organisms: 1,165 human and 173 non-human, of which *Mus musculus* accounts for 145. Restricting evaluation to the human subset reproduces the reported AUROC to within 0.001. The overall positive rate is 6.16% (2,448 positives), a class ratio of 1:15.23.

## Repository Structure

```
ScentOR/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── pairs.csv                         # M2OR raw export — NOT redistributed, see Data Availability
│   ├── final_dataset_weighted.csv        # Curated annotations (IDs, labels, weights; no raw strings)
│   └── final_embeddings_all.pkl          # Pre-computed embeddings (InChIKey-annotated ligands)
│
├── preprocessing/
│   ├── 00_restore_raw_strings.py         # (XAI helper) restore raw SMILES/sequences from M2OR export
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
│   ├── stats_utils.py                    # Shared statistical routines (Section S6 conventions)
│   ├── 01_threshold_fitting.py           # MCC-optimal thresholds from student checkpoints
│   ├── 02_significance_baseline.py       # Statistical testing for baselines (I–IV)
│   ├── 03_significance_kd.py             # Statistical testing for KD scenarios (V–VIII)
│   ├── 04_significance_modality.py       # Structural modality effect in the non-distilled baselines
│   ├── 05_uniparc_lookup.py              # UniParc resolution of curated OR sequences
│   ├── 06_species_assignment.py          # Organism assignment and near-identical variant groups
│   ├── 07_dataset_audit.py               # Dataset composition by organism and variant status
│   ├── 08_species_restricted_eval.py     # Metrics on species-defined subsets
│   ├── 09_near_identical_variants.py     # Opposite-label variant triples
│   ├── 10_panel_screening.py             # Panel-level screening metrics
│   ├── 11_additivity_decomposition.py    # Ligand / OR / interaction variance decomposition
│   ├── 12_structure_accuracy.py          # ESMFold OR51E2 against PDB 8F76
│   ├── 13_fn_margin.py                   # False-negative margins to the fold thresholds
│   ├── 14_modality_shap.py               # SHAP-based feature importance analysis
│   ├── 15_case_screening.py              # Robust TP screening & minimal pair finder
│   ├── 16_stereoisomer_exhaustive.py     # Exhaustive stereoisomer pair analysis
│   ├── 17_integrated_gradients.py        # Integrated gradients analysis
│   ├── 18_alanine_mutagenesis.py         # In silico alanine mutagenesis
│   ├── 19_attribution_sample.py          # Attribution sample selection
│   ├── 20_attribution_registry.py        # Profile similarity against the across-seed floor
│   ├── 21_untrained_control.py           # DRY motif rank under untrained weights
│   ├── 22_oof_validation.py              # OOF validation for specific pairs
│   ├── 23_ood_enantiomer.py              # Out-of-distribution enantiomer test
│   ├── 24_class1_candidates.py           # Class I case-selection filters
│   └── 25_determinism_check.py           # Single-seed determinism re-run
│
├── tests/
│   ├── canonical_values.yaml             # Values printed in the manuscript
│   └── test_regression.py                # Checks the stored outputs against them
│
├── tools/
│   └── check_manifest.py                 # Keeps this README and the tracked modules in step
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
python 02_significance_baseline.py    # Scenarios I–IV (baseline comparisons)
python 03_significance_kd.py          # Scenarios V–VIII (KD comparisons)
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
python 15_case_screening.py

# Step 2: Integrated Gradients (macroscopic + microscopic)
python 17_integrated_gradients.py

# Step 3: SHAP-based feature importance analysis
python 14_modality_shap.py

# Step 4: In silico alanine mutagenesis
python 18_alanine_mutagenesis.py

# Step 5: Threshold reconstruction from checkpoints
python 01_threshold_fitting.py

# Step 6 (Optional): OOF validation for specific pairs
python 22_oof_validation.py

# Step 7 (Optional): Out-of-distribution enantiomer test
python 23_ood_enantiomer.py
```

- `15_case_screening.py`: Identifies robust TP pairs (≥ 80% TP rate across 20 seeds) for macroscopic XAI and minimal pairs (high Tanimoto similarity, opposite predictions) for microscopic XAI
- `17_integrated_gradients.py`: Computes per-residue and per-atom integrated gradients via Captum, saving attribution vectors as `.npy` files
- `14_modality_shap.py`: Calculates SHAP-based feature importance decomposition (ligand/protein × semantic/structural) across all seeds and folds
- `18_alanine_mutagenesis.py`: Performs sequential alanine substitution (glycine for native alanine) to evaluate residue-level functional indispensability via ΔP
- `01_threshold_fitting.py`: Reconstructs MCC-optimal decision thresholds from trained student model checkpoints; required by `22_oof_validation.py` and `23_ood_enantiomer.py`
- `22_oof_validation.py`: Cross-validates OOF predictions for specific ligand–OR pairs (e.g., propionic acid with OR51E2)
- `23_ood_enantiomer.py`: Tests model behavior on out-of-distribution enantiomers (e.g., (R/S)-sotolon with OR8D1)

#### Phase 9: Extended Analyses

These scripts re-aggregate the stored out-of-fold predictions and attribution outputs produced in Phases 4–8. **No model is retrained and no threshold is re-optimized**; every analysis inherits the models and thresholds of the main pipeline. The one exception is `25_determinism_check.py`, which deliberately re-runs a single seed to confirm that the pipeline is deterministic.

```bash
cd analysis/

python 04_significance_modality.py
python 05_uniparc_lookup.py
python 06_species_assignment.py
python 07_dataset_audit.py
python 08_species_restricted_eval.py
python 09_near_identical_variants.py
python 10_panel_screening.py
python 11_additivity_decomposition.py
python 12_structure_accuracy.py
python 13_fn_margin.py
python 16_stereoisomer_exhaustive.py
python 19_attribution_sample.py
python 20_attribution_registry.py
python 21_untrained_control.py
python 24_class1_candidates.py
```

Shared statistical routines (Hodges–Lehmann estimates, distribution-free intervals, sign tests, cluster reduction, Benjamini–Hochberg correction) are collected in `stats_utils.py` and follow the conventions specified in Section S6 of the Supporting Information.

After changing any script or this README, check that the manifest below is still consistent and that the stored outputs still reproduce the published numbers:

```bash
python tools/check_manifest.py
python -m pytest tests/test_regression.py -q
```

## Mapping to the Manuscript

Script numbering reflects execution order and dependencies, not the location of the corresponding result in the paper. The table below is the authoritative mapping; it is the only place in this repository where manuscript locations are recorded. `analysis/stats_utils.py` is a library rather than a step and does not appear.

| Script | Main text | Supporting Information |
|---|---|---|
| `preprocessing/00_restore_raw_strings.py` | Dataset Curation and Preprocessing | S1 |
| `preprocessing/01_smiles_validation.py` | Dataset Curation and Preprocessing | S1, S1.1 |
| `preprocessing/02_embedding_generation.py` | Computational Methods | S2.1 |
| `preprocessing/03_weight_assignment.py` | Dataset Curation and Preprocessing | S1.1 |
| `preprocessing/04_ligand_3d_embedding.py` | Computational Methods | S2.3.5 |
| `preprocessing/05_protein_3d_embedding.py` | Computational Methods | S2.3.5 |
| `preprocessing/06_merge_embeddings.py` | Computational Methods | S2.1 |
| `training/gridsearch/gridsearch_lgb.py` | Table 5 | S5.1, Figures S1–S4 |
| `training/gridsearch/gridsearch_xgb.py` | Table 5 | S5.1, Figures S1–S4 |
| `training/gridsearch/gridsearch_cat.py` | Table 5 | S5.1, Figures S1–S4 |
| `training/teacher/train_teacher_lgb.py` | Tables 3, 4 | S2.2 |
| `training/teacher/train_teacher_xgb.py` | Tables 3, 4 | S2.2 |
| `training/teacher/train_teacher_cat.py` | Tables 3, 4 | S2.2 |
| `training/student/train_kd_student.py` | Table 5 | S2.3.4, S5.3 |
| `training/student/train_baseline_student.py` | Table 4 | S2.3.4 |
| `evaluation/evaluate_baseline_teacher.py` | Tables 3, 4 | S3 |
| `evaluation/aggregate.py` | Tables 3–5 | S6.10 |
| `analysis/01_threshold_fitting.py` | — | S5.3 |
| `analysis/02_significance_baseline.py` | Tables 3, 4 | S6.2, Tables S7, S8 |
| `analysis/03_significance_kd.py` | Table 5 | S6.3, Table S9 |
| `analysis/04_significance_modality.py` | Structural Modality in the Non-Distilled Baselines | S6.4, Table S10 |
| `analysis/05_uniparc_lookup.py` | — | S8.1 |
| `analysis/06_species_assignment.py` | — | S8.1, S1.2 |
| `analysis/07_dataset_audit.py` | Dataset Curation and Preprocessing | S8.2, S8.3, Figure S5 |
| `analysis/08_species_restricted_eval.py` | — | S9, Figure S6 |
| `analysis/09_near_identical_variants.py` | — | S10 |
| `analysis/10_panel_screening.py` | Figure 4 | S11, S11.2, Table S11 |
| `analysis/11_additivity_decomposition.py` | Figure 7 | S14, S14.1 |
| `analysis/12_structure_accuracy.py` | Figure 5 | S12, S13.7, Table S12 |
| `analysis/13_fn_margin.py` | — | S15, Figure S7 |
| `analysis/14_modality_shap.py` | Figure 3 | S2.3.7 |
| `analysis/15_case_screening.py` | — | S4.1 |
| `analysis/16_stereoisomer_exhaustive.py` | Figure 8 | S4, S4.6, Table S6 |
| `analysis/17_integrated_gradients.py` | Figures 6, 7 | S13.2, S13.3, S17 |
| `analysis/18_alanine_mutagenesis.py` | Figures 6, 7 | S13.2, S13.3 |
| `analysis/19_attribution_sample.py` | — | S13.1, Table S13 |
| `analysis/20_attribution_registry.py` | Attributions Are Determined by the OR | S13.2, Table S14 |
| `analysis/21_untrained_control.py` | Figure 7 | S13.5, S13.6, Table S15 |
| `analysis/22_oof_validation.py` | — | S16.4 |
| `analysis/23_ood_enantiomer.py` | — | S15 |
| `analysis/24_class1_candidates.py` | — | S13.10, Table S16 |
| `analysis/25_determinism_check.py` | — | S16.2 |

## Random Seeds

The 20 random seeds used throughout this work are listed below for exact reproducibility. Several values were chosen as mnemonic references to familiar scientific constants or milestones.

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

## Data and Software Availability

### Primary Data: M2OR Database

This work is based on the **M2OR database**, available at [https://m2or.chemsensim.fr/](https://m2or.chemsensim.fr/):

> Lalis, M. et al. M2OR: a database of olfactory receptor–odorant pairs for understanding the molecular mechanisms of olfaction. *Nucleic Acids Res.* **2024**, *52*(D1), D1370–D1379.

Because the M2OR website does not explicitly specify redistribution terms for its source records, **the raw M2OR export (`pairs.csv`) is not included in this repository.** To run the full pipeline from raw data, download the M2OR dataset directly from the M2OR website and place the export at `data/pairs.csv`. The curation scripts (`01_smiles_validation.py`–`03_weight_assignment.py`) reproduce the curated dataset from this export, applying the selection and filtering criteria described in the manuscript (39,723 pairs retained from 53,444 initial entries).

### Curated Annotation Table

`data/final_dataset_weighted.csv` contains the curated dataset **annotations** required for all model training and evaluation (Phase 3 onward):

| Column | Description |
|--------|-------------|
| `main_compounds_id` | M2OR compound identifier (lookup key for ligand embeddings) |
| `main_receptors_id` | M2OR receptor identifier (lookup key for protein embeddings) |
| `responsive` | Binary responsiveness label (1 = responsive, 0 = non-responsive) |
| `data_quality` | M2OR assay quality flag |
| `sample_weight` | Confidence-based sample weight |

Raw molecular structures, SMILES strings, protein sequences, and the original M2OR export are not redistributed by this repository. The raw SMILES and protein sequences can be restored by joining `main_compounds_id` / `main_receptors_id` against the M2OR export. The training and evaluation scripts (Phase 3–6) do not require these strings and run directly on this table together with the precomputed embeddings.

### Precomputed Embeddings

`data/final_embeddings_all.pkl` provides all four embedding types, indexed by M2OR compound/receptor identifiers, to enable exact reproduction from Phase 3 onward without GPU-intensive embedding generation or external API access (embedding generation can vary slightly across hardware):

| Key | Description | Dim |
|-----|-------------|-----|
| `ligand_embeddings` | MoLFormer-XL semantic embeddings | 768 |
| `protein_embeddings` | ProtT5-XL semantic embeddings | 1024 |
| `ligand_structure_embeddings` | EGNN structural control embeddings | 512 |
| `protein_structure_embeddings` | SE(3)-Transformer structural control embeddings | 512 |
| `metadata_ligands` | Ligand InChIKeys (for identification) | — |
| `metadata_proteins` | Reserved (sequences not redistributed) | — |

Ligand entries are annotated with **InChIKeys** for structure identification. Raw SMILES strings and protein sequences are not stored in this file; retrieve them from the M2OR export via the provided identifiers if needed.

### Explainability (XAI) Analyses

The XAI scripts (`analysis/15_case_screening.py`, `analysis/17_integrated_gradients.py`, `analysis/18_alanine_mutagenesis.py`, `analysis/22_oof_validation.py`, `analysis/23_ood_enantiomer.py`) require raw SMILES/sequences for a small number of specific pairs reported in the manuscript. A helper script, `preprocessing/00_restore_raw_strings.py`, restores these strings locally by joining the curated annotation table against the M2OR export you download yourself. No raw molecular structures, SMILES strings, protein sequences, or original M2OR export files are redistributed by this repository.

1. Obtain the M2OR export as described above and place it at `data/pairs.csv`.
2. Run the helper to produce a local working copy with raw strings restored:

   ```bash
   python preprocessing/00_restore_raw_strings.py \
       --m2or data/pairs.csv \
       --curated data/final_dataset_weighted.csv \
       --out data/final_dataset_weighted_with_strings.csv \
       --canonicalize
   ```

   The `--canonicalize` flag standardizes SMILES with RDKit to exactly match the representation produced by the curation pipeline (recommended).
3. Point the XAI scripts' `RAW_DATA_CSV` at `data/final_dataset_weighted_with_strings.csv`.

Out-of-distribution test structures (e.g., (R/S)-sotolon) are specified directly within `analysis/23_ood_enantiomer.py` and require no external data.

### Model Weights

Trained teacher (`.model`) and student (`.pt`) weights for all 20 seeds × 10 folds are archived on Zenodo:

> Kim, E. C., & Jung, Y. (2026). ScentOR: Trained Model Weights [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20019062

| File | Contents | Size (approx.) |
|------|----------|----------------|
| `teacher_models.tar.gz` | LightGBM, XGBoost, CatBoost `.model` files | 536.1 MB |
| `student_models.tar.gz` | PyTorch `.pt` checkpoints for Scenarios V–VIII | 17.9 GB |

```bash
wget https://zenodo.org/records/20019062/files/teacher_models.tar.gz
wget https://zenodo.org/records/20019062/files/student_models.tar.gz
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

The source code, documentation, configuration files, scripts, curated annotation table, and precomputed embeddings distributed in this GitHub repository are released under the MIT License, except where otherwise noted.

The trained model weights archived on Zenodo are released under the Creative Commons Attribution 4.0 International License (CC-BY 4.0).

This repository does not redistribute the raw M2OR database. The preprocessing scripts provide a reproducible workflow for reconstructing the curated dataset from the original M2OR source. Users are responsible for complying with the terms of the upstream data sources, pretrained models, and software dependencies cited in the manuscript and this repository.

## Citation

If you use ScentOR in your research, please cite:

```bibtex
@misc{kim2026scentor,
  title={ScentOR: Semantic-Only Heterogeneous Knowledge Distillation for Odorant Receptor Deorphanization},
  author={Kim, Eun Cheol and Jung, YounJoon},
  year={2026},
  note={Under review}
}
```
