# Project Timeline Ledger

## Purpose

This ledger records the project's evolution in chronological order. It preserves major states, decisions, milestones, discoveries, and changes in direction. It is historical: later changes do not rewrite earlier states.

---

## State 0 — Initial Project State

**Project:** EEG-based machine-learning study for stroke-severity classification.

**Research objective:** Predict stroke severity from EEG patterns. The initial target framework was NIHSS-derived severity classes: mild, moderate, and severe. The central hypothesis was that interhemispheric asymmetry in EEG spectral power could act as a biomarker of clinically assessed stroke severity.

**Initial pipeline state:**
- Figshare stroke EEG dataset was the primary dataset.
- UCLH stroke EEG data was planned for later external validation.
- Data acquisition, preprocessing/epoching, spectral feature extraction, patient-level aggregation, and NIHSS label merging were already completed for Figshare.
- `features/figshare_features.csv` existed as the modeling-ready table.
- Modeling and evaluation had not yet been performed.

The project was therefore a research/analysis pipeline rather than a finished model or deployed application.

---

## State 1 — Pre-Phase-5 Dataset Audit

The Figshare feature tables were audited before modeling.

**Verified dataset state:**
- `features/figshare_features.csv`: 50 rows × 63 columns, 50 unique subjects.
- `features/figshare_patient_features.csv`: 50 rows × 61 columns, 50 unique subjects.
- No missing values, infinite values, duplicate subjects, or constant columns were found.
- The modeling table was identified as `figshare_features.csv` because it contains the merged labels.

**Feature-scope decision:** The Phase 5 model input was fixed to exactly 19 `all_*` engineered features:
16 interhemispheric spectral asymmetry features across four bands and four homologous channel pairs, plus `all_pdBSI`, `all_DAR`, and `all_DTABR`. Hand-specific feature groups and epoch-count columns were excluded.

A leakage concern around epoch-count features was identified and resolved by excluding them from the model input.

---

## State 2 — Phase 5 Model Design and Implementation

`src/train_models.py` was created as the Phase 5 modeling pipeline.

The benchmark was expanded to include logistic regression alongside SVM, KNN, and Random Forest models. The candidate set became 20 variants including the dummy baseline.

Key implementation decisions were fixed before the canonical run:
- repeated stratified 5-fold cross-validation;
- seed 42 and 10 repeats for the canonical experiment;
- preprocessing and feature selection inside sklearn pipelines;
- binary ROC-AUC scoring;
- both KBest-selected and all-19-feature variants;
- a separate 3-feature global representation;
- explicit model/feature-set naming;
- model export and metadata generation;
- exploratory confusion-matrix and feature-importance outputs.

---

## State 3 — Phase 5 Review and Data Validation

Independent review identified several implementation and scientific safeguards that needed to be addressed before trusting the experiment.

The project adopted the following corrections:
- saved SVC configuration should match the CV-evaluated configuration;
- CV should fail explicitly on invalid folds/configurations;
- class-count and KNN fold-size checks should be performed before CV;
- non-canonical runs should be clearly identified;
- post-selection confusion matrices and full-data importance plots should be treated as exploratory/descriptive;
- saved-model metadata should include a feature-construction contract sufficient for later UCLH reuse.

The NIHSS audit was then completed:

| Severity class | Subjects | NIHSS range |
|---|---:|---:|
| mild | 33 | 1–4 |
| moderate | 17 | 5–11 |
| severe | 0 | none |

The observed data therefore supports a binary mild-vs-moderate Phase 5 problem. No severe subjects are present in the Figshare cohort.

---

## State 4 — Canonical Spectral Benchmark Completed

The canonical Phase 5 experiment was run using 5-fold repeated stratified CV with 10 repeats and seed 42.

**Dummy baseline:**
- macro-F1: 0.397059
- balanced accuracy: 0.500000
- ROC-AUC: 0.500000

**Best macro-F1 result:** `knn5_kbest8`
- macro-F1: 0.499927 ± 0.157864
- balanced accuracy: 0.520476 ± 0.142947
- ROC-AUC: 0.492738 ± 0.210890

**Highest ROC-AUC result:** `logreg_l2_global3`
- macro-F1: 0.477291 ± 0.137659
- balanced accuracy: 0.487619 ± 0.139610
- ROC-AUC: 0.584405 ± 0.173687

The spectral feature family therefore did not demonstrate strong or stable classification performance. The scores remain close to the dummy baseline and have high fold-to-fold variability.

This result established spectral asymmetry as the baseline to beat rather than motivating additional post-hoc hyperparameter tuning.

---

## State 5 — Second Feature Family Added: Graph/Connectivity Branch

The research scope was expanded with a second EEG feature family based on functional connectivity and hemispheric graph structure.

A read-only dataset audit established:
- 30 EEG channels are present in the preprocessed epochs.
- The montage contains 12 left-hemisphere, 12 right-hemisphere, and 6 midline channels.
- 12 homologous left/right channel pairs are available.
- Stored `*-epo.fif` files retain full-channel time-domain epoch signals, so graph connectivity can operate on those epochs without re-epoching.

A major artifact-quality risk was identified: the existing feature workflow is centered on eight core channels, so previously unused peripheral channels may not have equivalent artifact validation.

The graph branch was therefore deliberately restricted to the same eight core channels:
F3, C3, P3, O1 and F4, C4, P4, O2.

The approved graph pipeline is:

**band-limited epochs → wPLI connectivity → left/right hemispheric graphs → homologous pairing/deformation → compact network-level features**

The planned graph representation contains 16 subject-level features across delta, theta, alpha, and beta:
- global-efficiency asymmetry;
- mean clustering-coefficient asymmetry;
- mean homotopic connectivity;
- normalized left/right Frobenius-distance.

The intended later ablation is:
1. spectral-only (19 features);
2. graph-only (16 features);
3. combined spectral + graph (35 features).

External UCLH validation remains downstream of this comparison.

---

## State 6 — Current Project Position

The project has moved from the completed spectral baseline into the graph/connectivity branch.

The current research progression is:

**Figshare preprocessing and spectral features → spectral ML benchmark completed → graph feature branch designed/audited → graph feature extractor implemented → execution and ablation next → UCLH external validation later.**

No claim has yet been made that graph features improve classification; that is the purpose of the next experiment.

---

## State 7 — Graph Feature Extractor Implemented

`src/graph_features.py` was created as the Phase 3b graph feature pipeline.

**Implementation decisions fixed:**

- Node set restricted to the same 8 channels used by the spectral pipeline (F3, C3, P3, O1 | F4, C4, P4, O2), matching the channels with established artifact quality and maximising UCLH montage compatibility.
- wPLI computed with `mne_connectivity.spectral_connectivity_epochs`, `method='wpli'`, `mode='multitaper'`, `faverage=True` (band-averaged scalar per channel pair).
- Adjacency matrices (8×8) are symmetrised and diagonal-zeroed before sub-matrix extraction.
- Left sub-matrix: rows/cols 0–3; right sub-matrix: rows/cols 4–7; homotopic block: cross-hemisphere off-diagonal elements.
- Graph metrics use NetworkX: global efficiency (Dijkstra on 1/weight distances), mean weighted clustering coefficient (Onnela et al. 2005 geometric-mean formulation).
- All asymmetry features use the same AI formula as the spectral pipeline: (L − R) / (L + R), with a safe-divide fallback of 0.0.
- Frobenius distance is normalised: ‖A_L − A_R‖_F / (‖A_L‖_F + ‖A_R‖_F).
- Sampling-rate guard: raises if sfreq deviates more than 1 Hz from 500 Hz.
- Per-subject failures are logged and reported without aborting the batch.
- The 16 canonical feature names are defined in the module constant `GRAPH_FEATURE_NAMES` (deterministic order: delta/theta/alpha/beta × GlobalEff_AI/ClustCoeff_AI/MeanHomotopic/FrobeniusDist).

**Outputs (written at runtime):**

- `features/figshare_graph_features.csv` — 50 rows × 17 columns (subject_id + 16 features).
- `features/figshare_combined_features.csv` — inner join of spectral CSV and graph CSV on subject_id; retains all spectral columns (including severity_class) plus the 16 graph features.

The script has not yet been executed. Execution and output verification are the next step.

---

## State 8 — Graph CSV Validation and Bias Discovery

An independent read-only audit of the generated graph features revealed the following:
- **Graph CSV Sanity:** Zero NaN/inf counts, no constant columns, and plausible feature value ranges.
- **Epoch-count Bias:** `MeanHomotopic` features are strongly negatively correlated (up to -0.58) with `n_epochs_all`. `graph_features.py` utilizes the standard `wpli` method without addressing the epoch-count bias (no epoch balancing/sub-sampling was implemented).
- **Metadata Availability:** `participants.tsv` is missing from the repository, so patient lesion side, stroke type, and lesion location metadata cannot be verified.
- **UCLH Labels:** External UCLH labels available are verified as Stroke vs Control and NIHSS, while lesion side remains unverified.
