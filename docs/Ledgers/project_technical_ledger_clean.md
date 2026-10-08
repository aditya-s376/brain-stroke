# Project Technical Ledger

## Purpose

This document describes only the **current technical state** of the project. It is continuously rewritten as the implementation changes. Historical evolution belongs in the Timeline Ledger.

---

# I. Technical Details

### Project objective and data

The project is an EEG-based machine-learning study for stroke-severity classification. The current Figshare cohort contains 50 subjects and supports only a binary problem in practice: 33 mild subjects and 17 moderate subjects; no severe subjects are present.

The primary target is `severity_class`, encoded as:
- mild → 0
- moderate → 1

NIHSS validation currently shows:
- mild: NIHSS 1–4
- moderate: NIHSS 5–11

The planned external-validation dataset is UCLH.

### Repository and important files

The project is a Python scientific/ML repository. The important current files/components are:

- `PROJECT_EXECUTION_GUIDE.md`: overall execution plan and phase definitions.
- `AGENTS.md`: repository-level constraints for the modeling work.
- `docs/PHASE5_CONTEXT.md`: Phase 5 implementation context.
- `docs/PROGRESS.md`: concise project progress record.
- `src/load_figshare.py`: loads the Figshare EEG data and establishes the available EEG channels.
- `src/preprocessing.py`: preprocessing, filtering, epoch creation, and saving of processed epochs.
- `src/feature_extraction.py`: spectral power, asymmetry, and global spectral feature calculations.
- `src/aggregate_patient_features.py`: converts epoch-level features into patient-level summaries.
- `src/extract_labels.py`: extracts clinical labels.
- `src/labels.py`: defines the NIHSS-to-severity mapping.
- `src/build_feature_table.py`: creates the final labeled Figshare feature table.
- `src/train_models.py`: Phase 5 model training and evaluation.
- `src/graph_features.py`: Phase 3b graph feature extractor (wPLI/multitaper, 16 network-level features). Implemented; not yet executed.
- `features/figshare_features.csv`: current 50-subject, labeled spectral modeling table.
- `features/figshare_patient_features.csv`: patient-level feature table without the final merged labels.
- `features/figshare_graph_features.csv`: graph-only subject-level table (written at runtime by graph_features.py).
- `features/figshare_combined_features.csv`: spectral + graph combined table, 35 features (written at runtime by graph_features.py).

### Spectral feature pipeline

The existing spectral pipeline uses four EEG frequency bands:

- delta: 1–4 Hz
- theta: 4–8 Hz
- alpha: 8–13 Hz
- beta: 13–30 Hz

Interhemispheric asymmetry is calculated using the established left-minus-right formulation:

\[
AI = \frac{P_{left}-P_{right}}{P_{left}+P_{right}}
\]

The primary homologous channel pairs used for the 19-feature model are:
- F3/F4
- C3/C4
- P3/P4
- O1/O2

The model input is restricted to the 19 `all_*` patient-level features:
16 band/channel asymmetry features plus:
- `all_pdBSI`
- `all_DAR`
- `all_DTABR`

The `left_hand_*` and `right_hand_*` feature families are not part of the Phase 5 model input, and epoch-count columns are excluded.

Spectral power is based on Welch PSD with `n_fft=256`. Current preprocessing uses a 1–40 Hz band-pass and 50 Hz notch filter. Epochs are 4.0 s, with the end time adjusted to account for the final sample. Patient-level features are obtained by mean aggregation over clean epochs per condition; the Phase 5 model uses the `all` condition.

### Phase 5 model family

`src/train_models.py` currently evaluates 20 candidate configurations including the dummy baseline.

The classifier families are:
- majority-class DummyClassifier;
- linear SVM;
- RBF SVM;
- KNN with k = 3, 5, 7;
- Random Forest;
- regularized logistic regression.

Feature representations include:
- all 19 spectral features;
- 8 features selected with `SelectKBest(f_classif, k=8)` from the 19;
- the 3 global spectral features (`all_pdBSI`, `all_DAR`, `all_DTABR`).

Scaling and feature selection are contained inside sklearn pipelines so that they are fitted within CV folds rather than before CV.

The canonical evaluation uses:
- `RepeatedStratifiedKFold`
- 5 folds
- 10 repeats
- random seed 42
- macro-F1
- balanced accuracy
- binary ROC-AUC

The implementation also contains model export and metadata generation. SVC models use `decision_function` for scoring rather than relying on probability calibration.

### Current spectral-model result

The canonical run establishes the following current benchmark:

- Dummy: macro-F1 0.3971, balanced accuracy 0.5000, ROC-AUC 0.5000.
- Best macro-F1: `knn5_kbest8`, macro-F1 0.4999 ± 0.1579.
- Highest ROC-AUC: `logreg_l2_global3`, ROC-AUC 0.5844 ± 0.1737, with balanced accuracy 0.4876 ± 0.1396.

The current interpretation is that spectral asymmetry alone has weak and unstable predictive value for this 50-subject mild-vs-moderate classification problem.

### Graph/connectivity branch — `src/graph_features.py` (implemented; not yet run)

The second feature family uses the stored time-domain epochs rather than PSD tables.

The audited Figshare montage contains 30 channels:

Left hemisphere: FP1, F3, F7, FC3, FT7, C3, T3, CP3, TP7, P3, T5, O1  
Right hemisphere: FP2, F4, F8, FC4, FT8, C4, T4, CP4, TP8, P4, T6, O2  
Midline: Fz, FCz, Cz, CPz, Pz, Oz

**Node set (fixed; same 8 as spectral pipeline):**
- Left: F3, C3, P3, O1  (indices 0–3 in 8-node graph)
- Right: F4, C4, P4, O2  (indices 4–7 in 8-node graph)

**Homologous pairs (cross-hemisphere):** F3↔F4, C3↔C4, P3↔P4, O1↔O2.

**Connectivity:** `mne_connectivity.spectral_connectivity_epochs`, `method='wpli'`, `mode='multitaper'`, `faverage=True`. One 8×8 adjacency matrix per band per subject, symmetrised, diagonal zeroed, clipped to [0, 1].

**Graph metrics (NetworkX):**
- Global efficiency: Dijkstra on 1/weight distances, normalised by n(n−1).
- Mean clustering coefficient: Onnela et al. 2005 geometric-mean formulation.

**Asymmetry formula (consistent with spectral pipeline):** AI = (L − R) / (L + R); safe-divide returns 0.0.

**Frobenius distance:** ‖A_L − A_R‖_F / (‖A_L‖_F + ‖A_R‖_F).

**The 16 canonical feature names** (module constant `GRAPH_FEATURE_NAMES`, deterministic order):
- `wPLI_delta_GlobalEff_AI`, `wPLI_delta_ClustCoeff_AI`, `wPLI_delta_MeanHomotopic`, `wPLI_delta_FrobeniusDist`
- `wPLI_theta_GlobalEff_AI`, `wPLI_theta_ClustCoeff_AI`, `wPLI_theta_MeanHomotopic`, `wPLI_theta_FrobeniusDist`
- `wPLI_alpha_GlobalEff_AI`, `wPLI_alpha_ClustCoeff_AI`, `wPLI_alpha_MeanHomotopic`, `wPLI_alpha_FrobeniusDist`
- `wPLI_beta_GlobalEff_AI`, `wPLI_beta_ClustCoeff_AI`, `wPLI_beta_MeanHomotopic`, `wPLI_beta_FrobeniusDist`

**Sampling-rate guard:** raises ValueError if sfreq deviates > 1 Hz from 500 Hz.

**Outputs (written at runtime):**
- `features/figshare_graph_features.csv` — 50 × 17 (subject_id + 16 features).
- `features/figshare_combined_features.csv` — inner join on subject_id with spectral CSV; retains all spectral columns (including severity_class) + 16 graph features = 35 ML-candidate features.

### Expected experimental structure

The graph branch will eventually be evaluated through a controlled three-way ablation:

**Spectral only:** 19 features  
**Graph only:** 16 features  
**Combined:** 35 features

The comparisons should use the same CV philosophy and leakage safeguards as the spectral baseline.

---

# II. Notes on Future Implementations and Things to Keep in Mind

The immediate execution task is running `src/graph_features.py` on the existing `*-epo.fif` files and verifying the 50×17 graph CSV output. The script does not require re-epoching.

After verification, the ablation study requires extending `src/train_models.py` (or a new script) to evaluate three fixed feature families under the same CV protocol:
1. Spectral only (19 features from `figshare_features.csv`).
2. Graph only (16 features from `figshare_graph_features.csv`).
3. Combined (35 features from `figshare_combined_features.csv`).

The graph feature contract (node order, pair orientation, formula, band edges) must match this implementation exactly when reproducing features on UCLH data.

The eight-channel restriction should remain in place unless a future artifact audit confirms additional peripheral channels are reliably clean.

The combined experiment should not respond to results by repeatedly expanding the model/search space. The purpose of the ablation is to compare fixed feature families under the same evaluation protocol, not to tune until one representation wins.

The model metadata/feature contract should remain the authoritative description of the spectral feature-generation assumptions needed for UCLH transfer, including the asymmetry formula, channel-pair orientation, bands, PSD settings, filtering, epoch settings, aggregation, feature names, and label map.

The graph branch should explicitly verify EDF sampling frequency and the actual epoch timing relative to the motor-imagery cue before implementation is treated as final.

---

# III. Limitations / Drawbacks

The primary limitation is sample size: N=50 with only 17 moderate subjects. This produces substantial variance in model estimates and makes selection among many candidate models intrinsically unstable.

The Figshare dataset contains no severe cases, so the current empirical study cannot support a three-class mild/moderate/severe classifier. Its demonstrated task is binary mild-vs-moderate classification.

The spectral baseline is near chance. The small numerical advantages of individual models are not sufficient to establish clinical or cross-dataset generalization.

`n_fft=256` at the assumed 500 Hz sampling rate gives approximately 1.95 Hz frequency resolution. Consequently, the 1–4 Hz delta band is represented by only a small number of frequency bins. This is a known spectral-resolution trade-off of the existing feature pipeline.

The 500 Hz sampling-rate value is documented in project context but has not been independently asserted by the preprocessing source itself. It therefore remains a verification item rather than an unquestioned implementation fact.

Artifact rejection/quality control is strongest for the eight core channels used by the established spectral pipeline. Extending connectivity analysis to the remaining peripheral channels could introduce undetected artifacts, which is why the graph branch is currently restricted to the eight core channels.

Adding graph features increases dimensionality from 19 to 35 candidate features in the combined table. With N=50, this increases overfitting risk; the design therefore avoids hundreds of edge-level ML features and keeps the representation at network level.

The graph metrics may still be sensitive to channel availability and montage differences. UCLH may not contain an identical montage, so cross-dataset transfer requires a carefully defined channel intersection and feature-construction contract.

The confusion matrix and full-dataset RF Gini/logistic coefficient plots from Phase 5 are descriptive/exploratory outputs rather than unbiased estimates of generalization performance.

The project is not yet a validated clinical tool. The current evidence establishes a reproducible research baseline and motivates testing whether functional connectivity adds information beyond spectral asymmetry.
 
**Recent Findings (State 8 Audit):**
- **wPLI Epoch Bias:** Graph features were extracted using the standard `wpli` method without addressing epoch-count sensitivity (e.g., via sub-sampling). As a result, `MeanHomotopic` connectivity is strongly negatively correlated with the number of clean epochs.
- **Missing Metadata:** `participants.tsv` is missing from the repository. Consequently, patient lesion side, stroke type, and lesion location cannot be verified or incorporated.
