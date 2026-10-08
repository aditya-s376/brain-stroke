# Immediate Operational Context & Handoff Guide

## Role & Purpose of This File
This file serves as the **single source of immediate context, active operational constraints, feature specifications, and execution instructions** required by the next AI prompt or subagent working on this repository. Incoming agents read this file to immediately execute current tasks without needing to inspect background log files or project ledgers.

---

## Active Task Goal: Graph Feature Generation & 3-Way Feature Ablation
- **Current Phase:** Phase 3b (Graph / Connectivity Branch) & Phase 5 Feature Ablation.
- **Immediate Task 1:** Execute `src/graph_features.py` to extract 16 wPLI network features from stored epoch FIFs (`data/processed/figshare/*-epo.fif`) and write `features/figshare_graph_features.csv` (50×17) and `features/figshare_combined_features.csv` (50×36).
- **Immediate Task 2:** Run a 3-way feature family ablation study comparing Spectral-Only (19 features), Graph-Only (16 features), and Combined (35 features) using `src/train_models.py` under identical cross-validation protocol.

---

## Technical Constraints & Guardrails
1. **Target Problem:** Binary Classification (`0: mild` [NIHSS 1–4, n=33] vs `1: moderate` [NIHSS 5–11, n=17]). Severe stroke count on Figshare is 0.
2. **Feature Matrix ($X$):**
   - **Spectral Set (19 features):** `all_AI_delta_F3F4`, `all_AI_delta_C3C4`, `all_AI_delta_P3P4`, `all_AI_delta_O1O2`, `all_AI_theta_F3F4`, `all_AI_theta_C3C4`, `all_AI_theta_P3P4`, `all_AI_theta_O1O2`, `all_AI_alpha_F3F4`, `all_AI_alpha_C3C4`, `all_AI_alpha_P3P4`, `all_AI_alpha_O1O2`, `all_AI_beta_F3F4`, `all_AI_beta_C3C4`, `all_AI_beta_P3P4`, `all_AI_beta_O1O2`, `all_pdBSI`, `all_DAR`, `all_DTABR`.
   - **Graph Set (16 features):** `wPLI_{band}_{metric}` where bands are `delta`, `theta`, `alpha`, `beta` and metrics are `GlobalEff_AI`, `ClustCoeff_AI`, `MeanHomotopic`, `FrobeniusDist`.
   - **Combined Set (35 features):** 19 Spectral + 16 Graph.
   - **Strict Separation:** Column names containing `nihss`, `severity`, `subject_id`, or epoch counts must NEVER enter $X$.
3. **Pipeline Integrity:** All scaling (`StandardScaler`) and feature selection (`SelectKBest`) MUST live inside an `sklearn.pipeline.Pipeline` to guarantee zero data leakage across CV folds.
4. **Cross-Validation Setup:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`.
5. **Evaluation Metrics:** `f1_macro`, `balanced_accuracy`, `roc_auc` (binary mode). Always benchmark against `DummyClassifier(strategy='most_frequent')`.
6. **Model Constraints:** Classical ML only (Dummy, SVM, KNN, Random Forest, Logistic Regression). No deep learning. No heavy hyperparameter tuning. Skip SHAP for distance-based models.
7. **Graph Feature Bias:** Note that `MeanHomotopic` wPLI features were calculated without epoch-count sub-sampling and are strongly negatively correlated with the number of available epochs (`n_epochs_all`). Consider this bias during interpretation.

---

## File & Data Input/Output Locations
- **Processed Epoch Files:** `data/processed/figshare/*-epo.fif` (50 subjects).
- **Spectral Feature Table:** `features/figshare_features.csv` (50 rows × 63 columns).
- **Graph Feature Table:** `features/figshare_graph_features.csv` (50 rows × 17 columns).
- **Combined Feature Table:** `features/figshare_combined_features.csv` (50 rows × 36 columns).
- **Graph Extractor:** `src/graph_features.py`.
- **ML Trainer:** `src/train_models.py`.

---

## Immediate Handoff Instructions for Next Prompt
1. **Run Graph Feature Extraction:**
   Run `python src/graph_features.py`. Confirm `features/figshare_graph_features.csv` (50×17) and `features/figshare_combined_features.csv` (50×36) are generated with zero `NaN` or `inf` values.
2. **Execute Feature Family Ablation:**
   Run `src/train_models.py` (or adapt it) across the three feature representations (Spectral-19, Graph-16, Combined-35) under identical 5×10 CV protocol.
3. **Save Results & Update Ledgers:**
   Store output metrics under `results/metrics/` and document findings in `docs/Ledgers/project_technical_ledger_clean.md` and `docs/Ledgers/project_timeline_ledger_clean.md`.