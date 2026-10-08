# Agent Rules (All AI Tools Read This First)

- **Project Context:** EEG asymmetry -> stroke severity. Read `docs/PHASE5_CONTEXT.md` and `PROJECT_EXECUTION_GUIDE.md` before performing any task.
- **Task Goal:** Binary classification (`0=mild`, `1=moderate`). The severe class has 0 samples on Figshare.
- **Feature Matrix ($X$):** Use exactly the 19 engineered features. NEVER include NIHSS scores, severity labels, subject IDs, or epoch count columns in $X$.
- **Pipeline Integrity:** All scaling (`StandardScaler`) and feature selection (`SelectKBest`) MUST live inside a `sklearn.pipeline.Pipeline` to guarantee zero data leakage across CV folds.
- **Cross-Validation:** Use `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`.
- **Metrics:** `f1_macro`, `balanced_accuracy`, `roc_auc` (binary mode). Always benchmark against `DummyClassifier(strategy='most_frequent')`.
- **Model Constraints:** Classical ML only (SVM, RF, KNN). No deep learning. No heavy hyperparameter tuning. Skip SHAP for distance-based models (use RF/permutation importances only).
- **Scope Limit:** Do NOT modify Phase 0–4 files in `src/` or `features/` unless explicitly instructed.
- **Logging Rule:** Maintain running technical and timeline logs in `docs/Ledgers/` and update operational handoff context in `docs/PHASE5_CONTEXT.md`.