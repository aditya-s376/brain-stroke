
"""
src/train_models.py — Phase 5 / Ablation: Classical ML benchmarking on Figshare EEG features.

NOTE (selection optimism): This script evaluates multiple model configurations and picks the
best by RepeatedStratifiedKFold CV score. On N=50 subjects this introduces mild winner's-curse
(positive-selection bias). CV metrics are exploratory, not final test-set performance.

Usage:
    # Spectral-only (reproduces results_spectral_v1 baseline — regression reference):
    python src/train_models.py --feature-set spectral --output-dir results
    # Graph-only:
    python src/train_models.py --feature-set graph --output-dir results
    # Combined spectral + graph (35 features):
    python src/train_models.py --feature-set combined --output-dir results
    # Override CSV path:
    python src/train_models.py --feature-set spectral --features-csv path/to/custom.csv

Outputs per feature set are written to: <output-dir>/<feature_set>/
Ablation summary (all 3 runs) is written to: <output-dir>/ablation_summary.csv
"""

import argparse
import datetime
import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    permutation_test_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ─── Feature constants ─────────────────────────────────────────────────────────

# 19 spectral features: 16 asymmetry indices (4 pairs × 4 bands) + pdBSI + DAR + DTABR.
# Never assembled via column-drop — prevents banned columns from entering X by construction.
SPECTRAL_FEATURES: list[str] = [
    "all_AI_delta_F3F4", "all_AI_delta_C3C4", "all_AI_delta_P3P4", "all_AI_delta_O1O2",
    "all_AI_theta_F3F4", "all_AI_theta_C3C4", "all_AI_theta_P3P4", "all_AI_theta_O1O2",
    "all_AI_alpha_F3F4", "all_AI_alpha_C3C4", "all_AI_alpha_P3P4", "all_AI_alpha_O1O2",
    "all_AI_beta_F3F4",  "all_AI_beta_C3C4",  "all_AI_beta_P3P4",  "all_AI_beta_O1O2",
    "all_pdBSI", "all_DAR", "all_DTABR",
]  # exactly 19 entries

# 16 wPLI graph features — names taken verbatim from features/figshare_graph_features.csv.
GRAPH_FEATURES: list[str] = [
    "wPLI_delta_GlobalEff_AI",  "wPLI_delta_ClustCoeff_AI",
    "wPLI_delta_MeanHomotopic", "wPLI_delta_FrobeniusDist",
    "wPLI_theta_GlobalEff_AI",  "wPLI_theta_ClustCoeff_AI",
    "wPLI_theta_MeanHomotopic", "wPLI_theta_FrobeniusDist",
    "wPLI_alpha_GlobalEff_AI",  "wPLI_alpha_ClustCoeff_AI",
    "wPLI_alpha_MeanHomotopic", "wPLI_alpha_FrobeniusDist",
    "wPLI_beta_GlobalEff_AI",   "wPLI_beta_ClustCoeff_AI",
    "wPLI_beta_MeanHomotopic",  "wPLI_beta_FrobeniusDist",
]  # exactly 16 entries

# Combined = spectral 19 + graph 16 = 35. Order is stable for pipeline reproducibility.
COMBINED_FEATURES: list[str] = SPECTRAL_FEATURES + GRAPH_FEATURES

assert len(SPECTRAL_FEATURES) == 19,  "SPECTRAL_FEATURES must have exactly 19 entries"
assert len(GRAPH_FEATURES) == 16,     "GRAPH_FEATURES must have exactly 16 entries"
assert len(COMBINED_FEATURES) == 35,  "COMBINED_FEATURES must have exactly 35 entries"

# Map feature-set name → (feature list, default CSV relative path).
FEATURE_SET_CONFIG: dict[str, tuple[list[str], str]] = {
    "spectral": (SPECTRAL_FEATURES, "features/figshare_features.csv"),
    "graph":    (GRAPH_FEATURES,    "features/figshare_graph_features.csv"),
    "combined": (COMBINED_FEATURES, "features/figshare_combined_features.csv"),
}

# Three global summary features — spectral-only *_global3 diagnostic variants.
# SelectKBest(k=8) is omitted from global3 pipelines because k=8 > n_features=3.
GLOBAL_ONLY: list[str] = ["all_pdBSI", "all_DAR", "all_DTABR"]

# Pre-registered primary model for permutation tests.
PRIMARY_MODEL: str = "logreg_l2_kbest8"

# Columns that must NEVER appear in X — asserted explicitly after assembly.
BANNED_IN_X: frozenset[str] = frozenset({
    "subject_id", "nihss_raw", "severity_class",
    "n_epochs_all", "n_epochs_left_hand", "n_epochs_right_hand",
})

LABEL_MAP: dict[str, int] = {"mild": 0, "moderate": 1}


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5 / Ablation — Figshare EEG stroke severity classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--feature-set",
        choices=["spectral", "graph", "combined"],
        default="spectral",
        help=(
            "Feature family to evaluate. "
            "'spectral'=19 spectral asymmetry features (Phase 5 baseline); "
            "'graph'=16 wPLI graph features; "
            "'combined'=35 spectral+graph features."
        ),
    )
    p.add_argument(
        "--features-csv", type=Path, default=None,
        help=(
            "Override path to the feature CSV. "
            "Defaults to the canonical CSV for the selected --feature-set."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Root output directory. A per-feature-set sub-directory is created automatically.",
    )
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--n-repeats", type=int, default=10,
        help="Number of repeats for RepeatedStratifiedKFold",
    )
    p.add_argument(
        "--n-permutations", type=int, default=500,
        help="Number of permutations for permutation_test_score (primary model only)",
    )
    return p.parse_args()


# ─── Data loading & validation ─────────────────────────────────────────────────

def load_and_validate(
    csv_path: Path,
    features: list[str],
    seed: int,
    include_global3: bool = False,
) -> tuple[np.ndarray, "np.ndarray | None", np.ndarray, pd.DataFrame]:
    """
    Load a feature CSV and validate all preconditions.

    Parameters
    ----------
    csv_path       : path to the feature CSV.
    features       : ordered list of feature column names that form X.
    include_global3: if True, also extract X_global3 for spectral *_global3 variants.

    Returns
    -------
    X_main    : float array, shape (50, len(features))
    X_global3 : float array (50, 3) if include_global3 else None
    y         : int array (50,) — 0=mild, 1=moderate
    df        : raw DataFrame (retained for debugging; not used in CV)
    """
    df = pd.read_csv(csv_path)

    assert len(df) == 50, f"Expected 50 rows, got {len(df)}"
    assert df["subject_id"].nunique() == 50, "Duplicate subject IDs detected in CSV"

    if "severity_class" not in df.columns:
        project_root = Path(__file__).resolve().parents[1]
        canonical_csv_path = project_root / "features" / "figshare_features.csv"
        df_labels = pd.read_csv(canonical_csv_path)[["subject_id", "severity_class"]]
        df = df.merge(df_labels, on="subject_id", how="left")
        assert df["severity_class"].notna().all(), "Failed to merge severity_class for all subjects"
        assert len(df) == 50, "Merge altered row count unexpectedly"

    # ── Assemble X from the supplied features list.
    # LEAKAGE-SAFE: X is built from the hard-coded constant, not via column-drop.
    # No identifier, label, or epoch-count column can enter X by construction.
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns missing from CSV: {missing_cols}")

    # Belt-and-suspenders: assert no banned column accidentally overlaps the feature list.
    banned_overlap = BANNED_IN_X & set(features)
    if banned_overlap:
        raise AssertionError(
            f"BANNED columns found in feature list — fix the constant: {banned_overlap}"
        )

    X_main = df[features].to_numpy(dtype=float)

    assert X_main.shape == (50, len(features)), (
        f"Expected X shape (50, {len(features)}), got {X_main.shape}"
    )
    assert not np.isnan(X_main).any(),  "NaN values detected in X"
    assert np.isfinite(X_main).all(),   "Infinite values detected in X"

    X_global3 = None
    if include_global3:
        missing_g3 = [f for f in GLOBAL_ONLY if f not in df.columns]
        if missing_g3:
            print(f"  NOTE: global3 columns absent ({missing_g3}); *_global3 variants skipped.")
        else:
            X_global3 = df[GLOBAL_ONLY].to_numpy(dtype=float)

    # ── Label encoding
    unknown = set(df["severity_class"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected severity classes in CSV: {unknown}")
    y = df["severity_class"].map(LABEL_MAP).to_numpy(dtype=int)
    assert set(np.unique(y)) == {0, 1}, f"Expected binary labels {{0,1}}, got {set(np.unique(y))}"

    counts = pd.Series(y).value_counts().sort_index()
    print("Class counts:")
    print(f"  0 = mild     : {counts.get(0, 0)}")
    print(f"  1 = moderate : {counts.get(1, 0)}")

    # ── Preflight: class balance for CV
    min_class_count = int(np.bincount(y).min())
    assert min_class_count >= 5, (
        f"Smallest class has only {min_class_count} samples — "
        "StratifiedKFold(n_splits=5) requires at least 5."
    )

    np.random.seed(seed)  # seed global numpy state; sklearn estimators use their own random_state
    return X_main, X_global3, y, df


# ─── Pipeline component helpers ────────────────────────────────────────────────

def _svc(kernel: str, seed: int, probability: bool = False) -> SVC:
    """SVC with balanced class weights. probability=False during CV for speed & no Platt distortion."""
    return SVC(
        kernel=kernel,
        probability=probability,
        class_weight="balanced",
        random_state=seed,
    )


def _knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, weights="distance")


def _rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        class_weight="balanced",
        random_state=seed,
    )


def _kbest() -> SelectKBest:
    """SelectKBest(f_classif, k=8). Only used when n_input_features >= 8."""
    return SelectKBest(f_classif, k=8)


def _logreg(
    l1_ratio: float,
    C: float,
    seed: int,
    solver: str = "lbfgs",
) -> LogisticRegression:
    """LogisticRegression with balanced class weights and generous max_iter for convergence."""
    return LogisticRegression(
        l1_ratio=l1_ratio,
        C=C,
        solver=solver,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )


# ─── Pipeline factory ──────────────────────────────────────────────────────────

def build_cv_variants(
    seed: int, feature_set_name: str, active_features: list[str]
) -> list[tuple[str, str, list[str], Pipeline]]:
    """
    Build all 16 named (name, feature_set_name, feature_list, Pipeline) tuples.

    Naming convention: <model>_<variant_suffix>
      *_kbest8  — SelectKBest(k=8) inside the pipeline, all-19 inputs
      *_all19   — no feature selection, all-19 inputs
      *_global3 — no feature selection, only 3 global features (pdBSI / DAR / DTABR)

    SVC probability=False for CV — AUC is computed via decision_function, avoiding
    Platt-scaling overhead and distortion in small folds.
    """
    V: list[tuple[str, str, list[str], Pipeline]] = []

    # ── Group 1: Dummy baseline
    V.append((
        "dummy", "all19", active_features,
        Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
    ))

    # ── Group 2: SelectKBest(k=8) variants — 19 inputs, 8 selected internally
    # LEAKAGE-SAFE: SelectKBest is a Pipeline step; fitted inside each CV fold.
    for vname, clf in [
        ("svm_linear", _svc("linear", seed)),
        ("svm_rbf",    _svc("rbf",    seed)),
    ]:
        V.append((
            f"{vname}_kbest8", "all19", active_features,
            Pipeline([
                ("scaler", StandardScaler()),
                ("kbest",  _kbest()),
                ("clf",    clf),
            ]),
        ))

    for k in [3, 5, 7]:
        V.append((
            f"knn{k}_kbest8", "all19", active_features,
            Pipeline([
                ("scaler", StandardScaler()),
                ("kbest",  _kbest()),
                ("clf",    _knn(k)),
            ]),
        ))

    V.append((
        "rf_kbest8", "all19", active_features,
        Pipeline([
            ("kbest", _kbest()),  # RF does not need scaling
            ("clf",   _rf(seed)),
        ]),
    ))

    # ── Group 3: All-19 variants — no SelectKBest
    for vname, clf in [
        ("svm_linear", _svc("linear", seed)),
        ("svm_rbf",    _svc("rbf",    seed)),
    ]:
        V.append((
            f"{vname}_all19", "all19", active_features,
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    clf),
            ]),
        ))

    for k in [3, 5, 7]:
        V.append((
            f"knn{k}_all19", "all19", active_features,
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    _knn(k)),
            ]),
        ))

    V.append((
        "rf_all19", "all19", active_features,
        Pipeline([("clf", _rf(seed))]),
    ))

    # ── Group 4: Global-3 variants — 3 features only; SelectKBest(k=8) omitted (k > n_features)
    if feature_set_name == "spectral":
        V.append((
            "svm_rbf_global3", "global3", GLOBAL_ONLY,
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    _svc("rbf", seed)),
            ]),
        ))
        V.append((
            "knn5_global3", "global3", GLOBAL_ONLY,
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    _knn(5)),
            ]),
        ))
        V.append((
            "rf_global3", "global3", GLOBAL_ONLY,
            Pipeline([("clf", _rf(seed))]),
        ))

    # ── Group 5: Logistic Regression variants
    # LEAKAGE-SAFE: StandardScaler and SelectKBest are Pipeline steps; fitted inside each CV fold.
    V.append((
        "logreg_l2_kbest8", "all19", active_features,
        Pipeline([
            ("scaler", StandardScaler()),
            ("kbest",  _kbest()),
            ("clf",    _logreg(0.0, 1.0, seed, solver="lbfgs")),
        ]),
    ))
    V.append((
        "logreg_l2_all19", "all19", active_features,
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    _logreg(0.0, 1.0, seed, solver="lbfgs")),
        ]),
    ))
    if feature_set_name == "spectral":
        V.append((
            "logreg_l2_global3", "global3", GLOBAL_ONLY,
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    _logreg(0.0, 1.0, seed, solver="lbfgs")),
            ]),
        ))
    V.append((
        "logreg_l1_all19", "all19", active_features,
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    _logreg(1.0, 0.5, seed, solver="liblinear")),
        ]),
    ))

    expected_count = 20 if feature_set_name == "spectral" else 16
    assert len(V) == expected_count, f"Expected {expected_count} variants, got {len(V)}"
    return V


# ─── CV evaluation ─────────────────────────────────────────────────────────────

def run_cv(
    variants: list[tuple[str, str, list[str], Pipeline]],
    X_main: np.ndarray,
    X_global3: "np.ndarray | None",
    y: np.ndarray,
    cv: RepeatedStratifiedKFold,
) -> pd.DataFrame:
    """
    Evaluate all variants via cross_validate and return a results DataFrame.

    # LEAKAGE-SAFE: StandardScaler and SelectKBest are Pipeline steps.
    # They are fitted independently inside each CV fold and never see held-out data.
    """
    datasets: dict[str, np.ndarray] = {"all19": X_main}
    if X_global3 is not None:
        datasets["global3"] = X_global3

    # Preflight: largest KNN n_neighbors must fit in every training fold.
    # With n_splits=5 on N=50, min training fold = 40 samples.
    max_knn_k = 7  # hard-coded to match KNN variant config above
    min_train_size = len(y) - len(y) // 5   # conservative floor for n_splits=5
    assert max_knn_k <= min_train_size, (
        f"KNN k={max_knn_k} exceeds min training fold size {min_train_size}"
    )

    # Use plain string scorers.
    # "roc_auc" (binary): uses decision_function for SVC(probability=False),
    # predict_proba for KNN / RF / Dummy — no make_scorer needed (sklearn >= 1.6 compatible).
    scoring: dict[str, str] = {
        "f1_macro":          "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc":           "roc_auc",
    }

    rows: list[dict] = []
    for name, feat_set, features, pipeline in variants:
        X = datasets[feat_set]
        print(f"  [{name}]  feat={feat_set}  n_feat={len(features)}", end=" ", flush=True)
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        f1_mean  = float(scores["test_f1_macro"].mean())
        f1_std   = float(scores["test_f1_macro"].std())
        auc_mean = float(scores["test_roc_auc"].mean())
        auc_std  = float(scores["test_roc_auc"].std())
        ba_mean  = float(scores["test_balanced_accuracy"].mean())
        ba_std   = float(scores["test_balanced_accuracy"].std())
        print(
            f"-> f1_macro={f1_mean:.4f}+/-{f1_std:.4f}  "
            f"roc_auc={auc_mean:.4f}+/-{auc_std:.4f}"
        )
        rows.append({
            "model":          name,
            "feature_set":    feat_set,
            "n_features":     len(features),
            "f1_macro_mean":  f1_mean,
            "f1_macro_std":   f1_std,
            "bal_acc_mean":   ba_mean,
            "bal_acc_std":    ba_std,
            "roc_auc_mean":   auc_mean,
            "roc_auc_std":    auc_std,
        })

    return pd.DataFrame(rows)


# ─── Best-model selection ──────────────────────────────────────────────────────

def select_best_model(results_df: pd.DataFrame) -> pd.Series:
    """
    Return the best non-dummy model row across ALL feature sets (including global3).

    Excludes:
      - "dummy" (trivial baseline, not a real model)

    Deterministic tie-break: sort by f1_macro_mean desc, then roc_auc_mean desc.
    """
    eligible = results_df[results_df["model"] != "dummy"].copy()
    if eligible.empty:
        raise RuntimeError("No eligible models after excluding dummy.")
    eligible = eligible.sort_values(
        ["f1_macro_mean", "roc_auc_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return eligible.iloc[0]


def select_best_global3(results_df: pd.DataFrame) -> pd.Series:
    """
    Return the best *_global3 model row.
    Deterministic tie-break: sort by f1_macro_mean desc, then roc_auc_mean desc.
    """
    g3 = results_df[results_df["model"].str.endswith("_global3")].copy()
    if g3.empty:
        raise RuntimeError("No *_global3 models found in results.")
    g3 = g3.sort_values(
        ["f1_macro_mean", "roc_auc_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return g3.iloc[0]


# ─── Refit pipeline ────────────────────────────────────────────────────────────

def build_refit_pipeline(cv_pipeline: Pipeline, seed: int) -> Pipeline:
    """
    Clone cv_pipeline unchanged (no parameter overrides).
    The saved pipeline is config-identical to the CV-evaluated one.
    For SVC models, Phase 9 inference must use decision_function (not predict_proba).
    See scoring_method in the meta JSON.
    """
    return sklearn.clone(cv_pipeline)


def _scoring_method(pipeline: Pipeline) -> str:
    """Return 'decision_function' for SVC, 'predict_proba' for all others."""
    # The classifier is always the last step in these pipelines.
    clf = pipeline.steps[-1][1]
    return "decision_function" if isinstance(clf, SVC) else "predict_proba"


# ─── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    best_row: pd.Series,
    variants_map: dict[str, tuple[str, list[str], Pipeline]],
    X_all19: np.ndarray,
    X_global3: np.ndarray,
    y: np.ndarray,
    seed: int,
    plots_dir: Path,
    filename: str = "confusion_matrix.png",
) -> None:
    """
    One StratifiedKFold(5, shuffle=True, random_state=seed) pass with cross_val_predict
    so every subject is predicted exactly once while held out.

    Uses the given model's own pipeline AND its own feature set (all19 or global3).
    # LEAKAGE-SAFE: cross_val_predict fits the full pipeline inside each fold;
    # no preprocessing step sees held-out data.
    """
    name     = best_row["model"]
    feat_set = best_row["feature_set"]
    _, _, pipeline = variants_map[name]
    X = {"all19": X_all19, "global3": X_global3}[feat_set]

    cv_cm = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv_cm, method="predict")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["mild (0)", "moderate (1)"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix -- {name} | feat_set={feat_set}\n"
        "StratifiedKFold(5, shuffle) — exploratory: model selected on same data",
        fontsize=8,
    )
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}  (model={name}, feat_set={feat_set})")


# ─── Feature importance ────────────────────────────────────────────────────────

def _best_rf_name(results_df: pd.DataFrame) -> str:
    """Return the name of the best RF variant (f1_macro_mean desc, roc_auc_mean desc)."""
    rf_rows = results_df[results_df["model"].str.startswith("rf_")].copy()
    rf_rows = rf_rows.sort_values(
        ["f1_macro_mean", "roc_auc_mean"], ascending=[False, False]
    )
    return rf_rows.iloc[0]["model"]


def _gini_in_input_space(pipeline: Pipeline, feature_names: list[str]) -> np.ndarray:
    """
    Return Gini feature importances aligned to the pipeline's input feature space.

    If the pipeline contains a SelectKBest step (kbest8 variants):
      - Use get_support() to obtain the boolean selection mask over feature_names.
      - Place Gini values at selected positions; unselected positions receive 0.
    Otherwise (all19 / global3 variants):
      - Gini is already aligned with feature_names.
    """
    rf_step = pipeline.named_steps["clf"]       # RF is always the "clf" step
    gini = rf_step.feature_importances_         # shape: (k,) or (n_input,)

    if "kbest" in pipeline.named_steps:
        support = pipeline.named_steps["kbest"].get_support()  # bool mask, length = n_input
        full = np.zeros(len(feature_names))
        full[support] = gini
        return full

    return gini  # already aligned with feature_names


def _permutation_importances_per_fold(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Compute permutation importance per outer CV fold and average across folds.

    For each fold:
      1. Clone and fit the pipeline on the train split (no leakage).
      2. Call permutation_importance on the test split (n_repeats=10,
         scoring="balanced_accuracy").
      3. Collect importances_mean (shape = n_input_features to the pipeline).
    Return the element-wise mean over all folds.

    Permuting pipeline inputs (before SelectKBest) naturally measures importance
    in the original feature space, even when internal selection is active.

    # LEAKAGE-SAFE: pipeline cloned per fold; fitted only on train split before permuting test.
    """
    cv_perm = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_means: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_perm.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_pipe = sklearn.clone(pipeline)  # fresh, unfitted copy for this fold
        fold_pipe.fit(X_train, y_train)

        result = permutation_importance(
            fold_pipe, X_test, y_test,
            n_repeats=10,
            scoring="balanced_accuracy",
            random_state=seed + fold_idx,   # distinct seed per fold for variety
            n_jobs=-1,
        )
        fold_means.append(result.importances_mean)  # shape: (n_input_features,)

    return np.mean(fold_means, axis=0)


def plot_feature_importance(
    rf_name: str,
    rf_pipeline_refit: Pipeline,
    rf_pipeline_cv: Pipeline,
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    plots_dir: Path,
) -> None:
    """
    Side-by-side horizontal bar charts for the best RF variant:
      Left  -- Gini importance (fit on all 50; descriptive, no train/test split).
      Right -- Permutation importance (mean over 5 CV test folds, n_repeats=10,
               scoring=balanced_accuracy).
    Both aligned to the pipeline's input feature space. Sorted descending by Gini.
    """
    gini = _gini_in_input_space(rf_pipeline_refit, feature_names)

    print(f"    Computing per-fold permutation importances for {rf_name} ...", flush=True)
    perm = _permutation_importances_per_fold(rf_pipeline_cv, X, y, seed)

    # Sort descending by Gini for consistent ordering
    order   = np.argsort(gini)[::-1]
    labels  = [feature_names[i] for i in order]
    gini_s  = gini[order]
    perm_s  = perm[order]

    # Compact label display: strip "all_AI_" -> "AI_"; "all_" -> ""
    short = [lbl.replace("all_AI_", "AI_").replace("all_", "") for lbl in labels]

    fig_height = max(6.0, len(labels) * 0.38)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))
    ypos = np.arange(len(labels))

    # ── Left: Gini
    axes[0].barh(ypos, gini_s[::-1], color="steelblue", alpha=0.85)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(short[::-1], fontsize=8)
    axes[0].set_xlabel("Mean Gini Impurity Decrease")
    axes[0].set_title(
        f"Gini Importance -- {rf_name}\n"
        "descriptive, fit on all 50, not held-out",
        fontsize=9,
    )
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")

    # ── Right: Permutation
    axes[1].barh(ypos, perm_s[::-1], color="darkorange", alpha=0.85)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(short[::-1], fontsize=8)
    axes[1].set_xlabel("Mean Permutation delta balanced_accuracy")
    axes[1].set_title(
        f"Permutation Importance -- {rf_name}\n"
        "Mean over 5 CV test folds (n_repeats=10 per fold)",
        fontsize=9,
    )
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")

    fig.suptitle(f"Feature Importance -- {rf_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved feature_importance.png  (RF variant={rf_name})")


# ─── Logistic regression coefficient plot ──────────────────────────────────────

def _best_logreg_name(results_df: pd.DataFrame) -> str:
    """Return the name of the best logreg_* variant (f1_macro_mean desc, roc_auc_mean desc)."""
    lr_rows = results_df[results_df["model"].str.startswith("logreg_")].copy()
    lr_rows = lr_rows.sort_values(
        ["f1_macro_mean", "roc_auc_mean"], ascending=[False, False]
    )
    return lr_rows.iloc[0]["model"]


def plot_logreg_coefficients(
    lr_name: str,
    lr_pipeline: Pipeline,
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    plots_dir: Path,
) -> None:
    """
    Standardized logistic regression coefficient bar chart for the best logreg variant.

    The pipeline is refitted on all 50 samples (descriptive; stated in plot title).
    If the pipeline has a SelectKBest step (kbest8 variant), coefficients are mapped
    back to the original input feature space: selected features get their coefficient,
    unselected features get 0.
    """
    # Fit on all 50 — descriptive only, not a held-out estimate.
    lr_pipeline.fit(X, y)
    lr_step = lr_pipeline.named_steps["clf"]
    # coef_ shape: (1, k) for binary; squeeze to (k,)
    coef = lr_step.coef_.ravel()

    if "kbest" in lr_pipeline.named_steps:
        support = lr_pipeline.named_steps["kbest"].get_support()  # bool mask, length=n_input
        full_coef = np.zeros(len(feature_names))
        full_coef[support] = coef
        coef = full_coef
    # coef is now aligned with feature_names in all cases

    order   = np.argsort(np.abs(coef))[::-1]   # sort by absolute magnitude
    labels  = [feature_names[i] for i in order]
    coef_s  = coef[order]
    short   = [lbl.replace("all_AI_", "AI_").replace("all_", "") for lbl in labels]

    colors = ["steelblue" if v >= 0 else "tomato" for v in coef_s[::-1]]
    fig_height = max(6.0, len(labels) * 0.38)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ypos = np.arange(len(labels))

    ax.barh(ypos, coef_s[::-1], color=colors, alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels(short[::-1], fontsize=8)
    ax.set_xlabel("Standardized Coefficient (blue=positive, red=negative)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Logistic Regression Coefficients -- {lr_name}\n"
        "descriptive, fit on all 50, not held-out\n"
        "Sorted by absolute magnitude; unselected features = 0 for kbest8",
        fontsize=9,
    )
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "logreg_coefficients.png", dpi=150)
    plt.close(fig)
    print(f"  Saved logreg_coefficients.png  (logreg variant={lr_name})")


# ─── Artifact writers ──────────────────────────────────────────────────────────

def _build_feature_contract(feature_list: list[str]) -> dict:
    """
    Build the feature contract for Phase 9 validation.
    NOTE: These values mirror constants in src/feature_extraction.py and
    src/preprocessing.py and must be updated together if those files change.
    """
    return {
        "schema_version": "1.0",
        "feature_names": feature_list,
        "n_features": len(feature_list),
        "ai_formula": "(P_left - P_right) / (P_left + P_right)",
        "channel_pairs": [["F3", "F4"], ["C3", "C4"], ["P3", "P4"], ["O1", "O2"]],
        "channel_pair_orientation": "left_minus_right",
        "bands_hz": {
            "delta": [1.0, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta":  [13.0, 30.0],
        },
        "psd_method": "Welch",
        "psd_n_fft": 256,
        "filter_bandpass_hz": [1.0, 40.0],
        "filter_notch_hz": 50.0,
        "epoch_length_s": 4.0,
        "tmax_adjusted": "epoch_length_s - 1/sfreq  (= 3.998 s at 500 Hz, 2000 samples)",
        "epoch_baseline": None,
        "aggregation": "mean over clean epochs per condition (all / left_hand / right_hand)",
        "model_uses_condition": "all",
        "label_map": {"mild": 0, "moderate": 1},
        "nihss_buckets": {
            "mild":     "NIHSS 0-4",
            "moderate": "NIHSS 5-15",
            "severe":   "NIHSS >15 (0 subjects in Figshare)",
        },
        "source_csv": "features/figshare_features.csv",
        "unverified_fields": [
            "sfreq_hz: assumed 500 from PHASE5_CONTEXT.md, not asserted in preprocessing.py",
        ],
    }


def save_results(results_df: pd.DataFrame, metrics_dir: Path) -> None:
    """Save model_comparison.csv sorted by f1_macro_mean desc, roc_auc_mean desc."""
    sorted_df = results_df.sort_values(
        ["f1_macro_mean", "roc_auc_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "model_comparison.csv"
    sorted_df.to_csv(out, index=False, float_format="%.6f")

    display_cols = ["model", "feature_set", "n_features",
                    "f1_macro_mean", "bal_acc_mean", "roc_auc_mean"]
    print("\nModel comparison (sorted by f1_macro_mean desc):")
    print(sorted_df[display_cols].to_string(index=False))
    print(f"\n  Saved {out}")


def save_best_model(
    best_row: pd.Series,
    refit_pipeline: Pipeline,
    feature_list: list[str],
    n_repeats: int,
    seed: int,
    models_dir: Path,
    stem: str = "best_model",
    canonical_run: bool = True,
) -> None:
    """
    Persist a model and its metadata.
    The saved pipeline uses probability=False for SVC steps (supports decision_function in Phase 9).
    stem controls output filenames: <stem>.joblib and <stem>_meta.json.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(refit_pipeline, models_dir / f"{stem}.joblib")
    print(f"  Saved {stem}.joblib  (model={best_row['model']})")

    contract = _build_feature_contract(feature_list)
    contract["pipeline_steps"] = [name for name, _ in refit_pipeline.steps]

    meta = {
        "model_name":       best_row["model"],
        "feature_set":      best_row["feature_set"],
        "scoring_method":   _scoring_method(refit_pipeline),
        "canonical_run":    canonical_run,
        "feature_names":    feature_list,
        "n_features":       len(feature_list),
        "cv_f1_macro_mean": round(float(best_row["f1_macro_mean"]), 6),
        "cv_f1_macro_std":  round(float(best_row["f1_macro_std"]),  6),
        "cv_bal_acc_mean":  round(float(best_row["bal_acc_mean"]),  6),
        "cv_bal_acc_std":   round(float(best_row["bal_acc_std"]),   6),
        "cv_roc_auc_mean":  round(float(best_row["roc_auc_mean"]), 6),
        "cv_roc_auc_std":   round(float(best_row["roc_auc_std"]),  6),
        "n_repeats":        n_repeats,
        "seed":             seed,
        "sklearn_version":  sklearn.__version__,
        "fit_date":         datetime.date.today().isoformat(),
        "selection_note": (
            f"best of 19 configs by CV (RepeatedStratifiedKFold 5x{n_repeats}), "
            "mildly optimistic at N=50"
        ),
        "feature_contract": contract,
    }
    meta_path = models_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved {stem}_meta.json")


def run_permutation_test(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: RepeatedStratifiedKFold,
    n_permutations: int,
    metrics_dir: Path,
) -> dict:
    """Run permutation testing for the primary model on two metrics."""
    results = {}
    for scoring in ["balanced_accuracy", "roc_auc"]:
        score, perm_scores, pvalue = permutation_test_score(
            pipeline, X, y, scoring=scoring, cv=cv, n_permutations=n_permutations, n_jobs=-1, random_state=42
        )
        results[scoring] = {
            "score": float(score),
            "null_mean": float(perm_scores.mean()),
            "null_std": float(perm_scores.std()),
            "pvalue": float(pvalue),
        }
        print(f"    {scoring}: obs={score:.4f}, p={pvalue:.4f}")

    df_perm = pd.DataFrame(results).T
    df_perm.index.name = "metric"
    df_perm.reset_index(inplace=True)
    out = metrics_dir / "permutation_test.csv"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df_perm.to_csv(out, index=False)
    print(f"  Saved {out}")
    return results


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)  # seed global numpy state; estimators use their own random_state

    CANONICAL_SEED, CANONICAL_REPEATS, CANONICAL_PERMUTATIONS = 42, 10, 500
    canonical_run = (
        args.seed == CANONICAL_SEED and
        args.n_repeats == CANONICAL_REPEATS and
        args.n_permutations == CANONICAL_PERMUTATIONS
    )
    if not canonical_run:
        print(
            "\n" + "!" * 64 +
            f"\n  WARNING: Non-canonical run detected."
            f"\n  seed={args.seed} (canonical={CANONICAL_SEED}), "
            f"n_repeats={args.n_repeats} (canonical={CANONICAL_REPEATS}), "
            f"n_permutations={args.n_permutations} (canonical={CANONICAL_PERMUTATIONS})."
            "\n  CV scores are NOT comparable to the canonical experiment."
            "\n" + "!" * 64 + "\n"
        )

    project_root = Path(__file__).resolve().parents[1]
    metrics_dir = args.output_dir / args.feature_set / "metrics"
    plots_dir   = args.output_dir / args.feature_set / "plots"
    models_dir  = args.output_dir / args.feature_set / "models"

    active_features, default_csv = FEATURE_SET_CONFIG[args.feature_set]
    csv_path = args.features_csv if args.features_csv else project_root / default_csv

    # ── Step 1: Load & validate ────────────────────────────────────────────────
    print("=" * 64)
    print(f"Phase 5 -- EEG Stroke Severity Classification (Figshare) | set={args.feature_set}")
    print("=" * 64)
    print("\n[1] Loading and validating data ...")
    X_main, X_global3, y, _ = load_and_validate(
        csv_path, active_features, args.seed, include_global3=(args.feature_set == "spectral")
    )
    datasets: dict[str, np.ndarray] = {"all19": X_main}
    if X_global3 is not None:
        datasets["global3"] = X_global3

    # ── Step 2: Build pipelines ────────────────────────────────────────────────
    print("\n[2] Building model variants ...")
    variants = build_cv_variants(args.seed, args.feature_set, active_features)
    # Index by name for O(1) lookup in plotting functions
    variants_map: dict[str, tuple[str, list[str], Pipeline]] = {
        name: (feat_set, features, pipeline)
        for name, feat_set, features, pipeline in variants
    }
    print(f"    {len(variants)} variants built.")

    # ── Step 3: Cross-validation ───────────────────────────────────────────────
    # LEAKAGE-SAFE: StandardScaler and SelectKBest are Pipeline steps;
    # fitted inside each CV fold -- held-out data is never seen by preprocessing.
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=args.n_repeats, random_state=args.seed
    )
    print(
        f"\n[3] Running CV  "
        f"(RepeatedStratifiedKFold 5x{args.n_repeats}, seed={args.seed}) ..."
    )
    results_df = run_cv(variants, X_main, X_global3, y, cv)

    # ── Step 4: Permutation testing ─────────────────────────────────────────────
    print(f"\n[4] Permutation testing {PRIMARY_MODEL} ({args.n_permutations} perms) ...")
    prim_feat_set, _, prim_pipeline = variants_map[PRIMARY_MODEL]
    prim_X = datasets[prim_feat_set]
    perm_results = run_permutation_test(
        prim_pipeline, prim_X, y, cv, args.n_permutations, metrics_dir
    )

    # ── Step 4b: Save results CSV ──────────────────────────────────────────────
    print("\n[4b] Saving model comparison ...")
    save_results(results_df, metrics_dir)

    # ── Step 5: Select best model (all non-dummy variants eligible, including global3) ──
    # Deterministic sort: f1_macro_mean desc, then roc_auc_mean desc.
    best_row = select_best_model(results_df)
    best_name = best_row["model"]
    best_feat_set, best_features, best_pipeline_cv = variants_map[best_name]
    best_X = datasets[best_feat_set]
    print(
        f"\n[5] Best model: {best_name}  "
        f"(feat_set={best_feat_set}, "
        f"f1_macro={best_row['f1_macro_mean']:.4f}+/-{best_row['f1_macro_std']:.4f}, "
        f"roc_auc={best_row['roc_auc_mean']:.4f}+/-{best_row['roc_auc_std']:.4f})"
    )

    # ── Step 5b: Select best *_global3 variant ────────────────────────────────
    g3_row = None
    if args.feature_set == "spectral":
        g3_row = select_best_global3(results_df)
        g3_name = g3_row["model"]
        g3_feat_set, g3_features, g3_pipeline_cv = variants_map[g3_name]
        g3_X = datasets[g3_feat_set]
        print(
            f"      Best global3: {g3_name}  "
            f"(f1_macro={g3_row['f1_macro_mean']:.4f}+/-{g3_row['f1_macro_std']:.4f}, "
            f"roc_auc={g3_row['roc_auc_mean']:.4f}+/-{g3_row['roc_auc_std']:.4f})"
        )

    # ── Step 6: Confusion matrix (overall best model) ─────────────────────────
    print("\n[6] Generating confusion matrices ...")
    plot_confusion_matrix(
        best_row, variants_map, X_main, X_global3, y, args.seed, plots_dir,
        filename="confusion_matrix.png",
    )

    # ── Step 6b: Confusion matrix for best global3 variant ────────────────────
    if g3_row is not None:
        plot_confusion_matrix(
            g3_row, variants_map, X_main, X_global3, y, args.seed, plots_dir,
            filename="confusion_matrix_global3.png",
        )

    # ── Step 7: Feature importance (best RF variant) ───────────────────────────
    rf_name = _best_rf_name(results_df)
    rf_feat_set, rf_features, rf_pipeline_cv = variants_map[rf_name]
    rf_X = datasets[rf_feat_set]

    print(f"\n[7] Feature importance for {rf_name} ...")
    # Refit on all 50 for Gini (descriptive; stated as such in the plot title).
    rf_pipeline_refit = build_refit_pipeline(rf_pipeline_cv, args.seed)
    rf_pipeline_refit.fit(rf_X, y)

    plot_feature_importance(
        rf_name,
        rf_pipeline_refit,
        rf_pipeline_cv,
        rf_features,
        rf_X,
        y,
        args.seed,
        plots_dir,
    )

    # ── Step 7b: Logistic regression coefficient plot (best logreg variant) ────
    lr_name = _best_logreg_name(results_df)
    lr_feat_set, lr_features, lr_pipeline_cv = variants_map[lr_name]
    lr_X = datasets[lr_feat_set]

    print(f"\n[7b] Logistic regression coefficients for {lr_name} ...")
    # Clone so the CV pipeline object stays unfitted.
    lr_pipeline_plot = sklearn.clone(lr_pipeline_cv)
    plot_logreg_coefficients(
        lr_name, lr_pipeline_plot, lr_features, lr_X, y, plots_dir
    )

    # ── Step 8: Final refit & model save ──────────────────────────────────────
    if args.feature_set == "spectral":
        print(f"\n[8] Refitting {best_name} on all 50 samples ...")
        best_pipeline_refit = build_refit_pipeline(best_pipeline_cv, args.seed)
        best_pipeline_refit.fit(best_X, y)
        save_best_model(
            best_row, best_pipeline_refit, best_features, args.n_repeats, args.seed,
            models_dir, stem="best_model", canonical_run=canonical_run,
        )

        if g3_row is not None:
            print(f"\n[8b] Refitting {g3_name} (global3) on all 50 samples ...")
            g3_pipeline_refit = build_refit_pipeline(g3_pipeline_cv, args.seed)
            g3_pipeline_refit.fit(g3_X, y)
            save_best_model(
                g3_row, g3_pipeline_refit, g3_features, args.n_repeats, args.seed,
                models_dir, stem="best_global3_model", canonical_run=canonical_run,
            )
    else:
        print(f"\n[8] Skipping best_model.joblib save for non-spectral feature set '{args.feature_set}'.")

    # ── Step 9: Ablation Summary ──────────────────────────────────────────────
    summary_path = args.output_dir / "ablation_summary.csv"
    summary_data = {
        "feature_set": args.feature_set,
        "best_model_name": best_name,
        "best_model_f1_macro": best_row["f1_macro_mean"],
        "best_model_roc_auc": best_row["roc_auc_mean"],
        "primary_model": PRIMARY_MODEL,
        "primary_bal_acc": perm_results["balanced_accuracy"]["score"],
        "primary_bal_acc_pval": perm_results["balanced_accuracy"]["pvalue"],
        "primary_roc_auc": perm_results["roc_auc"]["score"],
        "primary_roc_auc_pval": perm_results["roc_auc"]["pvalue"],
    }
    summary_df_new = pd.DataFrame([summary_data])
    
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        expected_cols = list(summary_df_new.columns)
        if list(existing_df.columns) != expected_cols:
            raise ValueError(
                f"Existing ablation summary has mismatched columns. "
                f"Expected {expected_cols}, got {list(existing_df.columns)}"
            )
        existing_df = existing_df[existing_df["feature_set"] != args.feature_set]
        summary_df = pd.concat([existing_df, summary_df_new], ignore_index=True)
    else:
        summary_df = summary_df_new

    summary_df.to_csv(summary_path, index=False)
    print(f"\n[9] Updated summary in {summary_path}")

    print("\n" + "=" * 64)
    print("Phase 5 complete.")
    print(f"  Metrics : {metrics_dir}")
    print(f"  Plots   : {plots_dir}")
    print(f"  Models  : {models_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()