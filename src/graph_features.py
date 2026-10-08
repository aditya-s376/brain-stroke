"""Extract wPLI-based hemispheric graph features from preprocessed Figshare epochs.

Pipeline
--------
*-epo.fif (full 30-ch epochs)
  -> pick 8 core channels (F3 C3 P3 O1 | F4 C4 P4 O2)
  -> spectral_connectivity_epochs (wPLI, multitaper) per band
  -> 4x4 left / 4x4 right adjacency matrices + cross-hemisphere homotopic block
  -> 4 network-level features x 4 bands = 16 subject-level features
  -> features/figshare_graph_features.csv
  -> merged with spectral 19 features -> features/figshare_combined_features.csv

Nodes (order is fixed; DO NOT reorder -- determines matrix slicing)
------
Left  : F3, C3, P3, O1   (indices 0-3 in 8-node graph)
Right : F4, C4, P4, O2   (indices 4-7 in 8-node graph)

Homologous pairs (left index -> right index)
  F3(0) <-> F4(4)
  C3(1) <-> C4(5)
  P3(2) <-> P4(6)
  O1(3) <-> O2(7)

Features per band
  wPLI_{band}_GlobalEff_AI   : (GE_L - GE_R) / (GE_L + GE_R)
  wPLI_{band}_ClustCoeff_AI  : (CC_L - CC_R) / (CC_L + CC_R)
  wPLI_{band}_MeanHomotopic  : mean wPLI across the 4 homologous pairs
  wPLI_{band}_FrobeniusDist  : ||A_L - A_R||_F / (||A_L||_F + ||A_R||_F)

Sampling-rate note
  PHASE5_CONTEXT.md documents sfreq = 500 Hz.  The script reads sfreq at
  runtime from each Epochs object and will raise if it deviates from 500 Hz,
  acting as an embedded guard rather than a silent assumption.

Phase 9 / UCLH note
  The 8-channel node set is identical to the spectral pipeline's
  FEATURE_CHANNELS, which are the intersection channels most likely present in
  both Figshare and any UCLH montage.  Extending to peripheral channels would
  require a new artifact-quality audit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCHS_ROOT = REPO_ROOT / "data" / "processed" / "figshare"
FEATURES_ROOT = REPO_ROOT / "features"
SPECTRAL_CSV = FEATURES_ROOT / "figshare_features.csv"
GRAPH_CSV = FEATURES_ROOT / "figshare_graph_features.csv"
COMBINED_CSV = FEATURES_ROOT / "figshare_combined_features.csv"


# ---------------------------------------------------------------------------
# Channel / node configuration  (order is load-bearing -- do not change)
# ---------------------------------------------------------------------------
LEFT_NODES: list[str] = ["F3", "C3", "P3", "O1"]
RIGHT_NODES: list[str] = ["F4", "C4", "P4", "O2"]
ALL_NODES: list[str] = LEFT_NODES + RIGHT_NODES   # L = indices 0-3, R = 4-7
N_LEFT: int = len(LEFT_NODES)                      # 4
N_RIGHT: int = len(RIGHT_NODES)                    # 4
N_NODES: int = len(ALL_NODES)                      # 8

# Homologous pair indices in the 8-node ordering (left_idx, right_idx)
HOMOTOPIC_PAIRS: list[tuple[int, int]] = [
    (0, 4),  # F3 <-> F4
    (1, 5),  # C3 <-> C4
    (2, 6),  # P3 <-> P4
    (3, 7),  # O1 <-> O2
]


# ---------------------------------------------------------------------------
# Frequency bands  (must match feature_extraction.py BANDS)
# ---------------------------------------------------------------------------
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# Guard: expected sampling frequency from PHASE5_CONTEXT.md
EXPECTED_SFREQ: float = 500.0


# ---------------------------------------------------------------------------
# Canonical feature names (16 total; fixed order)
# ---------------------------------------------------------------------------
GRAPH_FEATURE_NAMES: list[str] = [
    f"wPLI_{band}_{metric}"
    for band in BANDS
    for metric in ("GlobalEff_AI", "ClustCoeff_AI", "MeanHomotopic", "FrobeniusDist")
]


# ---------------------------------------------------------------------------
# Graph metric helpers
# ---------------------------------------------------------------------------

def _weighted_global_efficiency(adj: np.ndarray) -> float:
    """Weighted global efficiency of a 4-node undirected graph.

    Edge weight = connectivity strength (higher = shorter path).
    Distance = 1 / weight for Dijkstra.  Unreachable pairs contribute 0.

    Returns a value in [0, 1].
    """
    n = adj.shape[0]
    if n < 2:
        return 0.0
    G = nx.from_numpy_array(adj, create_using=nx.Graph())
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        G[u][v]["distance"] = 1.0 / w if w > 0.0 else np.inf

    total_inv = 0.0
    for src in G.nodes():
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="distance")
        for tgt, dist in lengths.items():
            if tgt != src and np.isfinite(dist) and dist > 0.0:
                total_inv += 1.0 / dist
    return total_inv / (n * (n - 1))


def _weighted_mean_clustering(adj: np.ndarray) -> float:
    """Mean weighted clustering coefficient (Onnela et al. 2005).

    Uses networkx geometric-mean formulation.  Returns a value in [0, 1].
    Requires n >= 3; returns 0.0 otherwise.
    """
    n = adj.shape[0]
    if n < 3:
        return 0.0
    G = nx.from_numpy_array(adj, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))
    clustering = nx.clustering(G, weight="weight")
    values = list(clustering.values())
    return float(np.mean(values)) if values else 0.0


def _safe_ai(left_val: float, right_val: float) -> float:
    """Return (L - R) / (L + R); returns 0.0 on zero or non-finite denom."""
    denom = left_val + right_val
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    ai = (left_val - right_val) / denom
    return float(ai) if np.isfinite(ai) else 0.0


def _frobenius_dist(left_adj: np.ndarray, right_adj: np.ndarray) -> float:
    """Normalized Frobenius distance: ||A_L - A_R||_F / (||A_L||_F + ||A_R||_F).

    Returns 0.0 when both matrices are all-zero.
    """
    diff_norm = float(np.linalg.norm(left_adj - right_adj, "fro"))
    denom = float(np.linalg.norm(left_adj, "fro")) + float(np.linalg.norm(right_adj, "fro"))
    if denom == 0.0 or not np.isfinite(denom):
        return 0.0
    result = diff_norm / denom
    return float(result) if np.isfinite(result) else 0.0


# ---------------------------------------------------------------------------
# Core: wPLI adjacency matrix for one subject + one band
# ---------------------------------------------------------------------------

def _wpli_adjacency(
    epochs: mne.Epochs,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Compute an 8x8 wPLI adjacency matrix averaged over epochs.

    Uses mne_connectivity.spectral_connectivity_epochs with
    method='wpli' and mode='multitaper'.

    Parameters
    ----------
    epochs : mne.Epochs
        Already-picked 8-channel epochs in ALL_NODES channel order.
    fmin, fmax : float
        Band edges in Hz.

    Returns
    -------
    np.ndarray of shape (8, 8)
        Symmetric wPLI matrix, diagonal = 0, values clipped to [0, 1].
    """
    from mne_connectivity import spectral_connectivity_epochs  # type: ignore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        con = spectral_connectivity_epochs(
            epochs,
            method="wpli",
            mode="multitaper",
            sfreq=epochs.info["sfreq"],
            fmin=fmin,
            fmax=fmax,
            faverage=True,    # collapse frequency axis -> one scalar per pair
            verbose=False,
        )

    # con.get_data(output='dense') -> (n_ch, n_ch) or (n_ch, n_ch, 1)
    data = con.get_data(output="dense")
    if data.ndim == 3:
        data = data[:, :, 0]

    adj = np.array(data, dtype=float)
    adj = np.maximum(adj, adj.T)      # enforce symmetry
    np.fill_diagonal(adj, 0.0)
    adj = np.clip(adj, 0.0, 1.0)     # wPLI in [0, 1]

    if adj.shape != (N_NODES, N_NODES):
        raise ValueError(
            f"Expected {N_NODES}x{N_NODES} adjacency, got {adj.shape}"
        )
    return adj


# ---------------------------------------------------------------------------
# Per-subject extraction
# ---------------------------------------------------------------------------

def extract_subject_graph_features(epochs_path: Path) -> dict[str, float]:
    """Load epochs for one subject and return the 16 graph features.

    Parameters
    ----------
    epochs_path : Path
        Path to a ``*-epo.fif`` file produced by preprocessing.py.

    Returns
    -------
    dict mapping GRAPH_FEATURE_NAMES -> float scalar.
    """
    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)

    # Sampling-rate guard
    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - EXPECTED_SFREQ) > 1.0:
        raise ValueError(
            f"{epochs_path.name}: sfreq={sfreq} Hz (expected {EXPECTED_SFREQ} Hz). "
            "Update EXPECTED_SFREQ if the dataset has a different sampling rate."
        )

    # Pick 8 nodes in canonical order
    missing = [ch for ch in ALL_NODES if ch not in epochs.ch_names]
    if missing:
        raise ValueError(f"{epochs_path.name}: missing channels {missing}")
    epochs = epochs.copy().pick(ALL_NODES)
    epochs = epochs.reorder_channels(ALL_NODES)

    features: dict[str, float] = {}

    for band_name, (fmin, fmax) in BANDS.items():
        adj = _wpli_adjacency(epochs, fmin=fmin, fmax=fmax)

        # Hemispheric sub-matrices
        left_adj = adj[:N_LEFT, :N_LEFT]   # (4, 4)
        right_adj = adj[N_LEFT:, N_LEFT:]  # (4, 4)

        # Feature 1: Global Efficiency Asymmetry
        ge_l = _weighted_global_efficiency(left_adj)
        ge_r = _weighted_global_efficiency(right_adj)
        features[f"wPLI_{band_name}_GlobalEff_AI"] = _safe_ai(ge_l, ge_r)

        # Feature 2: Mean Clustering Coefficient Asymmetry
        cc_l = _weighted_mean_clustering(left_adj)
        cc_r = _weighted_mean_clustering(right_adj)
        features[f"wPLI_{band_name}_ClustCoeff_AI"] = _safe_ai(cc_l, cc_r)

        # Feature 3: Mean Homotopic wPLI (cross-hemisphere block)
        homotopic_vals = [
            adj[left_idx, right_idx]
            for left_idx, right_idx in HOMOTOPIC_PAIRS
        ]
        features[f"wPLI_{band_name}_MeanHomotopic"] = float(np.mean(homotopic_vals))

        # Feature 4: Normalized Frobenius Distance
        features[f"wPLI_{band_name}_FrobeniusDist"] = _frobenius_dist(left_adj, right_adj)

    return features


# ---------------------------------------------------------------------------
# Subject-ID extraction
# ---------------------------------------------------------------------------

def _subject_id_from_path(path: Path) -> str:
    """Extract subject ID string from a filename like 'sub-01-epo.fif'."""
    # preprocessing.py: subject_id = Path(subject_file).name.split("_")[0]
    # output_path: OUTPUT_ROOT / f"{subject_id}-epo.fif"
    stem = path.stem                         # e.g. "sub-01-epo"
    subject_id = stem.replace("-epo", "")   # e.g. "sub-01"
    return subject_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    epoch_files = sorted(EPOCHS_ROOT.glob("*-epo.fif"))
    if not epoch_files:
        raise FileNotFoundError(
            f"No *-epo.fif files found under {EPOCHS_ROOT}. "
            "Run preprocessing.py first."
        )
    if len(epoch_files) != 50:
        raise RuntimeError(
            f"Expected 50 subject epoch files, found {len(epoch_files)}."
        )

    print(f"Found {len(epoch_files)} epoch files.")
    print(f"Node set : LEFT={LEFT_NODES}  RIGHT={RIGHT_NODES}")
    print(f"Bands    : {list(BANDS.keys())}")
    print(f"Features : {GRAPH_FEATURE_NAMES}\n")

    records: list[dict] = []
    failed: list[str] = []

    for i, epo_path in enumerate(epoch_files, start=1):
        subject_id = _subject_id_from_path(epo_path)
        print(f"[{i:02d}/50] {subject_id} ...", end=" ", flush=True)
        try:
            feats = extract_subject_graph_features(epo_path)
            feats["subject_id"] = subject_id
            records.append(feats)
            print("OK")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            failed.append(f"{subject_id}: {exc}")

    if failed:
        print(f"\n{len(failed)} subject(s) failed:")
        for msg in failed:
            print(f"  {msg}")

    if not records:
        raise RuntimeError("No subjects processed successfully. Cannot write CSVs.")

    # ------------------------------------------------------------------
    # Write graph-only CSV
    # ------------------------------------------------------------------
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)

    graph_df = pd.DataFrame(records)
    graph_df = graph_df[["subject_id"] + GRAPH_FEATURE_NAMES]
    graph_df.to_csv(GRAPH_CSV, index=False)
    print(f"\nGraph features -> {GRAPH_CSV}")
    print(f"  Shape: {graph_df.shape}  (expected 50 x 17)")

    # ------------------------------------------------------------------
    # Merge with spectral features -> combined CSV
    # ------------------------------------------------------------------
    if not SPECTRAL_CSV.exists():
        print(
            f"\nWARNING: {SPECTRAL_CSV} not found. "
            "Skipping combined CSV generation."
        )
        return

    spectral_df = pd.read_csv(SPECTRAL_CSV)

    if "subject_id" not in spectral_df.columns:
        raise KeyError(
            f"Column 'subject_id' missing from {SPECTRAL_CSV}. "
            f"Available: {list(spectral_df.columns)}"
        )

    combined_df = spectral_df.merge(
        graph_df,
        on="subject_id",
        how="inner",
        validate="1:1",
    )
    if len(combined_df) != len(spectral_df):
        raise RuntimeError(
            f"Merge produced {len(combined_df)} rows (expected {len(spectral_df)}). "
            "Check subject_id alignment between CSVs."
        )

    combined_df.to_csv(COMBINED_CSV, index=False)
    print(f"Combined features -> {COMBINED_CSV}")
    print(f"  Shape: {combined_df.shape}")
    print(
        f"\nSummary:\n"
        f"  Spectral cols (original) : {spectral_df.shape[1]}\n"
        f"  Graph features added     : {len(GRAPH_FEATURE_NAMES)}\n"
        f"  Combined total columns   : {combined_df.shape[1]}\n"
        f"  Subjects                 : {combined_df.shape[0]}\n"
    )


if __name__ == "__main__":
    main()
