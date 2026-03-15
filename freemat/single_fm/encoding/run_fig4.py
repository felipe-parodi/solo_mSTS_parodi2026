"""Figure 4: Behavioral motif analyses for single-FM sessions.

This module wires motif ethograms to neural bins and runs:
  - Motif tuning heatmaps (per-unit mean firing by motif)
  - Motif decoding (spikes -> motif labels)
  - Lag sweep and next-syllable prediction around transitions

Example:
    >>> from freemat.single_fm.encoding.run_fig4 import Fig4Config, run_fig4_session
    >>> cfg = Fig4Config(bin_s=0.5, top_n_motifs=10)
    >>> run_fig4_session(
    ...     session_yaml="configs/single_fm/session_hedy_0526.yaml",
    ...     output_dir="results/single_fm/fig4_pilot/single_0526_hedy",
    ...     config=cfg,
    ... )
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from analysis.syllables.alignment import align_syllables_to_bins, detect_bin_onsets
from analysis.syllables.decoding import (
    SyllableDecodingConfig,
    compute_permutation_null,
    decode_syllables_by_bank,
)
from analysis.syllables.lag_sweep import lag_sweep_by_bank
from analysis.syllables.segmentation import compute_transition_matrix, remap_to_top_n
from analysis.syllables.transitions import compute_next_syllable_accuracy, compute_onset_peth
from freemat.single_fm.data import SingleFMSessionLoader
from freemat_data import channels_from_unit_ids, load_banks_config, load_motifs_from_session


@dataclass
class Fig4Config:
    """Configuration for Figure 4 motif analyses.

    Attributes:
        bin_s: Bin size in seconds.
        top_n_motifs: Number of most frequent motifs to keep.
        exclude_motif_ids: Motif IDs to exclude (set to missing).
        missing_value: Sentinel for missing bins.
        other_id: ID for "other" (non-top-N) motifs.
        n_folds: CV folds for decoding and lag sweep.
        embargo_bins: Embargo bins around CV fold boundaries.
        n_permutations: Number of label shuffles for null distributions.
        lag_range_bins: Lags to test in bins (±lag_range_bins).
        classifier_type: Classifier for decoding ("logistic" or "svm").
        random_state: Random seed.
    """

    bin_s: float = 0.5
    top_n_motifs: int = 10
    exclude_motif_ids: Tuple[int, ...] = (0,)
    missing_value: int = -999
    other_id: int = -1
    n_folds: int = 5
    embargo_bins: int = 4
    n_permutations: int = 1000
    lag_range_bins: int = 4
    classifier_type: str = "logistic"
    random_state: int = 42


def _bin_edges(time_window: Tuple[float, float], bin_s: float) -> np.ndarray:
    """Return bin edges matching freemat_data.transforms.spikes.bin_events."""
    t_start, t_end = time_window
    return np.arange(float(t_start), float(t_end) + float(bin_s), float(bin_s), dtype=float)


def _build_channel_depth_map(banks_cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build channel->depth metadata mapping from banks.yaml shank_groups."""
    shank_groups = banks_cfg.get("shank_groups")
    if not isinstance(shank_groups, dict):
        raise ValueError("banks.yaml missing shank_groups for depth mapping.")

    channel_map: Dict[int, Dict[str, Any]] = {}
    pattern = re.compile(r"^([ab])(\d)_shank(\d+)$")
    for shank_name, channels in shank_groups.items():
        if not isinstance(channels, list):
            raise ValueError(f"Invalid shank list for {shank_name}")
        match = pattern.match(shank_name)
        if not match:
            raise ValueError(f"Unexpected shank name: {shank_name}")
        bank_letter, probe_num, shank_num = match.groups()
        bank = "ventral" if bank_letter == "a" else "dorsal"
        probe = f"{bank_letter.upper()}{probe_num}"
        shank = int(shank_num)
        for depth_position, channel in enumerate(channels):
            ch = int(channel)
            if ch in channel_map:
                raise ValueError(f"Channel {ch} appears in multiple shanks.")
            channel_map[ch] = {
                "bank": bank,
                "probe": probe,
                "shank": shank,
                "depth_position": int(depth_position),
                "shank_name": shank_name,
            }
    return channel_map


def build_unit_depth_table(
    unit_channels: np.ndarray,
    unit_banks: Optional[np.ndarray],
    banks_yaml: Optional[str] = None,
) -> pd.DataFrame:
    """Build depth metadata for each unit using configs/banks.yaml.

    Args:
        unit_channels: (U,) channel IDs (0-indexed Deuteron).
        unit_banks: Optional bank labels from spike loader.
        banks_yaml: Optional path to banks.yaml (default: configs/banks.yaml).

    Returns:
        DataFrame with per-unit depth metadata and sort rank.
    """
    banks_cfg = load_banks_config(banks_yaml or "configs/banks.yaml")
    channel_map = _build_channel_depth_map(banks_cfg)

    rows = []
    for unit_idx, ch in enumerate(unit_channels):
        if int(ch) not in channel_map:
            raise ValueError(f"Channel {ch} not found in banks.yaml shank_groups.")
        info = channel_map[int(ch)]
        bank = info["bank"]
        if unit_banks is not None and len(unit_banks) > unit_idx:
            bank_label = str(unit_banks[unit_idx])
            if bank_label not in ("ventral", "dorsal"):
                raise ValueError(f"Unknown bank label {bank_label} for unit {unit_idx}.")
            if bank_label != bank:
                raise ValueError(f"Bank mismatch for unit {unit_idx}: {bank_label} vs {bank}.")
        rows.append(
            {
                "unit_index": int(unit_idx),
                "channel": int(ch),
                "bank": bank,
                "probe": info["probe"],
                "shank": int(info["shank"]),
                "depth_position": int(info["depth_position"]),
                "shank_name": info["shank_name"],
            }
        )

    df = pd.DataFrame(rows)
    bank_order = {"ventral": 0, "dorsal": 1}
    probe_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
    df["sort_rank"] = (
        df["bank"].map(bank_order) * 1000
        + df["probe"].map(probe_order) * 100
        + df["shank"] * 10
        + df["depth_position"]
    )
    return df


def prepare_fig4_data(
    session_yaml: str,
    config: Fig4Config,
    *,
    debug: bool = False,
    canonical_alignment_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and align spikes with motif labels for one session."""
    loader = SingleFMSessionLoader(
        session_yaml,
        canonical_alignment_path=canonical_alignment_path,
        debug=debug,
    )
    _, _, bundle = loader._load_spikes()
    spikes_df = loader.get_spikes(bin_s=config.bin_s)

    if spikes_df.empty:
        raise ValueError(f"No spikes found for {session_yaml}")

    X_df = (
        spikes_df.pivot(index="bin_idx", columns="unit_index", values="fr")
        .sort_index()
    )
    X = X_df.to_numpy()

    bin_times = (
        spikes_df.drop_duplicates("bin_idx")
        .sort_values("bin_idx")["t_bin_s"]
        .to_numpy()
    )

    if X.shape[0] != bin_times.shape[0]:
        raise ValueError("Bin time count does not match binned spike matrix.")

    unit_banks = None
    if "unit_bank" in spikes_df.columns:
        unit_banks = (
            spikes_df.groupby("unit_index")["unit_bank"]
            .first()
            .reindex(X_df.columns)
            .to_numpy()
        )
    if unit_banks is None or len(unit_banks) != X.shape[1]:
        raise ValueError("unit_bank labels are required for bank-specific analyses.")

    unit_ids = bundle.spikes.unit_ids
    unit_count = X.shape[1]
    if unit_ids is None or len(unit_ids) != unit_count:
        unit_ids = np.array([int(i) for i in X_df.columns], dtype=int)

    unit_channels = None
    if bundle.spikes.meta and "unit_channels" in bundle.spikes.meta:
        unit_channels = np.asarray(bundle.spikes.meta["unit_channels"], dtype=int)
    if unit_channels is None or len(unit_channels) != unit_count:
        unit_channels = channels_from_unit_ids(unit_ids, expect_U=unit_count)
    if unit_channels is None or len(unit_channels) != unit_count:
        raise ValueError("Unable to resolve unit_channels from spikes metadata.")

    motifs = load_motifs_from_session(loader._session)
    if motifs is None or motifs.t is None:
        raise ValueError(f"Motifs not available for {loader.session_id}.")

    bin_edges = _bin_edges(loader.time_window, config.bin_s)
    if len(bin_edges) - 1 != X.shape[0]:
        raise ValueError("Bin edges length mismatch with spike bins.")

    bin_labels = align_syllables_to_bins(
        motifs.motif_id,
        motifs.t,
        bin_edges,
        method="center",
        missing_value=config.missing_value,
    )

    if config.exclude_motif_ids:
        excl = np.isin(bin_labels, np.array(config.exclude_motif_ids, dtype=int))
        bin_labels[excl] = config.missing_value

    missing_mask = bin_labels == config.missing_value
    valid_ids = bin_labels[~missing_mask]
    if valid_ids.size == 0:
        raise ValueError("No valid motif labels after exclusion.")

    remapped = bin_labels.copy()
    remapped_valid, mapping = remap_to_top_n(
        valid_ids,
        n=config.top_n_motifs,
        other_id=config.other_id,
    )
    remapped[~missing_mask] = remapped_valid

    valid_mask = (~missing_mask) & (remapped != config.other_id)
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid bins after top-N remapping.")

    return {
        "X": X,
        "y": remapped,
        "bin_times": bin_times,
        "bin_edges": bin_edges,
        "unit_index": np.asarray(X_df.columns, dtype=int),
        "unit_ids": np.asarray(unit_ids),
        "unit_channels": unit_channels,
        "unit_banks": unit_banks,
        "valid_mask": valid_mask,
        "motif_mapping": mapping,
        "session_id": loader.session_id,
    }


def compute_motif_tuning(
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-unit tuning across motifs.

    Returns:
        tuning_raw: (U, M) mean firing rate per motif.
        tuning_z: (U, M) z-scored tuning per unit.
        motif_ids: (M,) motif IDs.
        motif_counts: (M,) counts of bins per motif.
    """
    motif_ids = np.array(sorted(set(y[valid_mask])), dtype=int)
    n_units = X.shape[1]
    n_motifs = motif_ids.shape[0]
    tuning_raw = np.full((n_units, n_motifs), np.nan, dtype=float)
    motif_counts = np.zeros(n_motifs, dtype=int)

    for j, motif_id in enumerate(motif_ids):
        mask = valid_mask & (y == motif_id)
        motif_counts[j] = int(np.sum(mask))
        if motif_counts[j] == 0:
            continue
        tuning_raw[:, j] = X[mask].mean(axis=0)

    mu = np.nanmean(tuning_raw, axis=1, keepdims=True)
    sd = np.nanstd(tuning_raw, axis=1, keepdims=True)
    sd = np.where(sd <= 0, 1.0, sd)
    tuning_z = (tuning_raw - mu) / sd

    return tuning_raw, tuning_z, motif_ids, motif_counts


def compute_motif_cluster_order(tuning_z: np.ndarray, motif_ids: np.ndarray) -> List[int]:
    """Cluster motifs by correlation of tuning profiles."""
    if tuning_z.shape[1] <= 1:
        return [int(m) for m in motif_ids]
    distances = pdist(tuning_z.T, metric="correlation")
    Z = linkage(distances, method="average")
    order = leaves_list(Z)
    return [int(motif_ids[i]) for i in order]


def run_motif_decoding(
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    unit_banks: np.ndarray,
    config: Fig4Config,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run motif decoding and label-shuffle nulls."""
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    dec_cfg = SyllableDecodingConfig(
        n_folds=config.n_folds,
        embargo_bins=config.embargo_bins,
        n_permutations=config.n_permutations,
        classifier_type=config.classifier_type,
        top_n_syllables=config.top_n_motifs,
        other_id=config.other_id,
        random_state=config.random_state,
    )

    results = decode_syllables_by_bank(X_valid, y_valid, unit_banks, config=dec_cfg)

    import json as _json

    summary_rows = []
    confusion = {}
    class_labels = {}
    for bank, result in results.items():
        conf = result.confusion_matrix
        confusion[bank] = conf
        class_labels[bank] = np.array(result.class_labels, dtype=int)
        summary_rows.append(
            {
                "bank": bank,
                "balanced_accuracy": result.balanced_accuracy,
                "macro_f1": result.macro_f1,
                "n_samples": result.n_samples,
                "n_test_samples": result.n_test_samples,
                "n_classes": len(result.class_labels),
                "fold_accuracies": _json.dumps(
                    [float(a) for a in result.per_fold_accuracy]
                ),
            }
        )

    nulls = {}
    for bank in results.keys():
        if bank == "pooled":
            nulls[bank] = compute_permutation_null(X_valid, y_valid, config=dec_cfg)
        else:
            mask = unit_banks == bank
            nulls[bank] = compute_permutation_null(X_valid[:, mask], y_valid, config=dec_cfg)

    return pd.DataFrame(summary_rows), nulls, {"confusion": confusion, "class_labels": class_labels}


def run_lag_sweep(
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    unit_banks: np.ndarray,
    config: Fig4Config,
) -> Dict[str, Any]:
    """Run lag sweep decoding for each bank."""
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    lags_bins = np.arange(-config.lag_range_bins, config.lag_range_bins + 1, dtype=int)
    results = lag_sweep_by_bank(
        X_valid,
        y_valid,
        unit_banks,
        lags_bins=lags_bins,
        bin_size_sec=config.bin_s,
        n_folds=config.n_folds,
        embargo_bins=config.embargo_bins,
        random_state=config.random_state,
    )
    return results


def run_next_syllable_prediction(
    X: np.ndarray,
    y: np.ndarray,
    config: Fig4Config,
) -> Any:
    """Compute next-syllable decoding accuracy around transitions."""
    y_clean = y.copy()
    y_clean[y_clean == config.other_id] = config.missing_value
    onset_bins, post_syllables = detect_bin_onsets(
        y_clean, missing_value=config.missing_value
    )
    return compute_next_syllable_accuracy(
        X,
        y_clean,
        onset_bins,
        post_syllables,
        pre_bins=config.lag_range_bins,
        post_bins=config.lag_range_bins,
        config=None,
    )


def run_fig4_session(
    session_yaml: str,
    output_dir: str,
    config: Fig4Config,
    *,
    debug: bool = False,
    canonical_alignment_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run full Fig 4 pipeline for a single session."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_fig4_data(
        session_yaml, config,
        debug=debug,
        canonical_alignment_path=canonical_alignment_path,
    )
    tuning_raw, tuning_z, motif_ids, motif_counts = compute_motif_tuning(
        data["X"], data["y"], data["valid_mask"]
    )

    depth_df = build_unit_depth_table(
        data["unit_channels"],
        data["unit_banks"],
    )
    depth_df["unit_id"] = data["unit_ids"]
    depth_df["row_idx"] = np.arange(depth_df.shape[0], dtype=int)

    tuning_df = depth_df.copy()
    for j, motif_id in enumerate(motif_ids):
        tuning_df[f"motif_{motif_id}_raw"] = tuning_raw[:, j]
        tuning_df[f"motif_{motif_id}_z"] = tuning_z[:, j]
        tuning_df[f"motif_{motif_id}_count"] = motif_counts[j]
    tuning_df["session_id"] = data["session_id"]

    cluster_order = compute_motif_cluster_order(tuning_z, motif_ids)
    with open(out_dir / "fig4_cluster_order.json", "w") as f:
        json.dump({"motif_order": cluster_order}, f, indent=2)
    tuning_df.to_parquet(out_dir / "fig4_tuning.parquet", index=False)

    dec_summary, nulls, dec_aux = run_motif_decoding(
        data["X"], data["y"], data["valid_mask"], data["unit_banks"], config
    )
    dec_summary["session_id"] = data["session_id"]
    dec_summary.to_parquet(out_dir / "fig4_decoding_summary.parquet", index=False)
    np.savez(
        out_dir / "fig4_null_distributions.npz",
        **{k: v for k, v in nulls.items()},
    )
    np.savez(
        out_dir / "fig4_confusion_matrices.npz",
        **{f"{k}_confusion": v for k, v in dec_aux["confusion"].items()},
        **{f"{k}_labels": v for k, v in dec_aux["class_labels"].items()},
    )

    lag_results = run_lag_sweep(
        data["X"], data["y"], data["valid_mask"], data["unit_banks"], config
    )
    lag_rows = []
    for bank, result in lag_results.items():
        ci_low = result.accuracy_ci_low
        ci_high = result.accuracy_ci_high
        for j, (lag_sec, acc) in enumerate(zip(result.lags_sec, result.accuracy_curve)):
            row = {
                "bank": bank,
                "lag_sec": float(lag_sec),
                "accuracy": float(acc),
                "optimal_lag_sec": float(result.optimal_lag_sec),
                "optimal_accuracy": float(result.optimal_accuracy),
            }
            if ci_low is not None and ci_high is not None:
                row["accuracy_ci_low"] = float(ci_low[j])
                row["accuracy_ci_high"] = float(ci_high[j])
            lag_rows.append(row)
    lag_df = pd.DataFrame(lag_rows)
    lag_df["session_id"] = data["session_id"]
    lag_df.to_parquet(out_dir / "fig4_lag_sweep.parquet", index=False)

    next_result = run_next_syllable_prediction(data["X"], data["y"], config)
    next_df = pd.DataFrame(
        {
            "tau_bin": next_result.tau_bins,
            "tau_sec": next_result.tau_sec * config.bin_s,
            "accuracy": next_result.accuracy_curve,
        }
    )
    next_df["session_id"] = data["session_id"]
    next_df.to_parquet(out_dir / "fig4_next_syllable.parquet", index=False)

    # ── Transition matrix ──
    valid_labels = data["y"][data["valid_mask"]]
    clean_labels = valid_labels[
        (valid_labels != config.missing_value) & (valid_labels != config.other_id)
    ]
    trans_matrix, trans_labels = compute_transition_matrix(
        clean_labels, syllable_set=list(motif_ids), normalize=True,
    )
    np.savez(
        out_dir / "fig4_transition_matrix.npz",
        matrix=trans_matrix,
        labels=np.array(trans_labels, dtype=int),
    )
    if debug:
        print(f"  Transition matrix: {trans_matrix.shape}, "
              f"self-transition mean={np.diag(trans_matrix).mean():.3f}")

    # ── Onset-aligned PETH ──
    y_for_peth = data["y"].copy()
    y_for_peth[y_for_peth == config.other_id] = config.missing_value
    onset_bins, _ = detect_bin_onsets(y_for_peth, missing_value=config.missing_value)
    peth_pre = config.lag_range_bins * 2
    peth_post = config.lag_range_bins * 2
    if len(onset_bins) > 0:
        peth_matrix, peth_tau = compute_onset_peth(
            data["X"], onset_bins, pre_bins=peth_pre, post_bins=peth_post,
        )
    else:
        n_units = data["X"].shape[1]
        window = peth_pre + peth_post
        peth_matrix = np.full((n_units, window), np.nan)
        peth_tau = np.arange(-peth_pre, peth_post)
    np.savez(
        out_dir / "fig4_onset_peth.npz",
        peth=peth_matrix,
        tau_bins=peth_tau,
        onset_count=np.array(len(onset_bins)),
        unit_banks=data["unit_banks"],
    )
    if debug:
        print(f"  PETH: {peth_matrix.shape}, {len(onset_bins)} onsets")

    # ── Fine-grained PETH at 50ms for smooth heatmaps ──
    loader = SingleFMSessionLoader(
        session_yaml,
        canonical_alignment_path=canonical_alignment_path,
        debug=debug,
    )
    peth_bin_s = 0.05
    spikes_fine = loader.get_spikes(bin_s=peth_bin_s)
    X_fine = (
        spikes_fine.pivot(index="bin_idx", columns="unit_index", values="fr")
        .sort_index()
        .to_numpy()
    )
    bin_edges_fine = _bin_edges(loader.time_window, peth_bin_s)

    motifs = load_motifs_from_session(loader._session)
    labels_fine = align_syllables_to_bins(
        motifs.motif_id, motifs.t, bin_edges_fine,
        method="center", missing_value=config.missing_value,
    )
    labels_fine_remapped, _ = remap_to_top_n(
        labels_fine[labels_fine != config.missing_value],
        n=config.top_n_motifs,
        other_id=config.other_id,
    )
    # Rebuild full label vector with remapped values
    y_fine = np.full_like(labels_fine, config.missing_value)
    y_fine[labels_fine != config.missing_value] = labels_fine_remapped
    y_fine[y_fine == config.other_id] = config.missing_value

    onset_fine, _ = detect_bin_onsets(y_fine, missing_value=config.missing_value)
    peth_pre_fine = int(4.0 / peth_bin_s)   # 80 bins = 4s
    peth_post_fine = int(4.0 / peth_bin_s)  # 80 bins = 4s
    if len(onset_fine) > 0:
        peth_fine, tau_fine = compute_onset_peth(
            X_fine, onset_fine,
            pre_bins=peth_pre_fine, post_bins=peth_post_fine,
        )
    else:
        n_units_fine = X_fine.shape[1]
        window_fine = peth_pre_fine + peth_post_fine
        peth_fine = np.full((n_units_fine, window_fine), np.nan)
        tau_fine = np.arange(-peth_pre_fine, peth_post_fine)
    np.savez(
        out_dir / "fig4_onset_peth_fine.npz",
        peth=peth_fine,
        tau_bins=tau_fine,
        bin_s=np.array(peth_bin_s),
        onset_count=np.array(len(onset_fine)),
        unit_banks=data["unit_banks"],
    )
    if debug:
        print(f"  Fine PETH: {peth_fine.shape}, bin={peth_bin_s}s, "
              f"{len(onset_fine)} onsets")

    with open(out_dir / "fig4_config.json", "w") as f:
        json.dump(
            {
                "bin_s": config.bin_s,
                "top_n_motifs": config.top_n_motifs,
                "exclude_motif_ids": list(config.exclude_motif_ids),
                "missing_value": config.missing_value,
                "other_id": config.other_id,
                "n_folds": config.n_folds,
                "embargo_bins": config.embargo_bins,
                "n_permutations": config.n_permutations,
                "lag_range_bins": config.lag_range_bins,
                "classifier_type": config.classifier_type,
                "random_state": config.random_state,
            },
            f,
            indent=2,
        )

    return {
        "tuning": tuning_df,
        "decoding": dec_summary,
        "lag_sweep": lag_df,
        "next_syllable": next_df,
    }


def run_tgm(
    X: np.ndarray,
    y: np.ndarray,
    config: Fig4Config,
    *,
    pre_bins: int = 8,
    post_bins: int = 9,
    n_pca: int = 50,
) -> Dict[str, Any]:
    """Run Temporal Generalization Matrix analysis.

    Args:
        X: Neural activity matrix (n_bins, n_units).
        y: Binned motif labels (n_bins,).
        config: Fig4Config with CV parameters.
        pre_bins: Bins before transition onset (default 8 = 4s at 0.5s).
        post_bins: Bins after onset including onset (default 9 = 4s at 0.5s).
        n_pca: PCA components for dimensionality reduction (0 to skip).

    Returns:
        Dict with 'result' (TGMResult) and 'session_id' key if available.
    """
    from analysis.syllables.tgm import compute_tgm

    y_clean = y.copy()
    y_clean[y_clean == config.other_id] = config.missing_value
    onset_bins, post_syllables = detect_bin_onsets(
        y_clean, missing_value=config.missing_value
    )

    result = compute_tgm(
        X, y_clean, onset_bins, post_syllables,
        pre_bins=pre_bins,
        post_bins=post_bins,
        n_folds=config.n_folds,
        embargo_bins=config.embargo_bins,
        n_pca=n_pca,
        random_state=config.random_state,
    )
    return {"result": result}


def run_bout_duration_prediction(
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    bin_times: np.ndarray,
    config: Fig4Config,
    session_yaml: str,
    *,
    n_perm: int = 1000,
    canonical_alignment_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict bout duration from neural activity at bout onset.

    Args:
        X: Neural activity matrix (n_bins, n_units).
        y: Binned motif labels (n_bins,).
        valid_mask: Boolean mask for valid bins.
        bin_times: Time of each bin in seconds.
        config: Fig4Config.
        session_yaml: Path to session YAML for loading frame-level motifs.
        n_perm: Number of permutations for null distribution.
        canonical_alignment_path: Optional alignment path.

    Returns:
        Dict with 'r' (Pearson correlation), 'p' (permutation p-value),
        'n_bouts', 'durations', 'predicted'.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    loader = SingleFMSessionLoader(
        session_yaml,
        canonical_alignment_path=canonical_alignment_path,
    )
    motifs = load_motifs_from_session(loader._session)
    if motifs is None or motifs.t is None:
        return {"r": 0.0, "p": 1.0, "n_bouts": 0}

    fps = float(np.round(1.0 / np.median(np.diff(motifs.t)), 0))
    from analysis.syllables.segmentation import segment_bouts
    bouts = segment_bouts(motifs.motif_id, motifs.t, min_duration_sec=0.4, fps=fps)

    # Filter to top-N motifs only
    valid_motifs = set(np.unique(y[valid_mask]))
    valid_motifs.discard(config.missing_value)
    valid_motifs.discard(config.other_id)
    bouts = [b for b in bouts if b.syllable_id in valid_motifs]

    if len(bouts) < 20:
        return {"r": 0.0, "p": 1.0, "n_bouts": len(bouts)}

    # Map bout onset to nearest bin
    onset_bins_arr = []
    durations = []
    for bout in bouts:
        bin_idx = np.argmin(np.abs(bin_times - bout.start_sec))
        onset_bins_arr.append(bin_idx)
        durations.append(bout.duration_sec)

    onset_bins_arr = np.array(onset_bins_arr, dtype=int)
    durations = np.array(durations, dtype=float)
    log_dur = np.log(durations + 1e-3)

    # Extract neural activity at onset
    X_onset = X[onset_bins_arr]

    # Standardize + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_onset)
    n_pca = min(50, X_scaled.shape[1], X_scaled.shape[0] - 1)
    if n_pca > 1:
        pca = PCA(n_components=n_pca)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = X_scaled

    # Cross-validated Ridge regression
    kf = KFold(n_splits=min(5, len(bouts)), shuffle=True, random_state=config.random_state)
    predictions = np.full(len(bouts), np.nan)
    for train_idx, test_idx in kf.split(X_pca):
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_pca[train_idx], log_dur[train_idx])
        predictions[test_idx] = ridge.predict(X_pca[test_idx])

    # Pearson r
    valid_pred = ~np.isnan(predictions)
    r_obs = float(np.corrcoef(log_dur[valid_pred], predictions[valid_pred])[0, 1])

    # Permutation test
    rng = np.random.RandomState(config.random_state)
    r_null = []
    for _ in range(n_perm):
        perm_dur = rng.permutation(log_dur)
        perm_preds = np.full(len(bouts), np.nan)
        for train_idx, test_idx in kf.split(X_pca):
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_pca[train_idx], perm_dur[train_idx])
            perm_preds[test_idx] = ridge.predict(X_pca[test_idx])
        valid_perm = ~np.isnan(perm_preds)
        r_null.append(float(np.corrcoef(perm_dur[valid_perm], perm_preds[valid_perm])[0, 1]))

    p_val = float(np.mean(np.array(r_null) >= r_obs))

    return {
        "r": r_obs,
        "p": p_val,
        "n_bouts": len(bouts),
        "durations": durations,
        "log_durations": log_dur,
        "predictions": predictions,
        "r_null": np.array(r_null),
    }


def run_population_participation(
    tuning_parquet: str,
    motif_ids: List[int],
    z_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Compute population participation matrix from existing tuning data.

    Args:
        tuning_parquet: Path to fig4_tuning.parquet file.
        motif_ids: List of motif IDs to include.
        z_threshold: Threshold on |z-score| for "participating."

    Returns:
        Dict with 'binary_matrix' (n_units × n_motifs), 'participation_frac'
        (per-motif), 'bandwidth' (per-unit), 'motif_ids'.
    """
    tuning_df = pd.read_parquet(tuning_parquet)
    z_cols = [f"motif_{m}_z" for m in motif_ids if f"motif_{m}_z" in tuning_df.columns]
    valid_motifs = [m for m in motif_ids if f"motif_{m}_z" in tuning_df.columns]

    if not z_cols:
        return {"binary_matrix": np.array([]), "participation_frac": np.array([]),
                "bandwidth": np.array([]), "motif_ids": []}

    z_matrix = tuning_df[z_cols].to_numpy()
    binary = (np.abs(z_matrix) > z_threshold).astype(int)

    participation_frac = binary.mean(axis=0)  # per motif
    bandwidth = binary.sum(axis=1)  # per unit

    return {
        "binary_matrix": binary,
        "participation_frac": participation_frac,
        "bandwidth": bandwidth,
        "motif_ids": valid_motifs,
        "n_units": binary.shape[0],
        "banks": tuning_df["bank"].to_numpy() if "bank" in tuning_df.columns else None,
    }


__all__ = [
    "Fig4Config",
    "prepare_fig4_data",
    "compute_motif_tuning",
    "run_motif_decoding",
    "run_lag_sweep",
    "run_next_syllable_prediction",
    "run_fig4_session",
    "run_tgm",
    "run_bout_duration_prediction",
    "run_population_participation",
]
