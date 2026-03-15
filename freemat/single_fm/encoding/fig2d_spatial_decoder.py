"""Figure 2D spatial decoders: height and zone classification.

This module tests whether population position information remains after
residualizing nuisance regressors (VIS + BODY_KIN + HEAD_KIN + drift).

Example:
    >>> from freemat.single_fm.encoding.fig2d_spatial_decoder import (
    ...     Fig2DDecoderConfig,
    ...     run_fig2d_spatial_decoding,
    ... )
    >>> cfg = Fig2DDecoderConfig(bin_s=0.5, cv_folds=(5, 10), n_permutations=200)
    >>> df = run_fig2d_spatial_decoding(
    ...     session_yaml="configs/single_fm/session_M2_0526.yaml",
    ...     output_dir="results/single_fm/fig2d/single_0526_M2",
    ...     config=cfg,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from freemat.single_fm.data import SingleFMSessionLoader
from freemat.single_fm.encoding.cv import CVConfig, blocked_cv_folds
from freemat.single_fm.encoding.ridge import ridge_fit_predict
from freemat.single_fm.encoding.stats import block_permute_indices
from freemat.single_fm.encoding.temporal import build_lagged_features
from freemat.single_fm.encoding.visual_features import (
    BODY_KIN_COLS,
    HEAD_KIN_COLS,
    VIS_COLS,
)


_HEIGHT_LABELS: Tuple[str, str, str] = ("floor", "bench", "ceiling")

# Fold guard thresholds: (min_train_per_class, min_test_per_class)
_FOLD_GUARD_THRESHOLDS: Dict[str, Tuple[int, int]] = {
    "strict": (20, 10),
    "relaxed": (10, 5),
}


def _fold_passes_guard(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    guard_level: str,
) -> bool:
    """Check if a fold has enough samples per class for reliable decoding.

    Args:
        y_train: Train labels for this fold.
        y_test: Test labels for this fold.
        n_classes: Expected number of classes.
        guard_level: "strict" or "relaxed" threshold level.

    Returns:
        True if all classes have sufficient samples in both train and test.
    """
    min_train, min_test = _FOLD_GUARD_THRESHOLDS.get(guard_level, (20, 10))
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)

    # All expected classes must be present with enough samples
    for cls in range(n_classes):
        train_mask = train_labels == cls
        test_mask = test_labels == cls
        train_n = int(train_counts[train_mask][0]) if train_mask.any() else 0
        test_n = int(test_counts[test_mask][0]) if test_mask.any() else 0
        if train_n < min_train or test_n < min_test:
            return False
    return True


@dataclass
class Fig2DDecoderConfig:
    """Configuration for within-session spatial decoding.

    Args:
        bin_s: Spike/features bin size in seconds.
        cv_folds: Temporal blocked CV fold counts to evaluate.
        cv_block_s: Block size for temporal CV.
        cv_gap_s: Embargo gap between train/test blocks.
        n_permutations: Number of block-shuffle label permutations.
        perm_block_s: Label permutation block size in seconds.
        lag_taps: Total lag taps for nuisance regressors (odd integer, e.g. 5).
        include_time_drift: Append unlagged time drift to nuisance design.
        height_strategy: Height-tier strategy: "quantile" (balanced bins) or "fixed".
        fixed_height_cuts: (low, high) cutpoints for fixed height tiers (meters).
        n_zone_sectors: Number of allocentric angular sectors in XY.
        balance_train_bins: Downsample train bins to equal per-class counts.
        class_weight: Class weighting passed to logistic regression.
        max_iter: Max iterations for logistic regression.
        random_state: Random seed.
        fallback_alpha: Ridge alpha used if per-unit alpha is unavailable.
        debug: Print diagnostics.
    """

    bin_s: float = 0.5
    cv_folds: Tuple[int, ...] = (5, 10)
    cv_block_s: float = 30.0
    cv_gap_s: float = 2.0
    n_permutations: int = 200
    perm_block_s: float = 10.0
    lag_taps: int = 5
    include_time_drift: bool = True
    height_strategy: str = "quantile"
    fixed_height_cuts: Tuple[float, float] = (0.30, 0.60)
    n_zone_sectors: int = 3
    balance_train_bins: bool = True
    class_weight: str = "balanced"
    max_iter: int = 2000
    random_state: int = 42
    fallback_alpha: float = 1.0
    fold_guard_level: str = "strict"
    min_valid_folds: int = 3
    debug: bool = False

    @property
    def n_lags(self) -> int:
        """Half-window lag count used by build_lagged_features."""
        if self.lag_taps % 2 != 1:
            raise ValueError(f"lag_taps must be odd, got {self.lag_taps}")
        return (self.lag_taps - 1) // 2


def _zscore_train_test(
    train: np.ndarray,
    test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score train/test matrices using train statistics."""
    mu = np.nanmean(train, axis=0, keepdims=True)
    sd = np.nanstd(train, axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def _balanced_train_indices(
    y_train: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return class-balanced train indices by random downsampling."""
    labels, counts = np.unique(y_train, return_counts=True)
    min_count = int(counts.min())
    chunks = []
    for label in labels:
        idx = np.flatnonzero(y_train == label)
        chunks.append(rng.choice(idx, size=min_count, replace=False))
    return np.sort(np.concatenate(chunks))


def _build_height_and_zone_labels(
    com_x: np.ndarray,
    com_y: np.ndarray,
    com_z: np.ndarray,
    config: Fig2DDecoderConfig,
    *,
    arena_center_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """Build balanced 3-tier height labels and 9-class zone labels.

    Args:
        com_x: Center-of-mass x coordinates (world frame).
        com_y: Center-of-mass y coordinates (world frame).
        com_z: Center-of-mass z coordinates (floor-relative).
        config: Decoder configuration.
        arena_center_xy: Arena center (x, y) for centering angular zones.
            If None, defaults to (0, 0).

    Returns:
        Tuple of (height_id, zone_sector, zone_id, height_cuts).
    """
    finite = np.isfinite(com_x) & np.isfinite(com_y) & np.isfinite(com_z)
    if finite.sum() == 0:
        raise ValueError("No finite COM bins available for spatial labels.")

    if config.height_strategy == "quantile":
        low, high = np.nanquantile(com_z[finite], [1.0 / 3.0, 2.0 / 3.0])
    elif config.height_strategy == "fixed":
        low, high = config.fixed_height_cuts
    else:
        raise ValueError(f"Unknown height_strategy: {config.height_strategy}")

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Invalid height cutpoints (non-finite).")
    if high <= low:
        raise ValueError(f"Height cutpoints must satisfy high > low, got ({low}, {high}).")

    height_id = np.full(com_z.shape, -1, dtype=int)
    height_id[finite & (com_z < low)] = 0
    height_id[finite & (com_z >= low) & (com_z < high)] = 1
    height_id[finite & (com_z >= high)] = 2

    # Center COM at arena center before computing angular sectors
    cx, cy = arena_center_xy if arena_center_xy is not None else (0.0, 0.0)
    angle = np.mod(np.arctan2(com_y - cy, com_x - cx), 2.0 * np.pi)
    sector_size = (2.0 * np.pi) / float(config.n_zone_sectors)
    zone_sector = np.floor(angle / sector_size).astype(int)
    zone_sector = np.clip(zone_sector, 0, config.n_zone_sectors - 1)

    zone_id = np.full(com_z.shape, -1, dtype=int)
    valid = height_id >= 0
    zone_id[valid] = height_id[valid] * config.n_zone_sectors + zone_sector[valid]

    return height_id, zone_sector, zone_id, (float(low), float(high))


def _load_alpha_map(
    encoding_results_path: Optional[str],
    unit_ids: Sequence[int],
    fallback_alpha: float,
) -> np.ndarray:
    """Load per-unit alpha_selected map from Fig2B outputs."""
    alpha_map: Dict[int, float] = {}
    if encoding_results_path is not None and Path(encoding_results_path).exists():
        df = pd.read_parquet(encoding_results_path)
        if "alpha_selected" in df.columns and "unit_id" in df.columns:
            alpha_df = df[["unit_id", "alpha_selected"]].dropna()
            alpha_map = {
                int(uid): float(alpha)
                for uid, alpha in zip(alpha_df["unit_id"], alpha_df["alpha_selected"])
            }
    if alpha_map:
        default_alpha = float(np.median(list(alpha_map.values())))
    else:
        default_alpha = float(fallback_alpha)
    return np.array([alpha_map.get(int(uid), default_alpha) for uid in unit_ids], dtype=float)


def _fit_nuisance_residuals(
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_nuis_train: np.ndarray,
    x_nuis_test: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ridge nuisance model and return train/test residuals."""
    x_train_z, x_test_z = _zscore_train_test(x_nuis_train, x_nuis_test)
    y_pred_train = ridge_fit_predict(x_train_z, y_train, x_train_z, alpha)
    y_pred_test = ridge_fit_predict(x_train_z, y_train, x_test_z, alpha)
    return y_train - y_pred_train, y_test - y_pred_test


def _build_fold_feature_cache(
    fr_matrix: np.ndarray,
    x_nuis: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    unit_alphas: np.ndarray,
) -> List[Dict[str, np.ndarray]]:
    """Build train/test raw and residual feature matrices for each fold."""
    n_units = fr_matrix.shape[1]
    cache: List[Dict[str, np.ndarray]] = []
    for fold in folds:
        train_idx = fold["train"]
        test_idx = fold["test"]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        raw_train, raw_test = _zscore_train_test(fr_matrix[train_idx], fr_matrix[test_idx])

        resid_train = np.zeros((len(train_idx), n_units), dtype=float)
        resid_test = np.zeros((len(test_idx), n_units), dtype=float)
        x_train = x_nuis[train_idx]
        x_test = x_nuis[test_idx]

        for unit_col in range(n_units):
            train_resid, test_resid = _fit_nuisance_residuals(
                fr_matrix[train_idx, unit_col],
                fr_matrix[test_idx, unit_col],
                x_train,
                x_test,
                float(unit_alphas[unit_col]),
            )
            resid_train[:, unit_col] = train_resid
            resid_test[:, unit_col] = test_resid

        resid_train_z, resid_test_z = _zscore_train_test(resid_train, resid_test)
        cache.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "raw_train": raw_train,
                "raw_test": raw_test,
                "resid_train": resid_train_z,
                "resid_test": resid_test_z,
            }
        )
    return cache


def _compute_valid_fold_mask(
    fold_cache: List[Dict[str, np.ndarray]],
    labels: np.ndarray,
    n_classes: int,
    guard_level: str,
) -> List[bool]:
    """Pre-compute which folds pass the min-class guard on observed labels.

    Args:
        fold_cache: Pre-built fold feature matrices.
        labels: Observed spatial labels (NOT shuffled).
        n_classes: Expected number of classes.
        guard_level: "strict" or "relaxed".

    Returns:
        Boolean mask with one entry per fold in fold_cache.
    """
    mask: List[bool] = []
    for fold_data in fold_cache:
        y_train = labels[fold_data["train_idx"]]
        y_test = labels[fold_data["test_idx"]]
        mask.append(_fold_passes_guard(y_train, y_test, n_classes, guard_level))
    return mask


def _decode_accuracy(
    fold_cache: List[Dict[str, np.ndarray]],
    labels: np.ndarray,
    n_classes: int,
    model_key: str,
    config: Fig2DDecoderConfig,
    rng: np.random.Generator,
    return_predictions: bool,
    *,
    valid_fold_mask: Optional[List[bool]] = None,
) -> Tuple[float, np.ndarray, List[float]]:
    """Run blocked-CV logistic decoding and return balanced accuracy.

    When ``valid_fold_mask`` is provided (e.g. from permutation null),
    the fold guard is skipped and the mask is used directly. This ensures
    permutations use the same fold set as the observed run.

    Args:
        fold_cache: Pre-built fold feature matrices from _build_fold_feature_cache.
        labels: Spatial label array (height_id or zone_id).
        n_classes: Number of expected classes.
        model_key: "raw" or "resid".
        config: Decoder configuration.
        rng: Random number generator.
        return_predictions: If True, aggregate per-bin predictions.
        valid_fold_mask: Pre-computed fold validity mask. When provided,
            skips the per-fold guard and uses this mask instead.

    Returns:
        Tuple of (balanced_accuracy, predictions_array, per_fold_scores).
        balanced_accuracy is fold-mean BA (mean of per-fold scores).
    """
    if model_key not in ("raw", "resid"):
        raise ValueError(f"Unknown model_key: {model_key}")

    predictions = np.full(labels.shape[0], -1, dtype=int)
    fold_scores: List[float] = []

    for fold_idx, fold_data in enumerate(fold_cache):
        # Determine fold eligibility
        if valid_fold_mask is not None:
            if not valid_fold_mask[fold_idx]:
                continue
        else:
            train_idx = fold_data["train_idx"]
            test_idx = fold_data["test_idx"]
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            if not _fold_passes_guard(y_train, y_test, n_classes, config.fold_guard_level):
                if config.debug:
                    print(f"  [Fig2D] Skipping fold {fold_idx}: insufficient samples "
                          f"per class (guard={config.fold_guard_level})")
                continue

        train_idx = fold_data["train_idx"]
        test_idx = fold_data["test_idx"]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        x_train = fold_data[f"{model_key}_train"]
        x_test = fold_data[f"{model_key}_test"]

        if config.balance_train_bins:
            balanced_idx = _balanced_train_indices(y_train, rng)
            x_train_fit = x_train[balanced_idx]
            y_train_fit = y_train[balanced_idx]
        else:
            x_train_fit = x_train
            y_train_fit = y_train

        # Safety: shuffled labels can produce single-class folds
        n_unique_train = len(np.unique(y_train_fit))
        if n_unique_train < 2:
            if config.debug:
                print(f"  [Fig2D] Fold {fold_idx}: <2 classes in train after balance, skipping")
            continue

        classifier = LogisticRegression(
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            random_state=config.random_state,
            multi_class="multinomial",
            solver="lbfgs",
        )
        classifier.fit(x_train_fit, y_train_fit)
        y_pred = classifier.predict(x_test)

        fold_scores.append(float(balanced_accuracy_score(y_test, y_pred)))
        if return_predictions:
            predictions[test_idx] = y_pred

    # Require minimum number of valid folds
    if len(fold_scores) < config.min_valid_folds:
        if config.debug:
            print(f"  [Fig2D] Too few valid folds ({len(fold_scores)} < "
                  f"{config.min_valid_folds}), returning NaN")
        return float("nan"), predictions, fold_scores

    # Always return fold-mean BA as the primary score
    score = float(np.mean(fold_scores))

    # Also store per-bin predictions for concatenated BA (computed by caller)
    return score, predictions, fold_scores


def _permutation_null(
    fold_cache: List[Dict[str, np.ndarray]],
    labels: np.ndarray,
    n_classes: int,
    config: Fig2DDecoderConfig,
    rng: np.random.Generator,
    *,
    valid_fold_mask: Optional[List[bool]] = None,
) -> Dict[str, np.ndarray]:
    """Compute block-shuffled permutation null for raw and residual decoders.

    Args:
        fold_cache: Pre-built fold feature matrices.
        labels: Observed spatial labels.
        n_classes: Number of expected classes.
        config: Decoder configuration.
        rng: Random number generator.
        valid_fold_mask: Pre-computed fold validity mask from observed run.
            When provided, permutations use the same fold set as observed.

    Returns:
        Dict with keys "raw", "resid" (null distributions) and
        "n_null_valid_raw", "n_null_valid_resid" (count of finite perms).
    """
    block_len = max(1, int(round(config.perm_block_s / config.bin_s)))
    null_raw = np.full(config.n_permutations, np.nan, dtype=float)
    null_resid = np.full(config.n_permutations, np.nan, dtype=float)
    for perm_idx in range(config.n_permutations):
        perm = block_permute_indices(labels.shape[0], block_len, rng)
        shuffled_labels = labels[perm]
        null_raw[perm_idx], _, _ = _decode_accuracy(
            fold_cache,
            shuffled_labels,
            n_classes,
            "raw",
            config,
            rng,
            return_predictions=False,
            valid_fold_mask=valid_fold_mask,
        )
        null_resid[perm_idx], _, _ = _decode_accuracy(
            fold_cache,
            shuffled_labels,
            n_classes,
            "resid",
            config,
            rng,
            return_predictions=False,
            valid_fold_mask=valid_fold_mask,
        )
    return {
        "raw": null_raw,
        "resid": null_resid,
        "n_null_valid_raw": int(np.isfinite(null_raw).sum()),
        "n_null_valid_resid": int(np.isfinite(null_resid).sum()),
    }


def _prepare_bin_data(
    merged_df: pd.DataFrame,
    config: Fig2DDecoderConfig,
    *,
    arena_center_xy: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], np.ndarray]:
    """Prepare aligned firing matrix, nuisance matrix, and spatial labels.

    Args:
        merged_df: Aligned session data with firing_rate, COM, and nuisance columns.
        config: Decoder configuration.
        arena_center_xy: Arena center (x, y) for centering angular zones.

    Returns:
        Tuple of (fr_matrix, x_nuisance, height_labels, zone_labels, t_bins,
        height_cuts, unit_ids).
    """
    required_nuisance = VIS_COLS + BODY_KIN_COLS + HEAD_KIN_COLS
    required_columns = ["t_bin_s", "firing_rate", "unit_id", "com_x", "com_y", "com_z"] + required_nuisance
    missing = [col for col in required_columns if col not in merged_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Fig2D decoder: {missing}")

    fr_pivot = (
        merged_df.pivot(index="t_bin_s", columns="unit_id", values="firing_rate")
        .sort_index()
        .fillna(0.0)
    )
    unit_ids = fr_pivot.columns.to_numpy(dtype=int)
    t_bins = fr_pivot.index.to_numpy(dtype=float)

    feature_bins = (
        merged_df.drop_duplicates(subset=["t_bin_s"])
        .set_index("t_bin_s")
        .reindex(fr_pivot.index)
    )
    com_x = feature_bins["com_x"].to_numpy(dtype=float)
    com_y = feature_bins["com_y"].to_numpy(dtype=float)
    com_z = feature_bins["com_z"].to_numpy(dtype=float)

    height_id, zone_sector, zone_id, height_cuts = _build_height_and_zone_labels(
        com_x,
        com_y,
        com_z,
        config,
        arena_center_xy=arena_center_xy,
    )

    x_nuisance = feature_bins[required_nuisance].to_numpy(dtype=float)
    if config.n_lags > 0:
        x_nuisance = build_lagged_features(x_nuisance, config.n_lags)
        trim = slice(config.n_lags, -config.n_lags)
        x_nuisance = x_nuisance[trim]
        fr_values = fr_pivot.to_numpy(dtype=float)[trim]
        t_bins = t_bins[trim]
        height_id = height_id[trim]
        zone_sector = zone_sector[trim]
        zone_id = zone_id[trim]
    else:
        fr_values = fr_pivot.to_numpy(dtype=float)

    if config.include_time_drift:
        x_nuisance = np.hstack([x_nuisance, t_bins.reshape(-1, 1)])

    valid = (
        np.all(np.isfinite(fr_values), axis=1)
        & np.all(np.isfinite(x_nuisance), axis=1)
        & (height_id >= 0)
        & (zone_id >= 0)
    )

    return (
        fr_values[valid],
        x_nuisance[valid],
        height_id[valid],
        zone_id[valid],
        t_bins[valid],
        height_cuts,
        unit_ids,
    )


def run_fig2d_spatial_decoding(
    session_yaml: str,
    output_dir: str,
    *,
    config: Optional[Fig2DDecoderConfig] = None,
    encoding_results_path: Optional[str] = None,
    canonical_alignment_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run within-session height and 9-zone decoders for Figure 2D.

    Args:
        session_yaml: Path to session YAML.
        output_dir: Session output directory.
        config: Decoder configuration.
        encoding_results_path: Optional fig2b per-unit parquet for alpha_selected.
        canonical_alignment_path: Optional canonical alignment JSON path.

    Returns:
        DataFrame with one row per target/model/CV-fold setting.
    """
    if config is None:
        config = Fig2DDecoderConfig()

    loader = SingleFMSessionLoader(
        session_yaml,
        canonical_alignment_path=canonical_alignment_path,
        debug=config.debug,
    )
    merged_df = loader.get_aligned_data(
        bin_s=config.bin_s,
        floor_filter=False,
        floor_filter_before_features=False,
        binning_method="mean",
    )
    if merged_df.empty:
        raise ValueError(f"No aligned data for session: {loader.session_id}")

    merged_df = merged_df.rename(columns={"fr": "firing_rate", "unit_index": "unit_id"})

    # Extract arena center for angular zone centering
    # Use vertical_axis → horiz_idxs mapping matching feature_builder.py
    vert_axis = loader._get_vertical_axis()
    horiz_idxs = (0, 2) if vert_axis == 1 else (1, 2) if vert_axis == 0 else (0, 1)
    arena_center_xy: Optional[Tuple[float, float]] = None
    if hasattr(loader, "arena") and loader.arena is not None:
        center = loader.arena.center
        arena_center_xy = (float(center[horiz_idxs[0]]), float(center[horiz_idxs[1]]))

    fr_matrix, x_nuis, height_labels, zone_labels, t_bins, height_cuts, unit_ids = _prepare_bin_data(
        merged_df,
        config,
        arena_center_xy=arena_center_xy,
    )

    if config.debug:
        print(
            f"[Fig2D] {loader.session_id}: bins={fr_matrix.shape[0]}, units={fr_matrix.shape[1]}, "
            f"height_cuts={height_cuts}, arena_center={arena_center_xy}"
        )

    unit_alphas = _load_alpha_map(encoding_results_path, unit_ids, config.fallback_alpha)
    rng = np.random.default_rng(config.random_state)

    rows: List[Dict[str, object]] = []
    null_arrays: Dict[str, np.ndarray] = {}

    targets = {
        "height_id": (height_labels, 3),
        "zone_id": (zone_labels, 3 * config.n_zone_sectors),
    }

    for n_folds in config.cv_folds:
        cv_cfg = CVConfig(folds=n_folds, block_s=config.cv_block_s, gap_s=config.cv_gap_s)
        folds = blocked_cv_folds(fr_matrix.shape[0], bin_s=config.bin_s, cfg=cv_cfg)
        fold_cache = _build_fold_feature_cache(fr_matrix, x_nuis, folds, unit_alphas)

        for target_name, (target_labels, n_classes) in targets.items():
            # Pre-compute valid fold mask on OBSERVED labels (Fix 2)
            fold_mask = _compute_valid_fold_mask(
                fold_cache, target_labels, n_classes, config.fold_guard_level,
            )

            score_raw, pred_raw, fold_scores_raw = _decode_accuracy(
                fold_cache,
                target_labels,
                n_classes,
                "raw",
                config,
                rng,
                return_predictions=True,
                valid_fold_mask=fold_mask,
            )
            score_resid, pred_resid, fold_scores_resid = _decode_accuracy(
                fold_cache,
                target_labels,
                n_classes,
                "resid",
                config,
                rng,
                return_predictions=True,
                valid_fold_mask=fold_mask,
            )

            null_stats: Dict[str, object] = {
                "raw": np.array([], dtype=float),
                "resid": np.array([], dtype=float),
                "n_null_valid_raw": 0,
                "n_null_valid_resid": 0,
            }
            if config.n_permutations > 0:
                null_stats = _permutation_null(
                    fold_cache,
                    target_labels,
                    n_classes,
                    config,
                    rng,
                    valid_fold_mask=fold_mask,
                )
                null_arrays[f"{target_name}_cv{n_folds}_raw"] = null_stats["raw"]
                null_arrays[f"{target_name}_cv{n_folds}_resid"] = null_stats["resid"]

            for model_name, score_fold_mean, predictions, fold_scores in [
                ("raw", score_raw, pred_raw, fold_scores_raw),
                ("resid", score_resid, pred_resid, fold_scores_resid),
            ]:
                # Concatenated-prediction BA (secondary metric)
                valid = predictions >= 0
                if valid.sum() == 0:
                    acc_concat = float("nan")
                    conf = np.zeros((n_classes, n_classes), dtype=float)
                else:
                    acc_concat = float(balanced_accuracy_score(
                        target_labels[valid], predictions[valid],
                    ))
                    conf = confusion_matrix(
                        target_labels[valid],
                        predictions[valid],
                        labels=np.arange(n_classes),
                        normalize="true",
                    )

                # P-value: compare fold-mean BA against finite null values
                null_dist = null_stats[model_name]
                null_finite = null_dist[np.isfinite(null_dist)] if null_dist.size > 0 else null_dist
                n_null_valid = int(null_stats.get(f"n_null_valid_{model_name}", 0))
                chance_perm_mean = float(np.nan) if null_finite.size == 0 else float(np.mean(null_finite))
                chance_perm_low = float(np.nan) if null_finite.size == 0 else float(np.percentile(null_finite, 2.5))
                chance_perm_high = float(np.nan) if null_finite.size == 0 else float(np.percentile(null_finite, 97.5))
                perm_p = float(np.nan)
                if null_finite.size > 0 and np.isfinite(score_fold_mean):
                    perm_p = float((np.sum(null_finite >= score_fold_mean) + 1) / (null_finite.size + 1))

                rows.append(
                    {
                        "session_id": loader.session_id,
                        "target": target_name,
                        "model": model_name,
                        "cv_folds": n_folds,
                        "n_bins": int(fr_matrix.shape[0]),
                        "n_units": int(fr_matrix.shape[1]),
                        "n_classes": int(n_classes),
                        "acc": float(score_fold_mean),
                        "acc_fold_mean": float(score_fold_mean),
                        "acc_concat": float(acc_concat),
                        "acc_fold_std": float(np.nanstd(fold_scores)) if fold_scores else float("nan"),
                        "chance_theory": float(1.0 / n_classes),
                        "chance_perm_mean": chance_perm_mean,
                        "chance_perm_ci_low": chance_perm_low,
                        "chance_perm_ci_high": chance_perm_high,
                        "chance_perm_p": perm_p,
                        "n_null_valid": n_null_valid,
                        "height_cut_low": float(height_cuts[0]),
                        "height_cut_high": float(height_cuts[1]),
                        "lag_taps": int(config.lag_taps),
                        "balance_train_bins": bool(config.balance_train_bins),
                        "class_weight": str(config.class_weight),
                        "fold_guard_level": config.fold_guard_level,
                        "n_valid_folds": len(fold_scores),
                        "confusion_json": conf.tolist(),
                    }
                )

    scores_df = pd.DataFrame(rows)

    encoding_dir = Path(output_dir) / "encoding"
    encoding_dir.mkdir(parents=True, exist_ok=True)
    scores_path = encoding_dir / "fig2d_decoder_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)

    labels_df = pd.DataFrame(
        {
            "t_bin_s": t_bins,
            "height_id": height_labels,
            "height_label": [_HEIGHT_LABELS[int(h)] for h in height_labels],
            "zone_sector": zone_labels % config.n_zone_sectors,
            "zone_id": zone_labels,
        }
    )
    labels_df.to_parquet(encoding_dir / "fig2d_spatial_labels.parquet", index=False)

    if null_arrays:
        np.savez(encoding_dir / "fig2d_decoder_nulls.npz", **null_arrays)

    # Save metadata JSON with exact config, cutpoints, class counts
    import json

    height_classes, height_counts = np.unique(height_labels, return_counts=True)
    zone_classes, zone_counts = np.unique(zone_labels, return_counts=True)
    metadata = {
        "session_id": loader.session_id,
        "height_cuts": list(height_cuts),
        "arena_center_xy": list(arena_center_xy) if arena_center_xy else None,
        "n_bins": int(fr_matrix.shape[0]),
        "n_units": int(fr_matrix.shape[1]),
        "height_class_counts": {
            _HEIGHT_LABELS[int(c)]: int(n) for c, n in zip(height_classes, height_counts)
        },
        "zone_class_counts": {int(c): int(n) for c, n in zip(zone_classes, zone_counts)},
        "config": {
            "bin_s": config.bin_s,
            "cv_folds": list(config.cv_folds),
            "n_permutations": config.n_permutations,
            "lag_taps": config.lag_taps,
            "height_strategy": config.height_strategy,
            "fixed_height_cuts": list(config.fixed_height_cuts),
            "n_zone_sectors": config.n_zone_sectors,
            "fold_guard_level": config.fold_guard_level,
            "min_valid_folds": config.min_valid_folds,
            "balance_train_bins": config.balance_train_bins,
            "random_state": config.random_state,
        },
    }
    meta_path = encoding_dir / "fig2d_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return scores_df


__all__ = [
    "Fig2DDecoderConfig",
    "run_fig2d_spatial_decoding",
]

