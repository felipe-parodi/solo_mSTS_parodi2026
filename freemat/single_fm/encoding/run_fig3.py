"""Fig 3: Reference frame encoding — allocentric vs body-centric vs head-centric.

Compares how well mid-STS neurons encode behavioral features (COM velocity,
displacement-to-center) when expressed in different spatial reference frames.

Features (6 per frame):
    vel_x, vel_y, vel_z — COM velocity in frame
    disp_x, disp_y, disp_z — displacement to arena center in frame

Frames:
    allocentric — world coordinates (no rotation)
    body — rotated by body-heading yaw (nose→hips)
    head — rotated by head-forward yaw

Per-unit outputs:
    r2_base, r2_allo, r2_body, r2_head
    delta_r2_allo, delta_r2_body, delta_r2_head
    winner_frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from freemat.single_fm.data.session_loader import SingleFMSessionLoader
from freemat.single_fm.encoding.cv import CVConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Fig3Config:
    """Configuration for Fig 3 reference frame encoding."""

    bin_s: float = 0.5
    cv_folds: int = 5
    cv_block_s: float = 30.0
    cv_gap_s: float = 2.0
    r2_threshold: float = 0.01
    min_bins_per_unit: int = 150
    n_jobs: int = -1
    use_gpu: bool = False
    floor_filter: bool = True
    floor_height_threshold: float = 0.5
    floor_filter_before_features: bool = False
    include_time_drift: bool = True
    debug: bool = False
    alphas: List[float] = field(default_factory=lambda: [
        1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0,
    ])
    smooth_com_win_s: float = 0.0  # smoothing on COM before diff (0 = none)
    alpha_mode: str = "per_frame"  # "per_frame" or "shared_allo"
    height_tier_strategy: str = "quantile"  # "quantile" or "fixed"
    fixed_height_cuts: Tuple[float, float] = (0.3, 0.6)
    rf_index_eps: float = 1e-9

    # Derived
    validate_coordinates: bool = True

    @property
    def frame_names(self) -> List[str]:
        return ["allo", "body", "head"]


# ---------------------------------------------------------------------------
# Yaw-only rotation helpers
# ---------------------------------------------------------------------------

def _yaw_rotate_2d(
    vx: np.ndarray,
    vy: np.ndarray,
    yaw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate (vx, vy) by -yaw (world-to-body/head transform).

    Args:
        vx: X component in allocentric frame (N,).
        vy: Y component in allocentric frame (N,).
        yaw: Yaw angle in radians (N,). Body or head heading yaw.

    Returns:
        (vx_rotated, vy_rotated) in the body/head frame.
    """
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    vx_rot = vx * cos_y + vy * sin_y
    vy_rot = -vx * sin_y + vy * cos_y
    return vx_rot, vy_rot


def _compute_yaw_from_heading_2d(
    heading_x: np.ndarray,
    heading_y: np.ndarray,
) -> np.ndarray:
    """Compute yaw angle from 2D heading vector.

    Args:
        heading_x: X component of heading (N,).
        heading_y: Y component of heading (N,).

    Returns:
        Yaw angle in radians (N,).
    """
    return np.arctan2(heading_y, heading_x)


# ---------------------------------------------------------------------------
# Frame-level feature computation
# ---------------------------------------------------------------------------

def compute_frame_features(
    com: np.ndarray,
    fps: float,
    arena_center: np.ndarray,
    body_heading: np.ndarray,
    head_forward_3d: np.ndarray,
    vertical_idx: int,
    horiz_idxs: Tuple[int, int],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute velocity and displacement features in 3 reference frames.

    All computed at FRAME level (not binned).

    Args:
        com: Center-of-mass (T, 3) in world coords.
        fps: Frame rate for velocity computation.
        arena_center: Arena center (3,) in world coords.
        body_heading: Body heading 2D unit vectors (T, 2) in horizontal plane.
        head_forward_3d: Head forward 3D vectors (T, 3).
        vertical_idx: Index of vertical axis (0, 1, or 2).
        horiz_idxs: Indices of horizontal axes.

    Returns:
        Dict[frame_name, Dict[feature_name, np.ndarray(T,)]]
        frame_name in {"allo", "body", "head"}
        feature_name in {"vel_x", "vel_y", "vel_z",
                         "disp_x", "disp_y", "disp_z"}
    """
    T = com.shape[0]
    dt = 1.0 / fps

    # COM velocity via central difference
    vel = np.full_like(com, np.nan)
    for j in range(3):
        y = com[:, j].copy()
        m = np.isfinite(y)
        if m.sum() >= 2:
            idx = np.arange(T, dtype=float)
            y_f = np.interp(idx, idx[m], y[m])
            # Central difference
            d = np.empty_like(y_f)
            d[1:-1] = (y_f[2:] - y_f[:-2]) / (2.0 * dt)
            d[0] = (y_f[1] - y_f[0]) / dt
            d[-1] = (y_f[-1] - y_f[-2]) / dt
            vel[:, j] = d

    # Displacement to arena center
    disp = com - arena_center.reshape(1, 3)

    # Extract horizontal and vertical components
    # Allocentric: use world horizontal axes
    vel_hx = vel[:, horiz_idxs[0]]
    vel_hy = vel[:, horiz_idxs[1]]
    vel_vz = vel[:, vertical_idx]

    disp_hx = disp[:, horiz_idxs[0]]
    disp_hy = disp[:, horiz_idxs[1]]
    disp_vz = disp[:, vertical_idx]

    # Allocentric frame features
    allo = {
        "vel_x": vel_hx.copy(),
        "vel_y": vel_hy.copy(),
        "vel_z": vel_vz.copy(),
        "disp_x": disp_hx.copy(),
        "disp_y": disp_hy.copy(),
        "disp_z": disp_vz.copy(),
    }

    # Body yaw from body_heading (2D)
    body_yaw = _compute_yaw_from_heading_2d(body_heading[:, 0], body_heading[:, 1])

    # Head yaw from head_forward_3d projected to horizontal plane
    head_hx = head_forward_3d[:, horiz_idxs[0]]
    head_hy = head_forward_3d[:, horiz_idxs[1]]
    head_yaw = _compute_yaw_from_heading_2d(head_hx, head_hy)

    # Body-centric: rotate by -body_yaw
    vel_bx, vel_by = _yaw_rotate_2d(vel_hx, vel_hy, body_yaw)
    disp_bx, disp_by = _yaw_rotate_2d(disp_hx, disp_hy, body_yaw)
    body = {
        "vel_x": vel_bx,
        "vel_y": vel_by,
        "vel_z": vel_vz.copy(),
        "disp_x": disp_bx,
        "disp_y": disp_by,
        "disp_z": disp_vz.copy(),
    }

    # Head-centric: rotate by -head_yaw
    vel_hcx, vel_hcy = _yaw_rotate_2d(vel_hx, vel_hy, head_yaw)
    disp_hcx, disp_hcy = _yaw_rotate_2d(disp_hx, disp_hy, head_yaw)
    head = {
        "vel_x": vel_hcx,
        "vel_y": vel_hcy,
        "vel_z": vel_vz.copy(),
        "disp_x": disp_hcx,
        "disp_y": disp_hcy,
        "disp_z": disp_vz.copy(),
    }

    return {"allo": allo, "body": body, "head": head}


# ---------------------------------------------------------------------------
# Per-unit processing
# ---------------------------------------------------------------------------

def _process_single_unit_fig3(
    unit_id: int,
    unit_data: pd.DataFrame,
    frame_feature_cols: Dict[str, List[str]],
    base_cols: List[str],
    cv_cfg: CVConfig,
    config: Fig3Config,
    session_id: str,
    rng_seed: int,
    unit_bank: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single unit for Fig 3 reference frame comparison.

    Args:
        unit_id: Unit identifier.
        unit_data: Pre-filtered DataFrame for this unit.
        frame_feature_cols: Dict mapping frame name -> list of column names.
        base_cols: Base covariate columns (speed, time_drift).
        cv_cfg: Cross-validation configuration.
        config: Fig 3 configuration.
        session_id: Session identifier.
        rng_seed: Random seed.
        unit_bank: Bank assignment.

    Returns:
        Dict with per-frame R2 results, or None if skipped.
    """
    # Drop rows with NaN in all needed columns
    all_cols = list(set(
        ["firing_rate"] + base_cols
        + [c for cols in frame_feature_cols.values() for c in cols]
    ))
    unit_data = unit_data.dropna(subset=[c for c in all_cols if c in unit_data.columns])

    if len(unit_data) < config.min_bins_per_unit:
        return None

    y = unit_data["firing_rate"].values.astype(np.float64)
    rng = np.random.default_rng(rng_seed)

    # Build base covariates (speed + optional time drift)
    X_base = unit_data[base_cols].values.astype(np.float64)

    # Z-score base covariates
    for j in range(X_base.shape[1]):
        col = X_base[:, j]
        finite = np.isfinite(col)
        if finite.any():
            m, s = col[finite].mean(), col[finite].std()
            s = s if s > 1e-12 else 1.0
            X_base[:, j] = (col - m) / s

    # Build per-frame feature matrices
    X_frames: Dict[str, np.ndarray] = {}
    for frame_name, cols in frame_feature_cols.items():
        X_f = unit_data[cols].values.astype(np.float64)
        # Z-score per feature
        for j in range(X_f.shape[1]):
            col = X_f[:, j]
            finite = np.isfinite(col)
            if finite.any():
                m, s = col[finite].mean(), col[finite].std()
                s = s if s > 1e-12 else 1.0
                X_f[:, j] = (col - m) / s
        X_frames[frame_name] = X_f

    # Fit base model
    r2_base, alpha_base = _fit_ridge_and_select_alpha(
        X_base, y, cv_cfg, config.alphas, config.use_gpu, rng,
        bin_s=config.bin_s,
    )

    # Fit base + allocentric (always with independent alpha selection)
    X_allo_full = np.hstack([X_base, X_frames["allo"]])
    r2_allo, alpha_allo = _fit_ridge_and_select_alpha(
        X_allo_full, y, cv_cfg, config.alphas, config.use_gpu, rng,
        bin_s=config.bin_s,
    )

    X_body_full = np.hstack([X_base, X_frames["body"]])
    X_head_full = np.hstack([X_base, X_frames["head"]])

    if config.alpha_mode == "per_frame":
        # Independent alpha selection for each frame
        r2_body, alpha_body = _fit_ridge_and_select_alpha(
            X_body_full, y, cv_cfg, config.alphas, config.use_gpu, rng,
            bin_s=config.bin_s,
        )
        r2_head, alpha_head = _fit_ridge_and_select_alpha(
            X_head_full, y, cv_cfg, config.alphas, config.use_gpu, rng,
            bin_s=config.bin_s,
        )
    else:
        # Shared alpha from allocentric model (legacy mode)
        r2_body = _fit_ridge_fixed_alpha(
            X_body_full, y, cv_cfg, alpha_allo, config.use_gpu,
            bin_s=config.bin_s,
        )
        alpha_body = alpha_allo
        r2_head = _fit_ridge_fixed_alpha(
            X_head_full, y, cv_cfg, alpha_allo, config.use_gpu,
            bin_s=config.bin_s,
        )
        alpha_head = alpha_allo

    # Delta R2
    delta_r2_allo = r2_allo - r2_base
    delta_r2_body = r2_body - r2_base
    delta_r2_head = r2_head - r2_base

    # Winner
    deltas = {"allo": delta_r2_allo, "body": delta_r2_body, "head": delta_r2_head}
    winner_frame = max(deltas, key=deltas.get)

    return {
        "unit_id": unit_id,
        "session_id": session_id,
        "unit_bank": unit_bank,
        "n_bins": len(y),
        "bin_s": config.bin_s,
        "r2_base": r2_base,
        "r2_allo": r2_allo,
        "r2_body": r2_body,
        "r2_head": r2_head,
        "delta_r2_allo": delta_r2_allo,
        "delta_r2_body": delta_r2_body,
        "delta_r2_head": delta_r2_head,
        "winner_frame": winner_frame,
        "alpha_selected": alpha_allo,  # backward compat
        "alpha_allo": alpha_allo,
        "alpha_body": alpha_body,
        "alpha_head": alpha_head,
        "alpha_mode": config.alpha_mode,
    }


def _fit_ridge_and_select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    cv_cfg: CVConfig,
    alphas: List[float],
    use_gpu: bool,
    rng: np.random.Generator,
    bin_s: float = 0.5,
) -> Tuple[float, float]:
    """Fit ridge with nested alpha selection via CV. Returns (r2, best_alpha)."""
    from freemat.single_fm.encoding.cv import blocked_cv_folds
    from freemat.single_fm.encoding.ridge import (
        compute_cv_r2,
        inner_cv_select_alpha,
    )

    n = len(y)
    folds = blocked_cv_folds(n, bin_s=bin_s, cfg=cv_cfg)

    # Use compute_cv_r2 which does nested alpha selection internally
    r2 = compute_cv_r2(X, y, folds, alphas=alphas, use_gpu=use_gpu, bin_s=bin_s)

    # Also select alpha explicitly to return it for reuse
    alpha_selected = inner_cv_select_alpha(
        X, y, folds, alphas, use_gpu=use_gpu,
    )

    return r2, alpha_selected


def _fit_ridge_fixed_alpha(
    X: np.ndarray,
    y: np.ndarray,
    cv_cfg: CVConfig,
    alpha: float,
    use_gpu: bool,
    bin_s: float = 0.5,
) -> float:
    """Fit ridge with a fixed alpha. Returns r2."""
    from freemat.single_fm.encoding.cv import blocked_cv_folds
    from freemat.single_fm.encoding.ridge import compute_cv_r2_fixed_alpha

    n = len(y)
    folds = blocked_cv_folds(n, bin_s=bin_s, cfg=cv_cfg)
    return compute_cv_r2_fixed_alpha(X, y, folds, alpha, use_gpu=use_gpu)


# ---------------------------------------------------------------------------
# Per-tier re-fit helpers
# ---------------------------------------------------------------------------


def _fit_ridge_cv_tier(
    X: np.ndarray,
    y: np.ndarray,
    full_folds: List[Dict[str, np.ndarray]],
    tier_indices: np.ndarray,
    alphas: List[float],
    use_gpu: bool,
    bin_s: float,
) -> Tuple[float, float, int]:
    """Fit ridge with per-tier fold intersection and independent alpha selection.

    Creates tier-restricted train/test sets by intersecting full-timeline folds
    with tier indices. Alpha is selected via inner CV on the tier-restricted
    training data.

    Args:
        X: Full design matrix (N_total, D) after dropna.
        y: Full target vector (N_total,) after dropna.
        full_folds: CV folds built on the full timeline.
        tier_indices: Indices (into X/y) belonging to this tier.
        alphas: Ridge alpha grid.
        use_gpu: Whether to use GPU acceleration.
        bin_s: Bin size in seconds for inner CV fold construction.

    Returns:
        Tuple of (r2, best_alpha, n_valid_folds).
    """
    from freemat.single_fm.encoding.cv import CVConfig, blocked_cv_folds
    from freemat.single_fm.encoding.ridge import (
        inner_cv_select_alpha,
        r2_score,
        ridge_fit_predict,
        ridge_fit_predict_gpu,
    )

    tier_set = set(tier_indices.tolist())
    ridge_fn = ridge_fit_predict_gpu if use_gpu else ridge_fit_predict

    # Intersect each outer fold with tier indices
    tier_folds = []
    for fold in full_folds:
        tr_global = fold["train"]
        te_global = fold["test"]
        tier_train = np.array([i for i in tr_global if i in tier_set], dtype=int)
        tier_test = np.array([i for i in te_global if i in tier_set], dtype=int)
        tier_folds.append({"train": tier_train, "test": tier_test})

    # Count valid folds: non-empty train+test, test has variance
    n_valid = 0
    for tf in tier_folds:
        if tf["train"].size > 0 and tf["test"].size > 0:
            if np.std(y[tf["test"]]) > 1e-6:
                n_valid += 1

    if n_valid < 2:
        return float("nan"), float("nan"), n_valid

    # Collect all tier train indices for alpha selection
    all_tier_train = np.unique(np.concatenate(
        [tf["train"] for tf in tier_folds if tf["train"].size > 0]
    ))

    if all_tier_train.size == 0:
        return float("nan"), float("nan"), 0

    # Inner CV for alpha selection: build local folds on the tier_train subset
    X_tier_train = X[all_tier_train]
    y_tier_train = y[all_tier_train]
    inner_cfg = CVConfig(folds=3, block_s=30.0, gap_s=2.0)
    inner_folds = blocked_cv_folds(len(all_tier_train), bin_s=bin_s, cfg=inner_cfg)
    best_alpha = inner_cv_select_alpha(
        X_tier_train, y_tier_train, inner_folds, alphas, use_gpu=use_gpu,
    )

    # Outer CV: fit on tier_train, predict on tier_test per fold
    all_test_true = []
    all_test_pred = []

    for tf in tier_folds:
        tr_idx, te_idx = tf["train"], tf["test"]
        if tr_idx.size == 0 or te_idx.size == 0:
            continue
        if np.std(y[te_idx]) <= 1e-6:
            continue

        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        # Standardize within fold
        mu = np.nanmean(X_train, axis=0, keepdims=True)
        sd = np.nanstd(X_train, axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd

        y_pred = ridge_fn(X_train, y_train, X_test, best_alpha)

        if np.all(np.isfinite(y_pred)):
            all_test_true.append(y_test)
            all_test_pred.append(y_pred)

    if not all_test_true:
        return float("nan"), float(best_alpha), 0

    y_true_all = np.concatenate(all_test_true)
    y_pred_all = np.concatenate(all_test_pred)
    r2 = r2_score(y_true_all, y_pred_all)

    return float(r2), float(best_alpha), n_valid


# ---------------------------------------------------------------------------
# Per-tier re-fit: per-unit processing
# ---------------------------------------------------------------------------


def _process_single_unit_fig3_per_tier_refit(
    unit_id: int,
    unit_data: pd.DataFrame,
    frame_feature_cols: Dict[str, List[str]],
    base_cols: List[str],
    cv_cfg: CVConfig,
    config: Fig3Config,
    height_tiers: np.ndarray,
    session_id: str,
    rng_seed: int,
    unit_bank: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Process a single unit for height-stratified RF-index with per-tier re-fit.

    For each height tier, fits independent ridge models (with independent alpha
    selection) on the tier-restricted subset of train bins, evaluates on
    tier-restricted test bins. This avoids the shared-model artifact where
    floor-dominated models produce negative R² for bench/ceiling tiers.

    CV folds are created on the FULL timeline to preserve temporal blocking,
    then intersected with tier indices.

    Args:
        unit_id: Unit identifier.
        unit_data: Pre-filtered DataFrame for this unit.
        frame_feature_cols: Dict mapping frame name -> list of column names.
        base_cols: Base covariate columns (speed, time_drift).
        cv_cfg: Cross-validation configuration.
        config: Fig 3 configuration.
        height_tiers: Height tier labels (0/1/2) aligned to unit_data rows
            BEFORE dropna. Will be filtered to match surviving rows.
        session_id: Session identifier.
        rng_seed: Random seed.
        unit_bank: Bank assignment.

    Returns:
        List of dicts (one per tier) with per-tier R² and RF-index,
        or None if the unit is skipped entirely.
    """
    from freemat.single_fm.encoding.cv import blocked_cv_folds

    # Build column mask and apply dropna (same logic as _process_single_unit_fig3)
    all_cols = list(set(
        ["firing_rate"] + base_cols
        + [c for cols in frame_feature_cols.values() for c in cols]
    ))
    keep_cols = [c for c in all_cols if c in unit_data.columns]
    valid_mask = unit_data[keep_cols].notna().all(axis=1).values
    unit_data = unit_data.loc[valid_mask].copy()
    # Critical: filter height_tiers with the SAME mask
    height_tiers = height_tiers[valid_mask]

    if len(unit_data) < config.min_bins_per_unit:
        return None

    y = unit_data["firing_rate"].values.astype(np.float64)

    # Z-score base covariates
    X_base = unit_data[base_cols].values.astype(np.float64)
    for j in range(X_base.shape[1]):
        col = X_base[:, j]
        finite = np.isfinite(col)
        if finite.any():
            m, s = col[finite].mean(), col[finite].std()
            s = s if s > 1e-12 else 1.0
            X_base[:, j] = (col - m) / s

    # Z-score per-frame features
    X_frames: Dict[str, np.ndarray] = {}
    for frame_name, cols in frame_feature_cols.items():
        X_f = unit_data[cols].values.astype(np.float64)
        for j in range(X_f.shape[1]):
            col = X_f[:, j]
            finite = np.isfinite(col)
            if finite.any():
                m, s = col[finite].mean(), col[finite].std()
                s = s if s > 1e-12 else 1.0
                X_f[:, j] = (col - m) / s
        X_frames[frame_name] = X_f

    # Build full-timeline folds (preserves temporal blocking)
    n = len(y)
    full_folds = blocked_cv_folds(n, bin_s=config.bin_s, cfg=cv_cfg)

    # Design matrices per model
    models = {
        "base": X_base,
        "allo": np.hstack([X_base, X_frames["allo"]]),
        "body": np.hstack([X_base, X_frames["body"]]),
        "head": np.hstack([X_base, X_frames["head"]]),
    }

    tier_labels = {0: "floor", 1: "bench", 2: "ceiling"}
    eps = config.rf_index_eps
    results: List[Dict[str, Any]] = []

    for tier_id in sorted(np.unique(height_tiers)):
        tier_indices = np.where(height_tiers == tier_id)[0]
        n_bins_tier = len(tier_indices)

        if n_bins_tier < config.min_bins_per_unit:
            continue

        # Fit each model independently on this tier
        tier_r2: Dict[str, float] = {}
        tier_alpha: Dict[str, float] = {}
        tier_n_valid: int = 0

        for model_name, X_model in models.items():
            r2, alpha, n_valid = _fit_ridge_cv_tier(
                X_model, y, full_folds, tier_indices,
                config.alphas, config.use_gpu, config.bin_s,
            )
            tier_r2[model_name] = r2
            tier_alpha[model_name] = alpha
            if model_name == "base":
                tier_n_valid = n_valid

        # Skip tier if base model couldn't fit
        if np.isnan(tier_r2.get("base", float("nan"))):
            continue

        r2_base_t = tier_r2["base"]
        r2_allo_t = tier_r2.get("allo", float("nan"))
        r2_body_t = tier_r2.get("body", float("nan"))
        r2_head_t = tier_r2.get("head", float("nan"))

        delta_allo = r2_allo_t - r2_base_t
        delta_body = r2_body_t - r2_base_t
        delta_head = r2_head_t - r2_base_t

        # RF-index per tier
        denom_size = abs(delta_allo) + abs(delta_body)
        rf_index = (delta_allo - delta_body) / (denom_size + eps)

        # Winner
        deltas = {"allo": delta_allo, "body": delta_body, "head": delta_head}
        winner_tier = max(deltas, key=lambda k: deltas[k])

        # QC: interpretable if models produce meaningful signal
        max_delta_r2 = max(delta_allo, delta_body)
        qc_interpretable = (
            max_delta_r2 > 0.01
            and denom_size > 0.01
            and tier_n_valid >= 2
        )

        results.append({
            "unit_id": unit_id,
            "session_id": session_id,
            "bank": unit_bank,
            "height_tier": tier_labels.get(int(tier_id), str(tier_id)),
            "height_tier_id": int(tier_id),
            "r2_base_tier": r2_base_t,
            "r2_allo_tier": r2_allo_t,
            "r2_body_tier": r2_body_t,
            "r2_head_tier": r2_head_t,
            "delta_r2_allo_tier": delta_allo,
            "delta_r2_body_tier": delta_body,
            "delta_r2_head_tier": delta_head,
            "rf_index": rf_index,
            "winner_tier": winner_tier,
            "n_bins_tier": n_bins_tier,
            "n_valid_folds": tier_n_valid,
            "alpha_allo_tier": tier_alpha.get("allo", float("nan")),
            "alpha_body_tier": tier_alpha.get("body", float("nan")),
            "alpha_head_tier": tier_alpha.get("head", float("nan")),
            "max_delta_r2": max_delta_r2,
            "denom_size": denom_size,
            "qc_interpretable": qc_interpretable,
            "bin_s": config.bin_s,
        })

    return results if results else None


# ---------------------------------------------------------------------------
# Session preparation helper (shared by encoding + height-stratified)
# ---------------------------------------------------------------------------


def _prepare_session_for_fig3(
    session_yaml: str,
    config: Fig3Config,
    *,
    canonical_alignment_path: Optional[str] = None,
    floor_filter: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str], str]:
    """Load session, compute frame features, merge to bins.

    Args:
        session_yaml: Path to session YAML.
        config: Fig 3 configuration.
        canonical_alignment_path: Optional alignment JSON.
        floor_filter: Override floor filter setting. If None, uses config.

    Returns:
        Tuple of (merged_df, frame_feature_cols, base_cols, session_id).
    """
    from freemat_data import load_session_yaml, compute_center_of_mass

    sref = load_session_yaml(session_yaml)
    session_id = str(getattr(sref, "session_id", "session"))

    if config.debug:
        print(f"[Fig3] Session: {session_id}")
        print(f"[Fig3] Bin size: {config.bin_s}s, CV folds: {config.cv_folds}")

    loader = SingleFMSessionLoader(
        session_yaml,
        canonical_alignment_path=canonical_alignment_path,
        debug=config.debug,
    )

    # Pose, arena, plane info
    pose = loader.get_pose_oriented()
    fps = float(pose.fps or 5.0)
    arena = loader.arena
    arena_center = np.asarray(arena.center, dtype=float)
    plane_dict = loader.plane_dict
    vertical_axis = loader._get_vertical_axis()
    horiz_idxs = (0, 2) if vertical_axis == 1 else (
        (1, 2) if vertical_axis == 0 else (0, 1)
    )

    if config.debug:
        print(f"[Fig3] Arena center: {arena_center}")
        print(f"[Fig3] Vertical axis: {vertical_axis}, horiz: {horiz_idxs}")

    # Frame-level features
    com = compute_center_of_mass(pose.X)

    from freemat.single_fm.features.canonical import (
        compute_body_heading,
        compute_head_forward,
    )
    body_heading = compute_body_heading(pose.X, vertical_axis)
    head_fwd_3d = compute_head_forward(pose.X, plane_dict)

    frame_feats = compute_frame_features(
        com, fps, arena_center, body_heading, head_fwd_3d,
        vertical_axis, horiz_idxs,
    )

    # Merged data (spikes + features)
    use_floor_filter = floor_filter if floor_filter is not None else config.floor_filter
    merged_df = loader.get_aligned_data(
        bin_s=config.bin_s,
        floor_filter=use_floor_filter,
        floor_height_threshold=config.floor_height_threshold,
        floor_filter_before_features=config.floor_filter_before_features,
    )

    if len(merged_df) == 0:
        return pd.DataFrame(), {}, [], session_id

    merged_df = merged_df.rename(columns={
        'fr': 'firing_rate',
        'unit_index': 'unit_id',
    })

    # Bin frame features to t_bins
    from freemat.single_fm.features.canonical import bin_features

    t_video_pose = loader.pose_video_time()
    t_bins_unique = merged_df["t_bin_s"].unique()
    t_bins_unique.sort()

    for frame_name, feats in frame_feats.items():
        binned = bin_features(feats, t_video_pose, t_bins_unique, method='mean')
        prefix = f"f3_{frame_name}_"
        for col in binned.columns:
            if col == "t_bin_s":
                continue
            t2v = dict(zip(t_bins_unique, binned[col].values))
            merged_df[prefix + col] = merged_df["t_bin_s"].map(t2v)

    # Base covariates
    base_cols: List[str] = []
    if "speed" in merged_df.columns:
        base_cols.append("speed")

    if config.include_time_drift:
        merged_df["_time_drift"] = merged_df["t_bin_s"].values.astype(float)
        base_cols.append("_time_drift")

    # Frame feature columns
    frame_feature_cols: Dict[str, List[str]] = {}
    for frame_name in ["allo", "body", "head"]:
        prefix = f"f3_{frame_name}_"
        cols = [c for c in merged_df.columns if c.startswith(prefix)]
        frame_feature_cols[frame_name] = cols

    if config.debug:
        for fn, cols in frame_feature_cols.items():
            print(f"[Fig3] Frame '{fn}' columns: {cols}")
        print(f"[Fig3] Base columns: {base_cols}")

    return merged_df, frame_feature_cols, base_cols, session_id


# ---------------------------------------------------------------------------
# Session-level runner
# ---------------------------------------------------------------------------

def run_fig3_encoding(
    session_yaml: str,
    output_dir: str,
    *,
    config: Optional[Fig3Config] = None,
    unit_ids: Optional[Sequence[int]] = None,
    save_results: bool = True,
    canonical_alignment_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run Fig 3 reference frame encoding for one session.

    Args:
        session_yaml: Path to session YAML.
        output_dir: Output directory.
        config: Analysis configuration.
        unit_ids: Optional subset of units.
        save_results: Whether to save parquet.

    Returns:
        DataFrame with per-unit results.
    """
    if config is None:
        config = Fig3Config()

    merged_df, frame_feature_cols, base_cols, session_id = _prepare_session_for_fig3(
        session_yaml, config,
        canonical_alignment_path=canonical_alignment_path,
    )

    if len(merged_df) == 0:
        print("[Fig3] WARNING: No merged data!")
        return pd.DataFrame()

    # Get units
    all_unit_ids = merged_df["unit_id"].unique()
    if unit_ids is not None:
        all_unit_ids = [u for u in all_unit_ids if u in unit_ids]

    if config.debug:
        print(f"[Fig3] Processing {len(all_unit_ids)} units...")

    # CV config
    cv_cfg = CVConfig(
        folds=config.cv_folds,
        block_s=config.cv_block_s,
        gap_s=config.cv_gap_s,
    )

    # Pre-extract unit data
    unit_data_dict = {
        uid: merged_df[merged_df["unit_id"] == uid].copy()
        for uid in all_unit_ids
    }

    # Bank mapping
    unit_bank_map: Dict[int, Optional[str]] = {}
    if "unit_bank" in merged_df.columns:
        unit_bank_map = merged_df.groupby("unit_id")["unit_bank"].first().to_dict()

    # Parallel processing
    rng_master = np.random.default_rng(42)
    unit_seeds = {uid: int(rng_master.integers(0, 2**31)) for uid in all_unit_ids}

    results = Parallel(n_jobs=config.n_jobs, verbose=10 if config.debug else 0)(
        delayed(_process_single_unit_fig3)(
            uid,
            unit_data_dict[uid],
            frame_feature_cols,
            base_cols,
            cv_cfg,
            config,
            session_id,
            unit_seeds[uid],
            unit_bank_map.get(uid),
        )
        for uid in all_unit_ids
    )

    results = [r for r in results if r is not None]

    if not results:
        print("[Fig3] WARNING: No valid units!")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if config.debug:
        print(f"[Fig3] Results: {len(df)} units")
        print(f"[Fig3] Median R2: base={df['r2_base'].median():.4f}, "
              f"allo={df['r2_allo'].median():.4f}, "
              f"body={df['r2_body'].median():.4f}, "
              f"head={df['r2_head'].median():.4f}")
        winner_counts = df["winner_frame"].value_counts()
        print(f"[Fig3] Winner counts: {dict(winner_counts)}")

    # Save
    if save_results and output_dir:
        out_path = Path(output_dir) / "encoding"
        out_path.mkdir(parents=True, exist_ok=True)
        parquet_path = out_path / "fig3_reference_frames.parquet"
        df.to_parquet(parquet_path, index=False)
        if config.debug:
            print(f"[Fig3] Saved: {parquet_path}")

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_fig3_results(
    session_dfs: List[pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate Fig 3 results across sessions.

    Args:
        session_dfs: List of per-session DataFrames.

    Returns:
        Combined DataFrame.
    """
    valid = [df for df in session_dfs if len(df) > 0]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)
