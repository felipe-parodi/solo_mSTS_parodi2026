"""Figure 5 follow-up: additive vs interaction encoding analysis.

Tests whether per-unit firing rates are better explained by additive tuning
for motif + height tier + continuous controls, or by explicit motif x tier
interaction terms. This is a direct encoding-model test of the paper's
joint state-space representation claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from freemat.single_fm.encoding.config import apply_fdr_correction
from freemat.single_fm.encoding.cv import CVConfig, blocked_cv_folds
from freemat.single_fm.encoding.ridge import (
    DEFAULT_ALPHAS,
    compute_cv_r2_fixed_alpha,
)
from freemat.single_fm.encoding.temporal import (
    _add_time_drift_covariate,
    build_lagged_features,
)
from freemat.single_fm.encoding.variance import compute_unique_variance_nway_shuffle
from freemat.single_fm.encoding.visual_features import (
    BODY_KIN_COLS,
    HEAD_KIN_COLS,
    VIS_COLS,
)

# Must match scripts/single_fm/run_fig5_cebra.py
HEIGHT_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "floor": (0.0, 0.5),
    "bench": (0.5, 1.2),
    "ceiling": (1.2, 3.0),
}


@dataclass
class InteractionConfig:
    """Configuration for additive vs interaction encoding analysis."""

    bin_s: float = 0.5
    cv_folds: int = 10
    cv_block_s: float = 30.0
    cv_gap_s: float = 2.0
    n_perm: int = 100
    n_perm_null: int = 500
    perm_block_s: float = 10.0
    alphas: Sequence[float] = DEFAULT_ALPHAS
    lag_s: float = 1.0
    n_lags: int = -1
    r2_threshold: float = 0.01
    effect_threshold: float = 0.005
    p_threshold: float = 0.05
    min_bins_per_unit: int = 150
    min_bins_per_cell: int = 20
    min_tiers_per_motif: int = 2
    n_jobs: int = -1
    use_gpu: bool = False
    sua_only: bool = False
    include_time_drift: bool = False
    include_rssi: bool = False
    include_antenna_distance: bool = False
    binning_method: str = "mean"
    debug: bool = False

    def __post_init__(self) -> None:
        if self.n_lags == -1:
            self.n_lags = round(self.lag_s / self.bin_s) if self.lag_s > 0 else 0


@dataclass
class InteractionDesign:
    """Shared session-wide design matrices for the interaction analysis."""

    X_controls: np.ndarray
    X_motif: np.ndarray
    X_tier: np.ndarray
    X_interaction: np.ndarray
    y_motif: np.ndarray
    height_tier: np.ndarray
    bin_times: np.ndarray
    retained_motifs: List[int]
    retained_cells: List[Tuple[int, int]]
    cell_counts: Dict[Tuple[int, int], int]
    control_col_names: List[str]
    motif_labels: List[int]
    interaction_labels: List[Tuple[int, int]]
    trim: int
    n_rows_pre_trim: int
    n_rows_post_trim: int
    n_rows_final: int

    @property
    def additive_matrix(self) -> np.ndarray:
        """Return the additive design matrix."""
        return np.hstack([self.X_controls, self.X_motif, self.X_tier])

    @property
    def full_matrix(self) -> np.ndarray:
        """Return the full design matrix with interaction terms."""
        return np.hstack(
            [self.X_controls, self.X_motif, self.X_tier, self.X_interaction]
        )

    def occupancy_records(self) -> List[Dict[str, int]]:
        """Return a flat occupancy table for serialization."""
        return [
            {"motif_id": int(mid), "tier": int(tier), "n_bins": int(count)}
            for (mid, tier), count in sorted(self.cell_counts.items())
        ]


def align_motifs_to_bins(
    motifs: Any,
    bin_times: np.ndarray,
    bin_s: float,
) -> np.ndarray:
    """Align motif labels to bin centers using majority vote within bin."""
    motif_t = motifs.t
    motif_id = motifs.motif_id
    n_bins = len(bin_times)
    y = np.zeros(n_bins, dtype=int)

    for i, t_center in enumerate(bin_times):
        t_lo = t_center - bin_s / 2
        t_hi = t_center + bin_s / 2
        mask = (motif_t >= t_lo) & (motif_t < t_hi)
        if not mask.any():
            continue

        labels_in_bin = motif_id[mask]
        nonzero = labels_in_bin[labels_in_bin > 0]
        if len(nonzero) > 0:
            y[i] = int(np.bincount(nonzero).argmax())

    return y


def build_height_tier(com_z: np.ndarray) -> np.ndarray:
    """Convert continuous height into the fixed Figure 5 tier labels."""
    com_z = np.asarray(com_z, dtype=np.float64)
    tier = np.full(com_z.shape, -1, dtype=int)

    for tier_idx, (_name, (lo, hi)) in enumerate(HEIGHT_THRESHOLDS.items()):
        mask = np.isfinite(com_z) & (com_z >= lo) & (com_z < hi)
        tier[mask] = tier_idx

    return tier


def _zscore_columns(X: np.ndarray) -> np.ndarray:
    """Z-score columns, keeping constant columns as zeros."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            out[:, j] = 0.0
            continue
        mu = np.mean(col[finite])
        sd = np.std(col[finite])
        sd = sd if sd > 1e-12 else 1.0
        out[:, j] = (col - mu) / sd
    return out


def _build_one_hot(labels: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Build a dense one-hot matrix."""
    labels = np.asarray(labels, dtype=int)
    classes = [int(c) for c in classes]
    X = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for col, cls in enumerate(classes):
        X[:, col] = (labels == cls).astype(np.float64)
    return X


def _build_interaction_block(
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    retained_cells: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Build one-hot motif x tier interaction columns."""
    X = np.zeros((len(y_motif), len(retained_cells)), dtype=np.float64)
    for col, (motif_id, tier_id) in enumerate(retained_cells):
        X[:, col] = ((y_motif == motif_id) & (height_tier == tier_id)).astype(
            np.float64
        )
    return X


def _compute_cell_counts(
    y_motif: np.ndarray,
    height_tier: np.ndarray,
) -> Dict[Tuple[int, int], int]:
    """Count bins per motif x tier cell."""
    counts: Dict[Tuple[int, int], int] = {}
    motif_ids = np.unique(y_motif[y_motif > 0])
    for motif_id in motif_ids:
        for tier in range(3):
            n = int(np.sum((y_motif == motif_id) & (height_tier == tier)))
            if n > 0:
                counts[(int(motif_id), tier)] = n
    return counts


def _retained_motifs_and_cells(
    y_motif: np.ndarray,
    height_tier: np.ndarray,
    min_bins_per_cell: int,
    min_tiers_per_motif: int,
) -> Tuple[List[int], List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Determine eligible motifs and motif x tier cells."""
    raw_counts = _compute_cell_counts(y_motif, height_tier)

    retained_motifs: List[int] = []
    for motif_id in sorted({mid for mid, _ in raw_counts.keys()}):
        eligible_tiers = [
            tier
            for tier in range(3)
            if raw_counts.get((motif_id, tier), 0) >= min_bins_per_cell
        ]
        if len(eligible_tiers) >= min_tiers_per_motif:
            retained_motifs.append(int(motif_id))

    retained_cells = [
        (int(motif_id), int(tier))
        for motif_id in retained_motifs
        for tier in range(3)
        if raw_counts.get((motif_id, tier), 0) >= min_bins_per_cell
    ]
    cell_counts = {
        cell: raw_counts[cell]
        for cell in retained_cells
    }
    return retained_motifs, retained_cells, cell_counts


def build_interaction_design(
    feat_df: pd.DataFrame,
    y_motif: np.ndarray,
    bin_times: np.ndarray,
    config: InteractionConfig,
) -> InteractionDesign:
    """Build the shared session-wide design matrices."""
    required_cols = VIS_COLS + BODY_KIN_COLS + HEAD_KIN_COLS + ["com_x", "com_y", "com_z"]
    missing = [col for col in required_cols if col not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    vis = feat_df[VIS_COLS].values.astype(np.float64)
    body = feat_df[BODY_KIN_COLS].values.astype(np.float64)
    head = feat_df[HEAD_KIN_COLS].values.astype(np.float64)
    xy = feat_df[["com_x", "com_y"]].values.astype(np.float64)
    com_z = feat_df["com_z"].values.astype(np.float64)

    height_tier = build_height_tier(com_z)
    xy = _zscore_columns(xy)

    if config.n_lags > 0:
        vis = build_lagged_features(vis, config.n_lags)
        body = build_lagged_features(body, config.n_lags)
        head = build_lagged_features(head, config.n_lags)

    trim = config.n_lags if config.n_lags > 0 else 0
    if trim > 0:
        vis = vis[trim:-trim]
        body = body[trim:-trim]
        head = head[trim:-trim]
        xy = xy[trim:-trim]
        height_tier = height_tier[trim:-trim]
        y_motif = y_motif[trim:-trim]
        bin_times = bin_times[trim:-trim]

    controls = np.hstack([vis, body, head, xy])
    control_col_names: List[str] = []
    if config.n_lags > 0:
        control_col_names.extend(
            [f"{col}_lag{lag}" for lag in range(-config.n_lags, config.n_lags + 1) for col in VIS_COLS]
        )
        control_col_names.extend(
            [f"{col}_lag{lag}" for lag in range(-config.n_lags, config.n_lags + 1) for col in BODY_KIN_COLS]
        )
        control_col_names.extend(
            [f"{col}_lag{lag}" for lag in range(-config.n_lags, config.n_lags + 1) for col in HEAD_KIN_COLS]
        )
    else:
        control_col_names.extend(VIS_COLS)
        control_col_names.extend(BODY_KIN_COLS)
        control_col_names.extend(HEAD_KIN_COLS)
    control_col_names.extend(["com_x", "com_y"])

    if config.include_time_drift:
        controls = _add_time_drift_covariate(bin_times.astype(np.float64), controls)
        control_col_names.append("time_drift")

    if config.include_rssi:
        if "rssi" not in feat_df.columns:
            raise ValueError("include_rssi=True but feat_df has no 'rssi' column.")
        rssi = feat_df["rssi"].values.astype(np.float64)
        if trim > 0:
            rssi = rssi[trim:-trim]
        rssi = _zscore_columns(rssi).reshape(-1, 1)
        controls = np.hstack([controls, rssi])
        control_col_names.append("rssi")

    if config.include_antenna_distance:
        if "d_antenna" not in feat_df.columns:
            raise ValueError(
                "include_antenna_distance=True but feat_df has no 'd_antenna' column."
            )
        d_ant = feat_df["d_antenna"].values.astype(np.float64)
        if trim > 0:
            d_ant = d_ant[trim:-trim]
        d_ant = _zscore_columns(d_ant).reshape(-1, 1)
        controls = np.hstack([controls, d_ant])
        control_col_names.append("d_antenna")

    finite_mask = (
        np.all(np.isfinite(controls), axis=1)
        & np.isfinite(bin_times)
        & (y_motif > 0)
        & (height_tier >= 0)
    )

    y_motif_valid = y_motif[finite_mask]
    tier_valid = height_tier[finite_mask]
    bin_times_valid = bin_times[finite_mask]
    controls_valid = controls[finite_mask]

    retained_motifs, retained_cells, cell_counts = _retained_motifs_and_cells(
        y_motif_valid,
        tier_valid,
        min_bins_per_cell=config.min_bins_per_cell,
        min_tiers_per_motif=config.min_tiers_per_motif,
    )

    if not retained_cells:
        raise ValueError(
            "No eligible motif x tier cells after applying session-qualified occupancy thresholds."
        )

    cell_mask = np.array(
        [(int(mid), int(tier)) in set(retained_cells) for mid, tier in zip(y_motif_valid, tier_valid)],
        dtype=bool,
    )

    controls_final = controls_valid[cell_mask]
    y_motif_final = y_motif_valid[cell_mask]
    tier_final = tier_valid[cell_mask]
    bin_times_final = bin_times_valid[cell_mask]

    X_motif = _build_one_hot(y_motif_final, retained_motifs)
    X_tier = _build_one_hot(tier_final, [0, 1, 2])
    X_interaction = _build_interaction_block(y_motif_final, tier_final, retained_cells)

    if len(controls_final) == 0:
        raise ValueError("No valid rows remain after motif x tier cell filtering.")

    return InteractionDesign(
        X_controls=controls_final,
        X_motif=X_motif,
        X_tier=X_tier,
        X_interaction=X_interaction,
        y_motif=y_motif_final,
        height_tier=tier_final,
        bin_times=bin_times_final,
        retained_motifs=retained_motifs,
        retained_cells=retained_cells,
        cell_counts=cell_counts,
        control_col_names=control_col_names,
        motif_labels=retained_motifs,
        interaction_labels=list(retained_cells),
        trim=trim,
        n_rows_pre_trim=len(feat_df),
        n_rows_post_trim=len(controls),
        n_rows_final=len(controls_final),
    )


def fit_unit_interaction_model(
    y: np.ndarray,
    design: InteractionDesign,
    config: InteractionConfig,
    *,
    unit_id: Optional[int] = None,
    unit_kind: Optional[str] = None,
    unit_bank: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit the additive-vs-interaction model for one unit."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) != design.n_rows_final:
        raise ValueError(
            f"Unit response length {len(y)} does not match design length {design.n_rows_final}."
        )
    if len(y) < config.min_bins_per_unit:
        raise ValueError(
            f"Unit has only {len(y)} bins after filtering; minimum is {config.min_bins_per_unit}."
        )

    groups = {
        "controls": design.X_controls,
        "motif": design.X_motif,
        "tier": design.X_tier,
        "motif_x_tier": design.X_interaction,
    }

    cv_cfg = CVConfig(
        folds=config.cv_folds,
        block_s=config.cv_block_s,
        gap_s=config.cv_gap_s,
    )

    uv = compute_unique_variance_nway_shuffle(
        groups,
        y,
        cv_cfg=cv_cfg,
        bin_s=config.bin_s,
        n_perm=config.n_perm,
        n_perm_null=config.n_perm_null,
        block_size_s=config.perm_block_s,
        alphas=config.alphas,
        use_gpu=config.use_gpu,
    )

    folds = blocked_cv_folds(len(y), config.bin_s, cv_cfg)
    additive_r2 = compute_cv_r2_fixed_alpha(
        design.additive_matrix,
        y,
        folds,
        float(uv.alpha_selected),
        use_gpu=config.use_gpu,
    )
    delta_r2_interaction = float(uv.full_r2 - additive_r2)
    result = uv.to_dict()

    interaction_unique = float(result.get("unique_motif_x_tier_shuffle", 0.0))
    interaction_p = result.get("p_motif_x_tier_shuffle")
    sig_interaction = bool(
        uv.full_r2 >= config.r2_threshold
        and interaction_unique >= config.effect_threshold
        and interaction_p is not None
        and interaction_p < config.p_threshold
    )

    result.update(
        {
            "unit_id": unit_id,
            "session_id": session_id,
            "unit_kind": unit_kind,
            "unit_bank": unit_bank,
            "n_bins": len(y),
            "r2_additive": float(additive_r2),
            "delta_r2_interaction": delta_r2_interaction,
            "sig_motif_x_tier": sig_interaction,
            "r2_threshold_passed": bool(uv.full_r2 >= config.r2_threshold),
            "n_retained_motifs": len(design.retained_motifs),
            "n_retained_cells": len(design.retained_cells),
        }
    )
    return result


def summarize_interaction_results(
    results_df: pd.DataFrame,
    design: InteractionDesign,
    session_id: str,
) -> Dict[str, Any]:
    """Create a compact session-level summary."""
    if results_df.empty:
        return {
            "session_id": session_id,
            "status": "no_units",
            "retained_motifs": design.retained_motifs,
            "retained_cells": [
                {"motif_id": int(mid), "tier": int(tier)}
                for mid, tier in design.retained_cells
            ],
            "occupancy": design.occupancy_records(),
        }

    summary = {
        "session_id": session_id,
        "n_units": int(len(results_df)),
        "n_retained_motifs": int(len(design.retained_motifs)),
        "n_retained_cells": int(len(design.retained_cells)),
        "retained_motifs": [int(m) for m in design.retained_motifs],
        "retained_cells": [
            {"motif_id": int(mid), "tier": int(tier)}
            for mid, tier in design.retained_cells
        ],
        "occupancy": design.occupancy_records(),
        "median_full_r2": float(results_df["full_r2"].median()),
        "median_r2_additive": float(results_df["r2_additive"].median()),
        "median_delta_r2_interaction": float(results_df["delta_r2_interaction"].median()),
        "median_unique_interaction": float(
            results_df["unique_motif_x_tier_shuffle"].median()
        ),
        "fraction_positive_delta": float(
            (results_df["delta_r2_interaction"] > 0).mean()
        ),
        "fraction_positive_unique": float(
            (results_df["unique_motif_x_tier_shuffle"] > 0).mean()
        ),
        "fraction_sig_interaction": float(results_df["sig_motif_x_tier"].mean()),
    }

    fdr_col = "p_motif_x_tier_fdr"
    sig_fdr_col = "sig_motif_x_tier_fdr"
    if fdr_col in results_df.columns:
        summary["median_p_interaction_fdr"] = float(results_df[fdr_col].median())
    if sig_fdr_col in results_df.columns:
        summary["fraction_sig_interaction_fdr"] = float(results_df[sig_fdr_col].mean())

    return summary


def finalize_results(
    results_df: pd.DataFrame,
    config: InteractionConfig,
) -> pd.DataFrame:
    """Apply FDR correction and keep a stable output schema."""
    if results_df.empty:
        return results_df

    df = apply_fdr_correction(
        results_df,
        p_col="p_motif_x_tier_shuffle",
        alpha=config.p_threshold,
    )
    if "sig_motif_x_tier_shuffle_fdr" in df.columns:
        df = df.rename(
            columns={"sig_motif_x_tier_shuffle_fdr": "sig_motif_x_tier_fdr"}
        )
    if "p_motif_x_tier_shuffle_fdr" in df.columns:
        df = df.rename(
            columns={"p_motif_x_tier_shuffle_fdr": "p_motif_x_tier_fdr"}
        )
    return df


__all__ = [
    "HEIGHT_THRESHOLDS",
    "InteractionConfig",
    "InteractionDesign",
    "align_motifs_to_bins",
    "build_height_tier",
    "build_interaction_design",
    "fit_unit_interaction_model",
    "summarize_interaction_results",
    "finalize_results",
]
