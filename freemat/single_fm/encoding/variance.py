"""Unique variance result containers and computation for 3-way partitioning.

Implements variance partitioning using the dropout (semi-partial R²) method
and the Testard-style regressor-shuffle method for 3-way partitioning:
VIS vs BODY_KIN vs HEAD_KIN.

Example:
    >>> from freemat.single_fm.encoding.variance import (
    ...     compute_unique_variance_3way,
    ...     UniqueVariance3WayResult,
    ... )
    >>> result = compute_unique_variance_3way(
    ...     X_vis, X_body_kin, X_head_kin, y, cv_cfg, bin_s=0.5, n_perm=1000
    ... )
    >>> print(f"Unique VIS: {result.unique_vis:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np

from freemat.single_fm.encoding.cv import CVConfig, blocked_cv_folds
from freemat.single_fm.encoding.ridge import (
    DEFAULT_ALPHAS,
    HAS_GPU,
    compute_cv_r2,
    compute_cv_r2_fixed_alpha,
    compute_cv_r2_fixed_alpha_batch,
    inner_cv_select_alpha,
)
from freemat.single_fm.encoding.stats import (
    block_permute_indices,
    permutation_test_unique_variance,
    permutation_test_unique_variance_3way,
)


@dataclass
class UniqueVarianceResult:
    """Result container for 2-way unique variance analysis (KIN vs VIS).

    Attributes:
        full_r2: CV-R² of the full model (KIN + VIS).
        kin_only_r2: CV-R² of KIN-only model.
        vis_only_r2: CV-R² of VIS-only model.
        unique_kin: R²_full - R²_vis_only (KIN contribution after VIS).
        unique_vis: R²_full - R²_kin_only (VIS contribution after KIN).
        shared: R²_full - unique_kin - unique_vis (collinear variance).
        p_kin: Permutation p-value for unique_kin (None if not computed).
        p_vis: Permutation p-value for unique_vis (None if not computed).
        null_dist_kin: Null distribution for unique_kin (if computed).
        null_dist_vis: Null distribution for unique_vis (if computed).
    """

    full_r2: float
    kin_only_r2: float
    vis_only_r2: float
    unique_kin: float
    unique_vis: float
    shared: float
    p_kin: Optional[float] = None
    p_vis: Optional[float] = None
    null_dist_kin: Optional[np.ndarray] = None
    null_dist_vis: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excludes null distributions for serialization)."""
        d = asdict(self)
        d.pop("null_dist_kin", None)
        d.pop("null_dist_vis", None)
        return d


@dataclass
class UniqueVariance3WayResult:
    """Result container for 3-way unique variance analysis (VIS vs BODY_KIN vs HEAD_KIN).

    Uses dropout method:
    - unique_vis = R²_full - R²_no_vis (VIS contribution after controlling for all KIN)
    - unique_body_kin = R²_full - R²_no_body_kin (BODY_KIN contribution)
    - unique_head_kin = R²_full - R²_no_head_kin (HEAD_KIN contribution)
    - shared_all = R²_full - unique_vis - unique_body_kin - unique_head_kin

    Attributes:
        full_r2: CV-R² of the full model (VIS + BODY_KIN + HEAD_KIN).
        vis_only_r2: CV-R² of VIS-only model.
        body_kin_only_r2: CV-R² of BODY_KIN-only model.
        head_kin_only_r2: CV-R² of HEAD_KIN-only model.
        no_vis_r2: CV-R² without VIS (BODY_KIN + HEAD_KIN).
        no_body_kin_r2: CV-R² without BODY_KIN (VIS + HEAD_KIN).
        no_head_kin_r2: CV-R² without HEAD_KIN (VIS + BODY_KIN).
        unique_vis: R²_full - R²_no_vis.
        unique_body_kin: R²_full - R²_no_body_kin.
        unique_head_kin: R²_full - R²_no_head_kin.
        shared_all: Remainder (may be negative due to CV noise).
        p_vis: Permutation p-value for unique_vis (None if not computed).
        p_body_kin: Permutation p-value for unique_body_kin (None if not computed).
        p_head_kin: Permutation p-value for unique_head_kin (None if not computed).
        null_dist_vis: Null distribution for unique_vis (if computed).
        null_dist_body_kin: Null distribution for unique_body_kin (if computed).
        null_dist_head_kin: Null distribution for unique_head_kin (if computed).
    """

    full_r2: float
    vis_only_r2: float
    body_kin_only_r2: float
    head_kin_only_r2: float
    no_vis_r2: float
    no_body_kin_r2: float
    no_head_kin_r2: float
    unique_vis: float
    unique_body_kin: float
    unique_head_kin: float
    shared_all: float
    p_vis: Optional[float] = None
    p_body_kin: Optional[float] = None
    p_head_kin: Optional[float] = None
    null_dist_vis: Optional[np.ndarray] = None
    null_dist_body_kin: Optional[np.ndarray] = None
    null_dist_head_kin: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excludes null distributions for serialization)."""
        d = asdict(self)
        d.pop("null_dist_vis", None)
        d.pop("null_dist_body_kin", None)
        d.pop("null_dist_head_kin", None)
        return d


@dataclass
class UniqueVariance3WayShuffleResult:
    """Result container for 3-way Testard-style regressor-shuffle analysis.

    Testard method (regressor-shuffle):
    - full_contrib_X: CV-R² when shuffling all groups EXCEPT X (measures total X contribution)
    - unique_X_shuffle: R²_full - CV-R²(X shuffled) (unique contribution of X)

    Also includes dropout results for comparison and permutation p-values
    from two null construction methods (block shuffle and circular shift).

    Attributes:
        full_r2: CV-R² of the full model (all features, no shuffling).

        # Dropout method results (for comparison)
        unique_vis_dropout: R²_full - R²_no_vis (VIS contribution via dropout).
        unique_body_kin_dropout: R²_full - R²_no_body_kin (BODY_KIN via dropout).
        unique_head_kin_dropout: R²_full - R²_no_head_kin (HEAD_KIN via dropout).

        # Testard shuffle method: Full contribution (shuffle others)
        full_contrib_vis: CV-R² with BODY_KIN and HEAD_KIN shuffled.
        full_contrib_body_kin: CV-R² with VIS and HEAD_KIN shuffled.
        full_contrib_head_kin: CV-R² with VIS and BODY_KIN shuffled.

        # Testard shuffle method: Unique contribution (shuffle group)
        unique_vis_shuffle: R²_full - R²(VIS shuffled).
        unique_body_kin_shuffle: R²_full - R²(BODY_KIN shuffled).
        unique_head_kin_shuffle: R²_full - R²(HEAD_KIN shuffled).

        # Block-shuffle permutation p-values
        p_vis_block: Block-shuffle null p-value for unique_vis.
        p_body_kin_block: Block-shuffle null p-value for unique_body_kin.
        p_head_kin_block: Block-shuffle null p-value for unique_head_kin.

        # Circular-shift permutation p-values
        p_vis_circular: Circular-shift null p-value for unique_vis.
        p_body_kin_circular: Circular-shift null p-value for unique_body_kin.
        p_head_kin_circular: Circular-shift null p-value for unique_head_kin.

        # Shuffle sample diagnostics (variance of estimates)
        null_95_vis: 95th percentile of shuffle samples for unique_vis.
        null_95_body_kin: 95th percentile of shuffle samples for unique_body_kin.
        null_95_head_kin: 95th percentile of shuffle samples for unique_head_kin.

        # Alpha selected from inner CV
        alpha_selected: Ridge alpha selected via inner CV on full model.
    """

    full_r2: float

    # Dropout method (for comparison/sanity check)
    unique_vis_dropout: float
    unique_body_kin_dropout: float
    unique_head_kin_dropout: float

    # Testard shuffle: full contribution (shuffle others, keep group)
    full_contrib_vis: float
    full_contrib_body_kin: float
    full_contrib_head_kin: float

    # Testard shuffle: unique contribution (shuffle group)
    unique_vis_shuffle: float
    unique_body_kin_shuffle: float
    unique_head_kin_shuffle: float

    # Block-shuffle permutation p-values
    p_vis_block: Optional[float] = None
    p_body_kin_block: Optional[float] = None
    p_head_kin_block: Optional[float] = None

    # Shuffle-sample empirical p-values (one-sided test: unique > 0)
    # Computed from n_perm shuffle samples; p = (count <= 0 + 1) / (n + 1)
    p_vis_shuffle: Optional[float] = None
    p_body_kin_shuffle: Optional[float] = None
    p_head_kin_shuffle: Optional[float] = None

    # Circular-shift permutation p-values (legacy, requires n_perm_null > 0)
    p_vis_circular: Optional[float] = None
    p_body_kin_circular: Optional[float] = None
    p_head_kin_circular: Optional[float] = None

    # Null 95th percentiles from shuffle samples
    null_95_vis: Optional[float] = None
    null_95_body_kin: Optional[float] = None
    null_95_head_kin: Optional[float] = None

    # Alpha selected from inner CV
    alpha_selected: Optional[float] = None

    # Full-model CV predictions (for residual analysis)
    full_model_predictions: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame row (excludes large arrays)."""
        d = asdict(self)
        d.pop("full_model_predictions", None)
        return d


def compute_unique_variance_3way(
    X_vis: np.ndarray,
    X_body_kin: np.ndarray,
    X_head_kin: np.ndarray,
    y: np.ndarray,
    cv_cfg: CVConfig,
    bin_s: float,
    n_perm: int = 0,
    perm_block_s: float = 10.0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
) -> UniqueVariance3WayResult:
    """Compute unique variance for 3-way partitioning: VIS vs BODY_KIN vs HEAD_KIN.

    Performs 7 model fits per unit:
    - Full model (all features)
    - 3 dropout models (no_vis, no_body_kin, no_head_kin)
    - 3 single-family models (vis_only, body_kin_only, head_kin_only) for diagnostics

    Args:
        X_vis: VIS feature matrix (T, D_vis).
        X_body_kin: BODY_KIN feature matrix (T, D_body).
        X_head_kin: HEAD_KIN feature matrix (T, D_head).
        y: Neural response vector (T,).
        cv_cfg: Cross-validation configuration.
        bin_s: Bin size in seconds.
        n_perm: Number of permutations for significance testing (0 to skip).
        perm_block_s: Block size for permutation in seconds.
        alphas: Ridge alpha grid for inner CV.
        rng: Random number generator for permutation test.
        use_gpu: Whether to use GPU acceleration (requires CuPy).

    Returns:
        UniqueVariance3WayResult with variance partitioning.
    """
    X_vis = np.asarray(X_vis, dtype=np.float64)
    X_body_kin = np.asarray(X_body_kin, dtype=np.float64)
    X_head_kin = np.asarray(X_head_kin, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    T = len(y)
    folds = blocked_cv_folds(T, bin_s, cv_cfg)

    # Full model
    X_full = np.hstack([X_vis, X_body_kin, X_head_kin])
    r2_full = compute_cv_r2(X_full, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)

    # Dropout models (for unique variance computation)
    X_no_vis = np.hstack([X_body_kin, X_head_kin])
    X_no_body_kin = np.hstack([X_vis, X_head_kin])
    X_no_head_kin = np.hstack([X_vis, X_body_kin])

    r2_no_vis = compute_cv_r2(X_no_vis, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
    r2_no_body_kin = compute_cv_r2(X_no_body_kin, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
    r2_no_head_kin = compute_cv_r2(X_no_head_kin, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)

    # Single-family models (for diagnostics)
    r2_vis_only = compute_cv_r2(X_vis, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
    r2_body_kin_only = compute_cv_r2(X_body_kin, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
    r2_head_kin_only = compute_cv_r2(X_head_kin, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)

    # Unique variance (dropout method)
    unique_vis = r2_full - r2_no_vis
    unique_body_kin = r2_full - r2_no_body_kin
    unique_head_kin = r2_full - r2_no_head_kin
    shared_all = r2_full - unique_vis - unique_body_kin - unique_head_kin

    # Permutation test if requested
    p_vis, p_body_kin, p_head_kin = None, None, None
    null_dist_vis, null_dist_body_kin, null_dist_head_kin = None, None, None

    if n_perm > 0:
        # Pre-compute optimal alpha ONCE on full model for permutation test
        inner_cfg = CVConfig(folds=3, block_s=30.0, gap_s=2.0)
        inner_folds = blocked_cv_folds(T, bin_s, inner_cfg)
        fixed_alpha = inner_cv_select_alpha(X_full, y, inner_folds, alphas, use_gpu=use_gpu)

        (
            p_vis, p_body_kin, p_head_kin,
            null_dist_vis, null_dist_body_kin, null_dist_head_kin
        ) = permutation_test_unique_variance_3way(
            X_vis, X_body_kin, X_head_kin, y, folds, bin_s,
            unique_vis, unique_body_kin, unique_head_kin,
            n_perm=n_perm,
            block_s=perm_block_s,
            rng=rng,
            use_gpu=use_gpu,
            fixed_alpha=fixed_alpha,
        )

    return UniqueVariance3WayResult(
        full_r2=r2_full,
        vis_only_r2=r2_vis_only,
        body_kin_only_r2=r2_body_kin_only,
        head_kin_only_r2=r2_head_kin_only,
        no_vis_r2=r2_no_vis,
        no_body_kin_r2=r2_no_body_kin,
        no_head_kin_r2=r2_no_head_kin,
        unique_vis=unique_vis,
        unique_body_kin=unique_body_kin,
        unique_head_kin=unique_head_kin,
        shared_all=shared_all,
        p_vis=p_vis,
        p_body_kin=p_body_kin,
        p_head_kin=p_head_kin,
        null_dist_vis=null_dist_vis,
        null_dist_body_kin=null_dist_body_kin,
        null_dist_head_kin=null_dist_head_kin,
    )


def compute_unique_variance_3way_shuffle(
    X_vis: np.ndarray,
    X_body_kin: np.ndarray,
    X_head_kin: np.ndarray,
    y: np.ndarray,
    cv_cfg: CVConfig,
    bin_s: float,
    n_perm: int = 100,
    n_perm_null: int = 0,
    block_size_s: float = 10.0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
) -> UniqueVariance3WayShuffleResult:
    """Compute Testard-style regressor-shuffle unique variance (3-way).

    .. deprecated:: Use compute_unique_variance_nway_shuffle instead.

    Testard et al. 2024 (Fig 3i) approach:
    - Full contribution: Fit model with OTHER groups shuffled, keeping target group intact.
      This measures what the target group can explain when others are noise.
    - Unique contribution: R²_full - R²(target group shuffled).
      This measures what is lost by destroying the target group's temporal structure.

    Also computes dropout method for comparison (sanity check).

    When n_perm_null > 0, runs permutation tests using both block-shuffle and
    circular-shift null construction methods to compute proper p-values.

    Args:
        X_vis: VIS feature matrix (T, D_vis).
        X_body_kin: BODY_KIN feature matrix (T, D_body).
        X_head_kin: HEAD_KIN feature matrix (T, D_head).
        y: Neural response vector (T,).
        cv_cfg: Cross-validation configuration.
        bin_s: Bin size in seconds.
        n_perm: Number of shuffle iterations for averaging (default: 100).
        n_perm_null: Number of permutations for null distributions / p-values (0 to skip).
        block_size_s: Block size for block-shuffle in seconds.
        alphas: Ridge alpha grid for inner CV.
        rng: Random number generator.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        UniqueVariance3WayShuffleResult with full and unique contributions.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    X_vis = np.asarray(X_vis, dtype=np.float64)
    X_body_kin = np.asarray(X_body_kin, dtype=np.float64)
    X_head_kin = np.asarray(X_head_kin, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    T = len(y)
    folds = blocked_cv_folds(T, bin_s, cv_cfg)
    block_len = max(1, int(round(block_size_s / bin_s)))

    # ==========================================================================
    # Step 1: Select fixed alpha and compute full model R²
    # ==========================================================================
    X_full = np.hstack([X_vis, X_body_kin, X_head_kin])

    # Select alpha once on full model (Testard approach: one alpha for all models)
    inner_cfg = CVConfig(folds=3, block_s=30.0, gap_s=2.0)
    inner_folds = blocked_cv_folds(T, bin_s, inner_cfg)
    fixed_alpha = inner_cv_select_alpha(X_full, y, inner_folds, alphas, use_gpu=use_gpu)

    # Compute r2_full with the SAME fixed alpha used for shuffle models
    # (ensures unique = r2_full - r2_shuffled uses a consistent alpha regime)
    r2_full, full_preds = compute_cv_r2_fixed_alpha(
        X_full, y, folds, fixed_alpha, use_gpu=use_gpu, return_predictions=True,
    )

    # ==========================================================================
    # Step 2: Dropout method (for comparison, uses same fixed_alpha)
    # ==========================================================================
    X_no_vis = np.hstack([X_body_kin, X_head_kin])
    X_no_body_kin = np.hstack([X_vis, X_head_kin])
    X_no_head_kin = np.hstack([X_vis, X_body_kin])

    r2_no_vis = compute_cv_r2_fixed_alpha(X_no_vis, y, folds, fixed_alpha, use_gpu=use_gpu)
    r2_no_body_kin = compute_cv_r2_fixed_alpha(X_no_body_kin, y, folds, fixed_alpha, use_gpu=use_gpu)
    r2_no_head_kin = compute_cv_r2_fixed_alpha(X_no_head_kin, y, folds, fixed_alpha, use_gpu=use_gpu)

    unique_vis_dropout = r2_full - r2_no_vis
    unique_body_kin_dropout = r2_full - r2_no_body_kin
    unique_head_kin_dropout = r2_full - r2_no_head_kin

    # ==========================================================================
    # Step 3: Testard shuffle method - Full contribution (shuffle OTHERS)
    # ==========================================================================
    D_total = X_full.shape[1]
    _use_batch = use_gpu and HAS_GPU and n_perm >= 4

    if _use_batch:
        # --- Batch GPU path: build all permuted matrices, solve in one call ---
        # VIS full contribution: VIS intact, body+head shuffled
        X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
        for i in range(n_perm):
            pb = block_permute_indices(T, block_len, rng)
            ph = block_permute_indices(T, block_len, rng)
            X_batch[i] = np.hstack([X_vis, X_body_kin[pb], X_head_kin[ph]])
        full_contrib_vis_samples = compute_cv_r2_fixed_alpha_batch(
            X_batch, y, folds, fixed_alpha,
        ).tolist()
        del X_batch

        # BODY_KIN full contribution: BODY_KIN intact, vis+head shuffled
        X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
        for i in range(n_perm):
            pv = block_permute_indices(T, block_len, rng)
            ph = block_permute_indices(T, block_len, rng)
            X_batch[i] = np.hstack([X_vis[pv], X_body_kin, X_head_kin[ph]])
        full_contrib_body_kin_samples = compute_cv_r2_fixed_alpha_batch(
            X_batch, y, folds, fixed_alpha,
        ).tolist()
        del X_batch

        # HEAD_KIN full contribution: HEAD_KIN intact, vis+body shuffled
        X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
        for i in range(n_perm):
            pv = block_permute_indices(T, block_len, rng)
            pb = block_permute_indices(T, block_len, rng)
            X_batch[i] = np.hstack([X_vis[pv], X_body_kin[pb], X_head_kin])
        full_contrib_head_kin_samples = compute_cv_r2_fixed_alpha_batch(
            X_batch, y, folds, fixed_alpha,
        ).tolist()
        del X_batch
    else:
        # --- Sequential path (CPU or small n_perm) ---
        full_contrib_vis_samples = []
        full_contrib_body_kin_samples = []
        full_contrib_head_kin_samples = []

        for _ in range(n_perm):
            perm_body = block_permute_indices(T, block_len, rng)
            perm_head = block_permute_indices(T, block_len, rng)
            X_full_vis_intact = np.hstack([X_vis, X_body_kin[perm_body], X_head_kin[perm_head]])
            r2_vis_full = compute_cv_r2_fixed_alpha(X_full_vis_intact, y, folds, fixed_alpha, use_gpu=use_gpu)
            full_contrib_vis_samples.append(r2_vis_full)

            perm_vis = block_permute_indices(T, block_len, rng)
            perm_head = block_permute_indices(T, block_len, rng)
            X_full_body_intact = np.hstack([X_vis[perm_vis], X_body_kin, X_head_kin[perm_head]])
            r2_body_full = compute_cv_r2_fixed_alpha(X_full_body_intact, y, folds, fixed_alpha, use_gpu=use_gpu)
            full_contrib_body_kin_samples.append(r2_body_full)

            perm_vis = block_permute_indices(T, block_len, rng)
            perm_body = block_permute_indices(T, block_len, rng)
            X_full_head_intact = np.hstack([X_vis[perm_vis], X_body_kin[perm_body], X_head_kin])
            r2_head_full = compute_cv_r2_fixed_alpha(X_full_head_intact, y, folds, fixed_alpha, use_gpu=use_gpu)
            full_contrib_head_kin_samples.append(r2_head_full)

    full_contrib_vis = float(np.mean(full_contrib_vis_samples))
    full_contrib_body_kin = float(np.mean(full_contrib_body_kin_samples))
    full_contrib_head_kin = float(np.mean(full_contrib_head_kin_samples))

    # ==========================================================================
    # Step 4: Testard shuffle method - Unique contribution (shuffle TARGET group)
    # ==========================================================================
    if _use_batch:
        # --- Batch GPU path ---
        # Unique VIS: shuffle VIS, keep others intact
        X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
        for i in range(n_perm):
            pv = block_permute_indices(T, block_len, rng)
            X_batch[i] = np.hstack([X_vis[pv], X_body_kin, X_head_kin])
        r2_vis_shuffled = compute_cv_r2_fixed_alpha_batch(
            X_batch, y, folds, fixed_alpha,
        )
        unique_vis_shuffle_samples = (r2_full - r2_vis_shuffled).tolist()
        del X_batch

        # Unique BODY_KIN: shuffle BODY_KIN, keep others intact
        X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
        for i in range(n_perm):
            pb = block_permute_indices(T, block_len, rng)
            X_batch[i] = np.hstack([X_vis, X_body_kin[pb], X_head_kin])
        r2_body_shuffled = compute_cv_r2_fixed_alpha_batch(
            X_batch, y, folds, fixed_alpha,
        )
        unique_body_kin_shuffle_samples = (r2_full - r2_body_shuffled).tolist()
        del X_batch

        # Unique HEAD_KIN: shuffle HEAD_KIN, keep others intact
        X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
        for i in range(n_perm):
            ph = block_permute_indices(T, block_len, rng)
            X_batch[i] = np.hstack([X_vis, X_body_kin, X_head_kin[ph]])
        r2_head_shuffled = compute_cv_r2_fixed_alpha_batch(
            X_batch, y, folds, fixed_alpha,
        )
        unique_head_kin_shuffle_samples = (r2_full - r2_head_shuffled).tolist()
        del X_batch
    else:
        # --- Sequential path ---
        unique_vis_shuffle_samples = []
        unique_body_kin_shuffle_samples = []
        unique_head_kin_shuffle_samples = []

        for _ in range(n_perm):
            perm_vis = block_permute_indices(T, block_len, rng)
            X_vis_shuffled = np.hstack([X_vis[perm_vis], X_body_kin, X_head_kin])
            r2_vis_shuffled = compute_cv_r2_fixed_alpha(X_vis_shuffled, y, folds, fixed_alpha, use_gpu=use_gpu)
            unique_vis_shuffle_samples.append(r2_full - r2_vis_shuffled)

            perm_body = block_permute_indices(T, block_len, rng)
            X_body_shuffled = np.hstack([X_vis, X_body_kin[perm_body], X_head_kin])
            r2_body_shuffled = compute_cv_r2_fixed_alpha(X_body_shuffled, y, folds, fixed_alpha, use_gpu=use_gpu)
            unique_body_kin_shuffle_samples.append(r2_full - r2_body_shuffled)

            perm_head = block_permute_indices(T, block_len, rng)
            X_head_shuffled = np.hstack([X_vis, X_body_kin, X_head_kin[perm_head]])
            r2_head_shuffled = compute_cv_r2_fixed_alpha(X_head_shuffled, y, folds, fixed_alpha, use_gpu=use_gpu)
            unique_head_kin_shuffle_samples.append(r2_full - r2_head_shuffled)

    unique_vis_shuffle = float(np.mean(unique_vis_shuffle_samples))
    unique_body_kin_shuffle = float(np.mean(unique_body_kin_shuffle_samples))
    unique_head_kin_shuffle = float(np.mean(unique_head_kin_shuffle_samples))

    # Empirical p-values from shuffle samples (one-sided test: unique > 0).
    # Under H0 the unique contribution is ~0; under H1 it is reliably > 0.
    _n = len(unique_vis_shuffle_samples)
    _vis_arr = np.asarray(unique_vis_shuffle_samples)
    _body_arr = np.asarray(unique_body_kin_shuffle_samples)
    _head_arr = np.asarray(unique_head_kin_shuffle_samples)
    p_vis_shuffle = float((np.sum(_vis_arr <= 0) + 1) / (_n + 1))
    p_body_kin_shuffle = float((np.sum(_body_arr <= 0) + 1) / (_n + 1))
    p_head_kin_shuffle = float((np.sum(_head_arr <= 0) + 1) / (_n + 1))

    # ==========================================================================
    # Step 5: Diagnostics from shuffle samples (variance of estimates)
    # ==========================================================================
    null_95_vis = float(np.percentile(unique_vis_shuffle_samples, 95))
    null_95_body_kin = float(np.percentile(unique_body_kin_shuffle_samples, 95))
    null_95_head_kin = float(np.percentile(unique_head_kin_shuffle_samples, 95))

    # ==========================================================================
    # Step 6: Permutation p-values (block + circular null methods)
    # ==========================================================================
    p_vis_block, p_body_kin_block, p_head_kin_block = None, None, None
    p_vis_circular, p_body_kin_circular, p_head_kin_circular = None, None, None

    if n_perm_null > 0:
        # Use a single-shot shuffle unique (first iteration) as the observed
        # statistic.  The null distribution is also built from single-shot
        # shuffle realizations (one permutation per null sample), so observed
        # and null are the *same estimator* — a matched comparison.
        #
        # The averaged shuffle unique (unique_*_shuffle) is reported
        # separately as a more stable effect-size estimate, but the p-value
        # must use a single-shot observed to match the single-shot null.
        obs_vis = unique_vis_shuffle_samples[0]
        obs_body = unique_body_kin_shuffle_samples[0]
        obs_head = unique_head_kin_shuffle_samples[0]

        # Block-shuffle null
        rng_block = np.random.default_rng(rng.integers(0, 2**32))
        (
            p_vis_block, p_body_kin_block, p_head_kin_block,
            _, _, _,
        ) = permutation_test_unique_variance_3way(
            X_vis, X_body_kin, X_head_kin, y, folds, bin_s,
            obs_vis, obs_body, obs_head,
            n_perm=n_perm_null,
            block_s=block_size_s,
            rng=rng_block,
            use_gpu=use_gpu,
            fixed_alpha=fixed_alpha,
            method="block",
            r2_full=r2_full,
        )

        # Circular-shift null
        rng_circ = np.random.default_rng(rng.integers(0, 2**32))
        (
            p_vis_circular, p_body_kin_circular, p_head_kin_circular,
            _, _, _,
        ) = permutation_test_unique_variance_3way(
            X_vis, X_body_kin, X_head_kin, y, folds, bin_s,
            obs_vis, obs_body, obs_head,
            n_perm=n_perm_null,
            block_s=block_size_s,
            rng=rng_circ,
            use_gpu=use_gpu,
            fixed_alpha=fixed_alpha,
            method="circular",
            r2_full=r2_full,
        )

    return UniqueVariance3WayShuffleResult(
        full_r2=r2_full,
        unique_vis_dropout=unique_vis_dropout,
        unique_body_kin_dropout=unique_body_kin_dropout,
        unique_head_kin_dropout=unique_head_kin_dropout,
        full_contrib_vis=full_contrib_vis,
        full_contrib_body_kin=full_contrib_body_kin,
        full_contrib_head_kin=full_contrib_head_kin,
        unique_vis_shuffle=unique_vis_shuffle,
        unique_body_kin_shuffle=unique_body_kin_shuffle,
        unique_head_kin_shuffle=unique_head_kin_shuffle,
        p_vis_shuffle=p_vis_shuffle,
        p_body_kin_shuffle=p_body_kin_shuffle,
        p_head_kin_shuffle=p_head_kin_shuffle,
        p_vis_block=p_vis_block,
        p_body_kin_block=p_body_kin_block,
        p_head_kin_block=p_head_kin_block,
        p_vis_circular=p_vis_circular,
        p_body_kin_circular=p_body_kin_circular,
        p_head_kin_circular=p_head_kin_circular,
        null_95_vis=null_95_vis,
        null_95_body_kin=null_95_body_kin,
        null_95_head_kin=null_95_head_kin,
        alpha_selected=fixed_alpha,
        full_model_predictions=full_preds,
    )


# =============================================================================
# N-way generalization (replaces 3-way for new analyses)
# =============================================================================


@dataclass
class UniqueVarianceNWayShuffleResult:
    """Result container for N-way Testard-style regressor-shuffle analysis.

    Dict-keyed fields allow any number of feature groups. The to_dict() method
    flattens to column-per-group format (e.g., unique_vis_shuffle,
    full_contrib_pos) for backward-compatible DataFrame serialization.

    Attributes:
        full_r2: CV-R2 of the full model (all features, no shuffling).
        unique_dropout: Per-group unique variance via dropout method.
        full_contrib: Per-group full contribution (others shuffled).
        unique_shuffle: Per-group unique contribution (target shuffled).
        p_shuffle: Per-group empirical p-values from shuffle samples.
        p_block: Per-group block-shuffle null p-values.
        p_circular: Per-group circular-shift null p-values.
        null_95: Per-group 95th percentile of shuffle samples.
        alpha_selected: Ridge alpha selected via inner CV on full model.
        full_model_predictions: CV predictions for residual analysis.
        group_names: Ordered list of group names (for consistent flattening).
    """

    full_r2: float
    unique_dropout: Dict[str, float]
    full_contrib: Dict[str, float]
    unique_shuffle: Dict[str, float]
    p_shuffle: Dict[str, Optional[float]]
    p_block: Dict[str, Optional[float]]
    p_circular: Dict[str, Optional[float]]
    null_95: Dict[str, Optional[float]]
    alpha_selected: Optional[float] = None
    full_model_predictions: Optional[Dict] = None
    group_names: Optional[list] = None

    def to_dict(self) -> Dict:
        """Flatten to column-per-group format for DataFrame compatibility.

        Produces columns like: unique_vis_dropout, unique_vis_shuffle,
        full_contrib_vis, p_vis_shuffle, p_vis_block, null_95_vis, etc.
        """
        d: Dict = {"full_r2": self.full_r2}
        names = self.group_names or sorted(self.unique_dropout.keys())
        for g in names:
            d[f"unique_{g}_dropout"] = self.unique_dropout.get(g)
            d[f"full_contrib_{g}"] = self.full_contrib.get(g)
            d[f"unique_{g}_shuffle"] = self.unique_shuffle.get(g)
            d[f"p_{g}_shuffle"] = self.p_shuffle.get(g)
            d[f"p_{g}_block"] = self.p_block.get(g)
            d[f"p_{g}_circular"] = self.p_circular.get(g)
            d[f"null_95_{g}"] = self.null_95.get(g)
        d["alpha_selected"] = self.alpha_selected
        return d


def compute_unique_variance_nway_shuffle(
    groups: Dict[str, np.ndarray],
    y: np.ndarray,
    cv_cfg: CVConfig,
    bin_s: float,
    n_perm: int = 100,
    n_perm_null: int = 0,
    block_size_s: float = 10.0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
) -> UniqueVarianceNWayShuffleResult:
    """Compute Testard-style regressor-shuffle unique variance (N-way).

    Generalized version of compute_unique_variance_3way_shuffle that accepts
    an arbitrary number of feature groups via a dict.

    Testard et al. 2024 (Fig 3i) approach:
    - Full contribution: Fit model with OTHER groups shuffled, target intact.
    - Unique contribution: R2_full - R2(target shuffled).

    Args:
        groups: Dict mapping group name to feature matrix, e.g.
            {"vis": X_vis, "body_kin": X_body_kin, "head_kin": X_head_kin,
             "pos": X_pos}. Each array has shape (T, D_group).
        y: Neural response vector (T,).
        cv_cfg: Cross-validation configuration.
        bin_s: Bin size in seconds.
        n_perm: Number of shuffle iterations for averaging (default: 100).
        n_perm_null: Number of permutations for null distributions (0 to skip).
        block_size_s: Block size for block-shuffle in seconds.
        alphas: Ridge alpha grid for inner CV.
        rng: Random number generator.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        UniqueVarianceNWayShuffleResult with full and unique contributions.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    group_names = list(groups.keys())
    group_arrays = {
        g: np.asarray(groups[g], dtype=np.float64) for g in group_names
    }
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    T = len(y)
    folds = blocked_cv_folds(T, bin_s, cv_cfg)
    block_len = max(1, int(round(block_size_s / bin_s)))

    # ======================================================================
    # Step 1: Select fixed alpha and compute full model R2
    # ======================================================================
    X_full = np.hstack([group_arrays[g] for g in group_names])
    D_total = X_full.shape[1]

    inner_cfg = CVConfig(folds=3, block_s=30.0, gap_s=2.0)
    inner_folds = blocked_cv_folds(T, bin_s, inner_cfg)
    fixed_alpha = inner_cv_select_alpha(
        X_full, y, inner_folds, alphas, use_gpu=use_gpu,
    )

    r2_full, full_preds = compute_cv_r2_fixed_alpha(
        X_full, y, folds, fixed_alpha, use_gpu=use_gpu, return_predictions=True,
    )

    # ======================================================================
    # Step 2: Dropout method (for comparison, uses same fixed_alpha)
    # ======================================================================
    unique_dropout: Dict[str, float] = {}
    for g in group_names:
        X_no_g = np.hstack([group_arrays[o] for o in group_names if o != g])
        r2_no_g = compute_cv_r2_fixed_alpha(
            X_no_g, y, folds, fixed_alpha, use_gpu=use_gpu,
        )
        unique_dropout[g] = r2_full - r2_no_g

    # ======================================================================
    # Step 3: Testard shuffle - Full contribution (shuffle OTHERS)
    # ======================================================================
    _use_batch = use_gpu and HAS_GPU and n_perm >= 4
    full_contrib_samples: Dict[str, list] = {}

    for g in group_names:
        if _use_batch:
            X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
            for i in range(n_perm):
                cols = []
                for o in group_names:
                    if o == g:
                        cols.append(group_arrays[o])
                    else:
                        perm = block_permute_indices(T, block_len, rng)
                        cols.append(group_arrays[o][perm])
                X_batch[i] = np.hstack(cols)
            samples = compute_cv_r2_fixed_alpha_batch(
                X_batch, y, folds, fixed_alpha,
            ).tolist()
            del X_batch
        else:
            samples = []
            for _ in range(n_perm):
                cols = []
                for o in group_names:
                    if o == g:
                        cols.append(group_arrays[o])
                    else:
                        perm = block_permute_indices(T, block_len, rng)
                        cols.append(group_arrays[o][perm])
                X_shuffled = np.hstack(cols)
                r2 = compute_cv_r2_fixed_alpha(
                    X_shuffled, y, folds, fixed_alpha, use_gpu=use_gpu,
                )
                samples.append(r2)
        full_contrib_samples[g] = samples

    full_contrib = {g: float(np.mean(s)) for g, s in full_contrib_samples.items()}

    # ======================================================================
    # Step 4: Testard shuffle - Unique contribution (shuffle TARGET)
    # ======================================================================
    unique_shuffle_samples: Dict[str, list] = {}

    for g in group_names:
        if _use_batch:
            X_batch = np.empty((n_perm, T, D_total), dtype=np.float64)
            for i in range(n_perm):
                cols = []
                for o in group_names:
                    if o == g:
                        perm = block_permute_indices(T, block_len, rng)
                        cols.append(group_arrays[o][perm])
                    else:
                        cols.append(group_arrays[o])
                X_batch[i] = np.hstack(cols)
            r2_shuffled = compute_cv_r2_fixed_alpha_batch(
                X_batch, y, folds, fixed_alpha,
            )
            samples = (r2_full - r2_shuffled).tolist()
            del X_batch
        else:
            samples = []
            for _ in range(n_perm):
                cols = []
                for o in group_names:
                    if o == g:
                        perm = block_permute_indices(T, block_len, rng)
                        cols.append(group_arrays[o][perm])
                    else:
                        cols.append(group_arrays[o])
                X_shuffled = np.hstack(cols)
                r2 = compute_cv_r2_fixed_alpha(
                    X_shuffled, y, folds, fixed_alpha, use_gpu=use_gpu,
                )
                samples.append(r2_full - r2)
        unique_shuffle_samples[g] = samples

    unique_shuffle = {
        g: float(np.mean(s)) for g, s in unique_shuffle_samples.items()
    }

    # ======================================================================
    # Step 5: Empirical p-values from shuffle samples
    # ======================================================================
    p_shuffle: Dict[str, Optional[float]] = {}
    null_95: Dict[str, Optional[float]] = {}
    for g in group_names:
        arr = np.asarray(unique_shuffle_samples[g])
        _n = len(arr)
        p_shuffle[g] = float((np.sum(arr <= 0) + 1) / (_n + 1))
        null_95[g] = float(np.percentile(arr, 95))

    # ======================================================================
    # Step 6: Permutation p-values (block + circular null)
    # ======================================================================
    p_block: Dict[str, Optional[float]] = {g: None for g in group_names}
    p_circular: Dict[str, Optional[float]] = {g: None for g in group_names}

    if n_perm_null > 0:
        # Use single-shot observed (first shuffle sample) matched to null
        obs = {g: unique_shuffle_samples[g][0] for g in group_names}

        for method_name, p_dict in [("block", p_block), ("circular", p_circular)]:
            rng_perm = np.random.default_rng(rng.integers(0, 2**32))
            for g in group_names:
                null_dist = []
                for _ in range(n_perm_null):
                    cols = []
                    for o in group_names:
                        if o == g:
                            perm = block_permute_indices(T, block_len, rng_perm)
                            cols.append(group_arrays[o][perm])
                        else:
                            cols.append(group_arrays[o])
                    X_null = np.hstack(cols)
                    r2_null = compute_cv_r2_fixed_alpha(
                        X_null, y, folds, fixed_alpha, use_gpu=use_gpu,
                    )
                    null_dist.append(r2_full - r2_null)
                null_arr = np.asarray(null_dist)
                p_dict[g] = float(
                    (np.sum(null_arr >= obs[g]) + 1) / (n_perm_null + 1)
                )

    return UniqueVarianceNWayShuffleResult(
        full_r2=r2_full,
        unique_dropout=unique_dropout,
        full_contrib=full_contrib,
        unique_shuffle=unique_shuffle,
        p_shuffle=p_shuffle,
        p_block=p_block,
        p_circular=p_circular,
        null_95=null_95,
        alpha_selected=fixed_alpha,
        full_model_predictions=full_preds,
        group_names=group_names,
    )


def compute_unique_variance_kin_vis(
    X_vis: np.ndarray,
    X_kin: np.ndarray,
    y: np.ndarray,
    cv_cfg: CVConfig,
    bin_s: float,
    n_perm: int = 0,
    perm_block_s: float = 10.0,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
) -> UniqueVarianceResult:
    """Compute unique variance for 2-way KIN vs VIS predictor families.

    Args:
        X_vis: VIS feature matrix (T, D_vis).
        X_kin: KIN feature matrix (T, D_kin).
        y: Neural response vector (T,).
        cv_cfg: Cross-validation configuration.
        bin_s: Bin size in seconds.
        n_perm: Number of permutations for significance testing (0 to skip).
        perm_block_s: Block size for permutation in seconds.
        alphas: Ridge alpha grid for inner CV.
        rng: Random number generator for permutation test.
        use_gpu: Whether to use GPU acceleration (requires CuPy).

    Returns:
        UniqueVarianceResult with variance partitioning.
    """
    X_vis = np.asarray(X_vis, dtype=np.float64)
    X_kin = np.asarray(X_kin, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    T = len(y)
    folds = blocked_cv_folds(T, bin_s, cv_cfg)

    X_full = np.hstack([X_vis, X_kin])
    r2_full = compute_cv_r2(X_full, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
    r2_kin_only = compute_cv_r2(X_kin, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)
    r2_vis_only = compute_cv_r2(X_vis, y, folds, alphas, bin_s=bin_s, use_gpu=use_gpu)

    unique_kin = r2_full - r2_vis_only
    unique_vis = r2_full - r2_kin_only
    shared = r2_full - unique_kin - unique_vis

    p_kin, p_vis = None, None
    null_dist_kin, null_dist_vis = None, None

    if n_perm > 0:
        inner_cfg = CVConfig(folds=3, block_s=30.0, gap_s=2.0)
        inner_folds = blocked_cv_folds(T, bin_s, inner_cfg)
        fixed_alpha = inner_cv_select_alpha(X_full, y, inner_folds, alphas, use_gpu=use_gpu)

        p_kin, p_vis, null_dist_kin, null_dist_vis = permutation_test_unique_variance(
            X_vis, X_kin, y, folds, bin_s,
            unique_kin, unique_vis,
            n_perm=n_perm,
            block_s=perm_block_s,
            alphas=alphas,
            rng=rng,
            use_gpu=use_gpu,
            fixed_alpha=fixed_alpha,
        )

    return UniqueVarianceResult(
        full_r2=r2_full,
        kin_only_r2=r2_kin_only,
        vis_only_r2=r2_vis_only,
        unique_kin=unique_kin,
        unique_vis=unique_vis,
        shared=shared,
        p_kin=p_kin,
        p_vis=p_vis,
        null_dist_kin=null_dist_kin,
        null_dist_vis=null_dist_vis,
    )


__all__ = [
    "UniqueVarianceResult",
    "UniqueVariance3WayResult",
    "UniqueVariance3WayShuffleResult",
    "UniqueVarianceNWayShuffleResult",
    "compute_unique_variance_kin_vis",
    "compute_unique_variance_3way",
    "compute_unique_variance_3way_shuffle",
    "compute_unique_variance_nway_shuffle",
]
