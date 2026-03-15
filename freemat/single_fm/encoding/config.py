"""Configuration and correction utilities for Figure 2B analysis.

Provides Fig2BConfig dataclass and FDR correction for encoding results.

Example:
    >>> from freemat.single_fm.encoding.config import Fig2BConfig
    >>> config = Fig2BConfig(bin_s=0.5, n_perm=1000, use_shuffle_method=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

# Optional statsmodels for FDR correction
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from freemat.single_fm.encoding.ridge import DEFAULT_ALPHAS


@dataclass
class Fig2BConfig:
    """Configuration for Figure 2B analysis.

    Uses N-way partitioning: VIS vs BODY_KIN vs HEAD_KIN (+ optional 3D POS).

    Args:
        bin_s: Bin size in seconds.
        cv_folds: Number of CV folds (default: 10 for Testard alignment).
        cv_block_s: CV block size in seconds.
        cv_gap_s: CV embargo gap in seconds.
        n_perm: Number of shuffle iterations for Testard method averaging (default: 100).
        n_perm_null: Number of permutations for null distributions / p-values (0 to skip).
            Uses both block-shuffle and circular-shift null methods. Default: 500.
        perm_block_s: Block size for permutation in seconds.
        effect_threshold: Minimum delta R² for significance.
        p_threshold: P-value threshold for significance.
        alphas: Ridge alpha grid.
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential).
        use_gpu: Whether to use GPU acceleration (requires CuPy).
        use_2d_only: If True, use 2D floor-plane features (excludes gaze_z only).
        floor_filter: If True, filter to time bins where monkey is on the floor.
        floor_height_threshold: Max height above floor (meters) to be considered "on floor".
        floor_filter_before_features: If True, NaN-mask off-floor frames in pose BEFORE
            feature computation (derivatives will have gaps at boundaries). If False
            (default), compute features on full timeline then NaN-mask (smooth derivatives,
            avoids velocity inflation from temporal gaps).
        include_time_drift: If True, add linear time drift as control covariate.
        use_gam: If True, use GAM with splines on VIS features instead of pure Ridge.
        gam_n_knots: Number of knots for GAM splines (only used if use_gam=True).
        use_shuffle_method: If True, use Testard-style regressor-shuffle for unique variance.
        lag_s: Temporal kernel half-window in seconds (0 = no lags).
            n_lags is computed as round(lag_s / bin_s).
            For lag_s=1.0, bin_s=0.5: n_lags=2 (±1.0s, 5 taps).
            For lag_s=1.0, bin_s=0.25: n_lags=4 (±1.0s, 9 taps).
        r2_threshold: Minimum full R² for unit inclusion (0.0 to include all).
        min_bins_per_unit: Minimum number of bins required per unit (skip units with fewer).
        generate_diagnostics: Whether to generate feature diagnostic plots.
        debug: Enable debug output.
    """

    bin_s: float = 0.5
    cv_folds: int = 10  # Changed to 10 for Testard alignment (was 5)
    cv_block_s: float = 30.0
    cv_gap_s: float = 2.0
    n_perm: int = 100  # Shuffle iterations for Testard method (averaging)
    n_perm_null: int = 500  # Permutations for null distribution (p-values); 0 to skip
    perm_block_s: float = 10.0
    effect_threshold: float = 0.005  # 0.5% delta R²
    p_threshold: float = 0.05
    alphas: Tuple[float, ...] = DEFAULT_ALPHAS
    n_jobs: int = -1  # -1 = all CPUs, 1 = sequential
    use_gpu: bool = False
    use_2d_only: bool = False
    floor_filter: bool = True  # Default to on-floor filtering
    floor_height_threshold: float = 0.5  # 50cm above floor
    floor_filter_before_features: bool = False  # False = compute-then-filter (current, smooth derivatives)
    include_time_drift: bool = False
    include_rssi: bool = False  # Include RSSI as control covariate (like time_drift)
    include_antenna_distance: bool = False  # 3D distance to Deuteron antenna as nuisance
    use_gam: bool = False
    gam_n_knots: int = 5
    use_shuffle_method: bool = True  # Testard-style regressor-shuffle (vs dropout)
    lag_s: float = 1.0  # Temporal kernel half-window in seconds (0.0 to disable)
    n_lags: int = -1  # Computed from lag_s / bin_s in __post_init__ (-1 = auto)
    r2_threshold: float = 0.01  # Minimum R² for inclusion
    min_bins_per_unit: int = 150  # Minimum bins required per unit (was hardcoded 200)
    sua_only: bool = False  # If True, filter to SUA units only (exclude MUA)
    mua_only: bool = False  # If True, filter to MUA units only (exclude SUA)
    binning_method: str = 'mean'  # 'mean' (safe, no cross-gap bridging) or 'interp' (smooth)
    include_pos: bool = True  # Include 3D position (COM) as a 4th feature group
    include_motif: bool = False  # Include motif one-hot as a 5th feature group (Fig 4C)
    save_residuals: bool = False  # Save per-bin residuals + COM position for spatial analysis
    validate_coordinates: bool = True  # FAIL-STOP gate: validate features before encoding
    generate_diagnostics: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        """Compute n_lags from lag_s and bin_s if not explicitly set."""
        if self.n_lags == -1:
            self.n_lags = round(self.lag_s / self.bin_s) if self.lag_s > 0 else 0


def apply_fdr_correction(
    results_df: pd.DataFrame,
    p_col: str = "p_kin",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        results_df: DataFrame with p-value column.
        p_col: Name of the p-value column (default: "p_kin").
        alpha: Significance level for FDR correction.

    Returns:
        DataFrame with added columns for FDR-corrected p-values and significance.
    """
    df = results_df.copy()

    # Determine the suffix for output columns based on p_col
    if p_col == "p_kin":
        fdr_col = "p_kin_fdr"
        sig_col = "sig_kin_fdr"
    elif p_col == "p_vis":
        fdr_col = "p_vis_fdr"
        sig_col = "sig_vis_fdr"
    else:
        fdr_col = f"{p_col}_fdr"
        sig_col = f"{p_col.replace('p_', 'sig_')}_fdr"

    if p_col not in df.columns:
        # No p-values to correct
        df[fdr_col] = np.nan
        df[sig_col] = False
        return df

    # Get p-values, handling NaN and None (when n_perm=0)
    pvals = df[p_col].values
    # Convert None to NaN and ensure float type for isnan
    pvals = np.array([np.nan if x is None else float(x) for x in pvals])
    valid_mask = ~np.isnan(pvals)

    if not valid_mask.any():
        df[fdr_col] = np.nan
        df[sig_col] = False
        return df

    if not HAS_STATSMODELS:
        # Fallback: just copy p-values without correction
        df[fdr_col] = pvals
        df[sig_col] = pvals < alpha
        return df

    # Apply BH-FDR correction to valid p-values
    valid_pvals = pvals[valid_mask]
    rejected, pvals_corrected, _, _ = multipletests(
        valid_pvals,
        alpha=alpha,
        method="fdr_bh",
    )

    # Store results
    fdr_values = np.full(len(pvals), np.nan)
    sig_values = np.zeros(len(pvals), dtype=bool)

    fdr_values[valid_mask] = pvals_corrected
    sig_values[valid_mask] = rejected

    df[fdr_col] = fdr_values
    df[sig_col] = sig_values

    return df


__all__ = [
    "Fig2BConfig",
    "apply_fdr_correction",
]
