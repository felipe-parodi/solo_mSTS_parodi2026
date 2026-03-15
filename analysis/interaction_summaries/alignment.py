"""Inter-brain representational alignment via windowed CCA.

Alignment captures "similarity between two animals' neural state spaces over time"
using Canonical Correlation Analysis in sliding windows.

Procedure:
    1. PCA on each monkey's neural data (fit on train, apply to all)
    2. Sliding windows over time
    3. CCA within each window
    4. Alignment = mean of top K canonical correlations

Controls:
    - Pseudo-pair: Match windows across different time periods (same regime/distance)
    - Residualization: Remove controls before CCA
    - Lag test: Check that alignment peaks near lag=0

Key hypothesis: Alignment increases during coordinated interaction (grooming)
and is not explained by shared movement.

Usage:
    from analysis.interaction_summaries.alignment import (
        compute_windowed_cca_alignment,
        compute_pseudo_pair_control,
    )

    result = compute_windowed_cca_alignment(
        X_m1=neural_rates_m1,
        X_m2=neural_rates_m2,
        window_sec=10.0,
        step_sec=1.0,
        n_pca_dims=20,
        n_cca_components=5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class AlignmentResult:
    """Result of inter-brain alignment computation.

    Attributes:
        alignment: Alignment time series (one value per window center)
        window_centers: Time indices of window centers
        canonical_correlations: Top K correlations per window, shape (n_windows, K)
        n_windows: Number of windows
        window_sec: Window size in seconds
        step_sec: Step size in seconds
        n_pca_dims: PCA dimensionality used
        n_cca_components: Number of CCA components used
    """

    alignment: np.ndarray
    window_centers: np.ndarray
    canonical_correlations: np.ndarray
    n_windows: int
    window_sec: float
    step_sec: float
    n_pca_dims: int
    n_cca_components: int

    def alignment_by_regime(
        self,
        regime_labels: np.ndarray,
        bin_size_sec: float,
    ) -> Dict[str, Dict[str, float]]:
        """Get alignment statistics grouped by dyadic regime.

        Args:
            regime_labels: Regime label per time bin
            bin_size_sec: Bin size in seconds

        Returns:
            Dict mapping regime to {mean, std, count}
        """
        stats: Dict[str, List[float]] = {}

        for i, center_bin in enumerate(self.window_centers):
            if np.isnan(self.alignment[i]):
                continue

            # Get regime at window center
            if center_bin < len(regime_labels):
                regime = str(regime_labels[center_bin])
            else:
                continue

            if regime not in stats:
                stats[regime] = []
            stats[regime].append(self.alignment[i])

        return {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "count": len(v),
            }
            for k, v in stats.items()
        }


def _pca_reduce(
    X: np.ndarray,
    n_components: int,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, PCA]:
    """Reduce dimensionality with PCA.

    Args:
        X: Data matrix, shape (n_samples, n_features)
        n_components: Target dimensionality
        train_mask: Fit PCA on these samples only

    Returns:
        Tuple of (reduced data, fitted PCA object)
    """
    n_samples, n_features = X.shape
    n_components = min(n_components, n_features, n_samples)

    if train_mask is None:
        train_mask = np.ones(n_samples, dtype=bool)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[train_mask])

    # Fit PCA on training data
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)

    # Transform all data
    X_all_scaled = scaler.transform(X)
    X_reduced = pca.transform(X_all_scaled)

    return X_reduced, pca


def _cca_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 5,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Compute CCA canonical correlations between two matrices.

    Args:
        X: First data matrix, shape (n_samples, n_features_x)
        Y: Second data matrix, shape (n_samples, n_features_y)
        n_components: Number of canonical components to return
        ridge: Regularization for numerical stability

    Returns:
        Array of top K canonical correlations
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.full(n_components, np.nan)

    # Center
    X_c = X - np.mean(X, axis=0, keepdims=True)
    Y_c = Y - np.mean(Y, axis=0, keepdims=True)

    n_x, n_y = X_c.shape[1], Y_c.shape[1]
    k = min(n_components, n_x, n_y)

    if k <= 0:
        return np.full(n_components, np.nan)

    # Covariance matrices
    Sxx = (X_c.T @ X_c) / max(1, n_samples - 1) + ridge * np.eye(n_x)
    Syy = (Y_c.T @ Y_c) / max(1, n_samples - 1) + ridge * np.eye(n_y)
    Sxy = (X_c.T @ Y_c) / max(1, n_samples - 1)

    # Inverse sqrt via eigendecomposition
    eigx, Vx = np.linalg.eigh(Sxx)
    eigx = np.maximum(eigx, ridge)
    Sxx_inv_sqrt = Vx @ np.diag(1.0 / np.sqrt(eigx)) @ Vx.T

    eigy, Vy = np.linalg.eigh(Syy)
    eigy = np.maximum(eigy, ridge)
    Syy_inv_sqrt = Vy @ np.diag(1.0 / np.sqrt(eigy)) @ Vy.T

    # Canonical correlations via SVD
    K = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    _, s, _ = np.linalg.svd(K, full_matrices=False)

    # Return top K correlations (padded with NaN if needed)
    result = np.full(n_components, np.nan)
    result[:min(len(s), k)] = s[:k]

    return result


def compute_windowed_cca_alignment(
    X_m1: np.ndarray,
    X_m2: np.ndarray,
    bin_size_sec: float,
    window_sec: float = 10.0,
    step_sec: float = 1.0,
    n_pca_dims: int = 20,
    n_cca_components: int = 5,
    train_mask: Optional[np.ndarray] = None,
) -> AlignmentResult:
    """Compute inter-brain alignment using windowed CCA.

    Args:
        X_m1: Neural rates for monkey 1, shape (n_bins, n_units_1)
        X_m2: Neural rates for monkey 2, shape (n_bins, n_units_2)
        bin_size_sec: Bin size in seconds
        window_sec: Window size for CCA in seconds
        step_sec: Step size between windows in seconds
        n_pca_dims: PCA dimensionality per monkey
        n_cca_components: Number of CCA components to use
        train_mask: Boolean mask for PCA training (optional)

    Returns:
        AlignmentResult with alignment time series
    """
    n_bins = min(X_m1.shape[0], X_m2.shape[0])
    X_m1 = X_m1[:n_bins]
    X_m2 = X_m2[:n_bins]

    window_bins = int(round(window_sec / bin_size_sec))
    step_bins = int(round(step_sec / bin_size_sec))

    # PCA reduce each monkey's data
    Z_m1, _ = _pca_reduce(X_m1, n_pca_dims, train_mask)
    Z_m2, _ = _pca_reduce(X_m2, n_pca_dims, train_mask)

    # Sliding windows
    window_centers = []
    all_correlations = []

    start = window_bins // 2
    end = n_bins - window_bins // 2

    for center in range(start, end, step_bins):
        w_start = center - window_bins // 2
        w_end = center + window_bins // 2

        if w_start < 0 or w_end > n_bins:
            continue

        # Extract window data
        Z1_win = Z_m1[w_start:w_end]
        Z2_win = Z_m2[w_start:w_end]

        # Compute CCA correlations
        corrs = _cca_correlations(Z1_win, Z2_win, n_cca_components)

        window_centers.append(center)
        all_correlations.append(corrs)

    # Convert to arrays
    window_centers = np.array(window_centers)
    canonical_correlations = np.array(all_correlations)  # shape (n_windows, K)

    # Alignment = mean of top K correlations
    alignment = np.nanmean(canonical_correlations, axis=1)

    return AlignmentResult(
        alignment=alignment,
        window_centers=window_centers,
        canonical_correlations=canonical_correlations,
        n_windows=len(window_centers),
        window_sec=window_sec,
        step_sec=step_sec,
        n_pca_dims=n_pca_dims,
        n_cca_components=n_cca_components,
    )


def compute_pseudo_pair_control(
    X_m1: np.ndarray,
    X_m2: np.ndarray,
    regime_labels: np.ndarray,
    bin_size_sec: float,
    window_sec: float = 10.0,
    step_sec: float = 1.0,
    n_pca_dims: int = 20,
    n_cca_components: int = 5,
    n_pseudo_pairs: int = 100,
    min_time_shift_sec: float = 60.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute pseudo-pair control for inter-brain alignment.

    Pairs M1 windows with M2 windows from different time periods but matched
    on dyadic regime. If alignment is due to true inter-brain coupling rather
    than shared behavior, pseudo-pair alignment should be lower.

    Args:
        X_m1: Neural rates for monkey 1
        X_m2: Neural rates for monkey 2
        regime_labels: Dyadic regime label per bin
        bin_size_sec: Bin size in seconds
        window_sec: Window size for CCA
        step_sec: Step size between windows
        n_pca_dims: PCA dimensionality
        n_cca_components: Number of CCA components
        n_pseudo_pairs: Number of pseudo-pairs to sample
        min_time_shift_sec: Minimum time shift for pseudo-pairing
        random_state: Random seed

    Returns:
        Dict with 'real_alignment', 'pseudo_alignment', 'difference'
    """
    rng = np.random.default_rng(random_state)

    n_bins = min(X_m1.shape[0], X_m2.shape[0], len(regime_labels))
    X_m1 = X_m1[:n_bins]
    X_m2 = X_m2[:n_bins]
    regime_labels = regime_labels[:n_bins]

    window_bins = int(round(window_sec / bin_size_sec))
    step_bins = int(round(step_sec / bin_size_sec))
    min_shift_bins = int(round(min_time_shift_sec / bin_size_sec))

    # PCA reduce
    Z_m1, _ = _pca_reduce(X_m1, n_pca_dims)
    Z_m2, _ = _pca_reduce(X_m2, n_pca_dims)

    # Identify valid window centers and their regimes
    valid_centers = []
    center_regimes = []

    start = window_bins // 2
    end = n_bins - window_bins // 2

    for center in range(start, end, step_bins):
        regime = str(regime_labels[center])
        if regime and regime != "":
            valid_centers.append(center)
            center_regimes.append(regime)

    valid_centers = np.array(valid_centers)
    center_regimes = np.array(center_regimes)

    # Compute real alignment
    real_result = compute_windowed_cca_alignment(
        X_m1, X_m2, bin_size_sec, window_sec, step_sec, n_pca_dims, n_cca_components
    )
    real_alignment = float(np.nanmean(real_result.alignment))

    # Compute pseudo-pair alignment
    pseudo_alignments = []

    unique_regimes = np.unique(center_regimes)

    for _ in range(n_pseudo_pairs):
        # For each regime, pair M1 windows with shifted M2 windows
        pseudo_corrs = []

        for regime in unique_regimes:
            regime_mask = center_regimes == regime
            regime_centers = valid_centers[regime_mask]

            if len(regime_centers) < 2:
                continue

            # Sample a pair of windows with sufficient time separation
            for _ in range(10):  # Try up to 10 times
                i, j = rng.choice(len(regime_centers), 2, replace=False)
                c1, c2 = regime_centers[i], regime_centers[j]

                if abs(c1 - c2) >= min_shift_bins:
                    # Pair M1 at c1 with M2 at c2
                    w1_start = c1 - window_bins // 2
                    w1_end = c1 + window_bins // 2
                    w2_start = c2 - window_bins // 2
                    w2_end = c2 + window_bins // 2

                    Z1_win = Z_m1[w1_start:w1_end]
                    Z2_win = Z_m2[w2_start:w2_end]

                    corrs = _cca_correlations(Z1_win, Z2_win, n_cca_components)
                    pseudo_corrs.append(np.nanmean(corrs))
                    break

        if pseudo_corrs:
            pseudo_alignments.append(np.mean(pseudo_corrs))

    pseudo_alignment = float(np.mean(pseudo_alignments)) if pseudo_alignments else np.nan
    difference = real_alignment - pseudo_alignment

    return {
        "real_alignment": real_alignment,
        "pseudo_alignment": pseudo_alignment,
        "difference": difference,
        "n_pseudo_pairs": len(pseudo_alignments),
    }


def compute_alignment_by_lag(
    X_m1: np.ndarray,
    X_m2: np.ndarray,
    bin_size_sec: float,
    window_sec: float = 10.0,
    lags_sec: np.ndarray = None,
    n_pca_dims: int = 20,
    n_cca_components: int = 5,
) -> Dict[str, np.ndarray]:
    """Compute alignment as a function of relative lag between monkeys.

    Alignment should peak near lag=0 if it reflects true inter-brain coupling.

    Args:
        X_m1: Neural rates for monkey 1
        X_m2: Neural rates for monkey 2
        bin_size_sec: Bin size in seconds
        window_sec: Window size for CCA
        lags_sec: Array of lags to test (default: -10 to +10 seconds)
        n_pca_dims: PCA dimensionality
        n_cca_components: Number of CCA components

    Returns:
        Dict with 'lags_sec', 'alignment_by_lag', 'peak_lag_sec'
    """
    if lags_sec is None:
        lags_sec = np.arange(-10, 11, 1)

    n_bins = min(X_m1.shape[0], X_m2.shape[0])
    window_bins = int(round(window_sec / bin_size_sec))

    # PCA reduce
    Z_m1, _ = _pca_reduce(X_m1[:n_bins], n_pca_dims)
    Z_m2, _ = _pca_reduce(X_m2[:n_bins], n_pca_dims)

    alignment_by_lag = []

    for lag_sec in lags_sec:
        lag_bins = int(round(lag_sec / bin_size_sec))

        # Compute alignment with M2 shifted by lag_bins
        corrs = []

        # Sample windows
        for center in range(window_bins, n_bins - window_bins, window_bins):
            m2_center = center + lag_bins

            if m2_center < window_bins or m2_center >= n_bins - window_bins:
                continue

            w1_start = center - window_bins // 2
            w1_end = center + window_bins // 2
            w2_start = m2_center - window_bins // 2
            w2_end = m2_center + window_bins // 2

            Z1_win = Z_m1[w1_start:w1_end]
            Z2_win = Z_m2[w2_start:w2_end]

            c = _cca_correlations(Z1_win, Z2_win, n_cca_components)
            corrs.append(np.nanmean(c))

        alignment_by_lag.append(np.mean(corrs) if corrs else np.nan)

    alignment_by_lag = np.array(alignment_by_lag)
    peak_idx = np.nanargmax(alignment_by_lag)
    peak_lag_sec = lags_sec[peak_idx]

    return {
        "lags_sec": lags_sec,
        "alignment_by_lag": alignment_by_lag,
        "peak_lag_sec": float(peak_lag_sec),
    }
