"""Core inter-brain synchrony metrics.

Provides lightweight, dependency-minimal metrics to quantify coupling between
two simultaneously recorded populations (M1 and M2). Functions favor simple,
vectorized NumPy and operate on time-by-unit matrices for clarity.

Core Metrics:

    Cosine Similarity (per time bin):
        CS(t) = (u(t) · v(t)) / (||u(t)|| × ||v(t)||)

        Where u(t), v(t) are z-scored population vectors at time t.
        CS ∈ [-1, 1], with 1 = identical direction.

    Instantaneous Population Coupling (IPC):
        IPC(t) = z₁(t) × z₂(t)

        Where z_i(t) = (mean_i(t) - μ_i) / σ_i is the z-scored population mean.
        IPC > 0 when both populations deviate in same direction from baseline.

    CCA-Aligned Cosine Similarity:
        1. Find projections W_x, W_y that maximize corr(X·W_x, Y·W_y)
        2. Project: Z_x = X·W_x, Z_y = Y·W_y
        3. CS_CCA(t) = (Z_x(t) · Z_y(t)) / (||Z_x(t)|| × ||Z_y(t)||)

        CCA computation via whitened cross-covariance:
            K = S_xx^(-½) · S_xy · S_yy^(-½)
            SVD: K = U·Σ·Vᵀ
            W_x = S_xx^(-½) · U[:,:k], W_y = S_yy^(-½) · V[:,:k]

Conventions:
    - Inputs are arrays shaped ``(T, U)`` where T is number of time bins and U
      is number of units for that subject.
    - When a metric requires matched dimensionality across subjects, we reduce
      to a common unit count by selecting the top-variance units per subject and
      truncating to the minimum U (reproducible).

Example:
    >>> import numpy as np
    >>> from data_utils.ibns_metrics import cosine_similarity_series, instantaneous_population_coupling
    >>> T, U1, U2 = 1000, 50, 80
    >>> m1 = np.random.randn(T, U1)
    >>> m2 = np.random.randn(T, U2)
    >>> cs = cosine_similarity_series(m1, m2)
    >>> ipc = instantaneous_population_coupling(m1, m2)
"""

from __future__ import annotations

from typing import Tuple, List

import numpy as np


def _match_dimensions_by_variance(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match unit dimensions of two matrices by variance ranking.

    Args:
        A: Time-by-unit rates for subject 1, shape ``(T, U_A)``.
        B: Time-by-unit rates for subject 2, shape ``(T, U_B)``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Matrices with matched unit dimension
        ``(T, U*)`` where ``U* = min(U_A, U_B)``.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays shaped (T, U)")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of time bins")

    UA = A.shape[1]
    UB = B.shape[1]
    if UA == 0 or UB == 0:
        return A[:, :0], B[:, :0]

    k = min(UA, UB)
    # Top-variance indices (stable, reproducible without randomness)
    varA = np.var(A, axis=0)
    varB = np.var(B, axis=0)
    idxA = np.argsort(varA)[-k:]
    idxB = np.argsort(varB)[-k:]
    return A[:, idxA], B[:, idxB]


def cosine_similarity_series(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compute per-bin cosine similarity between population vectors.

    Mathematical Definition:
        For each time bin t:
            CS(t) = (A_z(t) · B_z(t)) / (||A_z(t)|| × ||B_z(t)||)

        Where:
            - A_z, B_z are z-scored (across time, per unit) versions of inputs
            - A_z(t), B_z(t) are matched-dimension population vectors at time t
            - ||·|| denotes L2 norm

        Dimension matching: Top-k variance units selected, k = min(U1, U2).

    Args:
        m1: Time-by-unit rates for subject 1, shape ``(T, U1)``.
        m2: Time-by-unit rates for subject 2, shape ``(T, U2)``.

    Returns:
        np.ndarray: Cosine similarity time series of shape ``(T,)`` in ``[-1, 1]``.
    """
    if m1.shape[0] != m2.shape[0]:
        raise ValueError("m1 and m2 must have same T for cosine computation")

    A, B = _match_dimensions_by_variance(m1, m2)
    if A.shape[1] == 0:
        return np.full(A.shape[0], np.nan, dtype=float)

    # Z-score across time per unit to equalize scale
    def z(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True) + 1e-12
        return (x - mu) / sd

    Az = z(A)
    Bz = z(B)

    # Cosine similarity per time bin: dot(Az_t, Bz_t) / (||Az_t|| * ||Bz_t||)
    num = np.sum(Az * Bz, axis=1)
    den = np.linalg.norm(Az, axis=1) * np.linalg.norm(Bz, axis=1) + 1e-12
    return num / den


def instantaneous_population_coupling(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compute product of z-scored population means at each time bin.

    Mathematical Definition:
        IPC(t) = z₁(t) × z₂(t)

        Where:
            pop_i(t) = (1/U_i) × Σ_u rate_{i,u}(t)  (population mean at time t)
            z_i(t) = (pop_i(t) - μ_i) / σ_i         (z-scored)
            μ_i = mean_t(pop_i(t))
            σ_i = std_t(pop_i(t))

        Interpretation:
            IPC > 0: Both populations above or below their temporal means
            IPC < 0: Populations on opposite sides of their means
            IPC ≈ 0: No consistent coupling

        Note: Unlike cosine similarity, IPC is unbounded (can exceed ±1).

    Args:
        m1: Time-by-unit rates for subject 1, shape ``(T, U1)``.
        m2: Time-by-unit rates for subject 2, shape ``(T, U2)``.

    Returns:
        np.ndarray: Instantaneous coupling series of shape ``(T,)`` (unbounded).
    """
    if m1.shape[0] != m2.shape[0]:
        raise ValueError("m1 and m2 must have same T for IPC computation")

    pop1 = np.mean(m1, axis=1)
    pop2 = np.mean(m2, axis=1)
    z1 = (pop1 - np.mean(pop1)) / (np.std(pop1) + 1e-12)
    z2 = (pop2 - np.mean(pop2)) / (np.std(pop2) + 1e-12)
    return z1 * z2


def cca_cosine_series(m1: np.ndarray, m2: np.ndarray, n_components: int = 3, ridge: float = 1e-6) -> np.ndarray:
    """Compute a CCA-aligned cosine similarity time series.

    Mathematical Definition:
        Canonical Correlation Analysis finds projections W_x, W_y maximizing:
            corr(X·W_x, Y·W_y)

        Algorithm:
            1. Center data: X̃ = X - μ_X, Ỹ = Y - μ_Y
            2. Covariances (with ridge regularization):
                S_xx = X̃ᵀX̃/(T-1) + λI
                S_yy = ỸᵀỸ/(T-1) + λI
                S_xy = X̃ᵀỸ/(T-1)
            3. Whitened cross-covariance:
                K = S_xx^(-½) · S_xy · S_yy^(-½)
            4. SVD: K = U·Σ·Vᵀ
            5. Projections:
                W_x = S_xx^(-½) · U[:,:k]
                W_y = S_yy^(-½) · V[:,:k]
            6. Canonical variates: Z_x = X̃·W_x, Z_y = Ỹ·W_y
            7. Per-bin cosine:
                CS_CCA(t) = (Z_x(t)·Z_y(t)) / (||Z_x(t)||×||Z_y(t)||)

    Args:
        m1: Time-by-unit rates for subject 1, shape ``(T, U1)``.
        m2: Time-by-unit rates for subject 2, shape ``(T, U2)``.
        n_components: Number of canonical components k (<= min(U1, U2)).
        ridge: Diagonal regularization λ for numerical stability.

    Returns:
        np.ndarray: Cosine similarity time series in canonical space, shape ``(T,)``.
    """
    if m1.shape[0] != m2.shape[0]:
        raise ValueError("m1 and m2 must have same T for CCA computation")
    T = m1.shape[0]
    if T < 2:
        return np.full(T, np.nan)

    X = m1 - np.mean(m1, axis=0, keepdims=True)
    Y = m2 - np.mean(m2, axis=0, keepdims=True)
    U1 = X.shape[1]
    U2 = Y.shape[1]
    k = int(min(n_components, U1, U2))
    if k <= 0:
        return np.full(T, np.nan)

    # Covariances
    Sxx = (X.T @ X) / max(1, T - 1) + ridge * np.eye(U1)
    Syy = (Y.T @ Y) / max(1, T - 1) + ridge * np.eye(U2)
    Sxy = (X.T @ Y) / max(1, T - 1)

    # Inverse sqrt via eigendecomposition
    eigx, Vx = np.linalg.eigh(Sxx)
    eigh_small_x = eigx <= 0
    if np.any(eigh_small_x):
        eigx[eigh_small_x] = ridge
    Sxx_inv_sqrt = Vx @ np.diag(1.0 / np.sqrt(eigx)) @ Vx.T

    eigy, Vy = np.linalg.eigh(Syy)
    eigh_small_y = eigy <= 0
    if np.any(eigh_small_y):
        eigy[eigh_small_y] = ridge
    Syy_inv_sqrt = Vy @ np.diag(1.0 / np.sqrt(eigy)) @ Vy.T

    # Whitened cross-covariance and SVD
    K = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    U, _, Vt = np.linalg.svd(K, full_matrices=False)
    Wx = Sxx_inv_sqrt @ U[:, :k]
    Wy = Syy_inv_sqrt @ Vt.T[:, :k]

    # Project to canonical variates and compute per-bin cosine in latent space
    Zx = X @ Wx  # (T, k)
    Zy = Y @ Wy  # (T, k)
    num = np.sum(Zx * Zy, axis=1)
    den = np.linalg.norm(Zx, axis=1) * np.linalg.norm(Zy, axis=1) + 1e-12
    return num / den


def ccg_peak_mean_for_segment(
    A: np.ndarray,
    B: np.ndarray,
    start: int,
    end: int,
    max_lag_bins: int = 10,
    n_pairs: int = 500,
    rng_seed: int = 123,
) -> float:
    """Mean CCG peak (within ±max_lag) over subsampled M1×M2 unit pairs in a segment.

    Args:
        A: Rates or counts for subject 1, shape ``(T, U1)`` (coarse bins OK).
        B: Rates or counts for subject 2, shape ``(T, U2)``.
        start: Segment start index (inclusive).
        end: Segment end index (exclusive).
        max_lag_bins: Maximum lag in bins to search around zero.
        n_pairs: Number of unit pairs to subsample uniformly at random (capped by U1×U2).
        rng_seed: Seed for deterministic subsampling.

    Returns:
        float: Average peak cross-correlation across sampled pairs.
    """
    segA = A[start:end]
    segB = B[start:end]
    T = segA.shape[0]
    if T <= 1 or segA.shape[1] == 0 or segB.shape[1] == 0:
        return np.nan

    U1 = segA.shape[1]
    U2 = segB.shape[1]
    total_pairs = U1 * U2
    m = min(n_pairs, total_pairs)
    rng = np.random.default_rng(rng_seed)

    # Pre-whiten per unit to equalize scale
    def z(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True) + 1e-12
        return (x - mu) / sd

    ZA = z(segA)
    ZB = z(segB)

    # Sample pairs
    idx_i = rng.integers(0, U1, size=m)
    idx_j = rng.integers(0, U2, size=m)

    peaks: List[float] = []
    for i, j in zip(idx_i, idx_j):
        a = ZA[:, i]
        b = ZB[:, j]
        # Cross-correlation via FFT
        f = np.fft.fft(a, n=2 * T - 1)
        g = np.fft.fft(b[::-1], n=2 * T - 1)
        ccg = np.fft.ifft(f * g).real
        mid = T - 1
        lo = max(0, mid - max_lag_bins)
        hi = min(ccg.shape[0], mid + max_lag_bins + 1)
        peaks.append(float(np.max(ccg[lo:hi])))
    return float(np.mean(peaks)) if peaks else np.nan


def jpsth_coincidence_mean_for_segment(
    A: np.ndarray, B: np.ndarray, start: int, end: int, n_pairs: int = 500, rng_seed: int = 321
) -> float:
    """Mean JPSTH coincidence index across subsampled pairs in a segment.

    The coincidence index is defined here as diag(JPSTH) / sum(JPSTH), where
    JPSTH_ij = sum_t a_i(t) * b_j(t). This normalizes by overall co-activity.

    Args:
        A: Rates/counts for subject 1, shape ``(T, U1)``.
        B: Rates/counts for subject 2, shape ``(T, U2)``.
        start: Segment start index (inclusive).
        end: Segment end index (exclusive).
        n_pairs: Number of pairs to subsample.
        rng_seed: Random seed for subsampling.

    Returns:
        float: Mean coincidence index across sampled pairs.
    """
    segA = A[start:end]
    segB = B[start:end]
    T = segA.shape[0]
    if T <= 0 or segA.shape[1] == 0 or segB.shape[1] == 0:
        return np.nan

    U1 = segA.shape[1]
    U2 = segB.shape[1]
    total_pairs = U1 * U2
    m = min(n_pairs, total_pairs)
    rng = np.random.default_rng(rng_seed)

    # Ensure non-negative rates for interpretation (clip small negatives)
    Apos = np.clip(segA, a_min=0.0, a_max=None)
    Bpos = np.clip(segB, a_min=0.0, a_max=None)

    idx_i = rng.integers(0, U1, size=m)
    idx_j = rng.integers(0, U2, size=m)

    coins: List[float] = []
    for i, j in zip(idx_i, idx_j):
        ai = Apos[:, i]
        bj = Bpos[:, j]
        diag_sum = float(np.sum(ai * bj))
        total = float(np.sum(ai) * np.sum(bj))
        if total <= 0:
            continue
        coins.append(diag_sum / total)
    return float(np.mean(coins)) if coins else np.nan


def jpsth_coincidence(_: np.ndarray, __: np.ndarray) -> np.ndarray:
    """Placeholder for JPSTH coincidence metric.

    This function is intentionally left unimplemented in the scaffold. It is
    provided to define the public API and will raise at call time. Use the
    higher-level population metrics first (cosine/instantaneous).

    Raises:
        NotImplementedError: Always, in this scaffold.
    """
    raise NotImplementedError("JPSTH coincidence not implemented in scaffold")
