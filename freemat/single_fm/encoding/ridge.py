"""Ridge regression and R² utilities for encoding models.

Provides ridge regression with implicit intercept, GPU acceleration via CuPy,
Numba-accelerated R² scoring, spline design matrices, and cross-validated R²
with nested alpha selection.

Example:
    >>> from freemat.single_fm.encoding.ridge import compute_cv_r2, DEFAULT_ALPHAS
    >>> r2 = compute_cv_r2(X, y, folds, alphas=DEFAULT_ALPHAS)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from freemat.single_fm.encoding.cv import CVConfig, blocked_cv_folds

# Default alpha grid for ridge regression (extended range: 0.01 to 10000)
DEFAULT_ALPHAS: Tuple[float, ...] = tuple(np.logspace(-2, 4, 20).tolist())

# Optional Numba support for JIT compilation
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: identity decorator when Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# Optional CuPy support for GPU acceleration
try:
    import cupy as cp
    HAS_GPU = cp.cuda.is_available()
except ImportError:
    cp = None
    HAS_GPU = False


def ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit ridge regression and predict on test set.

    Includes implicit intercept via y-centering: fits ridge on (y - mean(y)),
    then adds mean(y) back to predictions.

    Args:
        X_train: Training design matrix (N_train, D).
        y_train: Training target (N_train,).
        X_test: Test design matrix (N_test, D).
        alpha: Ridge regularization strength.

    Returns:
        Predictions on test set (N_test,).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)

    if X_train.size == 0 or X_train.shape[1] == 0:
        return np.zeros(X_test.shape[0], dtype=np.float64)

    # Center y to handle intercept implicitly
    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean

    XtX = X_train.T @ X_train
    aI = float(alpha) * np.eye(XtX.shape[0])
    w = np.linalg.solve(XtX + aI, X_train.T @ y_centered)

    # Predictions = X @ w + intercept (where intercept = y_mean)
    return X_test @ w + y_mean


def ridge_fit_get_coeffs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit ridge regression and return coefficient vector.

    Same math as ``ridge_fit_predict`` but returns the weight vector ``w``
    instead of predictions.  Useful for interpreting per-feature contributions.

    Args:
        X_train: Design matrix (N, D).
        y_train: Target vector (N,).
        alpha: Ridge regularization strength.

    Returns:
        Coefficient vector w of shape (D,).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

    if X_train.size == 0 or X_train.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean

    XtX = X_train.T @ X_train
    aI = float(alpha) * np.eye(XtX.shape[0])
    return np.linalg.solve(XtX + aI, X_train.T @ y_centered)


def ridge_fit_predict_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """GPU-accelerated ridge regression using CuPy.

    Includes implicit intercept via y-centering. Falls back to CPU if GPU
    is not available.

    Args:
        X_train: Training design matrix (N_train, D).
        y_train: Training target (N_train,).
        X_test: Test design matrix (N_test, D).
        alpha: Ridge regularization strength.

    Returns:
        Predictions on test set (N_test,).
    """
    if not HAS_GPU or cp is None:
        return ridge_fit_predict(X_train, y_train, X_test, alpha)

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)

    if X_train.size == 0 or X_train.shape[1] == 0:
        return np.zeros(X_test.shape[0], dtype=np.float64)

    # Center y to handle intercept implicitly (on CPU, then transfer)
    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean

    # Transfer to GPU
    X_train_gpu = cp.asarray(X_train)
    y_centered_gpu = cp.asarray(y_centered)
    X_test_gpu = cp.asarray(X_test)

    # Ridge regression on GPU
    XtX = X_train_gpu.T @ X_train_gpu
    aI = float(alpha) * cp.eye(XtX.shape[0], dtype=cp.float64)
    w = cp.linalg.solve(XtX + aI, X_train_gpu.T @ y_centered_gpu)

    # Predict and transfer back to CPU, adding back intercept
    y_pred_gpu = X_test_gpu @ w + y_mean
    return cp.asnumpy(y_pred_gpu)


@njit(cache=True)
def _r2_score_numba(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Numba-accelerated R² computation."""
    n = len(y_true)
    if n == 0:
        return 0.0

    # Compute mean
    mu = 0.0
    for i in range(n):
        mu += y_true[i]
    mu /= n

    # Compute SS_tot and SS_res
    ss_tot = 0.0
    ss_res = 0.0
    for i in range(n):
        diff = y_true[i] - mu
        ss_tot += diff * diff
        resid = y_true[i] - y_pred[i]
        ss_res += resid * resid

    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R² score (can be negative if worse than mean).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if HAS_NUMBA:
        return _r2_score_numba(y_true, y_pred)

    # Fallback to numpy
    mu = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mu) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def build_spline_design(X: np.ndarray, n_knots: int = 5) -> np.ndarray:
    """Build cubic regression spline design matrix.

    Uses patsy for cubic regression splines with specified knots.
    Degrees of freedom per feature = n_knots + 3 (cubic spline).

    For features with zero or near-zero variance, the original feature is kept
    as-is instead of applying splines (spline fitting requires variance).

    Args:
        X: Predictor matrix (T, D).
        n_knots: Number of interior knots per feature.

    Returns:
        Spline design matrix. For features with variance, shape is (T, D' * (n_knots + 3))
        where D' is the number of features with variance. Constant features are kept as
        single columns.

    Raises:
        ImportError: If patsy is not installed.
    """
    try:
        import patsy
    except ImportError as e:
        raise ImportError("patsy is required for spline design; pip install patsy") from e

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    T, D = X.shape
    if D == 0:
        return np.zeros((T, 0), dtype=np.float64)

    df = n_knots + 3  # Cubic spline degrees of freedom
    cols = []
    min_variance = 1e-10  # Minimum variance threshold for spline fitting

    for j in range(D):
        col = X[:, j]
        col_var = np.var(col[np.isfinite(col)])

        if col_var < min_variance:
            # Feature has no variance - keep as single column (no spline)
            cols.append(col.reshape(-1, 1))
        else:
            # Apply cubic regression spline
            data = {"x": col}
            # Use cubic regression spline (cr) with df degrees of freedom
            # "-1" removes the intercept from the design matrix
            try:
                B = patsy.dmatrix(f"cr(x, df={df}) - 1", data, return_type="dataframe").values
                cols.append(B)
            except ValueError:
                # Fallback: if spline fails (e.g., not enough unique values), keep original
                cols.append(col.reshape(-1, 1))

    return np.concatenate(cols, axis=1) if cols else np.zeros((T, 0), dtype=np.float64)


def inner_cv_select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    alphas: Sequence[float],
    standardize: bool = True,
    use_gpu: bool = False,
) -> float:
    """Select optimal ridge alpha via inner cross-validation.

    Standardizes within each fold to match compute_cv_r2_fixed_alpha behavior,
    ensuring the selected alpha is on the correct feature scale.

    Args:
        X: Design matrix.
        y: Target vector.
        folds: Inner CV folds.
        alphas: Grid of alpha values.
        standardize: Whether to z-score features within each fold.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        Best alpha value.
    """
    best_alpha = alphas[0]
    best_score = -np.inf

    # Select ridge function based on GPU availability
    ridge_fn = ridge_fit_predict_gpu if use_gpu else ridge_fit_predict

    for alpha in alphas:
        scores = []
        for fold in folds:
            tr, te = fold["train"], fold["test"]
            if tr.size == 0 or te.size == 0:
                continue

            X_train, X_test = X[tr], X[te]

            # Standardize within fold (matches compute_cv_r2_fixed_alpha)
            if standardize:
                mu = np.nanmean(X_train, axis=0, keepdims=True)
                sd = np.nanstd(X_train, axis=0, keepdims=True)
                sd = np.where(sd < 1e-12, 1.0, sd)
                X_train = (X_train - mu) / sd
                X_test = (X_test - mu) / sd

            y_pred = ridge_fn(X_train, y[tr], X_test, float(alpha))
            scores.append(r2_score(y[te], y_pred))

        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

    return float(best_alpha)


def compute_cv_r2(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    standardize: bool = True,
    bin_s: float = 0.5,
    use_gpu: bool = False,
) -> float:
    """Compute cross-validated R² with nested alpha selection.

    Args:
        X: Design matrix (T, D).
        y: Target vector (T,).
        folds: Outer CV folds.
        alphas: Ridge alpha grid for inner CV.
        standardize: Whether to z-score features within each fold.
        bin_s: Bin size in seconds for inner CV fold construction.
        use_gpu: Whether to use GPU acceleration (requires CuPy).

    Returns:
        Cross-validated R².
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.size == 0 or X.shape[1] == 0:
        return 0.0

    # Select ridge function based on GPU availability
    ridge_fn = ridge_fit_predict_gpu if use_gpu else ridge_fit_predict

    all_test_true = []
    all_test_pred = []

    # Inner CV uses 3 folds for alpha selection
    inner_cfg = CVConfig(folds=3, block_s=30.0, gap_s=2.0)

    for fold in folds:
        tr, te = fold["train"], fold["test"]
        if tr.size == 0 or te.size == 0:
            continue

        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        # Standardize within fold
        if standardize:
            mu = np.nanmean(X_train, axis=0, keepdims=True)
            sd = np.nanstd(X_train, axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            X_train = (X_train - mu) / sd
            X_test = (X_test - mu) / sd

        # Build inner folds on training data
        inner_folds = blocked_cv_folds(len(tr), bin_s=bin_s, cfg=inner_cfg)
        # standardize=False: X_train is already z-scored by the outer fold above
        alpha_opt = inner_cv_select_alpha(
            X_train, y_train, inner_folds, alphas, standardize=False, use_gpu=use_gpu
        )

        # Predict on test
        y_pred = ridge_fn(X_train, y_train, X_test, alpha_opt)
        all_test_true.append(y_test)
        all_test_pred.append(y_pred)

    if not all_test_true:
        return 0.0

    y_true_all = np.concatenate(all_test_true)
    y_pred_all = np.concatenate(all_test_pred)
    return r2_score(y_true_all, y_pred_all)


def ridge_fit_predict_batch_gpu(
    X_train_batch: np.ndarray,
    y_train: np.ndarray,
    X_test_batch: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Batch GPU ridge: solve B independent problems sharing the same y_train.

    Uses CuPy batched linalg.solve to compute all B ridge regressions in
    parallel on GPU.  Falls back to a CPU loop when CuPy is unavailable.

    Args:
        X_train_batch: Training matrices (B, N_train, D).
        y_train: Shared target vector (N_train,).
        X_test_batch: Test matrices (B, N_test, D).
        alpha: Ridge regularization strength.

    Returns:
        Predictions array (B, N_test).
    """
    B = X_train_batch.shape[0]

    if not HAS_GPU or cp is None:
        # CPU fallback: loop over batch
        preds = [
            ridge_fit_predict(X_train_batch[b], y_train, X_test_batch[b], alpha)
            for b in range(B)
        ]
        return np.stack(preds)

    _, N_tr, D = X_train_batch.shape

    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean

    X_tr_gpu = cp.asarray(X_train_batch)       # (B, N_tr, D)
    y_gpu = cp.asarray(y_centered)              # (N_tr,)
    X_te_gpu = cp.asarray(X_test_batch)         # (B, N_te, D)

    # Batch XtX: (B, D, D)
    XtX = cp.matmul(X_tr_gpu.transpose(0, 2, 1), X_tr_gpu)

    # Regularize: broadcast (D, D) -> (B, D, D)
    aI = float(alpha) * cp.eye(D, dtype=cp.float64)
    XtX_reg = XtX + aI

    # Batch Xty: (B, D)
    Xty = cp.einsum("bdn,n->bd", X_tr_gpu.transpose(0, 2, 1), y_gpu)

    # Batch solve: (B, D, D) \ (B, D) -> (B, D)
    w = cp.linalg.solve(XtX_reg, Xty)

    # Batch predict: (B, N_te)
    preds_gpu = cp.einsum("bnd,bd->bn", X_te_gpu, w) + y_mean

    return cp.asnumpy(preds_gpu)


# Maximum batch size per GPU call to limit memory usage
_BATCH_CHUNK = 128


def compute_cv_r2_fixed_alpha_batch(
    X_batch: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    alpha: float,
    standardize: bool = True,
    use_gpu: bool = True,
) -> np.ndarray:
    """Batch CV-R² for B design matrices sharing the same target y.

    Computes cross-validated R² for each of B design matrices in one pass,
    using GPU-batched ridge regression per fold.  This is much faster than
    calling compute_cv_r2_fixed_alpha B times when the GPU is available.

    Falls back to sequential computation when GPU is unavailable.

    Args:
        X_batch: Batch of design matrices (B, T, D).
        y: Shared target vector (T,).
        folds: Outer CV folds.
        alpha: Fixed ridge regularization strength.
        standardize: Whether to z-score features within each fold.
        use_gpu: Whether to use GPU-batched ridge.

    Returns:
        Array of R² values (B,).
    """
    B = X_batch.shape[0]

    # Chunk large batches to limit GPU memory
    if B > _BATCH_CHUNK:
        r2_chunks = []
        for start in range(0, B, _BATCH_CHUNK):
            end = min(start + _BATCH_CHUNK, B)
            r2_chunks.append(
                compute_cv_r2_fixed_alpha_batch(
                    X_batch[start:end], y, folds, alpha, standardize, use_gpu,
                )
            )
        return np.concatenate(r2_chunks)

    if not use_gpu or not HAS_GPU:
        # CPU fallback: sequential
        return np.array([
            compute_cv_r2_fixed_alpha(X_batch[b], y, folds, alpha, standardize, use_gpu=False)
            for b in range(B)
        ])

    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Collect per-fold predictions
    all_test_true: List[List[np.ndarray]] = [[] for _ in range(B)]
    all_test_pred: List[List[np.ndarray]] = [[] for _ in range(B)]

    for fold in folds:
        tr, te = fold["train"], fold["test"]
        if tr.size == 0 or te.size == 0:
            continue

        X_trains = X_batch[:, tr, :].copy()  # (B, N_train, D)
        X_tests = X_batch[:, te, :].copy()   # (B, N_test, D)

        if standardize:
            mu = np.nanmean(X_trains, axis=1, keepdims=True)   # (B, 1, D)
            sd = np.nanstd(X_trains, axis=1, keepdims=True)    # (B, 1, D)
            sd = np.where(sd < 1e-12, 1.0, sd)
            X_trains = (X_trains - mu) / sd
            X_tests = (X_tests - mu) / sd

        # Batch GPU ridge
        preds = ridge_fit_predict_batch_gpu(X_trains, y[tr], X_tests, alpha)

        y_test = y[te]
        for b in range(B):
            all_test_true[b].append(y_test)
            all_test_pred[b].append(preds[b])

    # Compute R² per batch element
    r2_values = np.zeros(B)
    for b in range(B):
        if not all_test_true[b]:
            continue
        y_true_all = np.concatenate(all_test_true[b])
        y_pred_all = np.concatenate(all_test_pred[b])
        r2_values[b] = r2_score(y_true_all, y_pred_all)

    return r2_values


def compute_cv_r2_fixed_alpha(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Dict[str, np.ndarray]],
    alpha: float,
    standardize: bool = True,
    use_gpu: bool = False,
    return_predictions: bool = False,
) -> Union[float, Tuple[float, Dict[str, np.ndarray]]]:
    """Compute cross-validated R² with FIXED alpha (no inner CV).

    This is ~10x faster than compute_cv_r2() because it skips nested CV
    for alpha selection. Use when alpha is pre-computed or for permutation tests.

    Args:
        X: Design matrix (T, D).
        y: Target vector (T,).
        folds: Outer CV folds.
        alpha: Fixed ridge regularization strength.
        standardize: Whether to z-score features within each fold.
        use_gpu: Whether to use GPU acceleration.
        return_predictions: If True, also return per-bin predictions dict.

    Returns:
        Cross-validated R², or (R², predictions_dict) if return_predictions=True.
        predictions_dict has keys: y_true, y_pred, test_idx.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.size == 0 or X.shape[1] == 0:
        if return_predictions:
            return 0.0, {"y_true": np.array([]), "y_pred": np.array([]), "test_idx": np.array([], dtype=int)}
        return 0.0

    ridge_fn = ridge_fit_predict_gpu if use_gpu else ridge_fit_predict

    all_test_true = []
    all_test_pred = []
    all_test_idx = []

    for fold in folds:
        tr, te = fold["train"], fold["test"]
        if tr.size == 0 or te.size == 0:
            continue

        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]

        # Standardize within fold
        if standardize:
            mu = np.nanmean(X_train, axis=0, keepdims=True)
            sd = np.nanstd(X_train, axis=0, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            X_train = (X_train - mu) / sd
            X_test = (X_test - mu) / sd

        # Predict with fixed alpha (no inner CV)
        y_pred = ridge_fn(X_train, y_train, X_test, alpha)
        all_test_true.append(y_test)
        all_test_pred.append(y_pred)
        all_test_idx.append(te)

    if not all_test_true:
        if return_predictions:
            return 0.0, {"y_true": np.array([]), "y_pred": np.array([]), "test_idx": np.array([], dtype=int)}
        return 0.0

    y_true_all = np.concatenate(all_test_true)
    y_pred_all = np.concatenate(all_test_pred)
    r2 = r2_score(y_true_all, y_pred_all)

    if return_predictions:
        test_idx_all = np.concatenate(all_test_idx)
        return r2, {"y_true": y_true_all, "y_pred": y_pred_all, "test_idx": test_idx_all}
    return r2


__all__ = [
    "DEFAULT_ALPHAS",
    "HAS_GPU",
    "HAS_NUMBA",
    "ridge_fit_predict",
    "ridge_fit_predict_gpu",
    "ridge_fit_predict_batch_gpu",
    "r2_score",
    "build_spline_design",
    "inner_cv_select_alpha",
    "compute_cv_r2",
    "compute_cv_r2_fixed_alpha",
    "compute_cv_r2_fixed_alpha_batch",
]
