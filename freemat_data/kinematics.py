"""Project-agnostic kinematics utilities.

Lightweight, NumPy-only helpers for smoothing, differences, plane projections,
and angle unwrapping. Functions operate along the time axis (axis=0).

Example:
    >>> import numpy as np
    >>> from freemat_data.kinematics import central_diff, moving_average
    >>> t = np.linspace(0, 1, 11)
    >>> x = np.sin(2*np.pi*t)
    >>> v = central_diff(x[:, None], dt=float(np.median(np.diff(t))))[:, 0]
    >>> xs = moving_average(x, win_s=0.1, fps=1.0/np.median(np.diff(t)))
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _as_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure input is 2D (T, D), returning array and flag if originally 1D."""
    if X.ndim == 1:
        return X[:, None], True
    if X.ndim == 2:
        return X, False
    # Flatten trailing dims except time
    T = X.shape[0]
    return X.reshape(T, -1), False


def central_diff(X: np.ndarray, dt: float) -> np.ndarray:
    """Central difference along time with one-sided edges.

    Args:
        X: Array with time on axis 0. Any trailing dims are treated as features.
        dt: Time step in seconds (assumed constant).

    Returns:
        Array of same shape as `X` with dX/dt. NaNs propagate.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    X2, was_1d = _as_2d(np.asarray(X))
    T, D = X2.shape
    out = np.full_like(X2, np.nan, dtype=float)
    if T == 0:
        return X.copy()
    if T == 1:
        return np.zeros_like(X, dtype=float)

    # Finite masks for valid central differences
    fin = np.isfinite(X2)
    # Central differences for interior points
    Xp = X2[2:]
    Xm = X2[:-2]
    mask_mid = fin[2:] & fin[:-2]
    out[1:-1][mask_mid] = (Xp[mask_mid] - Xm[mask_mid]) / (2.0 * dt)

    # One-sided at boundaries
    m0 = fin[1] & fin[0]
    out[0][m0] = (X2[1][m0] - X2[0][m0]) / dt
    m1 = fin[-1] & fin[-2]
    out[-1][m1] = (X2[-1][m1] - X2[-2][m1]) / dt

    return out[:, 0] if was_1d else out.reshape(X.shape)


def moving_average(x: np.ndarray, win_s: float, fps: float) -> np.ndarray:
    """Centered moving-average with NaN-aware normalization.

    Args:
        x: Array with time on axis 0.
        win_s: Window size in seconds.
        fps: Sampling frequency in Hz (1/dt).

    Returns:
        Smoothed array with same shape. NaNs are ignored; if all values in the
        window are NaN, the output is NaN for that position.
    """
    X2, was_1d = _as_2d(np.asarray(x, dtype=float))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    win = int(max(1, round(float(win_s) * float(fps))))
    if win <= 1:
        return x.copy()
    # Force odd window for symmetric smoothing
    if win % 2 == 0:
        win += 1
    T, D = X2.shape
    k = np.ones(win, dtype=float)
    # NaN-aware convolution: sum and count
    Xn = np.where(np.isfinite(X2), X2, 0.0)
    C = np.where(np.isfinite(X2), 1.0, 0.0)
    Ys = np.vstack([np.convolve(Xn[:, d], k, mode="same") for d in range(D)]).T
    Cs = np.vstack([np.convolve(C[:, d], k, mode="same") for d in range(D)]).T
    out = Ys / np.where(Cs > 0, Cs, np.nan)
    return out[:, 0] if was_1d else out.reshape(x.shape)


def plane_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable in-plane orthonormal basis given a plane normal.

    Args:
        n: Normal vector, shape (3,) or (N, 3). If batched, uses row 0.

    Returns:
        (e1, e2): Two orthonormal 3D vectors spanning the plane orthogonal to n.
    """
    n = np.asarray(n, dtype=float)
    if n.ndim == 2:
        n = n[0]
    if n.shape != (3,):
        raise ValueError("n must have shape (3,) or (N,3)")
    n = n / (np.linalg.norm(n) + 1e-12)
    # Pick a reference vector least aligned with n for numerical stability
    ref_candidates = np.eye(3)
    dots = ref_candidates @ n
    ref = ref_candidates[np.argmin(np.abs(dots))]
    e1 = np.cross(n, ref)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    e2 /= (np.linalg.norm(e2) + 1e-12)
    return e1, e2


def project_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project vector(s) `v` onto plane with normal `n`.

    Args:
        v: Vector(s), shape (..., 3).
        n: Normal vector, shape (3,).

    Returns:
        Projection of `v` onto the plane (same shape as `v`).
    """
    v = np.asarray(v, dtype=float)
    n = np.asarray(n, dtype=float)
    if n.shape != (3,):
        raise ValueError("n must be shape (3,)")
    n_hat = n / (np.linalg.norm(n) + 1e-12)
    dot = np.sum(v * n_hat, axis=-1, keepdims=True)
    return v - dot * n_hat


def unwrap_phase(theta: np.ndarray) -> np.ndarray:
    """Unwrap a 1D or column vector of angles (radians), preserving NaNs.

    Args:
        theta: Array of angles (T,) or (T, 1) in radians.

    Returns:
        Unwrapped array with same shape.
    """
    th = np.asarray(theta, dtype=float)
    was_1d = False
    if th.ndim == 2 and th.shape[1] == 1:
        th = th[:, 0]
        was_1d = True
    elif th.ndim != 1:
        raise ValueError("theta must be 1D or (T,1)")
    out = np.full_like(th, np.nan)
    finite = np.isfinite(th)
    if finite.any():
        out[finite] = np.unwrap(th[finite])
    return out[:, None] if was_1d else out


__all__ = [
    "central_diff",
    "moving_average",
    "plane_basis_from_normal",
    "project_to_plane",
    "unwrap_phase",
]

