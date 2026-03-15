from __future__ import annotations

"""QC utilities for 3D pose.

Functions here are computation-only and return arrays/dicts; plotting is handled
in notebooks or visualization modules. These routines support limb-length and
velocity sanity checks and coverage summaries.
"""

from typing import Dict, List, Sequence, Tuple
import numpy as np


def coverage_metrics(valid_frac: np.ndarray) -> Dict[str, float | int | np.ndarray]:
    """Summarize pose coverage.

    Args:
        valid_frac: (T,) fraction of valid joints per frame.

    Returns:
        Dict with keys: 'valid_pct', 'longest_len', 'segment' (i0,i1).
    """
    vf = np.asarray(valid_frac, dtype=float)
    valid = vf >= 0.5
    T = valid.size
    best = (0, 0)
    best_len = 0
    k = 0
    while k < T:
        if not valid[k]:
            k += 1
            continue
        s = k
        while k < T and valid[k]:
            k += 1
        e = k
        if e - s > best_len:
            best_len = e - s
            best = (s, e)
    return {
        "valid_pct": float(np.mean(valid) * 100.0),
        "longest_len": int(best_len),
        "segment": np.array(best, dtype=np.int64),
    }


def limb_length_stats(
    X: np.ndarray,
    *,
    edges: Sequence[Tuple[int, int]],
    tol: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Compute robust limb-length baselines and per-frame deviations.

    Args:
        X: Pose positions (T, J, 3).
        edges: List of (j0,j1) bone edges.
        tol: Fractional deviation threshold for violation.

    Returns:
        Dict with 'median_len' (E,), 'deviation' (E,T), 'violation' (T,) flags.
    """
    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    meds: List[float] = []
    devs: List[np.ndarray] = []
    for j0, j1 in edges:
        P = X[:, [j0, j1], :]
        valid = np.isfinite(P).all(axis=(1, 2))
        L = np.linalg.norm(P[:, 0, :] - P[:, 1, :], axis=-1)
        med = np.nanmedian(L[valid]) if valid.any() else np.nan
        meds.append(float(med) if np.isfinite(med) else np.nan)
        if not np.isfinite(med) or med <= 1e-9:
            devs.append(np.zeros((T,), dtype=float))
        else:
            devs.append(np.abs(L / med - 1.0))
    D = np.vstack(devs) if devs else np.zeros((0, T), dtype=float)
    viol = (D > float(tol)).any(axis=0) if D.size else np.zeros((T,), dtype=bool)
    return {
        "median_len": np.asarray(meds, dtype=float),
        "deviation": D,
        "violation": viol,
    }


def velocity_outliers(
    X: np.ndarray,
    *,
    fps: float,
    z_thresh: float = 5.0,
    frac: float = 0.25,
) -> np.ndarray:
    """Flag frames with widespread per-joint speed spikes.

    Args:
        X: Pose positions (T, J, 3).
        fps: Frames per second.
        z_thresh: Z-score threshold for joint speed spikes.
        frac: Fraction of joints exceeding threshold to mark frame.

    Returns:
        (T,) boolean mask where True indicates an outlier frame.
    """
    X = np.asarray(X, dtype=float)
    if fps <= 0 or X.shape[0] <= 2:
        return np.zeros((X.shape[0],), dtype=bool)
    dt = 1.0 / float(fps)
    V = np.gradient(X, dt, axis=0, edge_order=2)
    S = np.linalg.norm(V, axis=-1)
    mu = np.nanmean(S, axis=0)
    sd = np.nanstd(S, axis=0) + 1e-12
    Z = (S - mu) / sd
    spike = np.isfinite(Z) & (Z > float(z_thresh))
    fr = spike.sum(axis=1).astype(float) / float(max(X.shape[1], 1))
    return fr >= float(frac)

