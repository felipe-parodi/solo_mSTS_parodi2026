"""Grooming reciprocity / social ledger analysis.

The ledger variable tracks history-dependent grooming imbalance:
    GroomImbalance(t) = GroomGiven(t-W:t) - GroomReceived(t-W:t)

Positive values mean subject has been "net giver" over the recent window.
This captures social reciprocity dynamics distinct from instantaneous state.

Two modes:
    - Cumulative: GroomImbalance grows unboundedly over session
    - Windowed: Fixed lookback window W (e.g., 60 seconds)

Neural Test:
    Does neural activity predict GroomImbalance above instantaneous state + movement?
    Ridge encoding model with controls, report delta-R^2 by bank.

Usage:
    from analysis.interaction_summaries.ledger import (
        compute_grooming_imbalance,
        compute_grooming_imbalance_windowed,
        encode_ledger,
    )

    # Cumulative imbalance
    imbalance = compute_grooming_imbalance(
        labels=behavior_labels,
        partner_labels=partner_behavior_labels,
        bin_size_sec=1.0,
    )

    # Windowed imbalance (60s lookback)
    imbalance_win = compute_grooming_imbalance_windowed(
        labels=behavior_labels,
        partner_labels=partner_behavior_labels,
        bin_size_sec=1.0,
        window_sec=60.0,
    )

    # Encode in neural activity
    result = encode_ledger(
        neural_rates=X,
        imbalance=imbalance,
        controls=control_covariates,
        subject='M1',
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from .controls import ControlCovariates


@dataclass
class LedgerEncodingResult:
    """Result of encoding grooming imbalance in neural activity.

    Attributes:
        coef_imbalance: Coefficient for imbalance predictor per unit
        z_coef_imbalance: Z-scored coefficients
        r2_full: R^2 with all predictors (imbalance + controls)
        r2_controls: R^2 with controls only
        delta_r2: Incremental R^2 from imbalance (r2_full - r2_controls)
        unit_indices: Unit indices for each coefficient
        bank_labels: Bank labels ('dorsal', 'ventral') for each unit
        n_units: Number of units
        n_samples: Number of samples used
    """

    coef_imbalance: np.ndarray
    z_coef_imbalance: np.ndarray
    r2_full: np.ndarray
    r2_controls: np.ndarray
    delta_r2: np.ndarray
    unit_indices: np.ndarray
    bank_labels: Optional[List[str]]
    n_units: int
    n_samples: int

    def summary_by_bank(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics grouped by bank."""
        if self.bank_labels is None:
            return {"all": self._compute_stats(np.arange(self.n_units))}

        summary = {}
        for bank in ["dorsal", "ventral"]:
            mask = np.array([b == bank for b in self.bank_labels])
            if mask.any():
                summary[bank] = self._compute_stats(np.where(mask)[0])

        return summary

    def _compute_stats(self, indices: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics for subset of units."""
        return {
            "n_units": len(indices),
            "delta_r2_mean": float(np.nanmean(self.delta_r2[indices])),
            "delta_r2_std": float(np.nanstd(self.delta_r2[indices])),
            "coef_mean": float(np.nanmean(self.coef_imbalance[indices])),
            "coef_std": float(np.nanstd(self.coef_imbalance[indices])),
            "z_coef_mean": float(np.nanmean(self.z_coef_imbalance[indices])),
            "frac_positive": float(np.mean(self.coef_imbalance[indices] > 0)),
        }


def _is_groom_give(label: Optional[str]) -> bool:
    """Check if label indicates grooming given."""
    if label is None:
        return False
    s = str(label).lower()
    return ("groom" in s) and (
        "give" in s or s == "groom-give" or s == "groom_give"
    )


def _is_groom_receive(label: Optional[str]) -> bool:
    """Check if label indicates grooming received."""
    if label is None:
        return False
    s = str(label).lower()
    return ("groom" in s) and (
        "receive" in s or "recv" in s or s == "groom-receive" or s == "groom_receive"
    )


def compute_grooming_imbalance(
    labels: List[Optional[str]],
    partner_labels: Optional[List[Optional[str]]] = None,
    bin_size_sec: float = 1.0,
    normalize: bool = False,
) -> np.ndarray:
    """Compute cumulative grooming imbalance time series.

    Imbalance = cumulative_give - cumulative_receive

    Args:
        labels: Behavior labels for focal subject (length n_bins)
        partner_labels: Behavior labels for partner (optional, used for fallback)
        bin_size_sec: Bin size in seconds (for scaling to duration)
        normalize: If True, return normalized: (give - recv) / (give + recv + eps)

    Returns:
        Grooming imbalance time series, shape (n_bins,)
    """
    n_bins = len(labels)

    # Detect give/receive from labels
    give_mask = np.array([_is_groom_give(lbl) for lbl in labels], dtype=bool)
    recv_mask = np.array([_is_groom_receive(lbl) for lbl in labels], dtype=bool)

    # Fallback: if no explicit give/receive, use partner labels
    if not give_mask.any() and partner_labels is not None:
        # Any "groom" in self → giving; any "groom" in partner → receiving
        give_mask = np.array(
            ["groom" in (str(lbl).lower() if lbl else "") for lbl in labels],
            dtype=bool,
        )
        recv_mask = np.array(
            ["groom" in (str(lbl).lower() if lbl else "") for lbl in partner_labels],
            dtype=bool,
        )

    # Cumulative durations
    cum_give = np.cumsum(give_mask.astype(float)) * bin_size_sec
    cum_recv = np.cumsum(recv_mask.astype(float)) * bin_size_sec

    if normalize:
        denom = cum_give + cum_recv + 1e-12
        return (cum_give - cum_recv) / denom

    return cum_give - cum_recv


def compute_grooming_imbalance_windowed(
    labels: List[Optional[str]],
    partner_labels: Optional[List[Optional[str]]] = None,
    bin_size_sec: float = 1.0,
    window_sec: float = 60.0,
    normalize: bool = False,
) -> np.ndarray:
    """Compute windowed grooming imbalance time series.

    Imbalance(t) = sum(give in [t-W:t]) - sum(receive in [t-W:t])

    Args:
        labels: Behavior labels for focal subject (length n_bins)
        partner_labels: Behavior labels for partner (optional, for fallback)
        bin_size_sec: Bin size in seconds
        window_sec: Lookback window in seconds
        normalize: If True, return normalized: (give - recv) / (give + recv + eps)

    Returns:
        Grooming imbalance time series, shape (n_bins,)
    """
    n_bins = len(labels)
    window_bins = int(round(window_sec / bin_size_sec))

    # Detect give/receive
    give_mask = np.array([_is_groom_give(lbl) for lbl in labels], dtype=bool)
    recv_mask = np.array([_is_groom_receive(lbl) for lbl in labels], dtype=bool)

    # Fallback
    if not give_mask.any() and partner_labels is not None:
        give_mask = np.array(
            ["groom" in (str(lbl).lower() if lbl else "") for lbl in labels],
            dtype=bool,
        )
        recv_mask = np.array(
            ["groom" in (str(lbl).lower() if lbl else "") for lbl in partner_labels],
            dtype=bool,
        )

    # Compute windowed sums using uniform filter (causal: sum over past W bins)
    # uniform_filter is centered, so we shift to make it causal
    give_float = give_mask.astype(float)
    recv_float = recv_mask.astype(float)

    # Causal windowed sum: use cumsum trick
    cum_give = np.cumsum(give_float)
    cum_recv = np.cumsum(recv_float)

    # sum(t-W:t) = cumsum(t) - cumsum(t-W-1), handle edge cases
    shifted_give = np.zeros_like(cum_give)
    shifted_give[window_bins:] = cum_give[:-window_bins]
    win_give = (cum_give - shifted_give) * bin_size_sec

    shifted_recv = np.zeros_like(cum_recv)
    shifted_recv[window_bins:] = cum_recv[:-window_bins]
    win_recv = (cum_recv - shifted_recv) * bin_size_sec

    if normalize:
        denom = win_give + win_recv + 1e-12
        return (win_give - win_recv) / denom

    return win_give - win_recv


def encode_ledger(
    neural_rates: np.ndarray,
    imbalance: np.ndarray,
    controls: ControlCovariates,
    subject: str,
    current_state_labels: Optional[List[str]] = None,
    unit_metadata: Optional[List[Dict[str, Any]]] = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    include_ar1: bool = True,
) -> LedgerEncodingResult:
    """Encode grooming imbalance in neural activity with controls.

    Fits ridge regression for each unit:
        y_u[t] ~ imbalance[t] + controls[t] + [current_state] + [AR(1)]

    Compares to controls-only model to compute delta-R^2.

    Args:
        neural_rates: Neural firing rates, shape (n_bins, n_units)
        imbalance: Grooming imbalance time series, shape (n_bins,)
        controls: ControlCovariates object
        subject: Subject identifier for control selection
        current_state_labels: Current behavior state labels (optional nuisance)
        unit_metadata: List of dicts with unit info (must have 'area' or 'bank' key)
        cv_folds: Number of CV folds for evaluation
        ridge_alpha: Ridge regularization strength
        include_ar1: If True, include AR(1) history term

    Returns:
        LedgerEncodingResult with encoding statistics
    """
    n_bins, n_units = neural_rates.shape

    if imbalance.shape[0] != n_bins:
        raise ValueError(f"Imbalance length {imbalance.shape[0]} != n_bins {n_bins}")

    # Get control features
    C = controls.get_for_subject(subject)

    # Z-score imbalance
    imb_z = (imbalance - np.nanmean(imbalance)) / (np.nanstd(imbalance) + 1e-12)

    # Build design matrix
    # Full model: [imbalance, controls, optional state dummies, optional AR(1)]
    X_imb = imb_z.reshape(-1, 1)
    X_full = np.hstack([X_imb, C])

    # Controls-only model
    X_ctrl = C.copy()

    # One-hot encode current state if provided
    if current_state_labels is not None:
        unique_states = sorted(set(s for s in current_state_labels if s))
        if len(unique_states) > 1:
            state_to_idx = {s: i for i, s in enumerate(unique_states)}
            state_onehot = np.zeros((n_bins, len(unique_states)), dtype=float)
            for t, s in enumerate(current_state_labels):
                if s in state_to_idx:
                    state_onehot[t, state_to_idx[s]] = 1.0
            X_full = np.hstack([X_full, state_onehot])
            X_ctrl = np.hstack([X_ctrl, state_onehot])

    # Create CV folds (temporal)
    fold_size = n_bins // cv_folds
    folds = []
    for i in range(cv_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < cv_folds - 1 else n_bins
        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_bins)])
        folds.append((train_idx, test_idx))

    # Fit per unit
    coef_imb = np.zeros(n_units)
    r2_full = np.zeros(n_units)
    r2_ctrl = np.zeros(n_units)

    for ui in range(n_units):
        y = neural_rates[:, ui]

        # Add AR(1) if requested
        if include_ar1:
            y_lag = np.roll(y, 1)
            y_lag[0] = np.nanmean(y)
            X_full_unit = np.hstack([X_full, y_lag.reshape(-1, 1)])
            X_ctrl_unit = np.hstack([X_ctrl, y_lag.reshape(-1, 1)])
        else:
            X_full_unit = X_full
            X_ctrl_unit = X_ctrl

        # CV evaluation
        ss_res_full, ss_res_ctrl, ss_tot = 0.0, 0.0, 0.0
        coef_sum = 0.0
        coef_count = 0

        for train_idx, test_idx in folds:
            # Fit full model
            X_tr, y_tr = X_full_unit[train_idx], y[train_idx]
            X_te, y_te = X_full_unit[test_idx], y[test_idx]

            XTX = X_tr.T @ X_tr
            reg = ridge_alpha * np.eye(XTX.shape[0])
            w_full = np.linalg.solve(XTX + reg, X_tr.T @ y_tr)

            y_pred_full = X_te @ w_full
            ss_res_full += np.sum((y_te - y_pred_full) ** 2)

            # Store imbalance coefficient (first column)
            coef_sum += w_full[0]
            coef_count += 1

            # Fit controls-only model
            X_tr_c, X_te_c = X_ctrl_unit[train_idx], X_ctrl_unit[test_idx]
            XTX_c = X_tr_c.T @ X_tr_c
            reg_c = ridge_alpha * np.eye(XTX_c.shape[0])
            w_ctrl = np.linalg.solve(XTX_c + reg_c, X_tr_c.T @ y_tr)

            y_pred_ctrl = X_te_c @ w_ctrl
            ss_res_ctrl += np.sum((y_te - y_pred_ctrl) ** 2)

            ss_tot += np.sum((y_te - np.mean(y_tr)) ** 2)

        coef_imb[ui] = coef_sum / max(1, coef_count)
        r2_full[ui] = 1 - ss_res_full / (ss_tot + 1e-12)
        r2_ctrl[ui] = 1 - ss_res_ctrl / (ss_tot + 1e-12)

    delta_r2 = r2_full - r2_ctrl

    # Z-score coefficients
    mu_coef = np.nanmean(coef_imb)
    sd_coef = np.nanstd(coef_imb)
    z_coef = (coef_imb - mu_coef) / (sd_coef + 1e-12)

    # Extract bank labels
    bank_labels = None
    if unit_metadata is not None:
        bank_labels = []
        for meta in unit_metadata:
            bank = meta.get("area", meta.get("bank", "unknown"))
            bank_labels.append(str(bank).lower())

    return LedgerEncodingResult(
        coef_imbalance=coef_imb,
        z_coef_imbalance=z_coef,
        r2_full=r2_full,
        r2_controls=r2_ctrl,
        delta_r2=delta_r2,
        unit_indices=np.arange(n_units),
        bank_labels=bank_labels,
        n_units=n_units,
        n_samples=n_bins,
    )
