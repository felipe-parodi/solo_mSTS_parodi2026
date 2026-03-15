"""Plotting helpers for Step-A exemplar figures.

Small, dependency-light plotting utilities that work with the returned model
objects from `gam_models.run_*` functions.
"""

from __future__ import annotations

import os
from typing import Sequence
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from freemat.single_fm.io.save import save_png_svg


def _domain_to_model_cols(X_domain: pd.DataFrame, colnames: list[str]) -> pd.DataFrame:
    """Map a domain-named DataFrame to the model's x0..xN column schema.

    Assumes `X_domain` columns are ordered to match training order for the
    model, and `colnames` are the patsy/pyGAM training names (e.g., x0..x6).
    """
    if not colnames:
        return X_domain.copy()
    D = len(colnames)
    # Use position-based mapping: ith domain column -> colnames[i]
    arr = X_domain.iloc[:, :D].to_numpy(float)
    return pd.DataFrame({colnames[i]: arr[:, i] for i in range(D)})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_theta_tuning(gam_M1: dict, unit_df: pd.DataFrame, out_path: str) -> None:
    """Plot tuning curve vs theta_look at low speed using M1 model.

    Builds a theta grid in [-pi, pi], sets yaw_rate=0, and predicts.
    """
    if not isinstance(gam_M1, dict):
        # Backwards compatibility: unsupported object without metadata
        return
    backend = gam_M1.get("backend", "")
    model = gam_M1.get("model")
    colnames = gam_M1.get("colnames", [])
    thetas = np.linspace(-np.pi, np.pi, 181)
    X_domain = pd.DataFrame({
        "sin1": np.sin(thetas),
        "cos1": np.cos(thetas),
        "sin2": np.sin(2 * thetas),
        "cos2": np.cos(2 * thetas),
        "sin3": np.sin(3 * thetas),
        "cos3": np.cos(3 * thetas),
        "yaw_rate": np.zeros_like(thetas),
    })
    # Training order for M1 was [sin1, cos1, sin2, cos2, sin3, cos3, yaw_rate]
    domain_cols = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "yaw_rate"]
    X_model = _domain_to_model_cols(X_domain[domain_cols], colnames)

    ci_lo = ci_hi = None
    if backend == "statsmodels" and model is not None:
        # The fit used only yaw_rate as smooth, Fourier as linear
        formula = gam_M1.get("formula")
        if not formula:
            return
        # Construct a compatible design matrix (no session scaling for this sweep)
        DM = dmatrix(formula, X_model, return_type="dataframe")
        y = model.predict(DM)
        try:
            sf = model.get_prediction(DM).summary_frame(alpha=0.05)
            ci_lo = sf["mean_ci_lower"].to_numpy(float)
            ci_hi = sf["mean_ci_upper"].to_numpy(float)
        except Exception:
            ci_lo = ci_hi = None
    elif backend == "pyGAM" and model is not None:
        Xnp = X_model[colnames].to_numpy(float) if colnames else X_domain[domain_cols].to_numpy(float)
        y = model.predict(Xnp)
        try:
            ci = model.confidence_intervals(Xnp, width=0.95)
            ci_lo, ci_hi = ci[:, 0], ci[:, 1]
        except Exception:
            ci_lo = ci_hi = None
    else:
        return

    _ensure_dir(out_path)
    fig = plt.figure(figsize=(4, 3))
    plt.plot(thetas, y, color="C0", lw=2)
    if ci_lo is not None and ci_hi is not None:
        plt.fill_between(thetas, ci_lo, ci_hi, color="C0", alpha=0.2, linewidth=0)
    plt.xlabel("theta_look (rad)")
    plt.ylabel("predicted FR")
    plt.title("M1: Look tuning @ low speed")
    plt.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def plot_partial(gam_obj: dict, unit_df: pd.DataFrame, term_name: str, out_path: str) -> None:
    """Plot a 1D partial dependence curve by varying one term.

    Holds other predictors at their sessionwide medians.
    """
    if not isinstance(gam_obj, dict):
        logging.warning("plot_partial: missing model dict for term %s; falling back to binned median.", term_name)
        _fallback_binned_partial(unit_df, term_name, out_path)
        return
    backend = gam_obj.get("backend", "")
    model = gam_obj.get("model")
    colnames = gam_obj.get("colnames", [])

    # Determine which model and columns to use
    if term_name in ("v_parallel", "v_perp", "v_z", "yaw_rate"):
        # M2 context: Fourier + yaw_rate + v_*
        domain_cols = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "yaw_rate", "v_parallel", "v_perp", "v_z"]
        base = {c: float(np.nanmedian(unit_df[c])) if c in unit_df.columns else 0.0 for c in domain_cols}
    elif term_name in ("align_cos", "speed"):
        # M3 context
        domain_cols = ["speed", "align_cos", "yaw_rate"]
        base = {c: float(np.nanmedian(unit_df[c])) if c in unit_df.columns else 0.0 for c in domain_cols}
    else:
        logging.warning("plot_partial: no usable model for term %s; falling back to binned median.", term_name)
        _fallback_binned_partial(unit_df, term_name, out_path)
        return

    v = unit_df[term_name].to_numpy(float) if term_name in unit_df.columns else np.array([])
    v = v[np.isfinite(v)]
    if v.size == 0:
        # Fallback grids for sparse predictors
        if term_name == "align_cos":
            grid = np.linspace(-1.0, 1.0, 100)
        elif term_name == "speed":
            sp = unit_df["speed"].to_numpy(float) if "speed" in unit_df.columns else np.array([])
            sp = sp[np.isfinite(sp)]
            if sp.size == 0:
                return
            lo, hi = np.nanpercentile(sp, [1.0, 99.0])
            grid = np.linspace(lo, hi, 100)
        else:
            return
    else:
        lo, hi = np.nanpercentile(v, [1.0, 99.0])
        grid = np.linspace(lo, hi, 100)

    rows = []
    for w in grid:
        r = base.copy()
        r[term_name] = float(w)
        rows.append(r)
    X_domain = pd.DataFrame(rows, columns=domain_cols)
    X_model = _domain_to_model_cols(X_domain[domain_cols], colnames)

    ci_lo = ci_hi = None
    if backend == "statsmodels" and model is not None:
        formula = gam_obj.get("formula")
        if not formula:
            logging.warning("plot_partial: statsmodels backend missing formula for %s; using fallback.", term_name)
            _fallback_binned_partial(unit_df, term_name, out_path)
            return
        DM = dmatrix(formula, X_model, return_type="dataframe")
        y = model.predict(DM)
        try:
            sf = model.get_prediction(DM).summary_frame(alpha=0.05)
            ci_lo = sf["mean_ci_lower"].to_numpy(float)
            ci_hi = sf["mean_ci_upper"].to_numpy(float)
        except Exception:
            ci_lo = ci_hi = None
    elif backend == "pyGAM" and model is not None:
        Xnp = X_model[colnames].to_numpy(float) if colnames else X_domain[domain_cols].to_numpy(float)
        y = model.predict(Xnp)
        try:
            ci = model.confidence_intervals(Xnp, width=0.95)
            ci_lo, ci_hi = ci[:, 0], ci[:, 1]
        except Exception:
            ci_lo = ci_hi = None
    else:
        logging.warning("plot_partial: unsupported backend for term %s; using fallback.", term_name)
        _fallback_binned_partial(unit_df, term_name, out_path)
        return

    # If predictions are all NaN or empty, fall back to binned visualization
    if y is None or (isinstance(y, np.ndarray) and not np.isfinite(y).any()):
        logging.warning("plot_partial: model produced no finite predictions for %s; using fallback.", term_name)
        _fallback_binned_partial(unit_df, term_name, out_path)
        return

    _ensure_dir(out_path)
    fig = plt.figure(figsize=(4, 3))
    plt.plot(grid, y, color="C1", lw=2)
    if ci_lo is not None and ci_hi is not None:
        plt.fill_between(grid, ci_lo, ci_hi, color="C1", alpha=0.2, linewidth=0)
    plt.xlabel(term_name)
    plt.ylabel("predicted FR")
    plt.title(f"Partial: {term_name}")
    plt.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def _fallback_binned_partial(unit_df: pd.DataFrame, term_name: str, out_path: str) -> None:
    """Fallback visualization: binned median FR vs term with IQR band.

    Used when a fitted model is unavailable; this shows the marginal
    relationship and guarantees a non-empty plot for debugging.
    """
    if term_name not in unit_df.columns or unit_df[[term_name, "fr"]].dropna().empty:
        logging.warning("plot_partial: no finite data for term %s; skipping plot.", term_name)
        return
    g = unit_df[[term_name, "fr"]].dropna()
    x = g[term_name].to_numpy(float)
    y = g["fr"].to_numpy(float)
    nbins = 25
    if term_name == "align_cos":
        lo, hi = -1.0, 1.0
    else:
        lo, hi = np.nanpercentile(x, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = np.nanmin(x), np.nanmax(x)
    edges = np.linspace(lo, hi, nbins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nbins, np.nan)
    p25 = np.full(nbins, np.nan)
    p75 = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if not np.any(m):
            continue
        yy = y[m]
        med[i] = np.nanmedian(yy)
        p25[i], p75[i] = np.nanpercentile(yy, [25.0, 75.0])
    _ensure_dir(out_path)
    fig = plt.figure(figsize=(4, 3))
    plt.hexbin(x, y, gridsize=40, cmap="Greys", mincnt=1)
    mfin = np.isfinite(med)
    if np.any(mfin):
        if np.isfinite(p25[mfin]).any() and np.isfinite(p75[mfin]).any():
            plt.fill_between(ctrs[mfin], p25[mfin], p75[mfin], color="C2", alpha=0.25, linewidth=0)
        plt.plot(ctrs[mfin], med[mfin], color="C2", lw=2)
    plt.xlabel(term_name)
    plt.ylabel("FR")
    plt.title(f"Fallback partial (marginal): {term_name}")
    plt.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


def scatter_yawrate_fr(unit_df: pd.DataFrame, out_path: str) -> None:
    """Hexbin scatter of yaw_rate vs FR with binned median and IQR bands."""
    g = unit_df[["yaw_rate", "fr"]].dropna()
    if g.empty:
        return
    x = g["yaw_rate"].to_numpy(float)
    y = g["fr"].to_numpy(float)

    # Define bins using central quantiles to avoid extreme tails
    q1, q99 = np.nanpercentile(x, [1.0, 99.0])
    if not np.isfinite(q1) or not np.isfinite(q99) or q1 >= q99:
        q1, q99 = np.nanmin(x), np.nanmax(x)
    nbins = 25
    edges = np.linspace(q1, q99, nbins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nbins, np.nan)
    p25 = np.full(nbins, np.nan)
    p75 = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if not np.any(m):
            continue
        yy = y[m]
        med[i] = np.nanmedian(yy)
        p25[i], p75[i] = np.nanpercentile(yy, [25.0, 75.0])

    _ensure_dir(out_path)
    fig = plt.figure(figsize=(4, 3))
    plt.hexbin(x, y, gridsize=40, cmap="viridis", mincnt=1)
    # IQR band and median line
    mfin = np.isfinite(med)
    if np.any(mfin):
        if np.isfinite(p25[mfin]).any() and np.isfinite(p75[mfin]).any():
            plt.fill_between(ctrs[mfin], p25[mfin], p75[mfin], color="w", alpha=0.25, linewidth=0)
        plt.plot(ctrs[mfin], med[mfin], color="w", lw=2)
    plt.xlabel("yaw_rate")
    plt.ylabel("FR")
    plt.title("Yaw rate vs FR (binned median ± IQR)")
    plt.tight_layout()
    save_png_svg(fig, out_path, dpi=150)
    plt.close(fig)


__all__ = ["plot_theta_tuning", "plot_partial", "scatter_yawrate_fr"]
