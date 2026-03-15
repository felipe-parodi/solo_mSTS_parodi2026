"""QC spike plot functions — extracted from qc_spikes.py (W1.4 refactor).

Contains all visualization functions for spike QC analysis:

- ``plot_fr_heatmap``: Firing rate heatmap sorted by mean FR
- ``plot_summary_panels``: 2x2 summary (active units, pop FR, histogram, breakdown)
- ``plot_active_units``: Active units over time
- ``plot_population_fr``: Population FR trace with smoothing
- ``plot_fr_hist``: Mean FR histogram (log x)
- ``plot_drop_breakdown``: Horizontal bar chart of drop reasons
- ``plot_scatter_panels``: 4-panel scatter of QC metrics
- ``raster_kept_units``: Spike raster for kept units with baseline shading
- ``try_plot``: Safe plot wrapper that logs errors without aborting
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .baseline_detection import (
    BaselineDetectionError,
    compute_raster_window,
    detect_baseline_span,
    validate_baseline_for_window,
)
from .. import (
    get_syncmap_from_session,
    infer_logger,
    load_banks_config,
    load_spikes_from_session,
    map_channels_to_bank,
    map_spikes_to_video,
)
from ..types import SessionRef


# ---------------------------------------------------------------------------
# Simple QC plots
# ---------------------------------------------------------------------------


def plot_fr_heatmap(
    fr: np.ndarray,
    centers: np.ndarray,
    mean_fr: np.ndarray,
    kept: np.ndarray,
    *,
    out_path: Path,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Save a FR heatmap (units x time), sorted by mean FR.

    Args:
        fr: (T, U) firing rate matrix.
        centers: (T,) bin centers (unused here, kept for parity with callers).
        mean_fr: (U,) mean FR per unit for sorting.
        kept: (U,) boolean mask of kept units.
        out_path: Output image path.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag (e.g., "M1") for title.
    """
    if fr.size == 0 or mean_fr.size == 0:
        return
    order = np.argsort(mean_fr)[::-1]
    X = fr[:, order].T  # (U,T)
    n_kept = int(np.sum(kept))
    n_total = int(kept.size)
    fig = plt.figure(figsize=(12, max(4, 0.15 * X.shape[0])))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(X, cmap="mako", cbar=True, ax=ax)
    ax.set_xlabel("Time (bins)")
    ax.set_ylabel("Units (sorted by mean FR)")
    title = f"FR heatmap (5-min bins). Kept {n_kept}/{n_total}"
    if subject:
        title = f"{subject} - {title}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_summary_panels(
    fr: np.ndarray,
    centers: np.ndarray,
    mean_fr: np.ndarray,
    keep: np.ndarray,
    *,
    active_threshold_hz: float,
    smooth_sigma_bins: float,
    breakdown: Dict[str, int],
    out_path: Path,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Save a 2x2 summary figure.

    Panels:
      - Active units over time (>= threshold)
      - Population FR trace (raw and optional smoothed)
      - Mean FR histogram (log x)
      - Drop breakdown by criterion

    Args:
        fr: (T, U) firing rate matrix.
        centers: (T,) bin centers in seconds.
        mean_fr: (U,) mean FR per unit.
        keep: (U,) boolean keep mask.
        active_threshold_hz: FR threshold for counting active units.
        smooth_sigma_bins: Gaussian sigma (bins) for pop FR smoothing.
        breakdown: Dict of drop-reason counts.
        out_path: Output image path.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag for title.
    """
    if fr.size == 0:
        return
    X = np.asarray(fr, dtype=float)
    c = np.asarray(centers, dtype=float).reshape(-1)
    if X.shape[0] != c.size and c.size > 0:
        m = int(min(X.shape[0], c.size))
        X = X[:m, :]
        c = c[:m]
    valid = np.isfinite(X)
    n_active = np.sum(valid & (X >= float(active_threshold_hz)), axis=1)
    pop = np.nansum(np.where(valid, X, 0.0), axis=1)
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        R = max(1, int(round(4 * smooth_sigma_bins)))
        kx = np.arange(-R, R + 1, dtype=float)
        k = np.exp(-0.5 * (kx / float(smooth_sigma_bins)) ** 2)
        k /= max(np.sum(k), 1.0)
        pop_sm = np.convolve(pop, k, mode="same")
    else:
        pop_sm = None

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    ax0 = axs[0, 0]
    x0 = c
    if x0.size != n_active.size:
        m = int(min(x0.size, n_active.size))
        x0 = x0[:m]
        n_active = n_active[:m]
    ax0.plot(x0, n_active, color="#2a9d8f")
    ax0.set_title(f"Active units over time (>={active_threshold_hz} Hz)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("# active units")

    ax1 = axs[0, 1]
    x1, y1 = c, pop
    if x1.size != y1.size:
        m = int(min(x1.size, y1.size))
        x1, y1 = x1[:m], y1[:m]
    ax1.plot(x1, y1, color="#1f77b4", alpha=0.8, label="raw")
    if pop_sm is not None:
        xs, ys = c, pop_sm
        if xs.size != ys.size:
            m = int(min(xs.size, ys.size))
            xs, ys = xs[:m], ys[:m]
        ax1.plot(xs, ys, color="#d62728", linewidth=2.0, label="smoothed")
    ax1.set_title("Population FR (sum across units)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Hz")
    ax1.legend(loc="upper right")

    ax2 = axs[1, 0]
    pos = np.isfinite(mean_fr) & (mean_fr > 0)
    if np.any(pos):
        vals_all = mean_fr[pos]
        vmin = max(float(np.min(vals_all)), 1e-3)
        vmax = float(np.max(vals_all))
        if vmin >= vmax:
            vmax = vmin * 10.0
        bins = np.logspace(np.log10(vmin), np.log10(max(vmax, vmin * 10.0)), 40)
        drop_vals = mean_fr[pos & (~keep)]
        kept_vals = mean_fr[pos & keep]
        if drop_vals.size:
            ax2.hist(drop_vals, bins=bins, color="#e76f51", alpha=0.6, label="dropped")
        if kept_vals.size:
            ax2.hist(kept_vals, bins=bins, color="#2a9d8f", alpha=0.6, label="kept")
        ax2.set_xscale("log")
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No positive FRs", ha="center", va="center")
    ax2.set_title("Mean FR distribution")
    ax2.set_xlabel("Hz (log)")
    ax2.set_ylabel("Units")

    ax3 = axs[1, 1]
    keys = ["low_mean_fr", "high_isi_violation", "low_presence", "high_fr_cv", "cv_nan"]
    counts = [int(breakdown.get(k, 0)) for k in keys]
    ax3.barh(keys, counts, color="#e76f51")
    ax3.set_title("Dropped units by criterion")
    ax3.set_xlabel("Count")
    maxc = max(counts) if any(counts) else 1
    for i, cv in enumerate(counts):
        ax3.text(cv + 0.02 * maxc, i, str(cv), va="center")

    title = f"Spike QC summary — kept {int(np.sum(keep))}/{keep.size}"
    if subject:
        title = f"{subject} - {title}"
    fig.suptitle(title)
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_active_units(
    fr: np.ndarray,
    centers: np.ndarray,
    *,
    threshold_hz: float,
    out_path: Path,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Save a single-panel active-units-over-time plot.

    Args:
        fr: (T, U) firing rate matrix.
        centers: (T,) bin centers in seconds.
        threshold_hz: FR threshold for counting active units.
        out_path: Output image path.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag for title.
    """
    valid = np.isfinite(fr)
    n_active = np.sum(valid & (fr >= float(threshold_hz)), axis=1)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(centers, n_active, color="#2a9d8f")
    title = f"Active units over time (>={threshold_hz} Hz)"
    if subject:
        title = f"{subject} - {title}"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("# active units")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_population_fr(
    fr: np.ndarray,
    centers: np.ndarray,
    *,
    smooth_sigma_bins: float,
    out_path: Path,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Save population FR trace plot with optional smoothing.

    Args:
        fr: (T, U) firing rate matrix.
        centers: (T,) bin centers in seconds.
        smooth_sigma_bins: Gaussian sigma (bins) for smoothing.
        out_path: Output image path.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag for title.
    """
    X = np.asarray(fr, dtype=float)
    c = np.asarray(centers, dtype=float).reshape(-1)
    if X.shape[0] != c.size and c.size > 0:
        m = int(min(X.shape[0], c.size))
        X = X[:m, :]
        c = c[:m]
    valid = np.isfinite(X)
    pop = np.nansum(np.where(valid, X, 0.0), axis=1)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    x, y = c, pop
    if x.size != y.size:
        m = int(min(x.size, y.size))
        x, y = x[:m], y[:m]
    ax.plot(x, y, color="#1f77b4", alpha=0.8, label="raw")
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        R = max(1, int(round(4 * smooth_sigma_bins)))
        kx = np.arange(-R, R + 1, dtype=float)
        k = np.exp(-0.5 * (kx / float(smooth_sigma_bins)) ** 2)
        k /= max(np.sum(k), 1.0)
        pop_sm = np.convolve(pop, k, mode="same")
        xs, ys = c, pop_sm
        if xs.size != ys.size:
            m = int(min(xs.size, ys.size))
            xs, ys = xs[:m], ys[:m]
        ax.plot(xs, ys, color="#d62728", linewidth=2.0, label="smoothed")
    title = "Population FR (sum across units)"
    if subject:
        title = f"{subject} - {title}"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_fr_hist(
    mean_fr: np.ndarray,
    keep: np.ndarray,
    *,
    out_path: Path,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Save histogram of per-unit mean firing rates (log x), colored by keep/dropped.

    Args:
        mean_fr: (U,) mean FR per unit.
        keep: (U,) boolean keep mask.
        out_path: Output image path.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag for title.
    """
    pos = np.isfinite(mean_fr) & (mean_fr > 0)
    vals = mean_fr[pos]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    if vals.size:
        vmin = max(float(np.min(vals)), 1e-3)
        vmax = float(np.max(vals))
        if vmin >= vmax:
            vmax = vmin * 10.0
        bins = np.logspace(np.log10(vmin), np.log10(max(vmax, vmin * 10.0)), 40)
        drop_vals = mean_fr[pos & (~keep)]
        kept_vals = mean_fr[pos & keep]
        if drop_vals.size:
            ax.hist(drop_vals, bins=bins, color="#e76f51", alpha=0.6, label="dropped")
        if kept_vals.size:
            ax.hist(kept_vals, bins=bins, color="#2a9d8f", alpha=0.6, label="kept")
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No positive FRs", ha="center", va="center")
    title = "Mean FR distribution"
    if subject:
        title = f"{subject} - {title}"
    ax.set_title(title)
    ax.set_xlabel("Hz (log)")
    ax.set_ylabel("Units")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_drop_breakdown(
    breakdown_any: Dict[str, int],
    *,
    out_path: Path,
    breakdown_exclusive: Optional[Dict[str, int]] = None,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Save a horizontal bar chart of drop reasons.

    If ``breakdown_exclusive`` is provided, plots side-by-side bars for
    'any (non-exclusive)' and 'primary (exclusive)' counts with a legend.

    Args:
        breakdown_any: Dict of non-exclusive drop counts.
        out_path: Output image path.
        breakdown_exclusive: Optional dict of exclusive primary counts.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag for title.
    """
    keys = ["low_mean_fr", "high_isi_violation", "low_presence", "high_fr_cv", "cv_nan"]
    any_counts = [int(breakdown_any.get(k, 0)) for k in keys]
    exc_counts = [int(breakdown_exclusive.get(k, 0)) for k in keys] if breakdown_exclusive else None
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)
    y_pos = np.arange(len(keys))
    if exc_counts is None:
        ax.barh(keys, any_counts, color="#e76f51", alpha=0.8)
    else:
        ax.barh(y_pos + 0.2, any_counts, height=0.4, color="#e76f51", alpha=0.8, label="dropped (any)")
        ax.barh(y_pos - 0.2, exc_counts, height=0.4, color="#fb8500", alpha=0.8, label="dropped (primary)")
        ax.set_yticks(y_pos, keys)
        ax.legend(loc="lower right")
    title = "Dropped units by criterion"
    if subject:
        title = f"{subject} - {title}"
    ax.set_title(title)
    ax.set_xlabel("Count")
    maxc = max(any_counts + (exc_counts or [])) if (any_counts or exc_counts) else 1
    for i, cv in enumerate(any_counts):
        ax.text(cv + 0.02 * maxc, i + (0.2 if exc_counts is not None else 0.0), str(cv), va="center")
    if exc_counts is not None:
        for i, cv in enumerate(exc_counts):
            ax.text(cv + 0.02 * maxc, i - 0.2, str(cv), va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


def plot_scatter_panels(
    mean_fr: np.ndarray,
    presence: np.ndarray,
    fr_cv: np.ndarray,
    isi_rate: np.ndarray,
    keep: np.ndarray,
    *,
    out_path: Path,
    close_fig: bool = True,
    subject: Optional[str] = None,
) -> None:
    """Scatter panels to visualize criteria against mean FR (kept vs dropped).

    Args:
        mean_fr: (U,) mean FR per unit.
        presence: (U,) presence ratio per unit.
        fr_cv: (U,) FR CV per unit.
        isi_rate: (U,) ISI violation rate per unit.
        keep: (U,) boolean keep mask.
        out_path: Output image path.
        close_fig: Close matplotlib figure after saving.
        subject: Optional subject/logger tag for title.
    """
    fig = plt.figure(figsize=(12, 8))
    if subject:
        fig.suptitle(f"{subject} - QC scatter panels", fontsize=14, y=0.995)

    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(mean_fr, presence, c=np.where(keep, "#2a9d8f", "#e76f51"), s=8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_title("Presence ratio vs Mean FR")
    ax.set_xlabel("Mean FR (Hz)")
    ax.set_ylabel("Presence ratio")

    ax = fig.add_subplot(2, 2, 2)
    y = np.where(np.isfinite(fr_cv), fr_cv, np.nan)
    ax.scatter(mean_fr, y, c=np.where(keep, "#2a9d8f", "#e76f51"), s=8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_title("FR CV vs Mean FR")
    ax.set_xlabel("Mean FR (Hz)")
    ax.set_ylabel("FR CV")

    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(mean_fr, isi_rate, c=np.where(keep, "#2a9d8f", "#e76f51"), s=8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_title("ISI violation vs Mean FR")
    ax.set_xlabel("Mean FR (Hz)")
    ax.set_ylabel("ISI viol frac")

    ax = fig.add_subplot(2, 2, 4)
    vals = mean_fr[np.isfinite(mean_fr) & (mean_fr > 0)]
    if vals.size:
        vmin = max(float(np.min(vals)), 1e-3)
        vmax = float(np.max(vals))
        if vmin >= vmax:
            vmax = vmin * 10.0
        bins = np.logspace(np.log10(vmin), np.log10(max(vmax, vmin * 10.0)), 40)
        drop_vals = mean_fr[(mean_fr > 0) & np.isfinite(mean_fr) & (~keep)]
        kept_vals = mean_fr[(mean_fr > 0) & np.isfinite(mean_fr) & keep]
        if drop_vals.size:
            ax.hist(drop_vals, bins=bins, color="#e76f51", alpha=0.6, label="dropped")
        if kept_vals.size:
            ax.hist(kept_vals, bins=bins, color="#2a9d8f", alpha=0.6, label="kept")
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "No positive FRs", ha="center", va="center")
    ax.set_title("Mean FR (log)")
    from matplotlib.patches import Patch
    handles = [Patch(color="#2a9d8f", label="kept"), Patch(color="#e76f51", label="dropped")]
    fig.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot wrapper
# ---------------------------------------------------------------------------


def try_plot(
    func,
    *,
    name: str,
    manifest: Dict[str, str],
    path: Path,
    errlog: Path,
) -> None:
    """Run a plotting function and capture errors without aborting the run.

    On success, adds the saved path to the manifest under the provided name.
    On failure, appends the exception to ``errlog`` and continues.

    Special case: :class:`BaselineDetectionError` is re-raised to halt
    the entire QC process.

    Args:
        func: Zero-arg callable that generates and saves the plot.
        name: Key name for the manifest dict.
        manifest: Dict to record successful plot paths.
        path: Expected output path of the plot.
        errlog: Path to error log file.
    """
    try:
        func()
        if path.exists():
            manifest[name] = str(path.resolve())
    except BaselineDetectionError:
        raise
    except Exception as e:  # pragma: no cover
        with errlog.open("a", encoding="utf-8") as f:
            f.write(f"[{name}] Plotting failed: {e}\n")


# ---------------------------------------------------------------------------
# Raster plot (uses baseline_detection)
# ---------------------------------------------------------------------------


def raster_kept_units(
    t: np.ndarray,
    ui: np.ndarray,
    keep_mask: np.ndarray,
    *,
    session: SessionRef,
    logger_tag: Optional[str] = None,
    out_path: Path,
    max_units: int = 100,
    banks_yaml: Optional[str] = None,
    events_text_path: Optional[str] = None,
    window_s: Optional[float] = 600.0,
    close_fig: bool = True,
) -> None:
    """Render a raster plot for up to ``max_units`` kept units.

    - If sync map is present, map to video seconds; otherwise, use logger seconds.
    - If banks are provided/derivable, color units by bank.
    - If an events text file is provided and baseline can be derived, shade it.

    Args:
        t: Spike times array.
        ui: Unit index array.
        keep_mask: Boolean keep mask.
        session: Loaded session YAML.
        logger_tag: Logger tag (e.g., 'm1', 'm2').
        out_path: Output image path.
        max_units: Maximum units to display.
        banks_yaml: Path to banks YAML config.
        events_text_path: Path to events text file for baseline detection.
        window_s: Time window (s) for windowed raster; None = full session.
        close_fig: Close matplotlib figure after saving.
    """
    if t.size == 0 or ui.size == 0 or not np.any(keep_mask):
        return

    # Map to video seconds if possible
    sync = get_syncmap_from_session(session)
    logger = (logger_tag or infer_logger(session)) if session is not None else (logger_tag or "m1")

    t_mapped: np.ndarray
    try:
        if sync is not None:
            t_mapped = np.asarray(map_spikes_to_video(
                type("_tmp", (), {"t": t}),
                sync,
                logger=logger,
                session=session,
            ), dtype=float)
        else:
            t_mapped = np.asarray(t, dtype=float)
    except Exception:
        t_mapped = np.asarray(t, dtype=float)

    kept_idx = np.where(keep_mask)[0]
    if kept_idx.size == 0:
        return

    # Bank labels if available
    colors = {"dorsal": "#1f77b4", "ventral": "#ff7f0e", "unknown": "#7f7f7f"}
    bank_by_unit = _resolve_bank_labels(session, banks_yaml)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Baseline detection and windowing
    errlog = out_path.parent / "qc_plot_error.txt"
    baseline_span: Optional[Tuple[float, float]] = None
    window_start: Optional[float] = None
    window_end: Optional[float] = None

    try:
        if events_text_path:
            baseline_span = detect_baseline_span(
                events_text_path,
                session=session,
                logger_tag=logger_tag,
                sync=sync,
                errlog=errlog,
            )
    except Exception as e:  # pragma: no cover
        import traceback
        with errlog.open("a", encoding="utf-8") as f:
            f.write(f"[raster_kept] baseline parsing failed: {e}\n")
            f.write(traceback.format_exc() + "\n")

    # Validate baseline for windowed rasters
    validate_baseline_for_window(
        baseline_span,
        window_s=window_s,
        events_text_path=events_text_path,
        logger_tag=logger_tag,
        errlog=errlog,
    )

    # Compute window if baseline detected and windowing requested
    if (
        baseline_span is not None
        and all(np.isfinite(baseline_span))
        and window_s is not None
        and window_s > 0
    ):
        t_min = float(np.nanmin(t_mapped)) if t_mapped.size else baseline_span[0]
        t_max = float(np.nanmax(t_mapped)) if t_mapped.size else baseline_span[1]
        window_start, window_end = compute_raster_window(
            baseline_span, window_s, t_min, t_max,
        )

    # Restrict spikes to window and rank by activity
    sel_time = np.ones_like(t_mapped, dtype=bool)
    if window_start is not None and window_end is not None:
        sel_time = (t_mapped >= window_start) & (t_mapped <= window_end)
        counts = [int(np.sum(sel_time & (ui == int(u)))) for u in kept_idx]
        order = np.argsort(np.asarray(counts))[::-1]
        kept_idx = kept_idx[order]

    # Cap to max_units with balanced bank selection
    kept_idx = _select_balanced_units(kept_idx, bank_by_unit, max_units)

    # Render raster
    y = 0
    bank_counts = {"dorsal": 0, "ventral": 0, "unknown": 0}
    for u in kept_idx:
        sel = (ui == int(u)) & sel_time
        if not np.any(sel):
            continue
        tt = t_mapped[sel]
        color = "#3a3a3a"
        if bank_by_unit is not None and u < len(bank_by_unit):
            b = str(bank_by_unit[u])
            color = colors.get(b, "#3a3a3a")
            if b in bank_counts:
                bank_counts[b] += 1
            else:
                bank_counts["unknown"] += 1
        ax.scatter(tt, np.full_like(tt, y, dtype=float), marker="|", s=40, c=color, linewidths=0.7)
        y += 1

    ax.set_ylim(-1, y + 1)
    ax.set_ylabel("Kept units (subset)")
    ax.set_xlabel("Time (video s)" if sync is not None else "Time (logger s)")

    # Shade baseline
    if baseline_span is not None and all(np.isfinite(baseline_span)):
        s0, s1 = baseline_span
        ax.axvspan(float(s0), float(s1), color="#ffd166", alpha=0.25, label="baseline")

    # Apply window x-limits
    if window_start is not None and window_end is not None:
        ax.set_xlim(float(window_start), float(window_end))

    title = f"Spike raster (kept subset up to {int(max_units)})"
    if logger_tag:
        title = f"{logger_tag.upper()} - {title}"
    if window_start is not None and window_end is not None:
        title += f" — window [{window_start:.1f}, {window_end:.1f}] s"
    if bank_by_unit is not None:
        title += f" — dorsal {bank_counts['dorsal']}, ventral {bank_counts['ventral']}, unk {bank_counts['unknown']}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if close_fig:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Internal helpers for raster
# ---------------------------------------------------------------------------


def _resolve_bank_labels(session: SessionRef, banks_yaml: Optional[str] = None):
    """Resolve bank labels for units from session metadata or banks.yaml.

    Returns:
        Array-like of bank labels per unit, or None.
    """
    try:
        unit_channels = getattr(session, "_spike_unit_channels", None)
        if unit_channels is None:
            spikes_obj = load_spikes_from_session(session)
            if spikes_obj is not None and isinstance(spikes_obj.meta, dict) and "unit_channels" in spikes_obj.meta:
                unit_channels = np.asarray(spikes_obj.meta["unit_channels"])
        if unit_channels is not None:
            cfg_path = None
            if banks_yaml is not None and os.path.isfile(str(banks_yaml)):
                cfg_path = banks_yaml
            elif os.path.isfile("configs/banks.yaml"):
                cfg_path = "configs/banks.yaml"
            if cfg_path is not None:
                cfg = load_banks_config(cfg_path)
                return map_channels_to_bank(unit_channels, cfg)
    except Exception:
        pass
    return None


def _select_balanced_units(
    kept_idx: np.ndarray,
    bank_by_unit,
    max_units: int,
) -> np.ndarray:
    """Select up to max_units with balanced bank representation.

    Args:
        kept_idx: Indices of kept units (already ranked by preference).
        bank_by_unit: Array-like of bank labels, or None.
        max_units: Maximum number of units to return.

    Returns:
        Subset of kept_idx, at most max_units.
    """
    if kept_idx.size <= int(max_units):
        return kept_idx

    if bank_by_unit is not None:
        dorsal_mask = np.array([
            bank_by_unit[u] == "dorsal" if u < len(bank_by_unit) else False
            for u in kept_idx
        ])
        ventral_mask = np.array([
            bank_by_unit[u] == "ventral" if u < len(bank_by_unit) else False
            for u in kept_idx
        ])
        dorsal_idx = kept_idx[dorsal_mask]
        ventral_idx = kept_idx[ventral_mask]
        other_idx = kept_idx[~dorsal_mask & ~ventral_mask]

        per_bank = int(max_units) // 2
        selected = []
        if dorsal_idx.size > 0:
            selected.append(dorsal_idx[:per_bank])
        if ventral_idx.size > 0:
            selected.append(ventral_idx[:per_bank])
        remaining = int(max_units) - sum(len(s) for s in selected)
        if remaining > 0 and other_idx.size > 0:
            selected.append(other_idx[:remaining])
        return np.concatenate(selected) if selected else kept_idx[:int(max_units)]

    return kept_idx[:int(max_units)]
