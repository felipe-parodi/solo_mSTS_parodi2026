from __future__ import annotations

"""Visualization helpers for spike events and rates.

These functions are intentionally minimal; downstream figure notebooks control
styling and saving (PNG/SVG) via external helpers.
"""

from typing import Iterable, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_event_raster(
    t: np.ndarray,
    unit_index: np.ndarray,
    *,
    t0: float,
    t1: float,
    unit_order: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a tick raster for events in [t0, t1).

    Args:
        t: Event times (E,) in seconds.
        unit_index: Unit indices (E,).
        t0: Start time (inclusive).
        t1: End time (exclusive).
        unit_order: Optional ordering of units (sequence of ints).
        ax: Optional Matplotlib axes.

    Returns:
        (fig, ax) for further customization/saving.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    else:
        fig = ax.figure
    mask = (t >= t0) & (t < t1)
    tt = t[mask]
    uu = unit_index[mask]
    # Determine ordering and filter events to units present in the ordering
    if unit_order is None:
        unit_order = np.unique(uu)
    unit_order = list(unit_order)
    order_map = {int(u): i for i, u in enumerate(unit_order)}
    if len(order_map) == 0 or tt.size == 0:
        # No events/units in window: draw empty axes
        ax.set_xlim(t0, t1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Units")
        return fig, ax
    # Keep only events for units included in unit_order
    keep_evt = np.array([int(u) in order_map for u in uu], dtype=bool)
    tt = tt[keep_evt]
    uu = uu[keep_evt]
    if tt.size == 0:
        ax.set_xlim(t0, t1)
        ax.set_ylim(-0.5, len(unit_order) - 0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Units")
        return fig, ax
    y = np.array([order_map[int(u)] for u in uu], dtype=float)
    # Use a LineCollection for efficient tick rendering
    seg0 = np.column_stack([tt, y - 0.4])
    seg1 = np.column_stack([tt, y + 0.4])
    segs = np.stack([seg0, seg1], axis=1)
    lc = LineCollection(segs, colors="k", linewidths=0.6)
    ax.add_collection(lc)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.5, len(unit_order) - 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Units")
    return fig, ax


def plot_rate_heatmap(
    fr: np.ndarray,
    t_axis: np.ndarray,
    *,
    unit_order: Optional[Sequence[int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a unit x time heatmap of firing rates.

    Args:
        fr: (T, U) array of firing rates (Hz) or z‑FR.
        t_axis: (T,) time in seconds (bin centers).
        unit_order: Optional ordering of units (indices into columns of fr).
        vmin, vmax: Color limits.
        ax: Optional axes.

    Returns:
        (fig, ax).
    """
    fr = np.asarray(fr)
    T, U = fr.shape
    if unit_order is None:
        unit_order = list(range(U))
    X = fr[:, unit_order].T  # (U, T)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    else:
        fig = ax.figure
    im = ax.imshow(
        X,
        aspect="auto",
        interpolation="nearest",
        extent=(float(t_axis[0]), float(t_axis[-1]), -0.5, len(unit_order) - 0.5),
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        origin="lower",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Units")
    fig.colorbar(im, ax=ax, label="FR" )
    return fig, ax


def plot_fr_histogram(
    fr: np.ndarray,
    *,
    bins: int = 50,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a population firing-rate histogram.

    Args:
        fr: Firing rates (N,) in Hz (NaNs allowed; ignored).
        bins: Number of histogram bins (default: 50).
        ax: Optional Matplotlib axes to draw on.

    Returns:
        (fig, ax) for further customization/saving.
    """
    fr = np.asarray(fr, dtype=float).reshape(-1)
    m = np.isfinite(fr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    else:
        fig = ax.figure
    if m.any():
        ax.hist(fr[m], bins=int(bins), color="#333333", alpha=0.9)
        mu = float(np.nanmean(fr[m]))
        ax.axvline(mu, color="#AA0000", linestyle="--", linewidth=1.0, label=f"mean={mu:.2f} Hz")
        try:
            import numpy as _np

            p95 = float(_np.nanpercentile(fr[m], 95.0))
            ax.axvline(p95, color="#5555AA", linestyle=":", linewidth=0.9, label=f"p95={p95:.2f} Hz")
        except Exception:
            pass
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Units")
    return fig, ax
