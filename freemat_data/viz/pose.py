from __future__ import annotations

"""Visualization helpers for 3D pose QA panels.

These helpers keep plotting minimal and let notebooks control styling
and saving.
"""

from typing import Optional, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
from ..transforms.pose import occupancy_projections


def plot_xy_occupancy(
    XY: np.ndarray,
    *,
    bins: int = 80,
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
):
    """2D occupancy histogram in the floor plane.

    Args:
        XY: (T, 2) array with X,Y in meters (Z‑up).
        bins: Number of bins per axis.
        range_x: Optional x‑axis range (min, max).
        range_y: Optional y‑axis range (min, max).
        ax: Optional axes.

    Returns:
        (fig, ax).
    """
    XY = np.asarray(XY, dtype=float).reshape(-1, 2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    else:
        fig = ax.figure
    rx = range_x if range_x is not None else (float(np.nanmin(XY[:, 0])), float(np.nanmax(XY[:, 0])))
    ry = range_y if range_y is not None else (float(np.nanmin(XY[:, 1])), float(np.nanmax(XY[:, 1])))
    h = ax.hist2d(XY[:, 0], XY[:, 1], bins=bins, range=[rx, ry], cmap="magma")
    fig.colorbar(h[3], ax=ax, label="Counts")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    return fig, ax


def plot_speed_timeseries(
    t: np.ndarray,
    speed: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
):
    """Plot speed vs time with simple labels.

    Args:
        t: Time (s), shape (T,).
        speed: Speed (m/s), shape (T,).
        ax: Optional axes.

    Returns:
        (fig, ax).
    """
    t = np.asarray(t, dtype=float)
    speed = np.asarray(speed, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2.5), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(t, speed, color="k", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    return fig, ax


def plot_valid_fraction(
    t: np.ndarray,
    valid_frac: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
):
    """Plot per-frame valid fraction over time.

    Args:
        t: Time (s), shape (T,).
        valid_frac: Fraction in [0,1], shape (T,).
        ax: Optional axes.

    Returns:
        (fig, ax).
    """
    t = np.asarray(t, dtype=float)
    valid_frac = np.asarray(valid_frac, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(t, valid_frac, lw=1.0, color="k")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Valid frac")
    return fig, ax


def plot_pose_traj_xy_xz(
    X: np.ndarray,
    *,
    joints: Optional[Sequence[int]] = None,
):
    """Plot XY and XZ trajectories for a subset of joints.

    Args:
        X: Pose positions (T, J, 3).
        joints: Optional joint indices; defaults to first min(5, J).

    Returns:
        (fig, axes) with two subplots (XY and XZ).
    """
    X = np.asarray(X, dtype=float)
    T, J, D = X.shape
    assert D == 3, "X must be (T,J,3)"
    if joints is None:
        k = min(5, J)
        joints = list(range(k))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for j in joints:
        axs[0].plot(X[:, j, 0], X[:, j, 1], alpha=0.7)
        axs[1].plot(X[:, j, 0], X[:, j, 2], alpha=0.7)
    axs[0].set_title("XY (top-down)")
    axs[1].set_title("XZ (side)")
    axs[0].set_xlabel("X"); axs[0].set_ylabel("Y")
    axs[1].set_xlabel("X"); axs[1].set_ylabel("Z")
    return fig, axs


def plot_occupancy_projections_grid(
    H: np.ndarray,
    *,
    edges: Optional[Sequence[np.ndarray]] = None,
    cmap: str = "magma",
    axarr: Optional[Sequence[plt.Axes]] = None,
):
    """Plot XY, XZ, and YZ occupancy projections side by side.

    Args:
        H: 3D occupancy counts (nx, ny, nz).
        cmap: Colormap for imshow (default: 'magma').
        axarr: Optional sequence of 3 Matplotlib axes; if None, a new figure
            with (1x3) axes is created.

    Returns:
        (fig, axes) where axes is a list [ax_xy, ax_xz, ax_yz].
    """
    H = np.asarray(H, dtype=float)
    if axarr is None:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    else:
        axes = list(axarr)
        fig = axes[0].figure
    Hxy, Hxz, Hyz = occupancy_projections(H)
    # Determine imshow extents if edges provided
    extent_xy = None
    extent_xz = None
    extent_yz = None
    if edges is not None and len(edges) == 3:
        xe, ye, ze = edges
        try:
            extent_xy = (float(xe[0]), float(xe[-1]), float(ye[0]), float(ye[-1]))
            extent_xz = (float(xe[0]), float(xe[-1]), float(ze[0]), float(ze[-1]))
            extent_yz = (float(ye[0]), float(ye[-1]), float(ze[0]), float(ze[-1]))
        except Exception:
            extent_xy = extent_xz = extent_yz = None
    im0 = axes[0].imshow(
        Hxy.T,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        aspect="equal",
        extent=extent_xy,
    )
    axes[0].set_title("XY")
    axes[0].set_xlabel("X (m)" if extent_xy else "X (bins)")
    axes[0].set_ylabel("Y (m)" if extent_xy else "Y (bins)")
    im1 = axes[1].imshow(
        Hxz.T,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        aspect="auto",
        extent=extent_xz,
    )
    axes[1].set_title("XZ")
    axes[1].set_xlabel("X (m)" if extent_xz else "X (bins)")
    axes[1].set_ylabel("Z (m)" if extent_xz else "Z (bins)")
    im2 = axes[2].imshow(
        Hyz.T,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        aspect="auto",
        extent=extent_yz,
    )
    axes[2].set_title("YZ")
    axes[2].set_xlabel("Y (m)" if extent_yz else "Y (bins)")
    axes[2].set_ylabel("Z (m)" if extent_yz else "Z (bins)")
    # Share a single colorbar
    for im in (im0, im1, im2):
        im.set_clim(vmin=float(np.nanmin(H)), vmax=float(np.nanmax(H)))
    cbar = fig.colorbar(im2, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("Counts")
    return fig, axes
