"""Matplotlib helpers for 3D environment plots (pose + cameras).

These utilities complement freemat_data.vis (geometry builders) by
providing thin Matplotlib renderers for camera gizmos, hex outlines,
and publication-ready 3D CoM trajectory panels.

Example:
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    >>> from freemat_data.vis_mpl import (
    ...     plot_com_trajectory_3d, draw_camera_gizmos_mpl,
    ...     plot_hex_outline_mpl, set_axes_equal_3d,
    ... )
    >>> fig = plt.figure(figsize=(4.0, 3.6))
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> plot_com_trajectory_3d(ax, com_xyz)
    >>> draw_camera_gizmos_mpl(ax, calib_upr, scale=1.0)
    >>> plot_hex_outline_mpl(ax, plane, hex_uv)
    >>> set_axes_equal_3d(ax)
    >>> plt.show()
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .vis import camera_gizmo_segments, uv_to_xyz, foot_of_perpendicular_to_plane
from .skeletons import skeleton_style_from_names, resolve_skeleton_style_for_pose


def _rgba_to_mpl(col: str) -> Tuple[float, float, float, float]:
    """Convert CSS-like 'rgba(r,g,b,a)' string to Matplotlib RGBA tuple.

    Args:
        col: Color string like 'rgba(80,80,80,0.7)'.

    Returns:
        (r, g, b, a) each in [0, 1]. Falls back to gray if parsing fails.
    """
    try:
        if not isinstance(col, str) or not col.startswith("rgba("):
            raise ValueError("unexpected color format")
        vals = col[5:-1].split(",")
        r = float(vals[0].strip()) / 255.0
        g = float(vals[1].strip()) / 255.0
        b = float(vals[2].strip()) / 255.0
        a = float(vals[3].strip())
        return (r, g, b, a)
    except Exception:
        return (0.5, 0.5, 0.5, 0.7)


def draw_camera_gizmos_mpl(
    ax: plt.Axes,
    calib_upr: Dict[str, dict],
    *,
    scale: float,
    ref_camera: Optional[str] = None,
    linewidth: float = 0.8,
) -> None:
    """Draw simple frustum wireframes for each camera.

    Args:
        ax: 3D Matplotlib axes (projection='3d').
        calib_upr: Upright calibration dict per camera.
        scale: Scene scale multiplier for frustum size.
        ref_camera: Optional camera name to highlight.
        linewidth: Line width for gizmo edges.
    """
    if not isinstance(calib_upr, dict) or not calib_upr:
        return
    segs = camera_gizmo_segments(calib_upr, scale=scale, ref_camera=ref_camera)
    for p0, p1, col in segs:
        c = _rgba_to_mpl(col)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=c, linewidth=linewidth, alpha=c[3])


def plot_hex_outline_mpl(
    ax: plt.Axes,
    plane: Dict[str, float],
    hex_uv: np.ndarray,
    *,
    color: str = "#888888",
    linewidth: float = 1.0,
) -> None:
    """Draw hex polygon outline on the upright floor plane.

    Args:
        ax: 3D Matplotlib axes (projection='3d').
        plane: Upright plane dict (a,b,c,d).
        hex_uv: Hex vertices (K,2).
        color: Line color.
        linewidth: Line width.
    """
    if hex_uv is None or not isinstance(hex_uv, np.ndarray) or hex_uv.ndim != 2 or hex_uv.shape[1] != 2:
        return
    H = uv_to_xyz(hex_uv[:, 0], hex_uv[:, 1], plane)
    Hc = np.vstack([H, H[0]])  # close polygon
    ax.plot(Hc[:, 0], Hc[:, 1], Hc[:, 2], color=color, linewidth=linewidth, alpha=0.9)


def set_axes_equal_3d(ax: plt.Axes, X: Optional[np.ndarray] = None, pad: float = 0.05) -> None:
    """Set equal aspect for 3D axes based on data extent.

    Args:
        ax: 3D Matplotlib axes (projection='3d').
        X: Optional (N,3) array to compute extents; if None, uses axis limits.
        pad: Fractional padding added to each axis range.
    """
    if X is not None and isinstance(X, np.ndarray) and X.size > 0:
        mn = np.nanmin(X, axis=0)
        mx = np.nanmax(X, axis=0)
    else:
        mn = np.array([ax.get_xlim3d()[0], ax.get_ylim3d()[0], ax.get_zlim3d()[0]], dtype=float)
        mx = np.array([ax.get_xlim3d()[1], ax.get_ylim3d()[1], ax.get_zlim3d()[1]], dtype=float)
    center = 0.5 * (mn + mx)
    span = np.max(mx - mn)
    span = float(span) * (1.0 + float(pad))
    lo = center - 0.5 * span
    hi = center + 0.5 * span
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    # For newer Matplotlib, also set box aspect
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def plot_com_trajectory_3d(
    ax: plt.Axes,
    com_xyz: np.ndarray,
    *,
    color: str = "#2E4057",
    line_alpha: float = 0.9,
    linewidth: float = 0.8,
    scatter_step: int = 10,
    scatter_alpha: float = 0.08,
    scatter_size: float = 2.0,
) -> None:
    """Plot a clean 3D CoM trajectory with a light occupancy trace.

    Args:
        ax: 3D Matplotlib axes (projection='3d').
        com_xyz: (T,3) CoM in meters (NaNs allowed; they are skipped).
        color: Line color for the main trajectory.
        line_alpha: Alpha for the line.
        linewidth: Line width.
        scatter_step: Subsampling for light occupancy points (<=1 disables scatter).
        scatter_alpha: Alpha for the scatter points.
        scatter_size: Marker size for scatter points.
    """
    if com_xyz is None:
        return
    C = np.asarray(com_xyz, dtype=float).reshape(-1, 3)
    m = np.isfinite(C).all(axis=1)
    if not m.any():
        return
    CC = C[m]
    ax.plot(CC[:, 0], CC[:, 1], CC[:, 2], color=color, alpha=line_alpha, linewidth=linewidth)
    if isinstance(scatter_step, int) and scatter_step > 1 and CC.shape[0] >= scatter_step:
        idx = np.arange(0, CC.shape[0], scatter_step)
        ax.scatter(CC[idx, 0], CC[idx, 1], CC[idx, 2], s=scatter_size, c=color, alpha=scatter_alpha, depthshade=False)


def draw_skeleton_mpl(
    ax: plt.Axes,
    X_frame: np.ndarray,
    *,
    joint_names: Optional[Sequence[str]] = None,
    confidences: Optional[np.ndarray] = None,
    min_conf: float = 0.0,
    linewidth: float = 1.2,
    markersize: float = 8.0,
    with_labels: bool = False,
    style: Optional[Dict[str, object]] = None,
) -> None:
    """Draw a single 3D pose frame with colored skeleton and keypoints.

    Args:
        ax: 3D axes (projection='3d').
        X_frame: (J,3) joint coordinates.
        joint_names: Optional list of J names.
        confidences: Optional (J,) in [0,1].
        min_conf: Hide joints/edges below this threshold.
        linewidth: Edge width.
        markersize: Joint marker size.
        with_labels: If True, annotate joint names.
    """
    if X_frame is None:
        return
    P = np.asarray(X_frame, dtype=float).reshape(-1, 3)
    J = P.shape[0]
    conf = None
    if confidences is not None:
        c = np.asarray(confidences, dtype=float).reshape(-1)
        conf = c if c.shape[0] == J else None
    # Resolve style with a pragmatic fallback for 17-joint poses if not provided
    if style is not None and isinstance(style, dict):
        sty = style
    else:
        sty = resolve_skeleton_style_for_pose(joint_names or [], J)
    edges = sty.get("edges", [])
    jcols = sty.get("joint_colors", [])
    ecols = sty.get("edge_colors", [])
    # Draw edges
    for k, (a, b) in enumerate(edges):
        if a >= J or b >= J:
            continue
        if conf is not None and (conf[a] < min_conf or conf[b] < min_conf):
            continue
        # Skip edges if either endpoint has non‑finite coordinates
        if not (np.isfinite(P[a]).all() and np.isfinite(P[b]).all()):
            continue
        col = ecols[k] if k < len(ecols) else (0.5, 0.5, 0.5, 0.9)
        if isinstance(col, str):
            col = col
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]], color=col, linewidth=linewidth, alpha=0.95)
    # Draw joints
    for j in range(J):
        if conf is not None and conf[j] < min_conf:
            continue
        if not np.isfinite(P[j]).all():
            continue
        col = jcols[j] if j < len(jcols) else '#7F7F7F'
        ax.plot([P[j, 0]], [P[j, 1]], [P[j, 2]], marker='o', markersize=markersize, color=col, markeredgecolor='white', markeredgewidth=0.4)
        if with_labels and joint_names is not None and j < len(joint_names):
            ax.text(P[j, 0], P[j, 1], P[j, 2], str(joint_names[j]), fontsize=7)


def draw_joint_height_lines_mpl(
    ax: plt.Axes,
    X_frame: np.ndarray,
    *,
    plane: Dict[str, float],
    color: str = "#777777",
    alpha: float = 0.7,
    linewidth: float = 0.8,
    max_lines: Optional[int] = None,
) -> None:
    """Draw vertical (normal) line segments from each joint to the floor plane.

    Args:
        ax: 3D Matplotlib axes (projection='3d').
        X_frame: (J,3) joint coordinates.
        plane: Upright plane dict (a,b,c,d).
        color: Line color.
        alpha: Line alpha.
        linewidth: Line width.
        max_lines: Optional cap on the number of joints to draw lines for.
    """
    if X_frame is None:
        return
    P = np.asarray(X_frame, dtype=float).reshape(-1, 3)
    if P.size == 0:
        return
    if isinstance(max_lines, int) and max_lines > 0 and P.shape[0] > max_lines:
        idx = np.linspace(0, P.shape[0] - 1, max_lines).astype(int)
        P = P[idx]
    F = foot_of_perpendicular_to_plane(P, plane)
    for i in range(P.shape[0]):
        ax.plot([P[i, 0], F[i, 0]], [P[i, 1], F[i, 1]], [P[i, 2], F[i, 2]], color=color, alpha=alpha, linewidth=linewidth)
