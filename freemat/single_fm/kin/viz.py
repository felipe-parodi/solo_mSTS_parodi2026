"""Extended QC visualization for kinematic features with Plotly HTML export.

Complements freemat/single_fm/viz/kin_diagnostics.py with:
- Feature correlation matrix (Plotly heatmap)
- Head-forward quiver plot in XY plane (Matplotlib)
- 3D trajectory with velocity vectors (Plotly)
- Occupancy heatmap (Matplotlib)
- Combined interactive HTML dashboard (Plotly)

Usage:
    >>> from freemat.single_fm.kin.viz import save_kin_qc_html
    >>> save_kin_qc_html(kin_df, X, head_forward, out_path="qc_dashboard.html")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore
    make_subplots = None  # type: ignore
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore
    HAS_MATPLOTLIB = False


# Feature groups for correlation matrix organization
DEFAULT_FEATURE_GROUPS: Dict[str, List[str]] = {
    "KIN": ["v_parallel", "v_perp", "v_z", "speed", "acceleration", "yaw_rate", "head_torso_delta"],
    "POS": ["pos_x", "pos_y", "pos_z", "distance_to_wall"],
    "GAZE": ["gaze_x", "gaze_y", "gaze_z", "gaze_elevation", "gaze_azimuth"],
}


def _ensure_plotly() -> None:
    """Raise ImportError if Plotly is not available."""
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for interactive visualizations. Install with: uv pip install plotly")


def _ensure_matplotlib() -> None:
    """Raise ImportError if Matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for static plots. Install with: uv pip install matplotlib")


def plot_correlation_matrix(
    kin_df: pd.DataFrame,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    title: str = "Feature Correlation Matrix",
) -> "go.Figure":
    """Create Plotly heatmap of feature correlations.

    Args:
        kin_df: DataFrame with kinematic features.
        feature_groups: Dict mapping group names to feature column names.
            If None, uses DEFAULT_FEATURE_GROUPS.
        title: Plot title.

    Returns:
        Plotly Figure with annotated heatmap.
    """
    _ensure_plotly()

    if feature_groups is None:
        feature_groups = DEFAULT_FEATURE_GROUPS

    # Collect all feature columns that exist in the DataFrame
    features = []
    for group_name in ["KIN", "POS", "GAZE"]:
        if group_name in feature_groups:
            for f in feature_groups[group_name]:
                if f in kin_df.columns:
                    features.append(f)

    if len(features) < 2:
        raise ValueError(f"Need at least 2 features for correlation matrix, found {len(features)}")

    # Compute correlation matrix
    corr_df = kin_df[features].corr()
    corr_matrix = corr_df.values

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(tickangle=45),
        width=700,
        height=600,
        margin=dict(l=100, r=50, t=80, b=100),
    )

    return fig


def plot_head_forward_quiver(
    com: np.ndarray,
    head_forward: np.ndarray,
    stride: int = 10,
    scale: float = 0.3,
    title: str = "Head-Forward Vectors (XY Plane)",
) -> Tuple["Figure", "Axes"]:
    """Create matplotlib quiver plot of head-forward vectors in XY plane.

    Args:
        com: Center of mass positions (T, 3).
        head_forward: Head-forward unit vectors (T, 3).
        stride: Plot every N-th vector to reduce clutter.
        scale: Arrow scale factor.
        title: Plot title.

    Returns:
        (Figure, Axes) tuple.
    """
    _ensure_matplotlib()

    com = np.asarray(com)
    head_forward = np.asarray(head_forward)

    # Subsample
    idx = np.arange(0, len(com), stride)
    x = com[idx, 0]
    y = com[idx, 1]
    u = head_forward[idx, 0]  # X component
    v = head_forward[idx, 1]  # Y component

    # Filter NaN
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    x, y, u, v = x[valid], y[valid], u[valid], v[valid]

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # Plot trajectory as faint line
    ax.plot(com[:, 0], com[:, 1], 'k-', alpha=0.2, lw=0.5, zorder=1)

    # Plot quiver
    q = ax.quiver(x, y, u, v, scale=1/scale, scale_units='xy',
                  angles='xy', color='#d62728', alpha=0.7, width=0.003, zorder=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_3d_trajectory_velocity(
    com: np.ndarray,
    speed: np.ndarray,
    stride: int = 5,
    title: str = "3D Trajectory Colored by Speed",
) -> "go.Figure":
    """Create Plotly 3D scatter of trajectory colored by speed.

    Args:
        com: Center of mass positions (T, 3).
        speed: Speed values (T,).
        stride: Plot every N-th point.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _ensure_plotly()

    com = np.asarray(com)
    speed = np.asarray(speed)

    # Subsample
    idx = np.arange(0, len(com), stride)
    x = com[idx, 0]
    y = com[idx, 1]
    z = com[idx, 2]
    c = speed[idx]

    # Filter NaN
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(c)
    x, y, z, c = x[valid], y[valid], z[valid], c[valid]

    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=c,
            colorscale='Viridis',
            colorbar=dict(title="Speed (m/s)"),
            opacity=0.7,
        ),
        hovertemplate="X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<br>Speed: %{marker.color:.2f} m/s<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=700,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def plot_occupancy_heatmap(
    com: np.ndarray,
    bins: int = 50,
    title: str = "XY Occupancy Heatmap",
) -> Tuple["Figure", "Axes"]:
    """Create 2D histogram of XY positions.

    Args:
        com: Center of mass positions (T, 3).
        bins: Number of bins per axis.
        title: Plot title.

    Returns:
        (Figure, Axes) tuple.
    """
    _ensure_matplotlib()

    com = np.asarray(com)
    x = com[:, 0]
    y = com[:, 1]

    # Filter NaN
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    h, xedges, yedges, im = ax.hist2d(
        x, y, bins=bins, cmap='hot', cmin=1
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Count")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect('equal')

    return fig, ax


def plot_feature_timeseries_plotly(
    kin_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    t_col: str = "t_bin_s",
    max_duration_s: float = 300.0,
    title: str = "Feature Time Series (First 5 min)",
) -> "go.Figure":
    """Create Plotly time series of multiple features.

    Args:
        kin_df: DataFrame with kinematic features.
        features: List of feature columns to plot. If None, uses defaults.
        t_col: Time column name.
        max_duration_s: Maximum duration to plot in seconds.
        title: Plot title.

    Returns:
        Plotly Figure.
    """
    _ensure_plotly()

    if features is None:
        features = ["v_parallel", "v_perp", "speed", "yaw_rate"]

    # Filter to existing columns
    features = [f for f in features if f in kin_df.columns]
    if len(features) == 0:
        raise ValueError("No valid features found in DataFrame")

    t = np.asarray(kin_df[t_col]) if t_col in kin_df.columns else np.arange(len(kin_df))

    # Limit duration
    mask = t <= (t[0] + max_duration_s) if len(t) > 0 else np.ones(len(kin_df), dtype=bool)
    t = t[mask]

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, feat in enumerate(features):
        y = np.asarray(kin_df[feat])[mask]
        fig.add_trace(go.Scatter(
            x=t, y=y,
            mode='lines',
            name=feat,
            line=dict(width=1, color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=900,
        height=400,
        margin=dict(l=60, r=50, t=80, b=60),
    )

    return fig


def save_kin_qc_html(
    kin_df: pd.DataFrame,
    com: np.ndarray,
    head_forward: np.ndarray,
    out_path: str,
    speed: Optional[np.ndarray] = None,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    session_id: str = "",
) -> None:
    """Save combined interactive HTML dashboard with all QC plots.

    Args:
        kin_df: DataFrame with kinematic features.
        com: Center of mass positions (T, 3).
        head_forward: Head-forward unit vectors (T, 3).
        out_path: Output HTML file path.
        speed: Optional speed array (T,). If None, computed from com.
        feature_groups: Feature group mapping for correlation matrix.
        session_id: Optional session ID for title.
    """
    _ensure_plotly()

    com = np.asarray(com)
    head_forward = np.asarray(head_forward)

    # Compute speed if not provided
    if speed is None:
        if "speed" in kin_df.columns:
            speed = np.asarray(kin_df["speed"])
        else:
            # Estimate from COM differences
            dt = 0.2  # Assume 5 Hz
            if "t_bin_s" in kin_df.columns:
                t = np.asarray(kin_df["t_bin_s"])
                dt = np.median(np.diff(t)) if len(t) > 1 else 0.2
            vel = np.gradient(com, dt, axis=0)
            speed = np.linalg.norm(vel, axis=1)

    # Create individual figures
    try:
        fig_corr = plot_correlation_matrix(kin_df, feature_groups)
    except Exception:
        fig_corr = None

    try:
        fig_3d = plot_3d_trajectory_velocity(com, speed)
    except Exception:
        fig_3d = None

    try:
        fig_ts = plot_feature_timeseries_plotly(kin_df)
    except Exception:
        fig_ts = None

    # Build combined HTML
    title = f"KIN QC Dashboard - {session_id}" if session_id else "KIN QC Dashboard"

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{title}</title>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #333; }",
        "h2 { color: #666; margin-top: 30px; }",
        ".plot-container { margin: 20px 0; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
    ]

    plot_id = 0

    if fig_ts is not None:
        html_parts.append("<h2>Feature Time Series</h2>")
        html_parts.append(f'<div id="plot{plot_id}" class="plot-container"></div>')
        html_parts.append("<script>")
        html_parts.append(f"Plotly.newPlot('plot{plot_id}', {fig_ts.to_json()});")
        html_parts.append("</script>")
        plot_id += 1

    if fig_corr is not None:
        html_parts.append("<h2>Feature Correlation Matrix</h2>")
        html_parts.append(f'<div id="plot{plot_id}" class="plot-container"></div>')
        html_parts.append("<script>")
        html_parts.append(f"Plotly.newPlot('plot{plot_id}', {fig_corr.to_json()});")
        html_parts.append("</script>")
        plot_id += 1

    if fig_3d is not None:
        html_parts.append("<h2>3D Trajectory (Colored by Speed)</h2>")
        html_parts.append(f'<div id="plot{plot_id}" class="plot-container"></div>')
        html_parts.append("<script>")
        html_parts.append(f"Plotly.newPlot('plot{plot_id}', {fig_3d.to_json()});")
        html_parts.append("</script>")
        plot_id += 1

    # Add summary stats
    n_valid = np.isfinite(head_forward).all(axis=1).sum()
    n_total = len(head_forward)
    html_parts.append("<h2>Summary Statistics</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Total frames: {n_total}</li>")
    html_parts.append(f"<li>Valid head-forward frames: {n_valid} ({100*n_valid/n_total:.1f}%)</li>")
    if "speed" in kin_df.columns:
        spd = kin_df["speed"]
        html_parts.append(f"<li>Mean speed: {spd.mean():.3f} m/s</li>")
        html_parts.append(f"<li>Max speed: {spd.max():.3f} m/s</li>")
    html_parts.append("</ul>")

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))


def save_matplotlib_plots(
    com: np.ndarray,
    head_forward: np.ndarray,
    out_dir: str,
    prefix: str = "kin_qc",
) -> Dict[str, str]:
    """Save matplotlib-based QC plots as PNG and SVG.

    Args:
        com: Center of mass positions (T, 3).
        head_forward: Head-forward unit vectors (T, 3).
        out_dir: Output directory.
        prefix: Filename prefix.

    Returns:
        Dict mapping plot names to saved paths.
    """
    import os
    _ensure_matplotlib()

    os.makedirs(out_dir, exist_ok=True)
    saved = {}

    # Head-forward quiver
    try:
        fig, ax = plot_head_forward_quiver(com, head_forward)
        base = os.path.join(out_dir, f"{prefix}_quiver")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base}.svg", bbox_inches="tight")
        plt.close(fig)
        saved["quiver"] = f"{base}.png"
    except Exception:
        pass

    # Occupancy heatmap
    try:
        fig, ax = plot_occupancy_heatmap(com)
        base = os.path.join(out_dir, f"{prefix}_occupancy")
        fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base}.svg", bbox_inches="tight")
        plt.close(fig)
        saved["occupancy"] = f"{base}.png"
    except Exception:
        pass

    return saved


__all__ = [
    "plot_correlation_matrix",
    "plot_head_forward_quiver",
    "plot_3d_trajectory_velocity",
    "plot_occupancy_heatmap",
    "plot_feature_timeseries_plotly",
    "save_kin_qc_html",
    "save_matplotlib_plots",
    "DEFAULT_FEATURE_GROUPS",
]
