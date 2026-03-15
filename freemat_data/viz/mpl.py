from __future__ import annotations

"""Matplotlib renderers (single namespace entry).

Consolidates MPL-based plotting for 3D scenes and 2D QA panels under
``freemat_data.viz`` while reusing existing implementation modules.
"""

# 3D scene renderers (from vis_mpl)
from ..vis_mpl import (
    plot_com_trajectory_3d,
    draw_camera_gizmos_mpl,
    plot_hex_outline_mpl,
    set_axes_equal_3d,
    draw_skeleton_mpl,
    draw_joint_height_lines_mpl,
)

# 2D QA panels (reuse existing viz modules)
from .pose import (
    plot_xy_occupancy,
    plot_speed_timeseries,
    plot_valid_fraction,
    plot_pose_traj_xy_xz,
    plot_occupancy_projections_grid,
)
from .spikes import (
    plot_event_raster,
    plot_rate_heatmap,
    plot_fr_histogram,
)

__all__ = [
    # 3D
    "plot_com_trajectory_3d",
    "draw_camera_gizmos_mpl",
    "plot_hex_outline_mpl",
    "set_axes_equal_3d",
    "draw_skeleton_mpl",
    "draw_joint_height_lines_mpl",
    # 2D
    "plot_xy_occupancy",
    "plot_speed_timeseries",
    "plot_valid_fraction",
    "plot_pose_traj_xy_xz",
    "plot_occupancy_projections_grid",
    "plot_event_raster",
    "plot_rate_heatmap",
    "plot_fr_histogram",
]
