"""CLI to load upright 3D pose, render a frame, and plot validity histogram.

This is a minimal, project‑agnostic pose QC entrypoint that:
- Loads 3D pose from a per‑session YAML via freemat_data
- Orients the pose upright (prefers floor_plane.json if present)
- Picks a robust non‑empty frame for visualization (connected, colored skeleton)
- Computes per‑frame validity (finite‑joint count) and saves:
  - bar plot of empty vs non‑empty frames
  - histogram of validity among non‑empty frames (fraction in [0,1])
- Writes a small JSON summary with basic metadata and plot paths

Example:
    python -m freemat_data.qc.qc_pose \
        --config configs/single_fm/session_M2_0526.yaml \
        --outdir results/single_fm/M2_0526/qc \
        --min-valid-frac 0.7 --conf-thresh 0.5 --no-show

Notes:
- This scaffold focuses on the two high‑value checks requested: a single upright
  frame and the validity histogram. A fuller pipeline (trim/interp/smooth/save
  processed NPZ) will be layered on next.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from .. import (
    load_session_yaml,
    load_pose3d_from_session,
    orient_with_floor,
    orient_pose_upright,
    rotation_from_floor_json,
    load_hex_uv,
    pose_time_seconds,
)
from ..transforms.pose import frame_valid_fraction
from ..viz.mpl import (
    draw_skeleton_mpl,
    draw_joint_height_lines_mpl,
    plot_hex_outline_mpl,
    set_axes_equal_3d,
)


@dataclass
class PoseQCSummary:
    """Minimal QC metadata for pose loading and validity histogram.

    Attributes:
        session_id: Session identifier.
        frames: Number of frames (T).
        joints: Number of joints (J).
        fps: Frames per second.
        up_axis: Declared up‑axis after orientation ("Z" preferred).
        plane: Upright floor plane coefficients if available.
        valid_frac_stats: Basic statistics over per‑frame valid fraction.
        chosen_frame: Index of the frame rendered (and its time seconds).
        plots: Mapping of saved plot names → absolute paths.
    """

    session_id: str
    frames: int
    joints: int
    fps: float
    up_axis: str
    plane: Optional[Dict[str, float]]
    valid_frac_stats: Dict[str, float]
    chosen_frame: Dict[str, float]
    plots: Dict[str, str]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_frame_index(valid_frac: np.ndarray, *, min_valid_frac: float = -1.0) -> int:
    """Pick a robust visualization frame index.

    Strategy:
        - Prefer median index among frames with valid_frac ≥ threshold
        - Fall back to argmax(valid_frac) if none meet the threshold

    Args:
        valid_frac: (T,) array in [0, 1]
        min_valid_frac: Threshold for a "good" frame

    Returns:
        Chosen frame index (int)
    """
    vf = np.asarray(valid_frac, dtype=float).reshape(-1)
    if vf.size == 0:
        return 0
    # If disabled (negative), use argmax validity without thresholding
    if min_valid_frac is None or float(min_valid_frac) < 0:
        # Prefer the first non-empty frame if any
        idx_first = np.where(np.isfinite(vf) & (vf > 0))[0]
        if idx_first.size:
            return int(idx_first[0])
        return int(np.nanargmax(vf))
    idx = np.where(vf >= float(min_valid_frac))[0]
    if idx.size:
        return int(idx[len(idx) // 2])
    return int(np.nanargmax(vf))


def _count_valid_joints(P: np.ndarray) -> int:
    """Count joints with finite XYZ only (no confidences).

    Args:
        P: Single frame joint positions (J, 3).

    Returns:
        Number of valid joints (int) where all XYZ are finite.
    """
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    valid_xyz = np.isfinite(P).all(axis=1)
    return int(valid_xyz.sum())


def _find_display_frame_index(
    X: np.ndarray,
    *,
    idx_hint: int,
    min_valid_joints: int = 4,
    search_radius: int = 60,
) -> int:
    """Refine the snapshot frame to ensure it's actually drawable.

    Starts from ``idx_hint`` and searches within ``±search_radius`` for a frame
    with at least ``min_valid_joints`` joints that have finite coordinates.
    Falls back to the global argmax of valid‑joint count if needed.

    Args:
        X: Pose positions (T, J, 3).
        idx_hint: Initial index to try.
        min_valid_joints: Minimum joints required to consider the frame drawable.
        search_radius: Number of frames to search around the hint.

    Returns:
        Index of the selected display frame.
    """
    T = int(X.shape[0])
    if T == 0:
        return 0
    idx0 = int(np.clip(int(idx_hint), 0, T - 1))
    # Quick accept if hint already has enough valid joints
    if _count_valid_joints(X[idx0]) >= min_valid_joints:
        return idx0
    # Local search around the hint
    lo = max(0, idx0 - int(search_radius))
    hi = min(T, idx0 + int(search_radius) + 1)
    best = idx0
    best_n = -1
    for k in range(lo, hi):
        n = _count_valid_joints(X[k])
        if n > best_n:
            best_n = n
            best = k
        if n >= min_valid_joints:
            return k
    # Global fallback: maximize number of valid joints across all frames
    vals = np.zeros(T, dtype=int)
    for k in range(T):
        vals[k] = _count_valid_joints(X[k])
    if np.any(vals > 0):
        return int(np.argmax(vals))
    return best


def _plot_empty_vs_nonempty(valid_counts: np.ndarray, *, out_path: Path) -> None:
    """Plot a simple bar chart: empty vs non‑empty frames.

    Args:
        valid_counts: (T,) integer counts of finite joints per frame.
        out_path: PNG path to save.
    """
    vc = np.asarray(valid_counts, dtype=float).reshape(-1)
    empty = int(np.sum(vc <= 0))
    nonempty = int(vc.size - empty)
    fig = plt.figure(figsize=(4.0, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(["empty", "non‑empty"], [empty, nonempty], color=["#d62728", "#2ca02c"], alpha=0.9)
    ax.set_ylabel("Frames")
    ax.set_title("Frames: empty vs non‑empty")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_valid_hist(valid_frac: np.ndarray, *, out_path: Path, title: str = "Pose validity histogram") -> None:
    """Plot a histogram of per‑frame valid fraction in [0, 1].

    Args:
        valid_frac: (T,) fraction per frame (non‑empty set is typical input).
        out_path: PNG path to save.
        title: Title for the figure.
    """
    vf = np.asarray(valid_frac, dtype=float).reshape(-1)
    m = np.isfinite(vf)
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(vf[m], bins=bins, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Per‑frame valid fraction")
    ax.set_ylabel("Frames")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_upright_frame(
    X: np.ndarray,
    *,
    joint_names: Optional[list[str]],
    plane: Optional[Dict[str, float]],
    hex_uv: Optional[np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    """Render one upright 3D skeleton frame with optional floor/hex overlays.

    Args:
        X: Pose coordinates (T, J, 3)
        joint_names: Optional joint names
        C: Optional confidences (T, J)
        plane: Optional upright floor plane dict
        hex_uv: Optional hex footprint in plane UV coords
        title: Figure title
        out_path: PNG path to save
    """
    if X.size == 0:
        return
    # Choose the middle frame of the provided X window for visual stability
    i = X.shape[0] // 2
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    # Draw without confidences; rely purely on finite coordinates
    P = np.asarray(X[i], dtype=float)
    finite_all = np.isfinite(P).all(axis=1) if P.ndim == 2 else False
    if np.any(finite_all):
        draw_skeleton_mpl(ax, P, joint_names=joint_names, confidences=None, linewidth=1.3, markersize=6.0)
        # Use a canonical Z-plane for height lines so they point straight down in Z
        try:
            zvals = P[finite_all, 2]
            z0 = float(np.nanpercentile(zvals, 5)) if zvals.size else 0.0
            plane_lines = {"a": 0.0, "b": 0.0, "c": 1.0, "d": -z0}
            draw_joint_height_lines_mpl(ax, P, plane=plane_lines, color="#777", alpha=0.6, linewidth=0.9)
        except Exception:
            pass
        # Keep hex outline (if available) in the plane's own basis
        try:
            if plane is not None and hex_uv is not None:
                plot_hex_outline_mpl(ax, plane, hex_uv)
        except Exception:
            pass
        try:
            set_axes_equal_3d(ax, P)
        except Exception:
            # Skip equal-axes if bounds are invalid (all-NaN frame, etc.)
            pass
    else:
        # No finite joints to render at this frame; leave a note in the panel
        try:
            ax.text2D(0.5, 0.5, "No finite joints in chosen frame", transform=ax.transAxes, ha="center", va="center")
        except Exception:
            pass
    ax.set_title(title)
    try:
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _render_frame_on_axes(
    ax: plt.Axes,
    P: np.ndarray,
    *,
    joint_names: Optional[list[str]],
    plane: Optional[Dict[str, float]],
    hex_uv: Optional[np.ndarray],
    title: str,
) -> None:
    """Render a single pose frame into an existing 3D axes.

    Draws joints/edges for finite points only and overlays optional plane helpers.
    """
    if P is None or P.size == 0:
        return
    Pint = np.asarray(P, dtype=float)
    finite_all = np.isfinite(Pint).all(axis=1) if (Pint.ndim == 2 and Pint.shape[1] == 3) else False
    if np.any(finite_all):
        draw_skeleton_mpl(ax, Pint, joint_names=joint_names, confidences=None, linewidth=1.1, markersize=5.0)
        # Vertical height lines relative to canonical Z-plane
        try:
            zvals = Pint[finite_all, 2]
            z0 = float(np.nanpercentile(zvals, 5)) if zvals.size else 0.0
            plane_lines = {"a": 0.0, "b": 0.0, "c": 1.0, "d": -z0}
            draw_joint_height_lines_mpl(ax, Pint, plane=plane_lines, color="#777", alpha=0.6, linewidth=0.8)
        except Exception:
            pass
        # Keep hex outline in original plane basis if provided
        try:
            if plane is not None and hex_uv is not None:
                plot_hex_outline_mpl(ax, plane, hex_uv)
        except Exception:
            pass
        try:
            set_axes_equal_3d(ax, Pint)
        except Exception:
            pass
    else:
        try:
            ax.text2D(0.5, 0.5, "No finite joints", transform=ax.transAxes, ha="center", va="center")
        except Exception:
            pass
    ax.set_title(title)
    try:
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    except Exception:
        pass


def _pick_indices_for_targets(
    valid_frac: np.ndarray,
    targets: list[float],
    *,
    bandwidth: float = 0.05,
) -> list[int]:
    """Pick representative indices whose validity is near the target fractions.

    For each target t, find frames with valid_frac in [t - bandwidth/2, t + bandwidth/2].
    Choose the median index among matches; if none, choose the closest frame by |vf - t|.
    """
    vf = np.asarray(valid_frac, dtype=float).reshape(-1)
    T = vf.size
    out: list[int] = []
    if T == 0:
        return out
    half = float(bandwidth) * 0.5
    for t in targets:
        lo = float(t) - half
        hi = float(t) + half
        idx = np.where((vf >= lo) & (vf <= hi))[0]
        if idx.size:
            out.append(int(idx[len(idx) // 2]))
        else:
            out.append(int(np.nanargmin(np.abs(vf - float(t)))))
    return out


def _plot_examples_for_bins(
    X: np.ndarray,
    *,
    joint_names: Optional[list[str]],
    plane: Optional[Dict[str, float]],
    hex_uv: Optional[np.ndarray],
    valid_frac: np.ndarray,
    targets: list[float],
    out_path: Path,
) -> None:
    """Plot representative skeleton frames for target validity fractions.

    Saves a single multi-panel figure.
    """
    idxs = _pick_indices_for_targets(valid_frac, targets, bandwidth=0.1)
    k = len(idxs)
    if k == 0:
        return
    cols = k
    fig = plt.figure(figsize=(4.0 * cols, 3.6))
    for j, (t, idx) in enumerate(zip(targets, idxs), start=1):
        ax = fig.add_subplot(1, cols, j, projection="3d")
        tit = f"vf≈{t:.2f} (idx {idx})"
        if 0 <= idx < X.shape[0]:
            _render_frame_on_axes(ax, X[idx], joint_names=joint_names, plane=plane, hex_uv=hex_uv, title=tit)
        else:
            ax.text2D(0.5, 0.5, f"index {idx} OOB", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(tit)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_qc_pose(
    config_path: str,
    *,
    outdir: Optional[str] = None,
    min_valid_frac: float = -1.0,
    conf_thresh: float = 0.5,
    do_snapshot: bool = True,
) -> Tuple[Path, Dict[str, str]]:
    """Load upright pose, save one snapshot and valid‑fraction histogram.

    Args:
        config_path: Path to per‑session YAML
        outdir: Output directory (defaults to results/<session_id>/qc)
        min_valid_frac: Threshold for picking an upright frame to render
        conf_thresh: Confidence threshold when computing validity

    Returns:
        (qc_json_path, plots_manifest)
    """
    session = load_session_yaml(config_path)
    pose_raw = load_pose3d_from_session(session, animal=0)
    if pose_raw is None:
        raise FileNotFoundError("run_qc_pose: pose_3d not resolved from session YAML")

    # Orient upright (prefer floor JSON when present)
    try:
        pose_upright, _used_axis = orient_with_floor(pose_raw)
    except Exception:
        pose_upright = orient_pose_upright(pose_raw, prefer="Z")
        _used_axis = pose_upright.up_axis

    # Floor plane + hex footprint (optional for overlays)
    plane: Optional[Dict[str, float]] = None
    hex_uv = None
    if pose_raw.path:
        npz_dir = os.path.dirname(os.path.abspath(pose_raw.path))
        floor_path = os.path.join(npz_dir, "floor_plane.json")
        if os.path.isfile(floor_path):
            try:
                _, plane_upr, _ = rotation_from_floor_json(floor_path)
                plane = None if plane_upr is None else {
                    "a": float(plane_upr.get("a", 0.0)),
                    "b": float(plane_upr.get("b", 0.0)),
                    "c": float(plane_upr.get("c", 1.0)),
                    "d": float(plane_upr.get("d", 0.0)),
                }
            except Exception:
                plane = None
            try:
                hex_uv = load_hex_uv(npz_dir)
            except Exception:
                hex_uv = None

    # Enforce Z-up for display; if pose claims Y-up, swap Y/Z and adjust plane coefficients
    if str(pose_upright.up_axis).strip().upper() != "Z":
        pose_upright = orient_pose_upright(pose_upright, prefer="Z")
        if isinstance(plane, dict):
            try:
                plane = {
                    "a": float(plane.get("a", 0.0)),
                    "b": float(plane.get("c", 1.0)),  # swap b<->c for Y/Z swap
                    "c": float(plane.get("b", 0.0)),
                    "d": float(plane.get("d", 0.0)),
                }
            except Exception:
                pass

    # Validity over the full session (finite-only; do not rely on confidences)
    valid_frac = frame_valid_fraction(pose_upright.X, None, conf_thresh=0.0)
    idx_hint = _pick_frame_index(valid_frac, min_valid_frac=float(min_valid_frac))
    # Refine to a frame that is actually drawable (enough finite joints)
    idx = _find_display_frame_index(
        pose_upright.X,
        idx_hint=int(idx_hint),
        min_valid_joints=4,
        search_radius=max(int(pose_upright.fps // 2) if pose_upright.fps > 0 else 30, 15),
    )
    t_pose = pose_time_seconds(pose_upright)
    t_sel = float(t_pose[idx]) if t_pose.size > idx else float(idx / max(pose_upright.fps, 1e-6))

    # Paths
    sid = session.session_id
    out_dir = Path(outdir) if outdir else (Path("results") / sid / "qc")
    _ensure_dir(out_dir)
    empty_bar_path = out_dir / f"pose_empty_vs_nonempty_{sid}.png"
    hist_path = out_dir / f"pose_valid_hist_{sid}.png"
    examples_path = out_dir / f"pose_examples_bins_{sid}.png"
    snap_path = out_dir / f"pose_frame_{sid}.png"

    # Plots
    valid_counts = (np.asarray(valid_frac, dtype=float) * float(pose_upright.J)).astype(int)
    _plot_empty_vs_nonempty(valid_counts, out_path=empty_bar_path)
    nonempty_mask = valid_counts > 0
    if np.any(nonempty_mask):
        _plot_valid_hist(valid_frac[nonempty_mask], out_path=hist_path, title="Pose validity (non‑empty frames)")
    else:
        _plot_valid_hist(valid_frac, out_path=hist_path)
    # Examples for key validity bins
    try:
        targets = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        _plot_examples_for_bins(
            pose_upright.X,
            joint_names=pose_upright.joint_names,
            plane=plane,
            hex_uv=hex_uv,
            valid_frac=valid_frac,
            targets=targets,
            out_path=examples_path,
        )
    except Exception:
        pass
    if do_snapshot:
        # Render a small window around idx for a stable snapshot (height lines and hex optional)
        s = max(0, int(idx))
        e = min(pose_upright.T, int(idx) + 1)
        title = f"Upright skeleton — t={t_sel:.2f}s, frame {idx}"
        _plot_upright_frame(
            pose_upright.X[s:e],
            joint_names=pose_upright.joint_names,
            plane=plane,
            hex_uv=hex_uv,
            title=title,
            out_path=snap_path,
        )

    plots_manifest: Dict[str, str] = {}
    if empty_bar_path.exists():
        plots_manifest["pose_empty_vs_nonempty"] = str(empty_bar_path.resolve())
    if hist_path.exists():
        plots_manifest["pose_valid_hist"] = str(hist_path.resolve())
    if do_snapshot and snap_path.exists():
        plots_manifest["pose_frame"] = str(snap_path.resolve())
    if examples_path.exists():
        plots_manifest["pose_examples_bins"] = str(examples_path.resolve())

    # Summary JSON
    stats = {
        "mean": float(np.nanmean(valid_frac)) if np.isfinite(valid_frac).any() else 0.0,
        "median": float(np.nanmedian(valid_frac)) if np.isfinite(valid_frac).any() else 0.0,
        "p05": float(np.nanpercentile(valid_frac, 5)) if np.isfinite(valid_frac).any() else 0.0,
        "p95": float(np.nanpercentile(valid_frac, 95)) if np.isfinite(valid_frac).any() else 0.0,
    }
    # Add empty/non-empty counts for clarity
    try:
        vc_all = (np.asarray(valid_frac, dtype=float) * float(pose_upright.J)).astype(int)
        stats.update({
            "empty_frames": int(np.sum(vc_all <= 0)),
            "nonempty_frames": int(vc_all.size - int(np.sum(vc_all <= 0))),
        })
    except Exception:
        pass
    qc = PoseQCSummary(
        session_id=sid,
        frames=int(pose_upright.T),
        joints=int(pose_upright.J),
        fps=float(pose_upright.fps),
        up_axis=str(pose_upright.up_axis),
        plane=plane,
        valid_frac_stats=stats,
        chosen_frame={"index": float(idx), "t_s": float(t_sel)},
        plots=plots_manifest,
    )
    qc_json_path = out_dir / f"qc_pose_{sid}.json"
    with qc_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(qc), f, indent=2)

    return qc_json_path, plots_manifest


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pose QC: upright frame + validity histogram")
    ap.add_argument("--config", required=True, help="Path to per‑session YAML")
    ap.add_argument("--outdir", default=None, help="Output directory (default: results/<session_id>/qc)")
    ap.add_argument("--min-valid-frac", type=float, default=-1.0, help="If >=0, threshold for selecting a visualization frame; if <0, pick argmax(validity)")
    ap.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold for validity if confidences exist")
    ap.add_argument("--skip-snapshot", action="store_true", help="Only save the histogram; skip the skeleton frame snapshot")
    ap.add_argument("--no-show", action="store_true", help="Do not show figures interactively")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)
    qc_json, plots = run_qc_pose(
        args.config,
        outdir=args.outdir,
        min_valid_frac=float(args.min_valid_frac),
        conf_thresh=float(args.conf_thresh),
        do_snapshot=(not args.skip_snapshot),
    )
    print(f"[qc_pose] wrote {qc_json}")
    for name, path in plots.items():
        print(f"[qc_pose] plot {name}: {path}")
    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
