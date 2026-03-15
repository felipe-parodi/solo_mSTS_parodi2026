"""Ridge coefficient extraction for Fig 2 and Fig 3 heatmaps.

Replicates the feature construction pipelines from ``run_fig2b`` and
``run_fig3``, fits ridge with the production-selected alpha, and returns
the standardised coefficient vector for each unit.

The coefficients are intended for visualisation only (supplementary
heatmaps) — they are **not** cross-validated predictions.

Caveat: production alpha was selected under fold-wise z-scoring;
coefficient extraction z-scores globally.  Differences are minimal
for visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from freemat.single_fm.encoding.ridge import ridge_fit_get_coeffs
from freemat.single_fm.encoding.temporal import (
    build_lagged_features,
    _add_time_drift_covariate,
)
from freemat.single_fm.encoding.visual_features import (
    BODY_KIN_COLS,
    HEAD_KIN_COLS,
    POS_COLS,
    get_feature_cols,
)


# ── Feature index builders ──────────────────────────────────────────


def build_feature_index_fig2(
    n_lags: int = 2,
    include_pos: bool = True,
    include_drift: bool = True,
    use_2d_only: bool = False,
) -> List[Dict[str, Any]]:
    """Build a feature index matching the Fig 2 X_full column order.

    Follows the exact build order in ``_process_single_unit``
    (run_fig2b.py:125-175):
      VIS lagged → [drift] → BODY_KIN lagged → [drift] →
      HEAD_KIN lagged → [drift] → POS (unlagged) → [drift]

    Args:
        n_lags: Number of lags in each direction.
        include_pos: Whether POS group is included.
        include_drift: Whether time_drift is appended per group.
        use_2d_only: Whether 2D-only feature set is used.

    Returns:
        List of dicts, one per column in X_full.  Keys:
        ``name``, ``group``, ``lag`` (int or None), ``display`` (bool).
    """
    vis_cols, _ = get_feature_cols(use_2d_only)
    lags = list(range(-n_lags, n_lags + 1)) if n_lags > 0 else [0]

    groups_spec = [
        ("vis", vis_cols, True),
        ("body_kin", list(BODY_KIN_COLS), True),
        ("head_kin", list(HEAD_KIN_COLS), True),
    ]
    if include_pos:
        groups_spec.append(("pos", list(POS_COLS), False))

    index: List[Dict[str, Any]] = []
    for group_name, base_cols, is_lagged in groups_spec:
        if is_lagged and n_lags > 0:
            for lag in lags:
                for col in base_cols:
                    index.append({
                        "name": col,
                        "group": group_name,
                        "lag": lag,
                        "display": True,
                    })
        else:
            for col in base_cols:
                index.append({
                    "name": col,
                    "group": group_name,
                    "lag": None,
                    "display": True,
                })
        if include_drift:
            index.append({
                "name": "time_drift",
                "group": group_name,
                "lag": None,
                "display": False,
            })

    return index


def build_feature_index_fig3(
    include_drift: bool = True,
) -> List[Dict[str, Any]]:
    """Build a feature index matching the Fig 3 per-frame model columns.

    Column order: [speed, (time_drift), vel_x, vel_y, vel_z,
                   disp_x, disp_y, disp_z]

    Args:
        include_drift: Whether time_drift base covariate is present.

    Returns:
        List of dicts per column.
    """
    index: List[Dict[str, Any]] = []
    index.append({"name": "speed", "group": "base", "lag": None, "display": True})
    if include_drift:
        index.append({"name": "time_drift", "group": "base", "lag": None, "display": False})
    for feat in ["vel_x", "vel_y", "vel_z"]:
        index.append({"name": feat, "group": "velocity", "lag": None, "display": True})
    for feat in ["disp_x", "disp_y", "disp_z"]:
        index.append({"name": feat, "group": "displacement", "lag": None, "display": True})
    return index


# ── Helpers ──────────────────────────────────────────────────────────


def _zscore_global(X: np.ndarray) -> np.ndarray:
    """Z-score each column of X using global (full-session) statistics."""
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


# ── Fig 2 coefficient extraction ────────────────────────────────────


def extract_fig2_coefficients(
    session_yaml: str,
    production_parquet: Path,
    *,
    bin_s: float = 0.5,
    n_lags: int = 2,
    include_pos: bool = True,
    include_drift: bool = True,
    use_2d_only: bool = False,
    floor_filter: bool = False,
    floor_height_threshold: float = 0.5,
    floor_filter_before_features: bool = False,
    min_bins_per_unit: int = 150,
) -> pd.DataFrame:
    """Extract per-unit ridge coefficients for Fig 2 (full model).

    Replicates feature construction from ``_process_single_unit``
    (run_fig2b.py:80-175), fits ridge with production ``alpha_selected``,
    and returns a wide DataFrame of coefficients.

    Args:
        session_yaml: Path to session YAML.
        production_parquet: Per-session ``fig2b_unique_variance.parquet``
            containing ``unit_id``, ``alpha_selected``, ``full_r2``,
            ``unit_bank``.
        bin_s: Bin width in seconds.
        n_lags: Temporal lag count.
        include_pos: Include 3D position group.
        include_drift: Include time drift covariates.
        use_2d_only: Use 2D feature set.
        floor_filter: Apply floor filter.
        floor_height_threshold: Floor filter height.
        floor_filter_before_features: Floor filter timing.
        min_bins_per_unit: Minimum bins per unit.

    Returns:
        Wide DataFrame with columns: unit_id, session_id, unit_bank,
        full_r2, alpha_selected, coeff_0 ... coeff_N.
    """
    from freemat.single_fm.data import SingleFMSessionLoader
    from freemat_data import load_session_yaml

    # Load production results
    prod_df = pd.read_parquet(production_parquet)
    prod_lookup = {
        int(row["unit_id"]): row
        for _, row in prod_df.iterrows()
    }

    # Load session
    sref = load_session_yaml(session_yaml)
    session_id = str(getattr(sref, "session_id", "session"))
    loader = SingleFMSessionLoader(session_yaml)

    merged_df = loader.get_aligned_data(
        bin_s=bin_s,
        floor_filter=floor_filter,
        floor_height_threshold=floor_height_threshold,
        floor_filter_before_features=floor_filter_before_features,
    )
    if len(merged_df) == 0:
        return pd.DataFrame()

    merged_df = merged_df.rename(columns={"fr": "firing_rate", "unit_index": "unit_id"})

    # Feature columns
    vis_cols_ref, _ = get_feature_cols(use_2d_only)
    vis_cols = [c for c in vis_cols_ref if c in merged_df.columns]
    body_kin_cols = [c for c in BODY_KIN_COLS if c in merged_df.columns]
    head_kin_cols = [c for c in HEAD_KIN_COLS if c in merged_df.columns]
    pos_cols = [c for c in POS_COLS if c in merged_df.columns] if include_pos else []

    feature_index = build_feature_index_fig2(
        n_lags=n_lags,
        include_pos=include_pos,
        include_drift=include_drift,
        use_2d_only=use_2d_only,
    )
    n_coeff = len(feature_index)

    rows: List[Dict[str, Any]] = []
    for unit_id in sorted(prod_lookup.keys()):
        prod_row = prod_lookup[unit_id]
        alpha = float(prod_row["alpha_selected"])
        full_r2 = float(prod_row["full_r2"])
        unit_bank = prod_row.get("unit_bank", None)

        ud = merged_df[merged_df["unit_id"] == unit_id].copy()

        # Fill NaN HEAD_KIN with 0 (rate features)
        for col in head_kin_cols:
            if col in ud.columns and ud[col].isna().all():
                ud[col] = 0.0

        # Drop NaN
        needed = vis_cols + body_kin_cols + head_kin_cols + ["firing_rate"]
        if "t_bin_s" in ud.columns:
            needed.append("t_bin_s")
        ud = ud.dropna(subset=[c for c in needed if c in ud.columns])

        if len(ud) < min_bins_per_unit:
            continue

        y = ud["firing_rate"].values.astype(np.float64)
        X_vis = ud[vis_cols].values.astype(np.float64)
        X_body_kin = ud[body_kin_cols].values.astype(np.float64)
        X_head_kin = ud[head_kin_cols].values.astype(np.float64)

        # POS: z-score inline (replicating run_fig2b.py:114-121)
        has_pos = len(pos_cols) > 0
        if has_pos:
            X_pos = ud[pos_cols].values.astype(np.float64)
            for j in range(X_pos.shape[1]):
                col_vals = X_pos[:, j]
                finite = np.isfinite(col_vals)
                if finite.any():
                    m, s = col_vals[finite].mean(), col_vals[finite].std()
                    s = s if s > 1e-12 else 1.0
                    X_pos[:, j] = (col_vals - m) / s

        # Lag expansion
        if n_lags > 0:
            X_vis = build_lagged_features(X_vis, n_lags)
            X_body_kin = build_lagged_features(X_body_kin, n_lags)
            X_head_kin = build_lagged_features(X_head_kin, n_lags)
            trim = n_lags
            X_vis = X_vis[trim:-trim]
            X_body_kin = X_body_kin[trim:-trim]
            X_head_kin = X_head_kin[trim:-trim]
            y = y[trim:-trim]
            if has_pos:
                X_pos = X_pos[trim:-trim]
            if len(y) < min_bins_per_unit:
                continue

        # Time drift (replicating run_fig2b.py:157-166)
        if include_drift and "t_bin_s" in ud.columns:
            t_raw = ud["t_bin_s"].values.astype(np.float64)
            if n_lags > 0:
                t_raw = t_raw[n_lags:-n_lags]
            X_vis = _add_time_drift_covariate(t_raw, X_vis)
            X_body_kin = _add_time_drift_covariate(t_raw, X_body_kin)
            X_head_kin = _add_time_drift_covariate(t_raw, X_head_kin)
            if has_pos:
                X_pos = _add_time_drift_covariate(t_raw, X_pos)

        # Assemble X_full in group order (matching variance.py:777)
        parts = [X_vis, X_body_kin, X_head_kin]
        if has_pos:
            parts.append(X_pos)
        X_full = np.hstack(parts)

        if X_full.shape[1] != n_coeff:
            print(
                f"[coefficients] WARNING: unit {unit_id} has "
                f"{X_full.shape[1]} cols, expected {n_coeff}. Skipping."
            )
            continue

        # Global z-score
        X_full = _zscore_global(X_full)

        # Fit ridge and get coefficients
        w = ridge_fit_get_coeffs(X_full, y, alpha)

        row: Dict[str, Any] = {
            "unit_id": int(unit_id),
            "session_id": session_id,
            "unit_bank": unit_bank,
            "full_r2": full_r2,
            "alpha_selected": alpha,
            "n_bins": len(y),
        }
        for i, val in enumerate(w):
            row[f"coeff_{i}"] = float(val)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Fig 3 coefficient extraction ────────────────────────────────────


def extract_fig3_coefficients(
    session_yaml: str,
    production_parquet: Path,
    *,
    bin_s: float = 0.5,
    include_drift: bool = True,
    floor_filter: bool = True,
    floor_height_threshold: float = 0.5,
    floor_filter_before_features: bool = False,
    min_bins_per_unit: int = 150,
) -> pd.DataFrame:
    """Extract per-unit ridge coefficients for Fig 3 (per-frame models).

    Replicates feature construction from ``_process_single_unit_fig3``
    (run_fig3.py:250-287) and ``run_fig3_encoding`` (run_fig3.py:425-514).

    For each unit, fits 3 ridge models (allo, body, head) using the
    production ``alpha_selected`` and returns coefficients for all frames.

    Args:
        session_yaml: Path to session YAML.
        production_parquet: Per-session ``fig3_reference_frames.parquet``
            containing unit_id, alpha_selected, r2_allo/body/head, unit_bank.
        bin_s: Bin width in seconds.
        include_drift: Include time drift base covariate.
        floor_filter: Apply floor filter.
        floor_height_threshold: Floor filter height.
        floor_filter_before_features: Floor filter timing.
        min_bins_per_unit: Minimum bins per unit.

    Returns:
        DataFrame with columns: unit_id, session_id, unit_bank,
        r2_allo, r2_body, r2_head, frame, coeff_0 ... coeff_N.
    """
    from freemat.single_fm.data.session_loader import SingleFMSessionLoader
    from freemat.single_fm.encoding.run_fig3 import compute_frame_features
    from freemat.single_fm.features.canonical import (
        bin_features,
        compute_body_heading,
        compute_head_forward,
    )
    from freemat_data import compute_center_of_mass, load_session_yaml

    # Load production results
    prod_df = pd.read_parquet(production_parquet)
    prod_lookup = {
        int(row["unit_id"]): row
        for _, row in prod_df.iterrows()
    }

    # Load session
    sref = load_session_yaml(session_yaml)
    session_id = str(getattr(sref, "session_id", "session"))
    loader = SingleFMSessionLoader(session_yaml)

    # Pose + arena (replicating run_fig3.py:428-456)
    pose = loader.get_pose_oriented()
    fps = float(pose.fps or 5.0)
    arena = loader.arena
    arena_center = np.asarray(arena.center, dtype=float)
    plane_dict = loader.plane_dict
    vertical_axis = loader._get_vertical_axis()
    horiz_idxs = (0, 2) if vertical_axis == 1 else (
        (1, 2) if vertical_axis == 0 else (0, 1)
    )

    com = compute_center_of_mass(pose.X)
    body_heading = compute_body_heading(pose.X, vertical_axis)
    head_fwd_3d = compute_head_forward(pose.X, plane_dict)

    frame_feats = compute_frame_features(
        com, fps, arena_center, body_heading, head_fwd_3d,
        vertical_axis, horiz_idxs,
    )

    # Aligned data
    merged_df = loader.get_aligned_data(
        bin_s=bin_s,
        floor_filter=floor_filter,
        floor_height_threshold=floor_height_threshold,
        floor_filter_before_features=floor_filter_before_features,
    )
    if len(merged_df) == 0:
        return pd.DataFrame()

    merged_df = merged_df.rename(columns={"fr": "firing_rate", "unit_index": "unit_id"})

    # Bin frame features (replicating run_fig3.py:479-493)
    t_video_pose = loader.pose_video_time()
    t_bins_unique = merged_df["t_bin_s"].unique()
    t_bins_unique.sort()

    for frame_name, feats in frame_feats.items():
        binned = bin_features(feats, t_video_pose, t_bins_unique, method="mean")
        prefix = f"f3_{frame_name}_"
        for col in binned.columns:
            if col == "t_bin_s":
                continue
            t2v = dict(zip(t_bins_unique, binned[col].values))
            merged_df[prefix + col] = merged_df["t_bin_s"].map(t2v)

    # Base columns
    base_cols: List[str] = []
    if "speed" in merged_df.columns:
        base_cols.append("speed")
    if include_drift:
        merged_df["_time_drift"] = merged_df["t_bin_s"].values.astype(float)
        base_cols.append("_time_drift")

    # Frame feature columns
    frame_feature_cols: Dict[str, List[str]] = {}
    for frame_name in ["allo", "body", "head"]:
        prefix = f"f3_{frame_name}_"
        cols = sorted(c for c in merged_df.columns if c.startswith(prefix))
        frame_feature_cols[frame_name] = cols

    feature_index = build_feature_index_fig3(include_drift=include_drift)
    n_coeff = len(feature_index)

    rows: List[Dict[str, Any]] = []
    for unit_id in sorted(prod_lookup.keys()):
        prod_row = prod_lookup[unit_id]
        # Per-frame alpha if available, fall back to alpha_selected, else raise
        alpha_map: Dict[str, float] = {}
        for _frame in ["allo", "body", "head"]:
            _key = f"alpha_{_frame}"
            if _key in prod_row and pd.notna(prod_row[_key]):
                alpha_map[_frame] = float(prod_row[_key])
            elif "alpha_selected" in prod_row and pd.notna(prod_row["alpha_selected"]):
                alpha_map[_frame] = float(prod_row["alpha_selected"])
            else:
                msg = f"No alpha found for unit {unit_id}, frame {_frame}"
                raise KeyError(msg)
        r2_allo = float(prod_row.get("r2_allo", np.nan))
        r2_body = float(prod_row.get("r2_body", np.nan))
        r2_head = float(prod_row.get("r2_head", np.nan))
        unit_bank = prod_row.get("unit_bank", None)

        ud = merged_df[merged_df["unit_id"] == unit_id].copy()

        # Drop NaN across all needed columns
        all_cols = list(set(
            ["firing_rate"] + base_cols
            + [c for cols in frame_feature_cols.values() for c in cols]
        ))
        ud = ud.dropna(subset=[c for c in all_cols if c in ud.columns])
        if len(ud) < min_bins_per_unit:
            continue

        y = ud["firing_rate"].values.astype(np.float64)

        # Base covariates (z-scored, replicating run_fig3.py:264-273)
        X_base = ud[base_cols].values.astype(np.float64)
        for j in range(X_base.shape[1]):
            col = X_base[:, j]
            finite = np.isfinite(col)
            if finite.any():
                m, s = col[finite].mean(), col[finite].std()
                s = s if s > 1e-12 else 1.0
                X_base[:, j] = (col - m) / s

        # Per-frame features (z-scored, replicating run_fig3.py:276-287)
        X_frames: Dict[str, np.ndarray] = {}
        for frame_name, cols in frame_feature_cols.items():
            X_f = ud[cols].values.astype(np.float64)
            for j in range(X_f.shape[1]):
                col = X_f[:, j]
                finite = np.isfinite(col)
                if finite.any():
                    m, s = col[finite].mean(), col[finite].std()
                    s = s if s > 1e-12 else 1.0
                    X_f[:, j] = (col - m) / s
            X_frames[frame_name] = X_f

        # Fit per-frame models
        for frame_name in ["allo", "body", "head"]:
            X = np.hstack([X_base, X_frames[frame_name]])
            if X.shape[1] != n_coeff:
                print(
                    f"[coefficients] WARNING: unit {unit_id} frame {frame_name} "
                    f"has {X.shape[1]} cols, expected {n_coeff}. Skipping."
                )
                continue

            w = ridge_fit_get_coeffs(X, y, alpha_map[frame_name])

            row: Dict[str, Any] = {
                "unit_id": int(unit_id),
                "session_id": session_id,
                "unit_bank": unit_bank,
                "r2_allo": r2_allo,
                "r2_body": r2_body,
                "r2_head": r2_head,
                "frame": frame_name,
                "alpha_selected": alpha_map[frame_name],
                "n_bins": len(y),
            }
            for i, val in enumerate(w):
                row[f"coeff_{i}"] = float(val)
            rows.append(row)

    return pd.DataFrame(rows)
