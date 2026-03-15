"""Temporal feature utilities and sensitivity analysis for Figure 2B.

Provides lagged feature construction, time drift covariates, feature
diagnostics, and subsample sensitivity analysis.

Example:
    >>> from freemat.single_fm.encoding.temporal import build_lagged_features
    >>> X_lagged = build_lagged_features(X, n_lags=2)  # 5 taps at ±2 bins
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from freemat.single_fm.encoding.visual_features import (
    VIS_COLS,
    KIN_COLS,
    BODY_KIN_COLS,
    HEAD_KIN_COLS,
)


def build_lagged_features(X: np.ndarray, n_lags: int) -> np.ndarray:
    """Stack lagged versions of feature matrix for temporal kernels.

    Creates a feature matrix with time-lagged copies of the original features,
    enabling the model to capture temporal dynamics (Testard et al. 2024 style).

    For bin_s=0.5 and n_lags=2, this creates lags [-2,-1,0,1,2] (±1.0s, 5 taps).
    For bin_s=1.0 and n_lags=1, this creates lags [-1,0,1] (±1.0s, 3 taps).

    Args:
        X: Feature matrix of shape (T, n_features).
        n_lags: Number of lags in each direction (total = 2*n_lags + 1).

    Returns:
        X_lagged: Feature matrix of shape (T, n_features * (2*n_lags + 1))
            with NaN padding at edges where lags extend beyond data.
    """
    if n_lags == 0:
        return X

    T, n_feat = X.shape
    lags = range(-n_lags, n_lags + 1)
    n_total_lags = len(lags)

    X_lagged = np.full((T, n_feat * n_total_lags), np.nan, dtype=np.float64)

    for i, lag in enumerate(lags):
        start_col = i * n_feat
        end_col = (i + 1) * n_feat

        if lag < 0:
            # Negative lag: past values (shift forward)
            X_lagged[-lag:, start_col:end_col] = X[:lag]
        elif lag > 0:
            # Positive lag: future values (shift backward)
            X_lagged[:-lag, start_col:end_col] = X[lag:]
        else:
            # lag == 0: no shift
            X_lagged[:, start_col:end_col] = X

    return X_lagged


def _add_time_drift_covariate(
    t_bins: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Add linear time drift regressor to design matrix.

    Args:
        t_bins: Time bin centers in seconds.
        X: Existing design matrix (T, D).

    Returns:
        Extended design matrix (T, D+1) with time_drift as last column.
    """
    # Normalize to [0, 1]
    t_min, t_max = t_bins.min(), t_bins.max()
    if t_max - t_min < 1e-12:
        time_drift = np.zeros_like(t_bins, dtype=np.float64)
    else:
        time_drift = (t_bins - t_min) / (t_max - t_min)
    return np.hstack([X, time_drift.reshape(-1, 1)])


def _generate_feature_diagnostics(
    session_yaml: str,
    session_id: str,
    kin_df: pd.DataFrame,
    vis_df: pd.DataFrame,
    output_dir: str,
    debug: bool = False,
) -> None:
    """Generate KIN/VIS feature diagnostic plots.

    Creates:
    - diagnostics/kin_qc.html (interactive dashboard)
    - diagnostics/kin_*.png (matplotlib plots)
    - diagnostics/vis_correlation.png (VIS-KIN correlation)

    Args:
        session_yaml: Path to session YAML.
        session_id: Session identifier.
        kin_df: KIN feature DataFrame.
        vis_df: VIS feature DataFrame.
        output_dir: Base output directory.
        debug: Enable debug output.
    """
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    if debug:
        print(f"[Fig2B] Generating diagnostics in: {diag_dir}")

    # 1. Static KIN diagnostics (matplotlib)
    try:
        from freemat.single_fm.viz.kin_diagnostics import save_kin_diagnostics
        save_kin_diagnostics(
            session_yaml=session_yaml,
            kin_df=kin_df,
            out_dir=diag_dir,
            include_speed=True,
        )
        if debug:
            print("[Fig2B] Saved KIN diagnostics (matplotlib)")
    except Exception as e:
        if debug:
            print(f"[Fig2B] Warning: KIN diagnostics failed: {e}")

    # 2. Interactive HTML dashboard (if pose data available)
    try:
        from freemat.single_fm.kin.viz import save_kin_qc_html, save_matplotlib_plots
        from freemat_data import (
            load_session_yaml,
            load_pose3d_from_session,
            load_pose3d,
            orient_pose_with_floor,
            compute_center_of_mass,
        )
        from freemat.single_fm.kin.head_forward import compute_head_forward

        sref = load_session_yaml(session_yaml)
        sid = str(getattr(sref, "session_id", "session"))
        pose_npz = os.path.join("results", "single_fm", sid, "pose_processed.npz")

        if os.path.isfile(pose_npz):
            pose0 = load_pose3d(pose_npz)
        else:
            pose0 = load_pose3d_from_session(sref)

        pose, _ = orient_pose_with_floor(pose0)
        com = compute_center_of_mass(pose.X)
        head_forward = compute_head_forward(pose)

        # Save interactive HTML
        html_path = os.path.join(diag_dir, "kin_qc.html")
        save_kin_qc_html(
            kin_df=kin_df,
            com=com,
            head_forward=head_forward,
            out_path=html_path,
            session_id=session_id,
        )
        if debug:
            print(f"[Fig2B] Saved interactive dashboard: {html_path}")

        # Save matplotlib plots (quiver, occupancy)
        save_matplotlib_plots(
            com=com,
            head_forward=head_forward,
            out_dir=diag_dir,
            prefix="kin_qc",
        )
        if debug:
            print("[Fig2B] Saved matplotlib QC plots")

    except Exception as e:
        if debug:
            print(f"[Fig2B] Warning: HTML dashboard failed: {e}")

    # 3. VIS-KIN cross-correlation matrix (VIS on y-axis, KIN on x-axis)
    try:
        import matplotlib.pyplot as plt

        # Combine VIS and KIN for correlation
        combined = pd.merge(
            vis_df[["t_bin_s"] + [c for c in VIS_COLS if c in vis_df.columns]],
            kin_df[["t_bin_s"] + [c for c in KIN_COLS if c in kin_df.columns]],
            on="t_bin_s",
            how="inner",
        )
        vis_cols_avail = [c for c in VIS_COLS if c in combined.columns]
        kin_cols_avail = [c for c in KIN_COLS if c in combined.columns]

        if len(vis_cols_avail) >= 1 and len(kin_cols_avail) >= 1:
            # Compute cross-correlation: VIS (rows) × KIN (cols)
            cross_corr = combined[vis_cols_avail + kin_cols_avail].corr()
            cross_corr = cross_corr.loc[vis_cols_avail, kin_cols_avail]

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cross_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(kin_cols_avail)))
            ax.set_yticks(range(len(vis_cols_avail)))
            ax.set_xticklabels(kin_cols_avail, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(vis_cols_avail, fontsize=10)
            ax.set_xlabel("KIN features", fontsize=12, fontweight="bold")
            ax.set_ylabel("VIS features", fontsize=12, fontweight="bold")
            ax.set_title("VIS × KIN Cross-Correlation", fontsize=14, fontweight="bold")

            # Add correlation values as text
            for i in range(len(vis_cols_avail)):
                for j in range(len(kin_cols_avail)):
                    val = cross_corr.iloc[i, j]
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

            fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
            plt.tight_layout()

            corr_path = os.path.join(diag_dir, "vis_kin_correlation")
            fig.savefig(f"{corr_path}.png", dpi=300, bbox_inches="tight")
            fig.savefig(f"{corr_path}.svg", bbox_inches="tight")
            plt.close(fig)

            if debug:
                print("[Fig2B] Saved VIS×KIN cross-correlation matrix")

    except Exception as e:
        if debug:
            print(f"[Fig2B] Warning: Correlation matrix failed: {e}")


def run_subsample_sensitivity(
    session_yamls: List[str],
    *,
    config: Optional[Any] = None,
    n_repeats: int = 100,
    seed: int = 42,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run subsample sensitivity analysis for coverage confound control.

    This addresses the coverage confound where sessions have different amounts
    of on-floor time (e.g., M2 ~80% vs M1 ~30%), leading to unequal bin
    counts. By subsampling each session to the minimum bin count, we can assess
    whether results are robust to this imbalance.

    Args:
        session_yamls: List of session YAML paths.
        config: Fig2BConfig instance (imported lazily to avoid circular imports).
        n_repeats: Number of subsampling repeats (default: 100).
        seed: Random seed for reproducibility.
        debug: Enable debug output.

    Returns:
        Dict with summary statistics across repeats.
    """
    from freemat.single_fm.encoding.config import Fig2BConfig
    from freemat.single_fm.data import SingleFMSessionLoader

    if config is None:
        config = Fig2BConfig()

    rng = np.random.default_rng(seed)

    # Step 1: Load aligned data for each session
    session_data: Dict[str, pd.DataFrame] = {}
    min_bins_per_session: Dict[str, int] = {}

    for yaml_path in session_yamls:
        loader = SingleFMSessionLoader(yaml_path, debug=debug)
        session_id = loader.session_id
        merged_df = loader.get_aligned_data(bin_s=config.bin_s)

        if len(merged_df) == 0:
            if debug:
                print(f"[Subsample] Skipping {session_id}: no aligned data")
            continue

        # Rename columns to match expected schema
        merged_df = merged_df.rename(columns={
            'fr': 'firing_rate',
            'unit_index': 'unit_id',
        })

        session_data[session_id] = merged_df
        # Count unique time bins (not rows, since rows = bins × units)
        n_bins = merged_df['t_bin_s'].nunique()
        min_bins_per_session[session_id] = n_bins

        if debug:
            print(f"[Subsample] {session_id}: {n_bins} bins, {merged_df['unit_id'].nunique()} units")

    if len(session_data) == 0:
        return {'error': 'No sessions with aligned data'}

    # Step 2: Find global minimum bin count across sessions
    global_min_bins = min(min_bins_per_session.values())

    if debug:
        print(f"[Subsample] Global min bins: {global_min_bins}")
        print(f"[Subsample] Running {n_repeats} subsampling repeats...")

    # Step 3-4: Subsample and run encoding for each repeat
    repeat_results: List[Dict[str, float]] = []

    for rep in range(n_repeats):
        rep_seed = rng.integers(0, 2**31)

        # Subsample each session to global_min_bins
        subsampled_dfs = []
        for session_id, df in session_data.items():
            # Get unique time bins
            unique_bins = df['t_bin_s'].unique()

            if len(unique_bins) <= global_min_bins:
                subsampled_dfs.append(df)
            else:
                # Deterministic seed from session_id
                session_seed = int.from_bytes(session_id.encode()[:8].ljust(8, b'\0'), 'little') % (2**31)
                rep_rng = np.random.default_rng(rep_seed + session_seed)
                sampled_bins = rep_rng.choice(unique_bins, size=global_min_bins, replace=False)
                subsampled = df[df['t_bin_s'].isin(sampled_bins)].copy()
                subsampled_dfs.append(subsampled)

        # Combine subsampled data
        combined_subsampled = pd.concat(subsampled_dfs, ignore_index=True)

        # Get feature columns
        vis_cols_available = [c for c in VIS_COLS if c in combined_subsampled.columns]
        body_kin_cols_available = [c for c in BODY_KIN_COLS if c in combined_subsampled.columns]
        head_kin_cols_available = [c for c in HEAD_KIN_COLS if c in combined_subsampled.columns]

        # Compute R² for each feature group (simplified, no CV for speed)
        unit_results = []
        for unit_id in combined_subsampled['unit_id'].unique():
            unit_df = combined_subsampled[combined_subsampled['unit_id'] == unit_id]

            if len(unit_df) < 50:
                continue

            y = unit_df['firing_rate'].values

            def r2_from_corr(X_cols):
                if not X_cols:
                    return 0.0
                X = unit_df[X_cols].values
                corrs = []
                for j in range(X.shape[1]):
                    valid = np.isfinite(X[:, j]) & np.isfinite(y)
                    if valid.sum() > 10:
                        corr = np.corrcoef(X[valid, j], y[valid])[0, 1]
                        if np.isfinite(corr):
                            corrs.append(abs(corr))
                return max(corrs) ** 2 if corrs else 0.0

            r2_vis = r2_from_corr(vis_cols_available)
            r2_body_kin = r2_from_corr(body_kin_cols_available)
            r2_head_kin = r2_from_corr(head_kin_cols_available)

            unit_results.append({
                'r2_vis': r2_vis,
                'r2_body_kin': r2_body_kin,
                'r2_head_kin': r2_head_kin,
            })

        if unit_results:
            unit_df_results = pd.DataFrame(unit_results)
            repeat_results.append({
                'mean_r2_vis': unit_df_results['r2_vis'].mean(),
                'mean_r2_body_kin': unit_df_results['r2_body_kin'].mean(),
                'mean_r2_head_kin': unit_df_results['r2_head_kin'].mean(),
            })

        if debug and (rep + 1) % 10 == 0:
            print(f"[Subsample] Completed repeat {rep + 1}/{n_repeats}")

    # Step 5: Compute summary statistics
    if not repeat_results:
        return {'error': 'No valid results from subsampling'}

    results_df = pd.DataFrame(repeat_results)

    summary = {
        'mean_r2_vis': float(results_df['mean_r2_vis'].mean()),
        'std_r2_vis': float(results_df['mean_r2_vis'].std()),
        'mean_r2_body_kin': float(results_df['mean_r2_body_kin'].mean()),
        'std_r2_body_kin': float(results_df['mean_r2_body_kin'].std()),
        'mean_r2_head_kin': float(results_df['mean_r2_head_kin'].mean()),
        'std_r2_head_kin': float(results_df['mean_r2_head_kin'].std()),
        'global_min_bins': global_min_bins,
        'min_bins_per_session': min_bins_per_session,
        'n_repeats': n_repeats,
        'full_results': repeat_results,
    }

    if debug:
        print(f"[Subsample] Results (n={n_repeats}):")
        print(f"  VIS R²: {summary['mean_r2_vis']:.4f} ± {summary['std_r2_vis']:.4f}")
        print(f"  BODY_KIN R²: {summary['mean_r2_body_kin']:.4f} ± {summary['std_r2_body_kin']:.4f}")
        print(f"  HEAD_KIN R²: {summary['mean_r2_head_kin']:.4f} ± {summary['std_r2_head_kin']:.4f}")

    return summary


def _add_rssi_covariate(
    rssi_z: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Append z-scored RSSI column to design matrix.

    Args:
        rssi_z: Z-scored RSSI array of shape (T,).
        X: Existing design matrix (T, D).

    Returns:
        Extended design matrix (T, D+1) with RSSI as last column.
    """
    return np.hstack([X, rssi_z.reshape(-1, 1)])


def load_rssi_for_session(
    session_yaml: str,
    t_bins: np.ndarray,
) -> np.ndarray:
    """Load RSSI from AutoLOG, align to video time, interpolate to bin centers.

    Parses the AutoLOG file referenced in the session YAML, converts
    PC timestamps to video time via the session sync map, and linearly
    interpolates the sparse RSSI samples (~1/min) to the dense bin
    centers used in encoding.

    Note:
        ``np.interp`` clamps to the first/last RSSI values outside the
        RSSI time range.  This is acceptable for a slowly varying nuisance
        regressor (~30 s extrapolation at recording edges).

    Args:
        session_yaml: Path to session YAML with ``autolog_txt`` in
            ``data_files``.
        t_bins: Bin center times in video seconds (1-D array, sorted).

    Returns:
        Z-scored RSSI interpolated to *t_bins* (same length).

    Raises:
        FileNotFoundError: If ``autolog_txt`` is absent from the YAML or
            the file does not exist on disk.
        ValueError: If no RSSI data is found in the AutoLOG.
    """
    from freemat_data import load_session_yaml, get_syncmap_from_session
    from freemat_data.loaders.events import parse_rssi

    sref = load_session_yaml(session_yaml)

    # --- Resolve autolog path from session YAML ---
    data_files = getattr(sref, "data_files", None) or {}
    autolog_path: Optional[str] = data_files.get("autolog_txt")
    if autolog_path is None:
        raise FileNotFoundError(
            f"Session YAML missing 'data_files.autolog_txt': {session_yaml}"
        )

    # --- Parse RSSI (recording window only) ---
    rssi_df = parse_rssi(autolog_path, recording_only=True)

    # --- Align to video time ---
    # Use anchored affine: video_sec = slope * (pc_sec - pc_flash) + T_vid_flash.
    # We do this directly because SessionSynchronizer.pc_to_video() has a
    # legacy fallback that omits the pc_flash anchor when "pc_to_video" slope
    # dict is absent from sync_map.json (producing seconds-of-day instead of
    # video-relative seconds).
    sync = get_syncmap_from_session(sref)
    pc_sec_arr = rssi_df["pc_sec"].to_numpy(dtype=np.float64)
    v = sync.map.get("video", {})
    pc_flash = float(v["pc_flash_sec"])
    tv_flash = float(v["T_vid_flash"])
    pc_to_vid = v.get("pc_to_video") or {}
    slope = float(pc_to_vid.get("slope", 1.0))
    t_video_rssi = slope * (pc_sec_arr - pc_flash) + tv_flash

    rssi_dbm = rssi_df["rssi_dbm"].to_numpy(dtype=np.float64)

    # Sort by video time (should already be monotonic, but be safe)
    sort_idx = np.argsort(t_video_rssi)
    t_video_rssi = t_video_rssi[sort_idx]
    rssi_dbm = rssi_dbm[sort_idx]

    # --- Interpolate to bin centers ---
    rssi_at_bins = np.interp(t_bins, t_video_rssi, rssi_dbm)

    # --- Z-score ---
    mu = rssi_at_bins.mean()
    std = rssi_at_bins.std()
    if std < 1e-12:
        rssi_z = np.zeros_like(rssi_at_bins)
    else:
        rssi_z = (rssi_at_bins - mu) / std

    return rssi_z


def load_antenna_distance_for_session(
    session_yaml: str,
    t_bins: np.ndarray,
    merged_df: pd.DataFrame,
    pipeline_floor_level: Optional[float] = None,
) -> np.ndarray:
    """Compute z-scored Euclidean distance from COM to antenna per bin.

    Loads the antenna position JSON for this session (stored in the
    encoding frame, matching com_x/com_y/com_z in ``merged_df``), then
    computes the per-bin distance and z-scores the result.

    When ``pipeline_floor_level`` is provided the antenna z-coordinate
    is corrected for the difference between the annotation tool's
    floor_level (ankle-based, stored in the JSON) and the pipeline's
    floor_level (camera-based, from ``compute_fresh_rotation``).

    Args:
        session_yaml: Path to session YAML with ``antenna_json`` in
            ``data_files``.
        t_bins: Sorted unique bin center times (1-D array).
        merged_df: DataFrame with ``com_x``, ``com_y``, ``com_z`` columns.
        pipeline_floor_level: Floor level used by the encoding pipeline
            (from ``session_loader.floor_level``).  If provided,
            the antenna z-coordinate is shifted so both COM and antenna
            reference the same floor.

    Returns:
        Z-scored antenna distance array (same length as *t_bins*).

    Raises:
        FileNotFoundError: If ``antenna_json`` is absent from the YAML
            or the JSON file does not exist on disk.
        ValueError: If JSON validation fails (wrong session, bad format,
            missing bins).
    """
    import json as _json

    from freemat_data import load_session_yaml

    sref = load_session_yaml(session_yaml)

    # --- Resolve antenna JSON path from session YAML ---
    data_files = getattr(sref, "data_files", None) or {}
    antenna_path: Optional[str] = data_files.get("antenna_json")
    if antenna_path is None:
        raise FileNotFoundError(
            f"Session YAML missing 'data_files.antenna_json': {session_yaml}"
        )
    if not os.path.isfile(antenna_path):
        raise FileNotFoundError(
            f"Antenna JSON not found on disk: {antenna_path}"
        )

    # --- Load and validate JSON ---
    with open(antenna_path, "r") as f:
        data = _json.load(f)

    if data.get("coordinate_frame") != "encoding":
        raise ValueError(
            f"antenna JSON coordinate_frame must be 'encoding', "
            f"got '{data.get('coordinate_frame')}'"
        )
    pos = np.array(data["position_encoding_frame_m"])
    if pos.shape != (3,) or not np.all(np.isfinite(pos)):
        raise ValueError(
            f"antenna position must be finite length-3, got {pos}"
        )
    json_session = data.get("session", "")
    yaml_session_id = getattr(sref, "session_id", "")
    if json_session != yaml_session_id:
        raise ValueError(
            f"antenna JSON session='{json_session}' != "
            f"YAML session='{yaml_session_id}'"
        )

    # --- Correct z-coordinate for floor_level mismatch ---
    # The annotation tool (viser_pose) uses ankle-based floor_level while
    # compute_fresh_rotation uses camera-based floor_level.  Both produce
    # enc_z = rot_world[vert] - floor, so a shift of (json_floor - pipeline_floor)
    # realigns the antenna to the pipeline's reference.
    json_floor = data.get("floor_level_m")
    if pipeline_floor_level is not None and json_floor is not None:
        floor_delta = json_floor - pipeline_floor_level
        if abs(floor_delta) > 1e-6:
            pos[2] += floor_delta

    # --- Extract per-bin COM (one-pass via drop_duplicates + reindex) ---
    com_df = (
        merged_df.drop_duplicates("t_bin_s")
        .sort_values("t_bin_s")[["t_bin_s", "com_x", "com_y", "com_z"]]
    )
    com_df = com_df.set_index("t_bin_s").reindex(t_bins)
    n_missing = int(com_df["com_x"].isna().sum())
    if n_missing > 0:
        pct = 100 * n_missing / len(t_bins)
        if pct > 50.0:
            raise ValueError(
                f"{n_missing}/{len(t_bins)} bins ({pct:.1f}%) missing COM after reindex"
            )
        print(f"  d_antenna: {n_missing}/{len(t_bins)} bins ({pct:.1f}%) missing COM "
              "-- will be NaN (dropped per-unit via dropna)")
    com_arr = com_df[["com_x", "com_y", "com_z"]].to_numpy(dtype=np.float64)

    # --- Euclidean distance ---
    d_antenna = np.linalg.norm(com_arr - pos, axis=1)

    # --- Z-score (nanmean/nanstd to handle missing COM bins) ---
    mu = np.nanmean(d_antenna)
    std = np.nanstd(d_antenna)
    if std < 1e-12:
        d_z = np.zeros_like(d_antenna)
    else:
        d_z = (d_antenna - mu) / std

    if len(d_z) != len(t_bins):
        raise ValueError(
            f"d_antenna length {len(d_z)} != t_bins length {len(t_bins)}"
        )

    return d_z


def _add_antenna_distance_covariate(
    d_antenna_z: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Append z-scored antenna distance column to design matrix.

    Args:
        d_antenna_z: Z-scored antenna distance array of shape (T,).
        X: Existing design matrix (T, D).

    Returns:
        Extended design matrix (T, D+1).
    """
    return np.hstack([X, d_antenna_z.reshape(-1, 1)])


__all__ = [
    "build_lagged_features",
    "run_subsample_sensitivity",
    "load_rssi_for_session",
    "_add_rssi_covariate",
    "load_antenna_distance_for_session",
    "_add_antenna_distance_covariate",
]
