"""Project-specific, light kinematic features for Step-A.

Implements a conservative, per-session computation of KIN-lite features from
aligned pose parquet rows.

Input columns per row (aligned per bin):
    - session_id, t_bin_s
    - P_x, P_y, P_z: COM position
    - fhead_x, fhead_y, fhead_z: head-forward vector (camera/world frame)
    - n_x, n_y, n_z: floor plane normal (ideally session-constant)

Output columns per row:
    ['session_id','t_bin_s','speed','v_parallel','v_perp','v_z',
     'theta_look','yaw_rate','align_cos',
     'sin1','cos1','sin2','cos2','sin3','cos3']

Notes:
    - Applies light moving-average smoothing on COM (default 50 ms) before
      central differences.
    - Uses a stable in-plane basis derived from the session-level median floor
      normal. All angle computations are done in that fixed basis.
    - NaNs are propagated; derivative edges use one-sided differences.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from freemat_data.kinematics import (
    central_diff,
    moving_average,
    plane_basis_from_normal,
    project_to_plane,
    unwrap_phase,
)
from freemat_data import load_pose3d, orient_pose_with_floor
from freemat_data.util import pose_time_seconds
from freemat_data import video_time_to_pose_axis
from freemat_data.calibration import rotation_from_floor_json
from freemat.single_fm.features.kin_features import (
    _compute_head_forward_robust as _fwd_robust,
    _canonicalize_joint_names as _canon_names,
)

__all__ = ["compute_kin_lite", "compute_kin_lite_dense"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)


def _median_normal(df: pd.DataFrame) -> np.ndarray:
    ncols = ["n_x", "n_y", "n_z"]
    n = df[ncols].to_numpy(dtype=float)
    med = np.nanmedian(n, axis=0)
    if not np.isfinite(med).all():
        # Fallback to Z-up
        med = np.array([0.0, 0.0, 1.0], dtype=float)
    return _unit(med)


def _median_dt_s(t: np.ndarray) -> float:
    dt = np.diff(t.astype(float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.nanmedian(dt)) if dt.size else 1.0


def _fourier_cols(theta: np.ndarray, K: int = 3) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in range(1, K + 1):
        out[f"sin{k}"] = np.sin(k * theta)
        out[f"cos{k}"] = np.cos(k * theta)
    return out


def _fill_nan_ffill_bfill(x: np.ndarray) -> np.ndarray:
    """Fill NaNs in 1D array by forward-fill then backward-fill.

    Args:
        x: 1D array

    Returns:
        Filled 1D array.
    """
    y = x.copy()
    finite = np.isfinite(y)
    if finite.all():
        return y
    # Forward fill
    idx = np.where(finite, np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    m = ~finite & (idx >= 0)
    y[m] = y[idx[m]]
    # Backward fill for leading NaNs
    finite = np.isfinite(y)
    if not finite.all():
        idx2 = np.where(finite, np.arange(len(y)), len(y))
        # reverse minimum accumulate
        idx2 = np.minimum.accumulate(idx2[::-1])[::-1]
        m2 = ~finite & (idx2 < len(y))
        y[m2] = y[idx2[m2]]
    return y


def _interp_short_gaps_1d(
    y: np.ndarray, t_axis: np.ndarray, max_gap_s: float,
) -> np.ndarray:
    """Linearly interpolate NaN runs shorter than or equal to max_gap_s.

    Args:
        y: 1D array of values.
        t_axis: Time vector aligned with y.
        max_gap_s: Maximum gap duration (seconds) to fill.

    Returns:
        1D array with only short gaps interpolated; longer gaps remain NaN.
    """
    yy = y.astype(float).copy()
    n = len(yy)
    if n == 0 or not np.any(~np.isfinite(yy)):
        return yy
    i = 0
    while i < n:
        if np.isfinite(yy[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isfinite(yy[j]):
            j += 1
        il = i - 1
        ir = j
        can_fill = il >= 0 and ir < n and np.isfinite(yy[il]) and np.isfinite(yy[ir])
        gap_s = (t_axis[ir] - t_axis[il]) if (can_fill and ir < n and il >= 0) else float("inf")
        if can_fill and gap_s <= float(max_gap_s):
            yy[i:j] = np.linspace(yy[il], yy[ir], (j - il + 1))[1:-1]
        i = j
    return yy


def _fill_gaps_limited(
    y: np.ndarray, t_axis: np.ndarray, max_s: float,
) -> np.ndarray:
    """Fill short NaN gaps by linear interpolation, edge-fill the rest.

    Args:
        y: 1D array (e.g. unwrapped angle).
        t_axis: Time vector aligned with y.
        max_s: Maximum gap duration (seconds) to interpolate.

    Returns:
        1D array with short gaps interpolated and edges forward/backward filled.
    """
    yy = y.copy()
    if not np.isfinite(yy).any() or max_s <= 0:
        return _fill_nan_ffill_bfill(yy)
    m = np.isfinite(yy)
    if m.all():
        return yy
    isn = ~m
    if not isn.any():
        return yy
    i = 0
    while i < len(yy):
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < len(yy) and isn[j]:
            j += 1
        gap_s = (t_axis[j - 1] - t_axis[i]) if j - 1 < len(t_axis) and i < len(t_axis) else float("inf")
        if gap_s <= max_s:
            il = i - 1
            ir = j
            if il >= 0 and ir < len(yy) and np.isfinite(yy[il]) and np.isfinite(yy[ir]):
                yy[i:j] = np.linspace(yy[il], yy[ir], j - il + 1)[1:-1]
            else:
                yy[i:j] = np.nan
        i = j
    return _fill_nan_ffill_bfill(yy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_kin_lite(
    pose_df: pd.DataFrame,
    *,
    smooth_win_s: float = 0.05,
    max_fill_s: float = 0.25,
    min_forward_norm: float = 0.05,
    com_max_gap_s: float = 2.0,
) -> pd.DataFrame:
    """Compute compact head-frame kinematics with robust forward fallback.

    When per-frame head-forward projected to the floor plane is missing or
    below ``min_forward_norm``, this function falls back to the CoM heading
    (projected and normalized) for those frames. This ensures ``theta_look`` and
    its Fourier terms have usable coverage in noisy head-pose regimes.

    Returns DataFrame with columns:
    ['session_id','t_bin_s','speed','v_parallel','v_perp','v_z',
     'theta_look','yaw_rate','align_cos',
     'sin1','cos1','sin2','cos2','sin3','cos3']
    """
    req = {
        "session_id",
        "t_bin_s",
        "P_x",
        "P_y",
        "P_z",
        "fhead_x",
        "fhead_y",
        "fhead_z",
        "n_x",
        "n_y",
        "n_z",
    }
    missing = req.difference(set(pose_df.columns))
    if missing:
        raise KeyError(f"compute_kin_lite missing required columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    for sid, g in pose_df.groupby("session_id", sort=False):
        g = g.sort_values("t_bin_s", kind="mergesort").copy()
        t = g["t_bin_s"].to_numpy(dtype=float)
        dt = _median_dt_s(t)
        fps = 1.0 / dt if dt > 0 else 1.0

        P = g[["P_x", "P_y", "P_z"]].to_numpy(dtype=float)
        # Bounded timewise gap fill for COM before smoothing/differencing.

        P_f = P.copy()
        for d in range(P.shape[1]):
            P_f[:, d] = _interp_short_gaps_1d(P[:, d], t, max_gap_s=float(com_max_gap_s))
        # Coverage and gap-length stats (overall COM finite across xyz)
        pre_fin = np.isfinite(P).all(axis=1)
        post_fin = np.isfinite(P_f).all(axis=1)
        cov_pre = float(np.mean(pre_fin))
        cov_post = float(np.mean(post_fin))
        # Gap durations (seconds) based on pre_fin mask
        gaps = []
        i = 0
        while i < len(pre_fin):
            if pre_fin[i]:
                i += 1
                continue
            j = i
            while j < len(pre_fin) and not pre_fin[j]:
                j += 1
            # duration from first to last invalid sample; approximate with time span
            if j - 1 < len(t) and i < len(t):
                dur = float(max(0.0, (t[j - 1] - t[i])))
                gaps.append(dur)
            i = j
        if gaps:
            gaps_arr = np.asarray(gaps, dtype=float)
            p50 = float(np.nanpercentile(gaps_arr, 50))
            p90 = float(np.nanpercentile(gaps_arr, 90))
            p99 = float(np.nanpercentile(gaps_arr, 99))
            n_short = int(np.sum(gaps_arr <= float(com_max_gap_s)))
            n_long = int(np.sum(gaps_arr > float(com_max_gap_s)))
            try:
                logging.info(
                    "[com_gaps] session=%s cov_pre=%.3f cov_post=%.3f n_gaps=%d short<=%.2fs=%d long>%.2fs=%d p50=%.2f p90=%.2f p99=%.2f max=%.2f",
                    str(sid), cov_pre, cov_post, len(gaps), float(com_max_gap_s), n_short, float(com_max_gap_s), n_long, p50, p90, p99, float(np.nanmax(gaps_arr)),
                )
            except (TypeError, ValueError):
                pass
        F = g[["fhead_x", "fhead_y", "fhead_z"]].to_numpy(dtype=float)
        n_med = _median_normal(g)

        # Stable in-plane basis once per session
        e1, e2 = plane_basis_from_normal(n_med)
        ez = n_med

        # Smooth COM before differencing
        P_s = moving_average(P_f, win_s=float(smooth_win_s), fps=float(fps))
        V = central_diff(P_s, dt=float(dt))

        # Head-forward projected to plane
        Fp = project_to_plane(F, n_med)
        # Gate by in-plane forward strength
        fnorm = np.linalg.norm(Fp, axis=1)
        weak = ~(np.isfinite(fnorm) & (fnorm >= min_forward_norm))
        # Robust fallback: CoM heading projected to plane for weak frames
        Vp_fb = project_to_plane(V, n_med)
        spd_fb = np.linalg.norm(Vp_fb, axis=1)
        good_fb = np.isfinite(Vp_fb).all(axis=1) & (spd_fb > 1e-9)
        use_fb = weak & good_fb
        if np.any(use_fb):
            Fp[use_fb] = Vp_fb[use_fb] / (spd_fb[use_fb, None] + 1e-12)
        # For any remaining weak frames without fallback velocity, keep NaN to allow gap-fill logic below
        still_weak = ~(np.isfinite(np.linalg.norm(Fp, axis=1)))
        if np.any(still_weak):
            Fp[still_weak] = np.array([np.nan, np.nan, np.nan])

        # Angles in fixed plane basis (raw, may contain NaNs)
        f1 = np.sum(Fp * e1[None, :], axis=1)
        f2 = np.sum(Fp * e2[None, :], axis=1)
        theta_look_wrapped = np.arctan2(f2, f1)
        theta_look_raw = unwrap_phase(theta_look_wrapped)
        # Fill only short gaps (<= max_fill_s)

        theta_look = _fill_gaps_limited(theta_look_raw, t, float(max_fill_s))
        frac_filled = float(np.mean(~np.isfinite(theta_look_raw) & np.isfinite(theta_look)))
        try:
            logging.info(
                "[compute_kin_lite] session=%s dt=%.3f fps=%.1f filled_theta_frac=%.3f",
                str(sid), float(dt), float(fps), frac_filled,
            )
        except (TypeError, ValueError):
            pass

        # Optional light smoothing on theta before building axes (Savitzky-Golay if available)
        theta_s = theta_look
        try:
            from scipy.signal import savgol_filter  # type: ignore
            w = int(7 if (7 % 2 == 1) else 7 + 1)
            theta_s = savgol_filter(theta_look, window_length=min(w, max(5, (len(theta_look)//10)*2+1)), polyorder=2, mode="interp")
        except (ImportError, ValueError):
            # fallback: small moving average in angle space
            k = max(3, int(round(0.07 * fps)) | 1)
            if k > 1 and k < len(theta_look):
                c = np.cos(theta_look)
                s = np.sin(theta_look)
                kc = np.ones(k) / k
                c = np.convolve(c, kc, mode="same")
                s = np.convolve(s, kc, mode="same")
                theta_s = np.arctan2(s, c)

        # Reconstruct in-plane head axes from smoothed theta
        cth_s = np.cos(theta_s)
        sth_s = np.sin(theta_s)
        ex = (cth_s[:, None] * e1[None, :]) + (sth_s[:, None] * e2[None, :])
        ex = _unit(ex)
        ey = _unit(np.cross(ez[None, :], ex))

        # Components of velocity along head frame
        v_parallel = np.sum(V * ex, axis=1)
        v_perp = np.sum(V * ey, axis=1)
        v_z = np.sum(V * ez[None, :], axis=1)
        speed = np.linalg.norm(V, axis=1)

        Vp = project_to_plane(V, n_med)
        v1 = np.sum(Vp * e1[None, :], axis=1)
        v2 = np.sum(Vp * e2[None, :], axis=1)
        theta_move_wrapped = np.arctan2(v2, v1)
        theta_move = unwrap_phase(theta_move_wrapped)

        # Yaw rate from smoothed, unwrapped look angle
        yaw_rate = central_diff(theta_s[:, None], dt=float(dt))[:, 0]

        align_cos = np.cos(theta_move - theta_look)

        out = pd.DataFrame(
            {
                "session_id": g["session_id"].to_numpy(),
                "t_bin_s": t,
                "speed": speed,
                "v_parallel": v_parallel,
                "v_perp": v_perp,
                "v_z": v_z,
                "theta_look": theta_look,
                "yaw_rate": yaw_rate,
                "align_cos": align_cos,
            }
        )
        # Pass-through pose_full_valid if present in input
        if "pose_full_valid" in g.columns:
            out["pose_full_valid"] = g["pose_full_valid"].to_numpy().astype(bool)
        # Per-feature coverage diagnostics
        try:
            cov = {k: float(np.isfinite(out[k]).mean()) for k in [
                "speed", "v_parallel", "v_perp", "v_z", "theta_look", "yaw_rate", "align_cos"
            ]}
            logging.info("[kin_cov] session=%s %s", str(sid), {k: round(v, 3) for k, v in cov.items()})
        except (KeyError, TypeError):
            pass
        four = _fourier_cols(theta_look, K=3)
        for k, arr in four.items():
            out[k] = arr
        rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "session_id",
                "t_bin_s",
                "speed",
                "v_parallel",
                "v_perp",
                "v_z",
                "theta_look",
                "yaw_rate",
                "align_cos",
                "sin1",
                "cos1",
                "sin2",
                "cos2",
                "sin3",
                "cos3",
            ]
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def compute_kin_lite_dense(
    pose_npz: str,
    neural_df: pd.DataFrame,
    *,
    smooth_win_s: float = 0.05,
    max_fill_s: float = 0.25,
    min_forward_norm: float = 0.05,
    com_max_gap_s: float = 2.0,
    agg: str = "mean",
) -> pd.DataFrame:
    """Compute KIN-lite at pose frame rate and downsample to neural bins.

    This path computes COM/forward, velocities, yaw_rate, and alignment at the
    native pose rate (e.g., 40 Hz), then aggregates within each neural bin
    centered at ``t_bin_s`` using a simple mean. Angle terms are aggregated via
    circular means by averaging sin/cos components.

    Args:
        pose_npz: Path to upright/leveled NPZ (Z-up after orient). If plane JSON
            is present, its normal is used as the floor normal.
        neural_df: Neural table for a single session containing ``t_bin_s`` and
            ``session_id`` (used for alignment and metadata).
        smooth_win_s: COM smoothing window before differencing (seconds).
        max_fill_s: Maximum gap (seconds) for head-direction interpolation prior
            to yaw_rate computation.
        min_forward_norm: Minimum in-plane head-forward norm to accept.
        com_max_gap_s: Maximum COM gap duration to linearly interpolate.
        agg: Aggregation method within bins (currently 'mean').

    Returns:
        DataFrame with the same columns as ``compute_kin_lite`` and aligned to
        ``t_bin_s`` present in ``neural_df``.
    """
    if neural_df is None or neural_df.empty:
        raise ValueError("compute_kin_lite_dense: empty neural_df")
    sessions = pd.unique(neural_df["session_id"].astype(str))
    if sessions.size != 1:
        raise ValueError("compute_kin_lite_dense expects single-session neural_df")
    sid = str(sessions[0])
    t_bins_vals = neural_df.loc[
        neural_df["session_id"].astype(str) == sid, "t_bin_s"
    ].dropna().to_numpy(float)
    t_bins_video = np.sort(pd.unique(t_bins_vals))
    if t_bins_video.size != t_bins_vals.size:
        try:
            logging.info(
                "[dense_kin] deduped t_bin_s per session: %d -> %d unique",
                int(t_bins_vals.size), int(t_bins_video.size),
            )
        except (TypeError, ValueError):
            pass
    if t_bins_video.size == 0:
        raise ValueError("compute_kin_lite_dense: no t_bin_s values")

    pose0 = load_pose3d(pose_npz)
    pose, _ = orient_pose_with_floor(pose0)
    t_pose = pose_time_seconds(pose)
    if t_pose.size != pose.T:
        t_pose = np.arange(pose.T, dtype=float) / float(pose.fps or 1.0)

    # Floor normal from floor_plane.json if present
    try:
        floor_json = os.path.join(os.path.dirname(os.path.abspath(pose_npz)), "floor_plane.json")
        if os.path.isfile(floor_json):
            _, plane_upr, _ = rotation_from_floor_json(floor_json)
            n_vec = np.array([float(plane_upr["a"]), float(plane_upr["b"]), float(plane_upr["c"])], dtype=float)
        else:
            n_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    except (FileNotFoundError, KeyError, ValueError):
        n_vec = np.array([0.0, 0.0, 1.0], dtype=float)

    # NaN-aware per-frame COM and robust head-forward
    X = np.asarray(pose.X, dtype=float)
    fin = np.isfinite(X).all(axis=2)  # (T, J)
    cnt = fin.sum(axis=1)  # (T,)
    sum_xyz = np.where(fin[:, :, None], X, 0.0).sum(axis=1)  # (T, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        com_all = sum_xyz / np.where(cnt[:, None] > 0, cnt[:, None], np.nan)
    # forward
    try:
        names = _canon_names(pose.joint_names or [])
        fwd = _fwd_robust(pose.X, names, n_vec)
    except (KeyError, ValueError, IndexError):
        from freemat_data import compute_head_forward as _hf
        fwd = _hf(pose.X, pose.joint_names)
    # Gap fill COM (short gaps only) then smooth and diff

    P = com_all.copy()
    for d in range(P.shape[1]):
        P[:, d] = _interp_short_gaps_1d(P[:, d], t_pose, float(com_max_gap_s))
    # Coverage and gap-length stats for COM at dense rate
    try:
        pre_fin = np.isfinite(com_all).all(axis=1)
        post_fin = np.isfinite(P).all(axis=1)
        cov_pre = float(np.mean(pre_fin))
        cov_post = float(np.mean(post_fin))
        gaps = []
        i = 0
        n = len(pre_fin)
        while i < n:
            if pre_fin[i]:
                i += 1
                continue
            j = i
            while j < n and not pre_fin[j]:
                j += 1
            if j - 1 < len(t_pose) and i < len(t_pose):
                gaps.append(float(max(0.0, (t_pose[j - 1] - t_pose[i]))))
            i = j
        if gaps:
            gaps_arr = np.asarray(gaps, dtype=float)
            p50 = float(np.nanpercentile(gaps_arr, 50))
            p90 = float(np.nanpercentile(gaps_arr, 90))
            p99 = float(np.nanpercentile(gaps_arr, 99))
            n_short = int(np.sum(gaps_arr <= float(com_max_gap_s)))
            n_long = int(np.sum(gaps_arr > float(com_max_gap_s)))
            logging.info(
                "[com_gaps] session=%s cov_pre=%.3f cov_post=%.3f n_gaps=%d short<=%.2fs=%d long>%.2fs=%d p50=%.2f p90=%.2f p99=%.2f max=%.2f",
                str(sid), cov_pre, cov_post, len(gaps), float(com_max_gap_s), n_short, float(com_max_gap_s), n_long, p50, p90, p99, float(np.nanmax(gaps_arr)),
            )
    except (TypeError, ValueError):
        pass
    # Basis and kinematics at dense rate
    dt = float(np.nanmedian(np.diff(t_pose))) if t_pose.size > 1 else 1.0
    fps = 1.0 / dt if dt > 0 else 1.0
    n_med = _unit(n_vec.reshape(1, 3))[0]
    e1, e2 = plane_basis_from_normal(n_med)
    ez = n_med
    P_s = moving_average(P, win_s=float(smooth_win_s), fps=float(fps))
    V = central_diff(P_s, dt=float(dt))
    # forward projected to plane with fallback to COM heading when weak
    Fp = project_to_plane(fwd, n_med)
    fnorm = np.linalg.norm(Fp, axis=1)
    weak = ~(np.isfinite(fnorm) & (fnorm >= float(min_forward_norm)))
    Vp_fb = project_to_plane(V, n_med)
    spd_fb = np.linalg.norm(Vp_fb, axis=1)
    good_fb = np.isfinite(Vp_fb).all(axis=1) & (spd_fb > 1e-9)
    use_fb = weak & good_fb
    if np.any(use_fb):
        Fp[use_fb] = Vp_fb[use_fb] / (spd_fb[use_fb, None] + 1e-12)
    still_weak = ~(np.isfinite(np.linalg.norm(Fp, axis=1)))
    if np.any(still_weak):
        Fp[still_weak] = np.array([np.nan, np.nan, np.nan])
    f1 = np.sum(Fp * e1[None, :], axis=1)
    f2 = np.sum(Fp * e2[None, :], axis=1)
    theta_look_wrapped = np.arctan2(f2, f1)
    theta_look_raw = unwrap_phase(theta_look_wrapped)

    # Limited gap fill of theta for yaw rate

    theta_look = _fill_gaps_limited(theta_look_raw, t_pose, float(max_fill_s))
    # Build head-aligned axes from smoothed theta for v components
    try:
        from scipy.signal import savgol_filter  # type: ignore
        w = int(7 if (7 % 2 == 1) else 7 + 1)
        theta_s = savgol_filter(theta_look, window_length=min(w, max(5, (len(theta_look)//10)*2+1)), polyorder=2, mode="interp")
    except (ImportError, ValueError):
        k = max(3, int(round(0.07 * fps)) | 1)
        if k > 1 and k < len(theta_look):
            c = np.cos(theta_look)
            s = np.sin(theta_look)
            kc = np.ones(k) / k
            c = np.convolve(c, kc, mode="same")
            s = np.convolve(s, kc, mode="same")
            theta_s = np.arctan2(s, c)
        else:
            theta_s = theta_look

    cth_s = np.cos(theta_s)
    sth_s = np.sin(theta_s)
    ex = _unit((cth_s[:, None] * e1[None, :]) + (sth_s[:, None] * e2[None, :]))
    ey = _unit(np.cross(ez[None, :], ex))
    v_parallel = np.sum(V * ex, axis=1)
    v_perp = np.sum(V * ey, axis=1)
    v_z = np.sum(V * ez[None, :], axis=1)
    speed = np.linalg.norm(V, axis=1)
    Vp = project_to_plane(V, n_med)
    v1 = np.sum(Vp * e1[None, :], axis=1)
    v2 = np.sum(Vp * e2[None, :], axis=1)
    theta_move = unwrap_phase(np.arctan2(v2, v1))
    yaw_rate = central_diff(theta_s[:, None], dt=float(dt))[:, 0]
    align_cos = np.cos(theta_move - theta_look)

    # Prepare per-frame arrays, including sin/cos harmonics
    sin1 = np.sin(theta_look)
    cos1 = np.cos(theta_look)
    sin2 = np.sin(2 * theta_look)
    cos2 = np.cos(2 * theta_look)
    sin3 = np.sin(3 * theta_look)
    cos3 = np.cos(3 * theta_look)
    full_valid = fin.all(axis=1)

    # Aggregate to neural bins using vectorized binning.
    pose_dur = float(t_pose[-1] - t_pose[0]) if t_pose.size else 0.0
    already_pose_axis = (
        (t_bins_video.min(initial=0.0) >= -0.25)
        and (t_bins_video.max(initial=0.0) <= t_pose[-1] + 0.25)
        and ((t_bins_video.max(initial=0.0) - t_bins_video.min(initial=0.0)) <= pose_dur + 1.0)
    )
    if already_pose_axis:
        tb_pose = t_bins_video
        try:
            logging.info("[dense_kin] bins appear pose-centered; no time shift applied")
        except (TypeError, ValueError, AttributeError):
            pass
    else:
        tb_pose = video_time_to_pose_axis(t_bins_video, pose)
        try:
            off = float(pose.fids[0]) / float(pose.fps or 1.0)
            logging.info("[dense_kin] shifting neural bins by -%.3fs to pose axis", off)
        except (TypeError, ValueError, AttributeError):
            pass
    tb = np.asarray(tb_pose, dtype=float)
    if tb.size < 2:
        # Fallback half-width: use half a second
        half = 0.5
    else:
        half = 0.5 * float(np.nanmedian(np.diff(tb)))
    edges = np.r_[tb[0] - half, (tb[1:] + tb[:-1]) / 2.0, tb[-1] + half]
    # Label each pose frame to its nearest neural bin based on pose-axis time
    labels = np.digitize(t_pose, edges) - 1  # 0..len(tb)-1
    labels = np.clip(labels, 0, len(tb) - 1)

    def _bin_mean(arr: np.ndarray) -> np.ndarray:
        m = np.isfinite(arr)
        w = np.where(m, arr, 0.0)
        s = np.bincount(labels[m], weights=w[m], minlength=len(tb))
        c = np.bincount(labels[m], minlength=len(tb))
        out = np.full(len(tb), np.nan, dtype=float)
        nz = c > 0
        out[nz] = s[nz] / c[nz]
        return out

    speed_b = _bin_mean(speed)
    vpar_b = _bin_mean(v_parallel)
    vperp_b = _bin_mean(v_perp)
    vz_b = _bin_mean(v_z)
    yaw_b = _bin_mean(yaw_rate)
    align_b = _bin_mean(align_cos)
    s1_b = _bin_mean(sin1)
    c1_b = _bin_mean(cos1)
    s2_b = _bin_mean(sin2)
    c2_b = _bin_mean(cos2)
    s3_b = _bin_mean(sin3)
    c3_b = _bin_mean(cos3)
    # Circular mean theta from mean sin/cos
    theta_b = np.arctan2(s1_b, c1_b)
    # Pose validity within bin (any frame valid)
    fv = full_valid.astype(int)
    fv_b = np.bincount(labels, weights=fv, minlength=len(tb)) > 0

    out = pd.DataFrame(
        {
            "session_id": sid,
            "t_bin_s": t_bins_video,
            "speed": speed_b,
            "v_parallel": vpar_b,
            "v_perp": vperp_b,
            "v_z": vz_b,
            "theta_look": theta_b,
            "yaw_rate": yaw_b,
            "align_cos": align_b,
            "sin1": s1_b,
            "cos1": c1_b,
            "sin2": s2_b,
            "cos2": c2_b,
            "sin3": s3_b,
            "cos3": c3_b,
            "pose_full_valid": fv_b.astype(bool),
        }
    )
    # Coverage diagnostics
    try:
        cov = {k: float(np.isfinite(out[k]).mean()) for k in [
            "speed", "v_parallel", "v_perp", "v_z", "theta_look", "yaw_rate", "align_cos"
        ]}
        logging.info("[kin_cov] session=%s %s", sid, {k: round(v, 3) for k, v in cov.items()})
    except (KeyError, TypeError):
        pass
    return out
