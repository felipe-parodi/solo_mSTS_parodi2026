"""3D spatial information analysis for encoding model residuals.

Provides occupancy-corrected 3D rate maps, Skaggs spatial information
(bits/spike), occupancy-weighted spatial heterogeneity for residuals,
split-half stability, and trajectory-preserving circular-shift significance
tests.

Core Algorithms:

    Arena Normalization:
        Horizontal: (com_xy - arena.center[horiz]) / arena.hexagon_radius -> ~[-1,1]
        Vertical:   com_z / (ceiling_height - floor_height) -> ~[0,1]

    Skaggs Spatial Information (bits/spike):
        SI = sum_i p_i * (lambda_i / lambda_bar) * log2(lambda_i / lambda_bar)
        where lambda_i = mean rate in voxel i, lambda_bar = overall mean,
        p_i = occupancy fraction.  Valid only for non-negative signals.

    Spatial Heterogeneity (for residuals):
        SH = sum_i p_i * (v_i - v_bar)^2
        Occupancy-weighted voxel variance. Valid for signed values.

    Split-Half Stability:
        Pearson r between first-half and second-half rate maps.

    Circular Shift Shuffle:
        Circularly shift spike train relative to position trajectory
        (preserves temporal autocorrelation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VoxelizedRateMap:
    """3D occupancy-corrected firing rate map.

    Attributes:
        rate_map_3d: Mean signal per voxel, masked where occupancy < min.
        occupancy_3d: Raw bin counts (n_z, n_y, n_x).
        p_occ: Occupancy fraction (sums to 1 over unmasked voxels).
        edges_x: Bin edges for normalized X.
        edges_y: Bin edges for normalized Y.
        edges_z: Bin edges for normalized Z.
        min_occupancy: Threshold used for masking.
    """

    rate_map_3d: np.ma.MaskedArray
    occupancy_3d: np.ndarray
    p_occ: np.ndarray
    edges_x: np.ndarray
    edges_y: np.ndarray
    edges_z: np.ndarray
    min_occupancy: int


@dataclass
class SpatialInfoResult:
    """Per-unit spatial analysis result.

    Attributes:
        unit_id: Integer unit identifier.
        spatial_metric: Skaggs SI (bits/spike) or spatial heterogeneity.
        metric_name: 'skaggs_si' or 'spatial_heterogeneity'.
        stability_r: Split-half Pearson correlation.
        p_value: From circular-shift null test.
        null_distribution: Null metric values from shuffles.
        signal_col: Which signal was analyzed ('y_true', 'y_pred', 'residual').
    """

    unit_id: int
    spatial_metric: float
    metric_name: str
    stability_r: float
    p_value: float
    null_distribution: np.ndarray
    signal_col: str


# ---------------------------------------------------------------------------
# Cross-session alignment
# ---------------------------------------------------------------------------

def align_coordinates_to_reference(
    com_x: np.ndarray,
    com_y: np.ndarray,
    com_z: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a rigid transform to COM coordinates for cross-session alignment.

    Transforms coordinates via ``R @ [x, y, z]^T + t`` so that multiple
    sessions share a common coordinate frame (e.g., aligned to a reference
    session using bottom-camera Procrustes).

    Call this **before** ``normalize_arena_coordinates()``.

    Args:
        com_x: X coordinates (meters).
        com_y: Y coordinates (meters).
        com_z: Z coordinates (meters).
        R: (3, 3) rotation matrix from Procrustes alignment.
        t: (3,) translation vector from Procrustes alignment.

    Returns:
        Tuple of (aligned_x, aligned_y, aligned_z).
    """
    pts = np.column_stack([com_x, com_y, com_z])  # (N, 3)
    aligned = (R @ pts.T).T + t  # (N, 3)
    return aligned[:, 0], aligned[:, 1], aligned[:, 2]


# ---------------------------------------------------------------------------
# Arena normalization
# ---------------------------------------------------------------------------

def _horiz_indices_from_up_axis(up_axis: str) -> Tuple[int, int]:
    """Map up_axis string to horizontal coordinate indices.

    Args:
        up_axis: 'X', 'Y', or 'Z'.

    Returns:
        Tuple of (h0, h1) indices into 3D coordinate arrays.
    """
    up = up_axis.upper()
    if up == "Y":
        return (0, 2)
    elif up == "X":
        return (1, 2)
    else:  # Z
        return (0, 1)


def normalize_arena_coordinates(
    com_x: np.ndarray,
    com_y: np.ndarray,
    com_z: np.ndarray,
    arena_center: np.ndarray,
    hexagon_radius: float,
    ceiling_height: float,
    floor_height: float,
    up_axis: str = "Y",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize COM coordinates to arena-relative space.

    Horizontal axes are centered on the arena center and divided by
    the hexagon circumradius, giving roughly [-1, 1].  Vertical axis
    (com_z, already floor-subtracted by the encoding pipeline) is
    divided by (ceiling_height - floor_height) giving [0, 1].

    Args:
        com_x: Horizontal coordinate 0 (meters).
        com_y: Horizontal coordinate 1 (meters).
        com_z: Vertical coordinate in raw calibration frame (meters).
        arena_center: Arena center point (3,) in the same frame as COM.
        hexagon_radius: Circumradius of the arena hexagon (meters).
        ceiling_height: Absolute ceiling height in arena frame (meters).
        floor_height: Absolute floor height in arena frame (meters).
        up_axis: Vertical axis label ('X', 'Y', or 'Z').

    Returns:
        Tuple of (norm_x, norm_y, norm_z) arrays.
    """
    h0, h1 = _horiz_indices_from_up_axis(up_axis)
    norm_x = (com_x - arena_center[h0]) / hexagon_radius
    norm_y = (com_y - arena_center[h1]) / hexagon_radius
    height_range = ceiling_height - floor_height
    norm_z = com_z / height_range if height_range > 0 else com_z
    return norm_x, norm_y, norm_z


def validate_normalization(
    normalized_coords_by_session: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    warn_threshold: float = 1.5,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Validate that normalized coordinates overlap across sessions.

    Args:
        normalized_coords_by_session: {session_id: (norm_x, norm_y, norm_z)}.
        warn_threshold: Flag sessions where any axis exceeds this range.

    Returns:
        Dict of {session_id: {axis: (min, max)}} range stats.
    """
    report: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for sid, (nx, ny, nz) in normalized_coords_by_session.items():
        ranges = {
            "norm_x": (float(np.nanmin(nx)), float(np.nanmax(nx))),
            "norm_y": (float(np.nanmin(ny)), float(np.nanmax(ny))),
            "norm_z": (float(np.nanmin(nz)), float(np.nanmax(nz))),
        }
        report[sid] = ranges

        for axis, (lo, hi) in ranges.items():
            if abs(lo) > warn_threshold or abs(hi) > warn_threshold:
                print(
                    f"  [WARN] {sid} {axis} out of range: "
                    f"[{lo:.3f}, {hi:.3f}] exceeds ±{warn_threshold}"
                )

    # Print summary table
    print("\n=== Normalization Validation ===")
    print(f"{'Session':<20} {'norm_x':>18} {'norm_y':>18} {'norm_z':>18}")
    print("-" * 76)
    for sid, ranges in report.items():
        nx_r = ranges["norm_x"]
        ny_r = ranges["norm_y"]
        nz_r = ranges["norm_z"]
        print(
            f"{sid:<20} [{nx_r[0]:+.3f}, {nx_r[1]:+.3f}] "
            f"[{ny_r[0]:+.3f}, {ny_r[1]:+.3f}] "
            f"[{nz_r[0]:+.3f}, {nz_r[1]:+.3f}]"
        )
    print()
    return report


# ---------------------------------------------------------------------------
# 3D voxelization
# ---------------------------------------------------------------------------

def voxelize_3d(
    norm_x: np.ndarray,
    norm_y: np.ndarray,
    norm_z: np.ndarray,
    signal: np.ndarray,
    n_bins_xy: int = 20,
    n_bins_z: int = 5,
    min_occupancy: int = 5,
    xy_range: Tuple[float, float] = (-1.2, 1.2),
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> VoxelizedRateMap:
    """Build 3D occupancy-corrected rate map.

    Args:
        norm_x: Normalized horizontal coordinate 0.
        norm_y: Normalized horizontal coordinate 1.
        norm_z: Normalized vertical coordinate.
        signal: Per-bin signal (firing rate or residual).
        n_bins_xy: Number of bins per horizontal axis.
        n_bins_z: Number of bins for vertical axis.
        min_occupancy: Minimum bin visits to include voxel.
        xy_range: (min, max) for horizontal axes.
        z_range: (min, max) for vertical axis.

    Returns:
        VoxelizedRateMap with masked rate_map_3d.
    """
    edges_x = np.linspace(xy_range[0], xy_range[1], n_bins_xy + 1)
    edges_y = np.linspace(xy_range[0], xy_range[1], n_bins_xy + 1)
    edges_z = np.linspace(z_range[0], z_range[1], n_bins_z + 1)

    # Filter to valid (finite) samples
    valid = (
        np.isfinite(norm_x) & np.isfinite(norm_y)
        & np.isfinite(norm_z) & np.isfinite(signal)
    )
    vx = norm_x[valid]
    vy = norm_y[valid]
    vz = norm_z[valid]
    vs = signal[valid]

    # Digitize into bins (clip to bounds)
    xi = np.clip(np.digitize(vx, edges_x) - 1, 0, n_bins_xy - 1)
    yi = np.clip(np.digitize(vy, edges_y) - 1, 0, n_bins_xy - 1)
    zi = np.clip(np.digitize(vz, edges_z) - 1, 0, n_bins_z - 1)

    # Accumulate
    occupancy = np.zeros((n_bins_z, n_bins_xy, n_bins_xy), dtype=int)
    sum_signal = np.zeros((n_bins_z, n_bins_xy, n_bins_xy), dtype=float)

    for x_i, y_i, z_i, s_i in zip(xi, yi, zi, vs):
        occupancy[z_i, y_i, x_i] += 1
        sum_signal[z_i, y_i, x_i] += s_i

    # Occupancy-corrected rate map
    mask = occupancy < min_occupancy
    rate_map = np.ma.masked_where(
        mask, sum_signal / np.maximum(occupancy, 1)
    )

    # Occupancy fraction (over unmasked voxels)
    total_occ = occupancy[~mask].sum()
    p_occ = np.zeros_like(occupancy, dtype=float)
    if total_occ > 0:
        p_occ[~mask] = occupancy[~mask] / total_occ

    return VoxelizedRateMap(
        rate_map_3d=rate_map,
        occupancy_3d=occupancy,
        p_occ=p_occ,
        edges_x=edges_x,
        edges_y=edges_y,
        edges_z=edges_z,
        min_occupancy=min_occupancy,
    )


# ---------------------------------------------------------------------------
# Spatial metrics
# ---------------------------------------------------------------------------

def _get_valid_mask(rate_map: np.ma.MaskedArray) -> np.ndarray:
    """Get boolean array of unmasked voxels, handling scalar nomask."""
    mask = np.ma.getmaskarray(rate_map)
    return ~mask


def skaggs_spatial_info(
    rate_map: np.ma.MaskedArray,
    p_occ: np.ndarray,
) -> float:
    """Skaggs et al. (1993) spatial information in bits/spike.

    SI = sum_i p_i * (lambda_i / lambda_bar) * log2(lambda_i / lambda_bar)

    Only valid for non-negative signals (y_true, y_pred).

    Args:
        rate_map: Masked 3D rate map (NZ, NY, NX).
        p_occ: Occupancy fraction array, same shape.

    Returns:
        Spatial information in bits/spike, or 0.0 if degenerate.
    """
    valid = _get_valid_mask(rate_map)
    if valid.sum() < 2:
        return 0.0

    lam = rate_map.data[valid]
    p = p_occ[valid]

    # Filter out zero-occupancy and zero-rate voxels
    nonzero = (p > 0) & (lam > 0)
    if nonzero.sum() < 2:
        return 0.0

    lam = lam[nonzero]
    p = p[nonzero]
    p = p / p.sum()  # Re-normalize after filtering

    lam_bar = np.sum(p * lam)
    if lam_bar <= 0:
        return 0.0

    ratio = lam / lam_bar
    si = np.sum(p * ratio * np.log2(ratio))
    return float(si)


def spatial_heterogeneity(
    rate_map: np.ma.MaskedArray,
    p_occ: np.ndarray,
) -> float:
    """Occupancy-weighted voxel variance for signed signals.

    SH = sum_i p_i * (v_i - v_bar)^2

    Valid for residuals (signed values) where Skaggs SI is invalid.

    Args:
        rate_map: Masked 3D rate map (may be signed).
        p_occ: Occupancy fraction array, same shape.

    Returns:
        Spatial heterogeneity (occupancy-weighted variance).
    """
    valid = _get_valid_mask(rate_map)
    if valid.sum() < 2:
        return 0.0

    v = rate_map.data[valid]
    p = p_occ[valid]

    nonzero = p > 0
    if nonzero.sum() < 2:
        return 0.0

    v = v[nonzero]
    p = p[nonzero]
    p = p / p.sum()

    v_bar = np.sum(p * v)
    sh = np.sum(p * (v - v_bar) ** 2)
    return float(sh)


# ---------------------------------------------------------------------------
# Split-half stability
# ---------------------------------------------------------------------------

def split_half_stability(
    norm_x: np.ndarray,
    norm_y: np.ndarray,
    norm_z: np.ndarray,
    signal: np.ndarray,
    n_bins_xy: int = 20,
    n_bins_z: int = 5,
    min_occupancy: int = 5,
    xy_range: Tuple[float, float] = (-1.2, 1.2),
    z_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """Compute split-half stability of spatial rate map.

    Splits time bins into odd/even indices so both halves sample the
    same movement patterns throughout the session, isolating firing-rate
    consistency from behavioral non-stationarity.

    Args:
        norm_x: Normalized X coordinates.
        norm_y: Normalized Y coordinates.
        norm_z: Normalized Z coordinates.
        signal: Per-bin signal values.
        n_bins_xy: Bins per horizontal axis.
        n_bins_z: Bins for vertical axis.
        min_occupancy: Minimum bin occupancy for each half.
        xy_range: Range for horizontal axes.
        z_range: Range for vertical axis.

    Returns:
        Pearson r between odd-bin and even-bin rate maps, or 0.0
        if insufficient data.
    """
    idx_odd = np.arange(0, len(signal), 2)
    idx_even = np.arange(1, len(signal), 2)

    vm1 = voxelize_3d(
        norm_x[idx_odd], norm_y[idx_odd], norm_z[idx_odd], signal[idx_odd],
        n_bins_xy=n_bins_xy, n_bins_z=n_bins_z,
        min_occupancy=max(1, min_occupancy // 2),
        xy_range=xy_range, z_range=z_range,
    )
    vm2 = voxelize_3d(
        norm_x[idx_even], norm_y[idx_even], norm_z[idx_even], signal[idx_even],
        n_bins_xy=n_bins_xy, n_bins_z=n_bins_z,
        min_occupancy=max(1, min_occupancy // 2),
        xy_range=xy_range, z_range=z_range,
    )

    # Correlate over voxels valid in BOTH halves
    both_valid = _get_valid_mask(vm1.rate_map_3d) & _get_valid_mask(vm2.rate_map_3d)
    if both_valid.sum() < 3:
        return 0.0

    v1 = vm1.rate_map_3d.data[both_valid]
    v2 = vm2.rate_map_3d.data[both_valid]

    r, _ = stats.pearsonr(v1, v2)
    return float(r) if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# Circular-shift shuffle test
# ---------------------------------------------------------------------------

def circular_shift_shuffle(
    signal: np.ndarray,
    n_shuffles: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate circularly shifted copies of a signal.

    Shift amount is drawn uniformly from [10%, 90%] of signal length
    to preserve temporal autocorrelation while disrupting
    position-firing alignment.

    Args:
        signal: 1D signal array.
        n_shuffles: Number of shuffled copies.
        rng: NumPy random generator.

    Returns:
        Array of shape (n_shuffles, len(signal)).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(signal)
    lo = max(1, int(0.1 * n))
    hi = max(lo + 1, int(0.9 * n))

    shifts = rng.integers(lo, hi, size=n_shuffles)
    out = np.empty((n_shuffles, n), dtype=signal.dtype)
    for i, s in enumerate(shifts):
        out[i] = np.roll(signal, s)
    return out


def compute_spatial_metrics_with_shuffles(
    norm_x: np.ndarray,
    norm_y: np.ndarray,
    norm_z: np.ndarray,
    signal: np.ndarray,
    unit_id: int,
    signal_col: str = "y_true",
    n_bins_xy: int = 20,
    n_bins_z: int = 5,
    min_occupancy: int = 5,
    n_shuffles: int = 100,
    xy_range: Tuple[float, float] = (-1.2, 1.2),
    z_range: Tuple[float, float] = (0.0, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> SpatialInfoResult:
    """Compute spatial metric with shuffle-based significance test.

    Auto-selects metric: Skaggs SI for non-negative signals (y_true,
    y_pred), spatial heterogeneity for signed signals (residual).

    Args:
        norm_x: Normalized X coordinates.
        norm_y: Normalized Y coordinates.
        norm_z: Normalized Z coordinates.
        signal: Per-bin signal values.
        unit_id: Unit identifier.
        signal_col: Signal name ('y_true', 'y_pred', 'residual').
        n_bins_xy: Bins per horizontal axis.
        n_bins_z: Bins for vertical axis.
        min_occupancy: Minimum voxel occupancy.
        n_shuffles: Number of circular-shift shuffles.
        xy_range: Range for horizontal axes.
        z_range: Range for vertical axis.
        rng: NumPy random generator.

    Returns:
        SpatialInfoResult with metric, stability, p-value, null distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Choose metric based on signal type
    use_skaggs = signal_col in ("y_true", "y_pred")
    metric_name = "skaggs_si" if use_skaggs else "spatial_heterogeneity"
    metric_fn = skaggs_spatial_info if use_skaggs else spatial_heterogeneity

    vox_kwargs = dict(
        n_bins_xy=n_bins_xy, n_bins_z=n_bins_z,
        min_occupancy=min_occupancy,
        xy_range=xy_range, z_range=z_range,
    )

    # Observed metric
    vm = voxelize_3d(norm_x, norm_y, norm_z, signal, **vox_kwargs)
    observed = metric_fn(vm.rate_map_3d, vm.p_occ)

    # Stability
    stab = split_half_stability(
        norm_x, norm_y, norm_z, signal, **vox_kwargs,
    )

    # Shuffle test: shift signal relative to fixed position
    shuffled_signals = circular_shift_shuffle(signal, n_shuffles, rng)
    null_dist = np.empty(n_shuffles)
    for i in range(n_shuffles):
        vm_shuf = voxelize_3d(
            norm_x, norm_y, norm_z, shuffled_signals[i], **vox_kwargs,
        )
        null_dist[i] = metric_fn(vm_shuf.rate_map_3d, vm_shuf.p_occ)

    # One-tailed p-value
    p_value = (1 + np.sum(null_dist >= observed)) / (1 + n_shuffles)

    return SpatialInfoResult(
        unit_id=unit_id,
        spatial_metric=observed,
        metric_name=metric_name,
        stability_r=stab,
        p_value=float(p_value),
        null_distribution=null_dist,
        signal_col=signal_col,
    )


# ---------------------------------------------------------------------------
# Session-level runner
# ---------------------------------------------------------------------------

def run_spatial_analysis_session(
    residuals_df: pd.DataFrame,
    arena_center: np.ndarray,
    hexagon_radius: float,
    ceiling_height: float,
    floor_height: float,
    up_axis: str = "Y",
    signal_col: str = "y_true",
    n_bins_xy: int = 20,
    n_bins_z: int = 5,
    min_occupancy: int = 5,
    n_shuffles: int = 100,
    xy_range: Tuple[float, float] = (-1.2, 1.2),
    z_range: Tuple[float, float] = (0.0, 1.0),
    session_id: str = "",
    subject: str = "",
    debug: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Run spatial analysis for all units in one session.

    Args:
        residuals_df: DataFrame with unit_id, com_x, com_y, com_z, and
            signal_col columns.
        arena_center: Arena center (3,).
        hexagon_radius: Arena hexagon circumradius (meters).
        ceiling_height: Arena ceiling height (meters, absolute).
        floor_height: Arena floor height (meters, absolute).
        up_axis: Vertical axis label.
        signal_col: Which signal to analyze.
        n_bins_xy: Bins per horizontal axis.
        n_bins_z: Bins for vertical axis.
        min_occupancy: Minimum voxel occupancy.
        n_shuffles: Number of shuffles per unit.
        xy_range: Range for horizontal axes.
        z_range: Range for vertical axis.
        session_id: Session identifier for output.
        subject: Subject name for output.
        debug: Print progress.

    Returns:
        Tuple of:
            results_df: Per-unit metrics (unit_id, spatial_metric, ...).
            norm_coords: {session_id: (norm_x, norm_y, norm_z)} for validation.
    """
    unit_ids = residuals_df["unit_id"].unique()
    n_units = len(unit_ids)

    if debug:
        print(
            f"[Spatial] {session_id}: {n_units} units, "
            f"signal={signal_col}, n_shuffles={n_shuffles}"
        )

    # Normalize coordinates (once per session, shared across units)
    norm_x, norm_y, norm_z = normalize_arena_coordinates(
        residuals_df["com_x"].values,
        residuals_df["com_y"].values,
        residuals_df["com_z"].values,
        arena_center, hexagon_radius, ceiling_height, floor_height,
        up_axis=up_axis,
    )

    # Store normalized coords per-row for later per-unit slicing
    residuals_df = residuals_df.copy()
    residuals_df["norm_x"] = norm_x
    residuals_df["norm_y"] = norm_y
    residuals_df["norm_z"] = norm_z

    rng = np.random.default_rng(42)
    results: List[Dict] = []

    for i, uid in enumerate(unit_ids):
        if debug and (i + 1) % 50 == 0:
            print(f"  Unit {i + 1}/{n_units}")

        mask = residuals_df["unit_id"] == uid
        ux = residuals_df.loc[mask, "norm_x"].values
        uy = residuals_df.loc[mask, "norm_y"].values
        uz = residuals_df.loc[mask, "norm_z"].values
        sig = residuals_df.loc[mask, signal_col].values

        res = compute_spatial_metrics_with_shuffles(
            ux, uy, uz, sig,
            unit_id=uid, signal_col=signal_col,
            n_bins_xy=n_bins_xy, n_bins_z=n_bins_z,
            min_occupancy=min_occupancy, n_shuffles=n_shuffles,
            xy_range=xy_range, z_range=z_range, rng=rng,
        )

        results.append({
            "unit_id": uid,
            "spatial_metric": res.spatial_metric,
            "metric_name": res.metric_name,
            "stability_r": res.stability_r,
            "p_value": res.p_value,
            "signal_col": signal_col,
            "session_id": session_id,
            "subject": subject,
        })

    results_df = pd.DataFrame(results)

    # FDR correction
    if len(results_df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, fdr_p, _, _ = multipletests(
            results_df["p_value"].values, method="fdr_bh",
        )
        results_df["p_fdr"] = fdr_p
        results_df["sig_spatial"] = fdr_p < 0.05

    # Build validation coords (session-level, deduplicated by time bin)
    dedup = residuals_df.drop_duplicates(subset=["t_bin_s"])
    norm_coords = {
        session_id: (
            dedup["norm_x"].values,
            dedup["norm_y"].values,
            dedup["norm_z"].values,
        )
    }

    if debug:
        n_sig = results_df["sig_spatial"].sum() if "sig_spatial" in results_df else 0
        print(
            f"  Done: {n_sig}/{n_units} significant "
            f"(median metric={results_df['spatial_metric'].median():.4f})"
        )

    return results_df, norm_coords


__all__ = [
    "VoxelizedRateMap",
    "SpatialInfoResult",
    "normalize_arena_coordinates",
    "validate_normalization",
    "voxelize_3d",
    "skaggs_spatial_info",
    "spatial_heterogeneity",
    "split_half_stability",
    "circular_shift_shuffle",
    "compute_spatial_metrics_with_shuffles",
    "run_spatial_analysis_session",
]
