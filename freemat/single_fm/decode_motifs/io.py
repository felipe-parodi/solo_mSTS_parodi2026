"""Spike data I/O, binning, and parsing helpers for motif decoding."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from freemat_data.banks import load_banks_config, map_channels_to_bank


# ---------------------------------------------------------------------------
# CLI / parsing utilities
# ---------------------------------------------------------------------------


def _parse_range_spec(spec: str) -> np.ndarray:
    """Parse range spec like "-300:300:20" -> array([-300, -280, ..., 300]).

    Args:
        spec: String in the form "start:stop:step" (ms)

    Returns:
        Array of floats with inclusive stop
    """
    s = str(spec).strip()
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid range spec '{spec}' (expected 'start:stop:step')")
    a, b, c = [float(x) for x in parts]
    xs = np.arange(a, b + 1e-9, c, dtype=float)
    return xs


def _parse_grid_csv(s: str) -> np.ndarray:
    """Parse a comma-separated list of floats into an array."""
    return np.array([float(x) for x in str(s).split(",") if str(x).strip() != ""], dtype=float)


def _banks_list(s: str) -> List[str]:
    opts = []
    for part in str(s).split(","):
        q = part.strip().lower()
        if q in ("dorsal", "ventral", "both"):
            opts.append(q)
    return list(dict.fromkeys(opts)) or ["dorsal", "ventral", "both"]


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Spike binning and loading
# ---------------------------------------------------------------------------


@dataclass
class SpikeBanks:
    t_edges: np.ndarray  # (T+1,) edges in seconds
    fr_dorsal: Optional[np.ndarray]  # (T, U_d)
    fr_ventral: Optional[np.ndarray]  # (T, U_v)
    bin_s: float
    bin_hz: float
    downgraded: bool


def _bin_event_times(times_per_unit: Sequence[np.ndarray], *, t_start: float, t_end: float, bin_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram event times into counts per unit.

    Args:
        times_per_unit: List of arrays (seconds) per unit
        t_start: Start time in seconds
        t_end: End time in seconds
        bin_s: Bin width in seconds

    Returns:
        (edges, counts) where counts is (T, U)
    """
    edges = np.arange(float(t_start), float(t_end) + 1e-12, float(bin_s))
    T = max(len(edges) - 1, 0)
    U = len(times_per_unit)
    counts = np.zeros((T, U), dtype=float)
    for j, t in enumerate(times_per_unit):
        if t is None:
            continue
        tj = np.asarray(t, dtype=float).reshape(-1)
        if tj.size == 0:
            continue
        h, _ = np.histogram(tj, bins=edges)
        counts[:, j] = h.astype(float)
    return edges, counts


def _counts_to_fr(counts: np.ndarray, bin_s: float) -> np.ndarray:
    return np.asarray(counts, dtype=float) / float(bin_s)


def _auto_downgrade_needed(counts: np.ndarray, thresh: float = 0.2) -> bool:
    """Return True if median spikes/bin/unit is below threshold."""
    if counts.size == 0:
        return False
    med = float(np.median(counts))
    return med < float(thresh)


def load_and_bin_spikes(npz: Mapping[str, Any], *, bin_hz: int) -> SpikeBanks:
    """Load spikes from NPZ and bin to target rate, with auto-downgrade.

    Supports either counts arrays at a raw dt, or per-unit event time arrays.

    Args:
        npz: Loaded NPZ mapping
        bin_hz: Desired binning frequency

    Returns:
        SpikeBanks struct with FR matrices and time edges
    """
    want_hz = int(bin_hz)
    target_bin_s = 1.0 / float(want_hz)

    fr_d: Optional[np.ndarray] = None
    fr_v: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None
    downgraded = False

    # Event-time inputs
    has_times_d = "spike_times_dorsal" in npz
    has_times_v = "spike_times_ventral" in npz

    # Count-matrix inputs
    has_counts_d = "spikes_dorsal" in npz
    has_counts_v = "spikes_ventral" in npz

    if (has_times_d or has_times_v):
        # Determine global time bounds from event lists
        tmins = []
        tmaxs = []
        times_d = None
        times_v = None
        if has_times_d:
            td = list(npz["spike_times_dorsal"])  # type: ignore[arg-type]
            times_d = [np.asarray(x, float).reshape(-1) for x in td]
            if any(len(x) for x in times_d):
                tmins.append(min([x.min() for x in times_d if x.size]))
                tmaxs.append(max([x.max() for x in times_d if x.size]))
        if has_times_v:
            tv = list(npz["spike_times_ventral"])  # type: ignore[arg-type]
            times_v = [np.asarray(x, float).reshape(-1) for x in tv]
            if any(len(x) for x in times_v):
                tmins.append(min([x.min() for x in times_v if x.size]))
                tmaxs.append(max([x.max() for x in times_v if x.size]))
        if not tmins or not tmaxs:
            raise ValueError("No spike event times present to determine global time range.")
        t_start = float(min(tmins))
        t_end = float(max(tmaxs))
        edges, cnt_d = (None, None)
        if times_d is not None:
            edges, cnt_d = _bin_event_times(times_d, t_start=t_start, t_end=t_end, bin_s=target_bin_s)
            fr_d = _counts_to_fr(cnt_d, target_bin_s)
        if times_v is not None:
            edges, cnt_v = _bin_event_times(times_v, t_start=t_start, t_end=t_end, bin_s=target_bin_s)
            fr_v = _counts_to_fr(cnt_v, target_bin_s)
        # Auto-downgrade if sparse
        if want_hz > 20:
            all_counts = []
            if times_d is not None:
                all_counts.append(cnt_d)
            if times_v is not None:
                all_counts.append(cnt_v)
            if len(all_counts):
                C = np.hstack([c for c in all_counts if c is not None])
                if _auto_downgrade_needed(C, thresh=0.2):
                    downgraded = True
                    target_bin_s = 1.0 / 20.0
                    if times_d is not None:
                        edges, cnt_d = _bin_event_times(times_d, t_start=t_start, t_end=t_end, bin_s=target_bin_s)
                        fr_d = _counts_to_fr(cnt_d, target_bin_s)
                    if times_v is not None:
                        edges, cnt_v = _bin_event_times(times_v, t_start=t_start, t_end=t_end, bin_s=target_bin_s)
                        fr_v = _counts_to_fr(cnt_v, target_bin_s)
    elif (has_counts_d or has_counts_v):
        if "dt_raw" not in npz:
            raise ValueError("dt_raw is required when providing spike count matrices.")
        dt_raw = float(npz["dt_raw"])  # seconds
        # If both present, ensure same T
        T_raw = None
        if has_counts_d:
            spikes_d = np.asarray(npz["spikes_dorsal"], float)  # (U_d, T_raw)
            U_d, Td = spikes_d.shape
            T_raw = Td if T_raw is None else T_raw
            if Td != T_raw:
                raise ValueError("spikes_dorsal and spikes_ventral must share T_raw")
        if has_counts_v:
            spikes_v = np.asarray(npz["spikes_ventral"], float)  # (U_v, T_raw)
            U_v, Tv = spikes_v.shape
            T_raw = Tv if T_raw is None else T_raw
            if Tv != T_raw:
                raise ValueError("spikes_dorsal and spikes_ventral must share T_raw")
        assert T_raw is not None
        t_start = 0.0
        t_end = float(T_raw) * dt_raw
        # Build edges at target
        edges = np.arange(t_start, t_end + 1e-12, target_bin_s)
        # Map raw bins to target bins by accumulation (downsample only). If target dt is smaller than raw, keep raw.
        if target_bin_s < dt_raw:
            warnings.warn(
                f"Requested bin_s={target_bin_s:.4f}s < dt_raw={dt_raw:.4f}s; keeping raw bins.",
                RuntimeWarning,
            )
            edges = np.arange(t_start, t_end + 1e-12, dt_raw)
            target_bin_s = dt_raw
        T_tgt = max(len(edges) - 1, 0)
        if has_counts_d:
            # Convert counts per raw bin to counts per target bin
            counts_d = np.zeros((T_tgt, spikes_d.shape[0]), float)
            # Map each raw index to target bin by left-edge
            centers_raw = (np.arange(T_raw) + 0.5) * dt_raw
            idx = np.searchsorted(edges, centers_raw, side="right") - 1
            idx = np.clip(idx, 0, T_tgt - 1)
            for i_raw in range(T_raw):
                counts_d[idx[i_raw]] += spikes_d[:, i_raw]
            fr_d = _counts_to_fr(counts_d, target_bin_s)
        if has_counts_v:
            counts_v = np.zeros((T_tgt, spikes_v.shape[0]), float)
            centers_raw = (np.arange(T_raw) + 0.5) * dt_raw
            idx = np.searchsorted(edges, centers_raw, side="right") - 1
            idx = np.clip(idx, 0, T_tgt - 1)
            for i_raw in range(T_raw):
                counts_v[idx[i_raw]] += spikes_v[:, i_raw]
            fr_v = _counts_to_fr(counts_v, target_bin_s)
        # Auto-downgrade if sparse (re-bin to 20 Hz)
        if want_hz > 20:
            C_parts = []
            if fr_d is not None:
                C_parts.append(fr_d * target_bin_s)
            if fr_v is not None:
                C_parts.append(fr_v * target_bin_s)
            if len(C_parts):
                C_all = np.hstack(C_parts)
                if _auto_downgrade_needed(C_all, thresh=0.2):
                    downgraded = True
                    target_bin_s = 1.0 / 20.0
                    edges = np.arange(t_start, t_end + 1e-12, target_bin_s)
                    T_tgt = max(len(edges) - 1, 0)
                    if has_counts_d:
                        counts_d = np.zeros((T_tgt, spikes_d.shape[0]), float)
                        centers_raw = (np.arange(T_raw) + 0.5) * dt_raw
                        idx = np.searchsorted(edges, centers_raw, side="right") - 1
                        idx = np.clip(idx, 0, T_tgt - 1)
                        for i_raw in range(T_raw):
                            counts_d[idx[i_raw]] += spikes_d[:, i_raw]
                        fr_d = _counts_to_fr(counts_d, target_bin_s)
                    if has_counts_v:
                        counts_v = np.zeros((T_tgt, spikes_v.shape[0]), float)
                        centers_raw = (np.arange(T_raw) + 0.5) * dt_raw
                        idx = np.searchsorted(edges, centers_raw, side="right") - 1
                        idx = np.clip(idx, 0, T_tgt - 1)
                        for i_raw in range(T_raw):
                            counts_v[idx[i_raw]] += spikes_v[:, i_raw]
                        fr_v = _counts_to_fr(counts_v, target_bin_s)
    else:
        raise ValueError("NPZ must contain spike_times_* or spikes_* arrays.")

    if edges is None:
        raise RuntimeError("Failed to construct time edges for binned spikes.")

    return SpikeBanks(
        t_edges=np.asarray(edges, float),
        fr_dorsal=fr_d,
        fr_ventral=fr_v,
        bin_s=float(target_bin_s),
        bin_hz=float(1.0 / float(target_bin_s)),
        downgraded=bool(downgraded),
    )


# ---------------------------------------------------------------------------
# Neural parquet normalization and resampling
# ---------------------------------------------------------------------------


def _normalize_neural_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize neural parquet to expected columns for FR reading.

    Expected inputs (any superset): [t|t_bin_s, unit_id|unit_index, bank|channel, fr|rate].
    Returns columns: t_bin_s, unit_id, bank, fr. Bank normalized to {dorsal, ventral} when possible.
    """
    out = df.copy()
    if "t_bin_s" not in out.columns:
        if "t" in out.columns:
            out = out.rename(columns={"t": "t_bin_s"})
        elif "time" in out.columns:
            out = out.rename(columns={"time": "t_bin_s"})
    if "unit_id" not in out.columns:
        if "unit_index" in out.columns:
            out = out.rename(columns={"unit_index": "unit_id"})
    if "fr" not in out.columns:
        for alt in ("rate", "zfr", "fr_pop_mean"):
            if alt in out.columns:
                out = out.rename(columns={alt: "fr"})
                break
    if "bank" not in out.columns:
        if "channel" in out.columns:
            try:
                banks_cfg = load_banks_config()
                out["bank"] = map_channels_to_bank(out["channel"].to_numpy(), banks_cfg)
            except (FileNotFoundError, KeyError, ValueError):
                out["bank"] = "unknown"
        else:
            out["bank"] = "unknown"
    out["bank"] = out["bank"].astype(str).str.lower()
    out.loc[~out["bank"].isin(["dorsal", "ventral"]), "bank"] = out.loc[
        ~out["bank"].isin(["dorsal", "ventral"]), "bank"
    ].replace({"d": "dorsal", "v": "ventral"}).astype(str)
    out["unit_id"] = out["unit_id"].astype(str)
    return out[["t_bin_s", "unit_id", "bank", "fr"]]


def _resample_fr_to_bins(df: pd.DataFrame, *, bin_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample per-unit firing rates to target bins.

    Returns:
        (t_centers, FR[T,U], banks_per_unit[U])
    """
    df = df.copy()
    df["t_bin_s"] = df["t_bin_s"].astype(float)
    unit_ids = np.array(sorted(df["unit_id"].astype(str).unique()), dtype=object)
    banks_u = []
    for uid in unit_ids:
        b = df.loc[df["unit_id"].astype(str) == str(uid), "bank"].dropna().astype(str)
        banks_u.append(b.iloc[0] if not b.empty else "unknown")
    banks_u = np.array(banks_u, dtype=object)
    t_src = np.sort(df["t_bin_s"].unique().astype(float))
    # Infer source dt
    dt = np.diff(np.sort(np.unique(t_src)))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    src_dt = float(np.nanmedian(dt)) if dt.size else 1.0
    tgt_dt = 1.0 / float(bin_hz)
    # Build matrix at source bins
    mat = np.full((t_src.size, unit_ids.size), np.nan, dtype=float)
    u_lut = {str(u): i for i, u in enumerate(unit_ids)}
    t_lut = {t: i for i, t in enumerate(t_src)}
    for _, r in df.iterrows():
        i = t_lut.get(float(r["t_bin_s"]))
        j = u_lut.get(str(r["unit_id"]))
        if i is not None and j is not None:
            mat[i, j] = float(r["fr"])  # Hz at source bins
    if abs(src_dt - tgt_dt) < 1e-9 or src_dt > tgt_dt:
        # same or coarser than target -> keep source
        edges = np.concatenate([
            [t_src[0] - src_dt / 2.0],
            0.5 * (t_src[:-1] + t_src[1:]),
            [t_src[-1] + src_dt / 2.0],
        ])
        return 0.5 * (edges[:-1] + edges[1:]), mat, banks_u
    # Downsample by counts accumulation
    counts_src = np.nan_to_num(mat, nan=0.0) * src_dt
    t_start = float(t_src.min())
    t_end = float(t_src.max()) + src_dt
    edges_tgt = np.arange(t_start, t_end + 1e-12, tgt_dt, dtype=float)
    Tt = max(len(edges_tgt) - 1, 0)
    counts_tgt = np.zeros((Tt, unit_ids.size), dtype=float)
    idx_src = np.searchsorted(edges_tgt, t_src, side="right") - 1
    idx_src = np.clip(idx_src, 0, Tt - 1)
    for i_src, i_tgt in enumerate(idx_src):
        counts_tgt[i_tgt] += counts_src[i_src]
    fr_tgt = counts_tgt / tgt_dt
    t_centers = 0.5 * (edges_tgt[:-1] + edges_tgt[1:])
    return t_centers, fr_tgt, banks_u


def _edges_from_centers(t_centers: np.ndarray) -> np.ndarray:
    if t_centers.size < 2:
        dt = 1.0
    else:
        dt = float(np.median(np.diff(t_centers)))
    return np.concatenate([[t_centers[0] - dt / 2], 0.5 * (t_centers[:-1] + t_centers[1:]), [t_centers[-1] + dt / 2]])


def load_spikes_from_neural_parquet(parquet_path: str, *, bin_hz: int) -> SpikeBanks:
    """Load aligned neural parquet and assemble per-bank FR matrices at target bin rate."""
    df = pd.read_parquet(parquet_path)
    df = _normalize_neural_df(df)
    t_cent, FR, banks_u = _resample_fr_to_bins(df, bin_hz=float(bin_hz))
    edges = _edges_from_centers(t_cent)
    # Split by bank
    mask_d = (banks_u == "dorsal")
    mask_v = (banks_u == "ventral")
    fr_d = FR[:, mask_d] if mask_d.any() else None
    fr_v = FR[:, mask_v] if mask_v.any() else None
    return SpikeBanks(
        t_edges=np.asarray(edges, float),
        fr_dorsal=fr_d,
        fr_ventral=fr_v,
        bin_s=float(np.median(np.diff(t_cent))) if t_cent.size > 1 else (1.0 / float(bin_hz)),
        bin_hz=float(1.0 / (float(np.median(np.diff(t_cent))) if t_cent.size > 1 else (1.0 / float(bin_hz)))),
        downgraded=False,
    )
