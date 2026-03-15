"""Neural data I/O and resampling for pose decoding (Single-FM)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from freemat_data.banks import load_banks_config, map_channels_to_bank


def parse_range_spec(spec: str) -> np.ndarray:
    """Parse range spec like ``'-300:300:20'`` to array.

    Args:
        spec: Colon-separated ``'start:stop:step'`` string.

    Returns:
        1-D array of values.

    Raises:
        ValueError: If *spec* does not have exactly 3 parts.
    """
    s = str(spec).strip()
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid range spec '{spec}' (expected 'start:stop:step')")
    a, b, c = [float(x) for x in parts]
    return np.arange(a, b + 1e-9, c, dtype=float)


def parse_alpha_grid(s: Optional[str]) -> np.ndarray:
    """Parse alpha grid specification.

    Args:
        s: Comma-separated values, ``'a:b:k'`` logspace, or single value.
           ``None`` returns default logspace(-3, 3, 13).
    """
    if not s:
        return np.logspace(-3, 3, 13)
    if "," in s:
        return np.array([float(x) for x in s.split(",")], dtype=float)
    if ":" in s:
        a, b, k = [float(x) for x in s.split(":")]
        return np.logspace(a, b, int(k))
    return np.array([float(s)], dtype=float)


def normalize_neural_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise neural parquet to standard columns.

    Expected inputs (any superset): ``[t|t_bin_s, unit_id|unit_index,
    bank|channel, fr|rate]``.  Returns columns: ``t_bin_s, unit_id, bank, fr``.
    Bank normalised to ``{dorsal, ventral}`` when possible.
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
                out["bank"] = map_channels_to_bank(
                    out["channel"].to_numpy(), banks_cfg,
                )
            except (FileNotFoundError, KeyError, ValueError):
                out["bank"] = "unknown"
        else:
            out["bank"] = "unknown"
    # Normalize bank strings
    out["bank"] = out["bank"].astype(str).str.lower()
    out.loc[~out["bank"].isin(["dorsal", "ventral"]), "bank"] = out.loc[
        ~out["bank"].isin(["dorsal", "ventral"]), "bank"
    ].replace({"d": "dorsal", "v": "ventral"}).astype(str)
    out["unit_id"] = out["unit_id"].astype(str)
    return out[["t_bin_s", "unit_id", "bank", "fr"]]


def infer_bin_s(t: np.ndarray) -> float:
    """Infer bin width from sorted unique timestamps."""
    dt = np.diff(np.sort(np.unique(t)))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.nanmedian(dt)) if dt.size else 1.0


def resample_fr_to_bins(
    df: pd.DataFrame, *, bin_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample per-unit firing rates to target bins.

    Args:
        df: Tidy DataFrame with ``[t_bin_s, unit_id, bank, fr]``.
        bin_hz: Target binning rate in Hz.

    Returns:
        ``(t_centers, FR[T,U], unit_ids[U], banks_per_unit[U])``
    """
    df = df.copy()
    df["t_bin_s"] = df["t_bin_s"].astype(float)
    unit_ids = np.array(
        sorted(df["unit_id"].astype(str).unique()), dtype=object,
    )
    banks_u = []
    for uid in unit_ids:
        b = (
            df.loc[df["unit_id"].astype(str) == str(uid), "bank"]
            .dropna()
            .astype(str)
        )
        banks_u.append(b.iloc[0] if not b.empty else "unknown")
    banks_u = np.array(banks_u, dtype=object)

    # Build source matrix
    t_src = np.sort(df["t_bin_s"].unique().astype(float))
    src_dt = infer_bin_s(t_src)
    tgt_dt = 1.0 / float(bin_hz)

    mat = np.full((t_src.size, unit_ids.size), np.nan, dtype=float)
    u_lut = {str(u): i for i, u in enumerate(unit_ids)}
    t_lut = {t: i for i, t in enumerate(t_src)}
    for _, r in df.iterrows():
        i = t_lut.get(float(r["t_bin_s"]))
        j = u_lut.get(str(r["unit_id"]))
        if i is not None and j is not None:
            mat[i, j] = float(r["fr"])

    # If same binning, return as-is
    if abs(src_dt - tgt_dt) < 1e-9:
        return t_src, mat, unit_ids, banks_u

    # If source coarser than target, keep source bins
    if src_dt > tgt_dt:
        print(
            f"[decode_pose] Warning: source dt={src_dt:.3f}s is coarser "
            f"than target dt={tgt_dt:.3f}s; keeping source bins.",
        )
        return t_src, mat, unit_ids, banks_u

    # Downsample via count aggregation
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
    return t_centers, fr_tgt, unit_ids, banks_u
