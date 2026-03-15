"""Flexible save utilities for Single-FM outputs.

This module provides small, testable helpers to persist aligned outputs
without embedding figure logic. Functions fail loudly on missing inputs
and create parent directories as needed.

Example:
    >>> import pandas as pd
    >>> from freemat.single_fm.io.save import (
    ...     save_aligned_parquet, save_qc_json, copy_session_yaml,
    ...     update_manifest_row,
    ... )
    >>> df = pd.DataFrame({"t": [0.0, 1.0], "fr": [5.0, 7.0]})
    >>> save_aligned_parquet(df, "results/single_fm/aligned/session_001_bin1p0.parquet",
    ...                      metadata={"session_id": "session_001", "bin_s": 1.0})
    >>> save_qc_json({"n_units": 128}, "results/single_fm/session_001/qc_session_001.json")
    >>> copy_session_yaml("configs/single_fm/session_001.yaml", "results/single_fm/session_001")
    'results/single_fm/session_001/session.yaml'
    >>> update_manifest_row("results/single_fm/_manifest.csv", "session_001",
    ...                    {"aligned_1p0": "results/single_fm/aligned/session_001_bin1p0.parquet"})

Notes:
    - Metadata for parquet is written as a sidecar JSON (`<path>.meta.json`).
      Embedding user metadata directly into parquet file metadata is engine-dependent;
      sidecar keeps behavior simple and explicit.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np


def _ensure_parent(path: Path) -> None:
    """Ensure parent directory exists for a path.

    Args:
        path: Target file path

    Raises:
        OSError: If directory creation fails
    """

    path.parent.mkdir(parents=True, exist_ok=True)


def save_aligned_parquet(
    df: pd.DataFrame, path: str | os.PathLike[str], *, metadata: Mapping[str, Any] | None = None
) -> None:
    """Save aligned dataframe to parquet, with optional sidecar metadata.

    Args:
        df: DataFrame to persist
        path: Output parquet path
        metadata: Optional dictionary to persist to `<path>.meta.json`

    Raises:
        ValueError: If dataframe is empty
        OSError: If writing the file fails
    """

    if df is None or df.empty:
        raise ValueError("save_aligned_parquet: dataframe is empty or None")
    out = Path(path)
    _ensure_parent(out)
    # Write without index for stable, tidy schema across notebooks
    df.to_parquet(out, index=False)
    if metadata is not None:
        meta_path = out.with_suffix(out.suffix + ".meta.json")
        meta_obj = {
            **dict(metadata),
            "parquet_path": str(out).replace("\\", "/"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta_obj, indent=2))


def save_qc_json(qc: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Save QC dictionary to JSON.

    Args:
        qc: QC metrics and notes
        path: Output JSON path

    Raises:
        ValueError: If qc is empty
        OSError: If writing the file fails
    """

    if not qc:
        raise ValueError("save_qc_json: qc payload is empty")
    out = Path(path)
    _ensure_parent(out)
    out.write_text(json.dumps(qc, indent=2))


def copy_session_yaml(src_yaml: str | os.PathLike[str], dst_dir: str | os.PathLike[str]) -> str:
    """Copy per-session YAML into results directory as `session.yaml`.

    Args:
        src_yaml: Source YAML path
        dst_dir: Destination directory where the YAML will be copied

    Returns:
        Destination file path as string

    Raises:
        FileNotFoundError: If source YAML does not exist
        OSError: If copying fails
    """

    src = Path(src_yaml)
    if not src.exists():
        raise FileNotFoundError(f"copy_session_yaml: missing source YAML: {src}")
    dst_path = Path(dst_dir) / "session.yaml"
    _ensure_parent(dst_path)
    shutil.copy2(src, dst_path)
    return str(dst_path)


def update_manifest_row(
    manifest_csv: str | os.PathLike[str], session_id: str, updates: Mapping[str, Any]
) -> None:
    """Update or add a row for `session_id` in the manifest CSV.

    Args:
        manifest_csv: Path to `_manifest.csv`
        session_id: Session identifier key
        updates: Columns to update for this session

    Raises:
        ValueError: If `session_id` or `updates` are invalid
        OSError: If reading or writing the file fails
    """

    if not session_id:
        raise ValueError("update_manifest_row: session_id is empty")
    if not updates:
        raise ValueError("update_manifest_row: updates mapping is empty")

    path = Path(manifest_csv)
    _ensure_parent(path)
    if path.exists():
        df = pd.read_csv(path)
        if "session_id" not in df.columns:
            raise ValueError("manifest is missing required 'session_id' column")
        mask = df["session_id"] == session_id
        if mask.any():
            idx = df.index[mask][0]
            for k, v in updates.items():
                # Ensure target column can hold strings/paths
                if k in df.columns and df[k].dtype != object:
                    df[k] = df[k].astype("object")
                df.loc[idx, k] = v
        else:
            row = {"session_id": session_id, **updates}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([{ "session_id": session_id, **updates }])
    # Normalize Windows paths to forward slashes for portability
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.replace("\\", "/") if isinstance(x, str) else x)
    df.to_csv(path, index=False)


def save_pose_npz_upright(
    data_or_src: Mapping[str, Any] | str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Save upright pose NPZ or copy an existing NPZ, with optional sidecar metadata.

    Args:
        data_or_src: Either a mapping of arrays (e.g., {"X": ndarray, ...}) to save,
            or a path to an existing `.npz` to copy.
        path: Output NPZ path
        meta: Optional metadata to write as `<path>.json`

    Raises:
        FileNotFoundError: If copying from a missing NPZ
        TypeError: If input mapping is not dict-like when not copying
        OSError: If writing fails
    """

    out = Path(path)
    _ensure_parent(out)
    if isinstance(data_or_src, (str, os.PathLike)):
        src = Path(data_or_src)
        if not src.exists():
            raise FileNotFoundError(f"save_pose_npz_upright: missing source NPZ: {src}")
        shutil.copy2(src, out)
    else:
        if not hasattr(data_or_src, "items"):
            raise TypeError("save_pose_npz_upright: data_or_src must be a mapping or path to .npz")
        # Use compressed NPZ; keys must be strings
        np.savez_compressed(out, **dict(data_or_src))
    if meta is not None:
        meta_path = out.with_suffix(".json")
        meta_obj = {
            **dict(meta),
            "npz_path": str(out).replace("\\", "/"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta_obj, indent=2))


__all__ = [
    "save_aligned_parquet",
    "save_qc_json",
    "copy_session_yaml",
    "update_manifest_row",
    "save_pose_npz_upright",
    "save_png_svg",
]


def save_png_svg(
    fig: "Any",  # matplotlib.figure.Figure, but avoid import here
    path_png: str | os.PathLike[str],
    *,
    dpi: int = 150,
    transparent: bool = False,
    bbox_inches: str = "tight",
) -> tuple[str, str]:
    """Save a Matplotlib figure as PNG and SVG.

    Args:
        fig: Matplotlib figure object
        path_png: Output PNG path
        dpi: Raster DPI for PNG
        transparent: Whether to save with transparent background
        bbox_inches: Matplotlib bbox mode

    Returns:
        Tuple of (png_path, svg_path)
    """

    out_png = Path(path_png)
    _ensure_parent(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    if str(out_png).lower().endswith(".png"):
        out_svg = out_png.with_suffix(".svg")
    else:
        out_svg = Path(str(out_png) + ".svg")
    fig.savefig(out_svg, bbox_inches=bbox_inches, transparent=transparent)
    return str(out_png), str(out_svg)
