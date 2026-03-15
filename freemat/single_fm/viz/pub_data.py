"""Publication data loading and shared constants for Figs 2, 3, 4, and 5.

Extracted from ``scripts/single_fm/generate_publication_figures.py`` (W5.2)
to make data loading and constants reusable across library modules without
importing the top-level script.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path defaults ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_PROD_DIR = _PROJECT_ROOT / "results" / "production"
DEFAULT_FIG4_DIR = DEFAULT_PROD_DIR / "fig4_all4"
DEFAULT_FIG5_DIR = _PROJECT_ROOT / "results" / "fig5_cebra"
DEFAULT_PUB_DIR = _PROJECT_ROOT / "results" / "publication_figures"

# ── Constants ────────────────────────────────────────────────────────────
FRAME_COLORS: Dict[str, str] = {"allo": "#457B9D", "body": "#E63946", "head": "#F4A261"}
FRAME_LABELS: Dict[str, str] = {"allo": "Allo", "body": "Body", "head": "Head"}
FRAMES: List[str] = ["allo", "body", "head"]

GROUP_COLORS: Dict[str, str] = {
    "vis": "#457B9D", "body_kin": "#E63946",
    "head_kin": "#F4A261", "pos": "#7B3294",
}
GROUP_LABELS: Dict[str, str] = {
    "vis": "VIS", "body_kin": "BODY_KIN",
    "head_kin": "HEAD_KIN", "pos": "3D POS",
}
GROUPS: List[str] = ["vis", "body_kin", "head_kin", "pos"]

SUBJECT_MAP: Dict[str, str] = {
    "M2_0526": "M2", "M2_0605": "M2",
    "M1_0525": "M1", "M1_0609": "M1", "M1_0613": "M1",
}
SUBJECT_COLORS: Dict[str, str] = {"M2": "#2D7D9A", "M1": "#D4A373"}

# Fig 4 session output directory mapping
FIG4_SESSION_DIRS: Dict[str, str] = {
    "M2_0526": "single_0526_M2",
    "M2_0605": "single_0605_M2",
    "M1_0525": "single_0525_M1",
    "M1_0609": "single_0609_M1",
    "M1_0613": "single_0613_M1",
}


# ── Utility functions ────────────────────────────────────────────────────
def setup_nature_style() -> None:
    """Apply Nature journal style globally (SVG-editable, transparent-ready).

    Includes ``svg.fonttype: 'none'`` so text remains editable in
    Adobe Illustrator, and ``font.family: 'sans-serif'`` for consistency.
    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "svg.fonttype": "none",
    })


def save_fig(
    fig: plt.Figure,
    name: str,
    directory: Path,
    transparent: bool = True,
) -> None:
    """Save figure as PNG + SVG, then close.

    Args:
        fig: Matplotlib figure to save.
        name: Base filename (without extension).
        directory: Output directory (created if missing).
        transparent: If True, save with transparent background.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        directory / f"{name}.png",
        dpi=300, bbox_inches="tight", transparent=transparent,
    )
    fig.savefig(
        directory / f"{name}.svg",
        bbox_inches="tight", transparent=transparent,
    )
    plt.close(fig)
    print(f"  -> {directory / name}.png")


def winsorize(arr: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Winsorize array at *pct* and *100-pct* percentiles.

    Args:
        arr: Input array.
        pct: Lower percentile for clipping (upper = 100 - pct).

    Returns:
        Clipped copy of *arr*.
    """
    if arr.size == 0:
        return arr
    lo, hi = np.nanpercentile(arr, [pct, 100 - pct])
    return np.clip(arr, lo, hi)


# ── Data loading ─────────────────────────────────────────────────────────
def load_and_merge_unit_kind(
    df: pd.DataFrame,
    prod_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge unit_kind (SUA/MUA) from lookup table.

    Args:
        df: Summary DataFrame with ``session_id`` and ``unit_id`` columns.
        prod_dir: Production results directory (default: ``DEFAULT_PROD_DIR``).

    Returns:
        DataFrame with ``unit_kind`` column merged in (if lookup exists).
    """
    root = prod_dir or DEFAULT_PROD_DIR
    uk_path = root / "unit_kind_lookup.parquet"
    if uk_path.exists():
        uk = pd.read_parquet(uk_path)
        df = df.merge(uk, on=["session_id", "unit_id"], how="left")
    return df


def load_fig2(
    scope: str = "M2",
    bin_s: str = "05",
    prod_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load Fig 2 summary parquet.

    Args:
        scope: Run scope (e.g., ``'M2'``, ``'all4'``).
        bin_s: Bin size string (e.g., ``'05'`` for 0.5 s).
        prod_dir: Production results directory.

    Returns:
        Summary DataFrame with ``subject`` column added.
    """
    root = prod_dir or DEFAULT_PROD_DIR
    path = root / f"{scope}_bin{bin_s}" / "summary" / "fig2b_summary.parquet"
    df = pd.read_parquet(path)
    df["subject"] = df["session_id"].map(SUBJECT_MAP)
    return load_and_merge_unit_kind(df, prod_dir=root)


def load_fig2d_decoders(
    scope: str = "all4",
    bin_s: str = "05",
    prod_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load Fig 2D spatial decoder summary parquet.

    Args:
        scope: Run scope (e.g. ``"all4"``).
        bin_s: Bin-size string (e.g. ``"05"`` for 0.5 s).
        prod_dir: Production results directory.

    Returns:
        Decoder summary DataFrame, or empty DataFrame if file is missing.
    """
    root = prod_dir or DEFAULT_PROD_DIR
    path = root / f"{scope}_bin{bin_s}" / "summary" / "fig2d_decoder_summary.parquet"
    if not path.exists():
        # Fallback: decoder results may live in a separate fig2d_aligned directory
        fallback = root.parent / "fig2d_aligned" / "summary" / "fig2d_decoder_summary.parquet"
        if fallback.exists():
            path = fallback
        else:
            return pd.DataFrame()
    return pd.read_parquet(path)


def load_fig3(
    bin_s: str = "05",
    prod_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load Fig 3 summary parquet.

    Args:
        bin_s: Bin size string (e.g., ``'05'`` for 0.5 s).
        prod_dir: Production results directory.

    Returns:
        Summary DataFrame with ``subject`` column added.
    """
    root = prod_dir or DEFAULT_PROD_DIR
    path = root / f"fig3_bin{bin_s}" / "summary" / "fig3_summary.parquet"
    df = pd.read_parquet(path)
    df["subject"] = df["session_id"].map(SUBJECT_MAP)
    return load_and_merge_unit_kind(df, prod_dir=root)


def load_residuals(
    scope: str = "M2",
    bin_s: str = "05",
    prod_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load per-session residuals from a production run config.

    Args:
        scope: Run scope (e.g., ``'M2'``, ``'all4'``).
        bin_s: Bin size string.
        prod_dir: Production results directory.

    Returns:
        Dict mapping session_id -> residuals DataFrame.
    """
    import glob

    root = prod_dir or DEFAULT_PROD_DIR
    pattern = str(root / f"{scope}_bin{bin_s}" / "*" / "encoding" / "fig2b_residuals.parquet")
    files = glob.glob(pattern)
    if not files:
        print(f"  [WARN] No residuals found for {scope}_bin{bin_s}")
        return {}

    _DIR_TO_SESSION: Dict[str, str] = {
        "single_0526_M2": "M2_0526",
        "single_0605_M2": "M2_0605",
        "single_0525_M1": "M1_0525",
        "single_0609_M1": "M1_0609",
        "single_0613_M1": "M1_0613",
    }
    result: Dict[str, pd.DataFrame] = {}
    for f in files:
        session_dir = Path(f).parent.parent.name
        session_id = _DIR_TO_SESSION.get(session_dir, session_dir)
        result[session_id] = pd.read_parquet(f)
    return result


def one_per_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one SUA per channel: the one with highest full_r2.

    Channel is inferred from ``unit_id // 10`` (Plexon convention).

    Args:
        df: Summary DataFrame with ``unit_kind``, ``unit_id``, ``session_id``.

    Returns:
        Filtered DataFrame with one SUA per channel.
    """
    if "unit_kind" not in df.columns:
        return df
    sua = df[df["unit_kind"] == "SUA"].copy()
    sua["channel"] = sua["unit_id"] // 10
    r2_col = "full_r2" if "full_r2" in sua.columns else "r2_allo"
    idx = sua.groupby(["session_id", "channel"])[r2_col].idxmax()
    return sua.loc[idx].drop(columns=["channel"])


def load_fig3_rf_index(
    rf_index_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load Fig 3 RF-index summary parquet.

    Args:
        rf_index_dir: Directory containing RF-index runner output
            (expects ``summary/fig3_rf_index_all.parquet``).

    Returns:
        Summary DataFrame with rf_index column, or empty DataFrame if missing.
    """
    root = rf_index_dir or (_PROJECT_ROOT / "results" / "fig3_rf")
    summary_path = root / "summary" / "fig3_rf_index_all.parquet"
    if not summary_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(summary_path)
    if "session_id" in df.columns:
        df["subject"] = df["session_id"].map(SUBJECT_MAP)
    return df


def load_fig4_similarity(
    similarity_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load Fig 4 similarity vs confusion results.

    Args:
        similarity_dir: Directory containing similarity runner output
            (expects ``summary/fig4_similarity_summary.json`` and
            ``summary/fig4_similarity_per_session.json``).

    Returns:
        Dict with keys ``summary`` and ``per_session``.
    """
    root = similarity_dir or (_PROJECT_ROOT / "results" / "fig4_similarity")
    result: Dict[str, Any] = {"summary": {}, "per_session": []}

    summary_path = root / "summary" / "fig4_similarity_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            result["summary"] = json.load(f)

    per_session_path = root / "summary" / "fig4_similarity_per_session.json"
    if per_session_path.exists():
        with open(per_session_path) as f:
            result["per_session"] = json.load(f)

    return result


def load_fig4_similarity_matrices(
    similarity_dir: Optional[Path] = None,
    session_key: str = "M2_0526",
) -> Optional[Dict[str, Any]]:
    """Load per-session aligned similarity/confusion matrices for plotting.

    Args:
        similarity_dir: Directory containing similarity runner output.
        session_key: Session to load matrices for.

    Returns:
        Dict with sim_sub, conf_sym, labels arrays, or None if not found.
    """
    root = similarity_dir or (_PROJECT_ROOT / "results" / "fig4_similarity")
    _session_output_map = {
        "M1_0525": "single_0525_M1",
        "M2_0526": "single_0526_M2",
        "M2_0605": "single_0605_M2",
        "M1_0609": "single_0609_M1",
        "M1_0613": "single_0613_M1",
    }
    # Normalize alternative session key names to canonical keys used by producer
    _session_key_canonical = {
        "M1_0525": "M1_0525",
        "M1_0609": "M1_0609",
        "M1_0613": "M1_0613",
        "M2_0605": "M2_0605",
    }
    canonical_key = _session_key_canonical.get(session_key, session_key)
    session_subdir = _session_output_map.get(canonical_key, canonical_key)
    npz_path = root / session_subdir / f"fig4_similarity_matrices_{canonical_key}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    return {
        "sim_sub": data["sim_sub"],
        "conf_sym": data["conf_sym"],
        "labels": data["labels"].tolist(),
    }


def load_fig4(
    fig4_dir: Optional[Path] = None,
    sessions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load Fig 4 results from session output directories.

    Args:
        fig4_dir: Root directory containing per-session subdirs.
        sessions: Session keys to load (default: all found).

    Returns:
        Dict with keys: tuning, decoding, nulls, lag_sweep, next_syllable,
        cluster_orders, confusion, class_labels, transition, peth.
    """
    root = fig4_dir or DEFAULT_FIG4_DIR
    if sessions is None:
        sessions = [k for k, v in FIG4_SESSION_DIRS.items()
                    if (root / v).exists()]

    all_tuning: List[pd.DataFrame] = []
    all_decoding: List[pd.DataFrame] = []
    all_lag: List[pd.DataFrame] = []
    all_next: List[pd.DataFrame] = []
    all_nulls: Dict[str, Dict[str, np.ndarray]] = {}
    all_cluster_orders: Dict[str, List[int]] = {}
    all_confusion: Dict[str, Dict[str, np.ndarray]] = {}
    all_class_labels: Dict[str, Dict[str, np.ndarray]] = {}
    all_transition: Dict[str, Dict[str, np.ndarray]] = {}
    all_peth: Dict[str, Dict[str, Any]] = {}

    for sess_key in sessions:
        sess_dir = root / FIG4_SESSION_DIRS[sess_key]
        if not sess_dir.exists():
            print(f"  [WARN] Fig 4 session dir not found: {sess_dir}")
            continue

        tuning_path = sess_dir / "fig4_tuning.parquet"
        if tuning_path.exists():
            all_tuning.append(pd.read_parquet(tuning_path))

        dec_path = sess_dir / "fig4_decoding_summary.parquet"
        if dec_path.exists():
            all_decoding.append(pd.read_parquet(dec_path))

        lag_path = sess_dir / "fig4_lag_sweep.parquet"
        if lag_path.exists():
            all_lag.append(pd.read_parquet(lag_path))

        next_path = sess_dir / "fig4_next_syllable.parquet"
        if next_path.exists():
            all_next.append(pd.read_parquet(next_path))

        null_path = sess_dir / "fig4_null_distributions.npz"
        if null_path.exists():
            npz = np.load(null_path)
            all_nulls[sess_key] = {k: npz[k] for k in npz.files}

        order_path = sess_dir / "fig4_cluster_order.json"
        if order_path.exists():
            with open(order_path) as fh:
                all_cluster_orders[sess_key] = json.load(fh)["motif_order"]

        cm_path = sess_dir / "fig4_confusion_matrices.npz"
        if cm_path.exists():
            cm_npz = np.load(cm_path)
            sess_cm: Dict[str, np.ndarray] = {}
            sess_lbl: Dict[str, np.ndarray] = {}
            for bank in ["ventral", "dorsal", "pooled"]:
                cm_key = f"{bank}_confusion"
                lbl_key = f"{bank}_labels"
                if cm_key in cm_npz.files:
                    sess_cm[bank] = cm_npz[cm_key]
                    sess_lbl[bank] = cm_npz[lbl_key]
            all_confusion[sess_key] = sess_cm
            all_class_labels[sess_key] = sess_lbl

        trans_path = sess_dir / "fig4_transition_matrix.npz"
        if trans_path.exists():
            t_npz = np.load(trans_path)
            all_transition[sess_key] = {
                "matrix": t_npz["matrix"],
                "labels": t_npz["labels"],
            }

        peth_path = sess_dir / "fig4_onset_peth.npz"
        if peth_path.exists():
            p_npz = np.load(peth_path, allow_pickle=True)
            all_peth[sess_key] = {
                "peth": p_npz["peth"],
                "tau_bins": p_npz["tau_bins"],
                "onset_count": int(p_npz["onset_count"]),
                "unit_banks": p_npz["unit_banks"],
            }

        peth_fine_path = sess_dir / "fig4_onset_peth_fine.npz"
        if peth_fine_path.exists():
            pf_npz = np.load(peth_fine_path, allow_pickle=True)
            if sess_key not in all_peth:
                all_peth[sess_key] = {}
            all_peth[sess_key]["peth_fine"] = pf_npz["peth"]
            all_peth[sess_key]["tau_bins_fine"] = pf_npz["tau_bins"]
            all_peth[sess_key]["bin_s_fine"] = float(pf_npz["bin_s"])
            all_peth[sess_key]["unit_banks_fine"] = pf_npz["unit_banks"]

    return {
        "tuning": pd.concat(all_tuning, ignore_index=True) if all_tuning else pd.DataFrame(),
        "decoding": pd.concat(all_decoding, ignore_index=True) if all_decoding else pd.DataFrame(),
        "lag_sweep": pd.concat(all_lag, ignore_index=True) if all_lag else pd.DataFrame(),
        "next_syllable": pd.concat(all_next, ignore_index=True) if all_next else pd.DataFrame(),
        "nulls": all_nulls,
        "cluster_orders": all_cluster_orders,
        "confusion": all_confusion,
        "class_labels": all_class_labels,
        "transition": all_transition,
        "peth": all_peth,
    }


# ── Fig 5 Constants ──────────────────────────────────────────────────────
METHOD_COLORS: Dict[str, str] = {
    "CEBRA": "#2D7D9A",
    "PCA-8D": "#D4A373",
    "Raw": "#888888",
}
HEIGHT_COLORS_FIG5: Dict[int, str] = {
    0: "#8B6914",  # floor — warm brown/gold
    1: "#2CA02C",  # bench — green
    2: "#7B2D8E",  # ceiling — purple
}
HEIGHT_NAMES: Dict[int, str] = {0: "Floor", 1: "Bench", 2: "Ceiling"}

MOTIF_NAMES_FIG5: Dict[int, str] = {
    1: "Rest", 2: "Stand", 3: "Pace", 4: "Inspect", 5: "Pivot",
    6: "Travel", 7: "Scan", 8: "Walk", 9: "Perch", 10: "Reach",
    11: "Sit", 12: "Walk(rare)",
}

ALL_SESSIONS: List[str] = [
    "M2_0526", "M2_0605", "M1_0525", "M1_0609", "M1_0613",
]


# ── Fig 5 Data Loading ──────────────────────────────────────────────────
def load_fig5_results(
    fig5_dir: Optional[Path] = None,
    sessions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load all Fig 5 results from per-session results.json files.

    Args:
        fig5_dir: Root directory containing per-session subdirs.
        sessions: Session keys to load (default: all found).

    Returns:
        Dict mapping session_key -> results dict.
    """
    root = fig5_dir or DEFAULT_FIG5_DIR
    if sessions is None:
        sessions = [s for s in ALL_SESSIONS if (root / s).exists()]

    results: Dict[str, Any] = {}
    for sess in sessions:
        rpath = root / sess / "results.json"
        if rpath.exists():
            with open(rpath) as f:
                results[sess] = json.load(f)
    return results


def load_fig5_embeddings(
    session_key: str,
    fig5_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Load embeddings.npz for a single session.

    Args:
        session_key: Session key (e.g. ``'M2_0526'``).
        fig5_dir: Root directory containing per-session subdirs.

    Returns:
        Dict with embedding arrays: emb_spatial_viz_3d, emb_behavior_viz_3d,
        emb_behavior_viz_8d, emb_time_3d, emb_pca_8d, y_motif, height_tier, t.
    """
    root = fig5_dir or DEFAULT_FIG5_DIR
    npz_path = root / session_key / "embeddings.npz"
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}
