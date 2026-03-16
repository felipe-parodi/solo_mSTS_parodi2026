"""Cross-session feature distribution comparison for Figure 2B validation."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..io.save import save_png_svg
from .fig2b_timeseries import VIS_COLS, KIN_COLS, SESSIONS, load_features_direct


EXPECTED_RANGES = {
    "speed": (0, 2.0), "v_parallel": (-1.5, 1.5), "v_perp": (-1.0, 1.0),
    "v_z": (-0.5, 0.5), "yaw_rate": (-180, 180), "delta_yaw": (-90, 90),
    "acceleration": (0, 5), "d_front": (0.1, 6), "gaze_x": (-1, 1),
    "gaze_y": (-1, 1), "gaze_z": (-1, 1), "v_forward": (-1.5, 1.5), "vis_motion": (-15, 15),
}


def _setup_style():
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none"})


def compute_cross_session_stats(session_data):
    rows = []
    all_cols = VIS_COLS + KIN_COLS
    for sid, (vis_df, kin_df) in session_data.items():
        for col in all_cols:
            df = vis_df if col in VIS_COLS else kin_df
            if col not in df.columns:
                continue
            vals = df[col].values
            valid = vals[np.isfinite(vals)]
            pct_nan = 100 * (1 - len(valid) / len(vals)) if len(vals) > 0 else 100
            exp_lo, exp_hi = EXPECTED_RANGES.get(col, (-np.inf, np.inf))
            n_outlier = np.sum((valid < exp_lo) | (valid > exp_hi))
            pct_outlier = 100 * n_outlier / len(valid) if len(valid) > 0 else 0
            rows.append({"session_id": sid, "feature": col,
                "mean": np.mean(valid) if len(valid) > 0 else np.nan,
                "std": np.std(valid) if len(valid) > 0 else np.nan,
                "min": np.min(valid) if len(valid) > 0 else np.nan,
                "max": np.max(valid) if len(valid) > 0 else np.nan,
                "pct_nan": pct_nan, "pct_outlier": pct_outlier})
    return pd.DataFrame(rows)


def plot_feature_distributions(session_data, output_dir, feature_type="kin"):
    _setup_style()
    cols = VIS_COLS if feature_type == "vis" else KIN_COLS
    sessions = list(session_data.keys())
    n_sessions, n_features = len(sessions), len(cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(8, 2*n_features), sharex=False)
    if n_features == 1:
        axes = [axes]
    session_colors = plt.cm.Set2(np.linspace(0, 1, n_sessions))
    
    for i, col in enumerate(cols):
        ax = axes[i]
        data_by_session, positions = [], []
        for j, sid in enumerate(sessions):
            vis_df, kin_df = session_data[sid]
            df = vis_df if col in VIS_COLS else kin_df
            if col in df.columns:
                vals = df[col].values
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    data_by_session.append(valid)
                    positions.append(j)
        if data_by_session:
            parts = ax.violinplot(data_by_session, positions=positions, showmeans=True, showmedians=True)
            for pc, c in zip(parts["bodies"], session_colors[:len(data_by_session)]):
                pc.set_facecolor(c)
                pc.set_alpha(0.7)
        exp_lo, exp_hi = EXPECTED_RANGES.get(col, (None, None))
        if exp_lo is not None:
            ax.axhline(exp_lo, color="red", linestyle="--", alpha=0.5, lw=1)
            ax.axhline(exp_hi, color="red", linestyle="--", alpha=0.5, lw=1)
        ax.set_ylabel(col)
        ax.set_xticks(range(n_sessions))
        ax.set_xticklabels(sessions, rotation=45, ha="right", fontsize=8)
        if i == 0:
            ax.set_title(f"{feature_type.upper()} Feature Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(output_dir) / f"{feature_type}_distributions.png"
    save_png_svg(fig, str(fname), dpi=300)
    plt.close(fig)
    return fname


def flag_anomalous_sessions(stats_df, cv_threshold=0.3, outlier_threshold=5.0):
    flags = []
    for feature in stats_df["feature"].unique():
        feat_df = stats_df[stats_df["feature"] == feature]
        means = feat_df["mean"].values
        grand_mean, grand_std = np.nanmean(means), np.nanstd(means)
        for _, row in feat_df.iterrows():
            flag_reasons = []
            if grand_std > 0 and np.isfinite(row["mean"]):
                z = abs(row["mean"] - grand_mean) / grand_std
                if z > 2:
                    flag_reasons.append(f"mean {z:.1f} SD from grand mean")
            if row["pct_outlier"] > outlier_threshold:
                flag_reasons.append(f"{row['pct_outlier']:.1f}% outliers")
            if row["pct_nan"] > 20:
                flag_reasons.append(f"{row['pct_nan']:.1f}% NaN")
            if flag_reasons:
                flags.append({"session_id": row["session_id"], "feature": feature, "flags": "; ".join(flag_reasons)})
    return pd.DataFrame(flags) if flags else pd.DataFrame(columns=["session_id", "feature", "flags"])


def run_distribution_diagnostics(output_dir, debug=True):
    session_data = {}
    for sid, pose_path, floor_path in SESSIONS:
        if debug:
            print(f"[Distributions] Loading {sid}...")
        try:
            vis_df, kin_df = load_features_direct(pose_path, floor_path, sid, debug=False)
            session_data[sid] = (vis_df, kin_df)
        except Exception as e:
            print(f"[Distributions] Error loading {sid}: {e}")
    if debug:
        print("[Distributions] Computing statistics...")
    stats_df = compute_cross_session_stats(session_data)
    if debug:
        print("[Distributions] Generating plots...")
    plot_feature_distributions(session_data, output_dir, "vis")
    plot_feature_distributions(session_data, output_dir, "kin")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(Path(output_dir) / "feature_statistics.csv", index=False)
    flags_df = flag_anomalous_sessions(stats_df)
    if len(flags_df) > 0:
        flags_df.to_csv(Path(output_dir) / "anomaly_flags.csv", index=False)
        if debug:
            print(f"[Distributions] Found {len(flags_df)} anomaly flags")
    return stats_df, flags_df


__all__ = ["compute_cross_session_stats", "plot_feature_distributions", "flag_anomalous_sessions", 
           "run_distribution_diagnostics", "EXPECTED_RANGES"]
