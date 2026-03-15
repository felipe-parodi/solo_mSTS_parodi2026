from __future__ import annotations

"""CLI for kinematic feature extraction (Single-FM).

Computes head-frame KIN covariates aligned to bin centers, with defaults:
- head only, 1.0 s smoothing, POS excluded, per-session z-scoring enabled
- saves base features and a long-form lagged table (±~800 ms by default)

Example:
    python -m freemat.single_fm.pipeline.kin_cli \
        --session-yaml configs/single_fm/session_001.yaml \
        --bin-s 0.05 --include-speed --lags-ms -800:800

Outputs:
    - Parquet: results/single_fm/<sid>/kin_features/kin_features_<sid>.parquet
    - Parquet: results/single_fm/<sid>/kin_features/kin_features_lagged_<sid>.parquet
    - QC JSON: results/single_fm/<sid>/kin_features/kin_qc_<sid>.json
"""

import argparse
import os
from typing import List, Optional

import numpy as np

from freemat_data import load_session_yaml
from freemat.single_fm.kin.features import KinParams, build_kin_features, make_lagged_long
from freemat.single_fm.io.save import save_aligned_parquet, save_qc_json
from freemat.single_fm.viz.kin_diagnostics import save_kin_diagnostics


def _parse_lags(arg: Optional[str], bin_ms: int) -> List[int]:
    """Parse lags from a string.

    Accepted formats:
        - None: symmetric range using step=bin_ms, approx ±800 ms
        - "-800:800" → inclusive range with step=bin_ms
        - comma-separated list: "-400,-200,0,200,400"
    """
    if arg is None or str(arg).strip() == "":
        # Default symmetric around 0 using the bin step, approx ±800 ms
        step = int(max(1, abs(int(bin_ms))))
        nmax = int(max(1, round(800.0 / step)))
        L = [i * step for i in range(-nmax, nmax + 1)]
        return L
    s = str(arg).strip()
    if "," in s:
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                out.append(int(tok))
        return out
    if ":" in s:
        a, b = s.split(":", 1)
        lo = int(a) if a.strip() else -800
        hi = int(b) if b.strip() else 800
        if lo > hi:
            lo, hi = hi, lo
        L = list(range(int(lo), int(hi) + 1, int(max(1, bin_ms))))
        # Ensure 0 is included
        if 0 not in L:
            L.append(0)
            L = sorted(L)
        return L
    # Single integer case → interpret as symmetric ±value with step=bin_ms
    try:
        m = abs(int(s))
        n = int(max(1, round(m / max(1, bin_ms))))
        return [i * bin_ms for i in range(-n, n + 1)]
    except Exception:
        # Fallback to default
        step = int(max(1, abs(int(bin_ms))))
        nmax = int(max(1, round(800.0 / step)))
        return [i * step for i in range(-nmax, nmax + 1)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute KIN features (head-frame)")
    p.add_argument("--session-yaml", required=True, help="Path to session YAML")
    p.add_argument("--bin-s", type=float, default=0.05, help="Bin size in seconds (default: 0.05). Ignored if --bins-parquet is set")
    p.add_argument("--bins-parquet", default=None, help="Existing parquet providing t_bin_s for alignment (aligned FR or VIS)")
    p.add_argument("--smooth-win-s", type=float, default=1.0, help="Smoothing window for COM/yaw (seconds; default: 1.0)")
    p.add_argument("--min-forward-norm", type=float, default=0.05, help="Minimum in-plane forward strength required for yaw/vel (default 0.05)")
    p.add_argument("--include-speed", action="store_true", help="Include speed magnitude in KIN features")
    p.add_argument("--no-zscore", action="store_true", help="Disable per-session z-scoring of features")
    p.add_argument("--lags-ms", default=None, help="Lags in ms: '-800:800' or comma-list (default ~±800ms with step=bin)")
    p.add_argument("--min-valid-frac", type=float, default=0.9, help="Min per-frame valid joint fraction for pose gating (default 0.9)")
    p.add_argument("--conf-thresh", type=float, default=0.0, help="Confidence threshold when computing valid fraction (default 0.0)")
    p.add_argument("--debug", action="store_true", default=True, help="Print debug information (on by default)")
    p.add_argument("--out-dir", default=None, help="Override output directory for parquet + QC")
    p.add_argument("--no-diagnostics", action="store_true", help="Reserved for future diagnostics (no-op)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sref = load_session_yaml(args.session_yaml)

    kparams = KinParams(
        bin_s=float(args.bin_s),
        smooth_win_s=float(args.smooth_win_s),
        min_forward_norm=float(args.min_forward_norm),
        zscore_per_session=(not args.no_zscore),
        include_speed=bool(args.include_speed),
        min_valid_frac=float(args.min_valid_frac),
        conf_thresh=float(args.conf_thresh),
        debug=bool(args.debug),
        ignore_confidence_for_validity=True,
    )
    df, qc = build_kin_features(
        args.session_yaml,
        params=kparams,
        bins_parquet=args.bins_parquet,
    )

    sid = str(getattr(sref, "session_id", qc.get("session_id", "session")))
    out_dir = args.out_dir or os.path.join("results", "single_fm", sid, "kin_features")
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, f"kin_features_{sid}.parquet")
    save_aligned_parquet(df, out_parquet, metadata={
        "session_id": sid,
        "bin_s": float(qc.get("bin_s", args.bin_s)),
        "smooth_win_s": float(args.smooth_win_s),
        "min_forward_norm": float(args.min_forward_norm),
        "include_speed": bool(args.include_speed),
        "zscore_per_session": (not args.no_zscore),
        "aligned_to": qc.get("aligned_to"),
    })

    # Build and save lagged long-form table
    bin_s_used = float(qc.get("bin_s", args.bin_s))
    bin_ms = int(round(bin_s_used * 1000.0))
    lags = _parse_lags(args.lags_ms, bin_ms)
    dfL = make_lagged_long(df, lags_ms=lags, bin_s=bin_s_used)
    out_parquet_lag = os.path.join(out_dir, f"kin_features_lagged_{sid}.parquet")
    save_aligned_parquet(dfL, out_parquet_lag, metadata={
        "session_id": sid,
        "bin_s": bin_s_used,
        "lags_ms": list(map(int, lags)),
        "base_parquet": out_parquet,
    })

    qc_out = dict(qc)
    qc_out.update({
        "lags_ms": list(map(int, lags)),
        "base_parquet": out_parquet,
        "lagged_parquet": out_parquet_lag,
    })
    qc_path = os.path.join(out_dir, f"kin_qc_{sid}.json")
    save_qc_json(qc_out, qc_path)

    if not args.no_diagnostics:
        try:
            save_kin_diagnostics(
                args.session_yaml,
                df,
                out_dir=out_dir,
                include_speed=bool(args.include_speed),
            )
        except Exception as e:
            print(f"Diagnostics failed: {e}")

    print(f"KIN features written: {out_parquet}")
    print(f"Lagged features written: {out_parquet_lag}")
    print(f"QC written: {qc_path}")


if __name__ == "__main__":
    main()
