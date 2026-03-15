from __future__ import annotations

"""CLI for visual feature extraction (Single-FM).

This command computes virtual head-view visual covariates for a session using
scene cameras and upright 3D head pose. It warps frames to a head-anchored
ground patch, computes dense flow inside a circular ROI, aggregates with
visibility weights, and optionally saves diagnostics.

Example:
    python -m freemat.single_fm.pipeline.vis_cli \
        --session-yaml configs/single_fm/session_001.yaml \
        --cams Cam_001 --bin-s 0.05 --roi-radius-px 60 --frame-stride 2

Outputs:
    - Parquet: results/single_fm/<sid>/vis_features/vis_features_<sid>.parquet
    - Diagnostics (PNG+SVG) in the same directory (unless disabled)
    - QC JSON: results/single_fm/<sid>/vis_features/vis_qc_<sid>.json
"""

import argparse
import os
from freemat_data import load_session_yaml
from freemat.single_fm.visual.features import build_visual_features, VisualParams
from freemat.single_fm.io.save import save_aligned_parquet, save_qc_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute VIS features (virtual head view)")
    p.add_argument("--session-yaml", required=True, help="Path to session YAML")
    p.add_argument("--cams", nargs="*", default=None, help="Restrict to these cameras (e.g., Cam_001 Cam_002)")
    p.add_argument("--bin-s", type=float, default=0.05, help="Bin size in seconds (default: 0.05)")
    p.add_argument("--flow", choices=["farneback", "tvl1", "farneback_cuda"], default="farneback", help="Optical flow backend (CPU/GPU)")
    p.add_argument("--roi-radius-px", type=int, default=60, help="ROI radius on the virtual patch (pixels)")
    p.add_argument("--patch-h", type=int, default=200, help="Virtual patch height (pixels)")
    p.add_argument("--patch-w", type=int, default=200, help="Virtual patch width (pixels)")
    p.add_argument("--mpp", type=float, default=0.005, help="Meters-per-pixel on the plane (default: 0.005)")
    p.add_argument("--me-k", type=int, default=6, help="Number of motion-energy PCs")
    p.add_argument("--lookahead-m", type=float, default=1.0, help="Forward offset along plane (meters)")
    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame-pair (default: 1)")
    p.add_argument("--workers", type=int, default=0, help="Parallel cameras with N threads (0=sequential)")
    p.add_argument("--profile", action="store_true", help="Profile and print top hotspots")
    p.add_argument("--start-s", type=float, default=None, help="Start time (s) on the pose timeline")
    p.add_argument("--end-s", type=float, default=None, help="End time (s) on the pose timeline")
    p.add_argument("--duration-s", type=float, default=None, help="Process only this many seconds; if start not set, begins at first valid pose time")
    p.add_argument("--debug", action="store_true", help="Verbose debug prints (camera files, time window, progress)")
    p.add_argument("--min-valid-frac", type=float, default=0.9, help="Minimum fraction of valid joints to consider a frame valid")
    p.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold (if pose.C present)")
    p.add_argument("--seek-preroll-s", type=float, default=2.0, help="Seek a bit earlier than start to stabilize decoder (seconds)")
    p.add_argument("--reindex-ts", action="store_true", help="Force re-generate per-camera timestamp CSVs")
    p.add_argument("--flow-ds", type=int, default=1, help="Downsample factor for optic flow (1=full, 2=half, etc.)")
    p.add_argument("--min-coverage", type=float, default=0.80, help="Minimum patch coverage fraction on both frames (default: 0.80)")
    p.add_argument("--forward-min-norm", type=float, default=0.05, help="Minimum in-plane forward norm for yaw validity (default: 0.05)")
    p.add_argument("--no-zscore", action="store_true", help="Disable per-session z-scoring of features")
    p.add_argument("--kin-smooth-win-s", type=float, default=0.50, help="Smoothing window for kinematics (s)")
    p.add_argument("--out-dir", default=None, help="Override output directory for parquet + diagnostics")
    p.add_argument("--no-diagnostics", action="store_true", help="Do not save diagnostic figures")
    p.add_argument("--patch-overlays", type=int, default=0, help="Save N patch overlay figures for the first selected camera (0=off)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    sref = load_session_yaml(args.session_yaml)

    params = VisualParams(
        bin_s=float(args.bin_s),
        flow_algo=str(args.flow),
        roi_radius_px=int(args.roi_radius_px),
        patch_hw=(int(args.patch_h), int(args.patch_w)),
        mpp=float(args.mpp),
        me_k=int(args.me_k),
        lookahead_m=float(args.lookahead_m),
        frame_stride=int(args.frame_stride),
        cams_include=list(args.cams) if args.cams else None,
        zscore_per_session=(not args.no_zscore),
        kin_smooth_win_s=float(args.kin_smooth_win_s),
        min_valid_frac=float(args.min_valid_frac),
        conf_thresh=float(args.conf_thresh),
        seek_preroll_s=float(args.seek_preroll_s),
        reindex_ts=bool(args.reindex_ts),
        flow_ds=int(args.flow_ds),
        min_coverage=float(args.min_coverage),
        min_forward_norm=float(args.forward_min_norm),
    )

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile(); pr.enable()
        df, qc = build_visual_features(
            args.session_yaml,
            params=params,
            workers=int(args.workers),
            start_s=args.start_s,
            end_s=args.end_s,
            duration_s=args.duration_s,
            debug=bool(args.debug),
        )
        pr.disable()
        s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(30)
        print(s.getvalue())
    else:
        df, qc = build_visual_features(
            args.session_yaml,
            params=params,
            workers=int(args.workers),
            start_s=args.start_s,
            end_s=args.end_s,
            duration_s=args.duration_s,
            debug=bool(args.debug),
        )

    sid = str(getattr(sref, "session_id", qc.get("session_id", "session")))
    out_dir = args.out_dir or os.path.join("results", "single_fm", sid, "vis_features")
    os.makedirs(out_dir, exist_ok=True)
    out_parquet = os.path.join(out_dir, f"vis_features_{sid}.parquet")
    save_aligned_parquet(df, out_parquet, metadata={
        "session_id": sid,
        "bin_s": float(args.bin_s),
        "flow_algo": str(args.flow),
        "roi_radius_px": int(args.roi_radius_px),
        "patch_hw": [int(args.patch_h), int(args.patch_w)],
        "mpp": float(args.mpp),
        "me_k": int(args.me_k),
        "lookahead_m": float(args.lookahead_m),
        "frame_stride": int(args.frame_stride),
        "cams_include": list(args.cams) if args.cams else None,
        "zscore_per_session": (not args.no_zscore),
        "kin_smooth_win_s": float(args.kin_smooth_win_s),
    })
    qc_path = os.path.join(out_dir, f"vis_qc_{sid}.json")
    save_qc_json(qc, qc_path)

    if not args.no_diagnostics:
        from freemat.single_fm.viz import save_vis_diagnostics, save_patch_overlays
        save_vis_diagnostics(args.session_yaml, df, out_dir=out_dir)
        if int(args.patch_overlays) > 0:
            first_cam = (list(args.cams)[0] if args.cams else (qc.get("cams_mapped") or [None])[0])
            if first_cam:
                try:
                    save_patch_overlays(
                        args.session_yaml,
                        out_dir=out_dir,
                        cam=str(first_cam),
                        params=VisualParams(
                            bin_s=float(args.bin_s),
                            flow_algo=str(args.flow),
                            roi_radius_px=int(args.roi_radius_px),
                            patch_hw=(int(args.patch_h), int(args.patch_w)),
                            mpp=float(args.mpp),
                            me_k=int(args.me_k),
                            lookahead_m=float(args.lookahead_m),
                            frame_stride=int(args.frame_stride),
                            cams_include=None,
                            zscore_per_session=(not args.no_zscore),
                            kin_smooth_win_s=float(args.kin_smooth_win_s),
                        ),
                        max_examples=int(args.patch_overlays),
                    )
                except Exception as e:
                    print(f"Patch overlays failed: {e}")

    print(f"VIS features written: {out_parquet}")
    print(f"QC written: {qc_path}")
    if not args.no_diagnostics:
        print(f"Diagnostics saved under: {out_dir}")


if __name__ == "__main__":
    main()
