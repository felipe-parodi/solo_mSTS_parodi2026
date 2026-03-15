"""QC for NeuroExplorer (.nex/.nex5) continuous data.

This module provides a small CLI to:
  - Load continuous analog data from a .nex/.nex5 file
  - Plot a short time‑series snippet for selected channels
  - Compute and plot PSDs (Welch) for those channels
  - Flag potential dead/flat channels based on relative variance

Examples:
    # Inspect first 10 seconds of a file and plot channels 0,8,16,24
    python -m freemat_data.qc.qc_nex \
        --file /path/to/recording.nex5 \
        --channels 0,8,16,24 \
        --duration 10 \
        --outdir results/nex_qc/demo

Notes:
    - This relies on ``neo`` for reading .nex/.nex5. Install with ``pip install neo``.
    - By design, the loader reads the first segment into memory. Use --duration to
      limit to a short snippet for rapid QC.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from ..loaders.nex import load_continuous_from_nex, read_nex_header, load_continuous_from_dir
from ..types import LFPTimeSeries


@dataclass
class DeadChannelHeuristics:
    """Heuristics for flagging dead/flat channels.

    Attributes:
        min_std_ratio: Flag channel when its std is below this fraction of the
            median channel std. Example: 0.05 flags channels <5% of median.
        min_absolute_std: Optional absolute std floor to consider a channel valid.
            Useful when data are known to be in microvolts. If None, only ratio is used.
    """

    min_std_ratio: float = 0.05
    min_absolute_std: Optional[float] = None


def compute_channel_std(x: np.ndarray) -> np.ndarray:
    """Compute per-channel standard deviation for (T, C) array.

    Args:
        x: Time by channel matrix.

    Returns:
        (C,) array of channel standard deviations.
    """
    if x.ndim != 2:
        raise ValueError("compute_channel_std expects x with shape (T, C)")
    return np.std(x, axis=0, ddof=1)


def detect_dead_channels(std_per_ch: np.ndarray, h: DeadChannelHeuristics) -> np.ndarray:
    """Flag channels with near-zero variance using relative/absolute thresholds.

    Args:
        std_per_ch: (C,) per-channel standard deviations.
        h: Dead channel heuristic parameters.

    Returns:
        Boolean mask of shape (C,) where True indicates a likely dead channel.
    """
    if std_per_ch.ndim != 1:
        raise ValueError("std_per_ch must be 1D")
    if std_per_ch.size == 0:
        return np.zeros((0,), dtype=bool)
    med = float(np.median(std_per_ch))
    by_ratio = std_per_ch < (h.min_std_ratio * med)
    if h.min_absolute_std is None:
        return by_ratio
    by_abs = std_per_ch < float(h.min_absolute_std)
    return np.logical_or(by_ratio, by_abs)


def welch_psd(x: np.ndarray, fs: float, *, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (frequency, PSD) for each channel using Welch's method.

    Args:
        x: (T, C) array.
        fs: Sampling rate in Hz.
        nperseg: Optional segment length; defaults to min(T, int(2*fs)).

    Returns:
        f: (F,) frequency vector in Hz.
        pxx: (F, C) power spectral density in data units^2/Hz.
    """
    if x.ndim != 2:
        raise ValueError("welch_psd expects x with shape (T, C)")
    T = x.shape[0]
    if nperseg is None:
        nperseg = max(8, min(T, int(2 * fs)))
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, axis=0, noverlap=nperseg // 2)
    return f, p


def bandpower_from_psd(f: np.ndarray, pxx: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Integrate PSD between fmin and fmax for each channel.

    Args:
        f: (F,) frequency axis.
        pxx: (F, C) PSD values.
        fmin: Lower bound (Hz).
        fmax: Upper bound (Hz).

    Returns:
        (C,) bandpower in units^2.
    """
    if pxx.ndim == 1:
        pxx = pxx[:, None]
    sel = (f >= fmin) & (f <= fmax)
    if not np.any(sel):
        return np.zeros((pxx.shape[1],), dtype=float)
    # Trapezoidal integration along frequency axis
    bp = np.trapz(pxx[sel, :], f[sel], axis=0)
    return bp


def compute_noise_scores(x: np.ndarray, fs: float, ch_idx: List[int]) -> dict:
    """Compute lightweight noise scores for selected channels.

    Returns a dict with per-channel arrays for:
      - line60_ratio: bandpower 59-61 Hz divided by 50-70 Hz (local window)
      - lowfreq_ratio: bandpower 0.5-10 Hz divided by 0.5-100 Hz
      - broadband_ratio: bandpower 300-3000 Hz divided by 30-300 Hz
      - clip_fraction: fraction of samples within 0.1% of min/max per channel
    """
    if x.ndim != 2:
        raise ValueError("compute_noise_scores expects x with shape (T, C)")
    X = x[:, ch_idx]
    if X.size == 0 or X.shape[0] == 0:
        raise RuntimeError("compute_noise_scores: zero samples in selection; check --fs and --duration")
    f, pxx = welch_psd(X, fs)
    # Line noise ratio near 60 Hz
    bp_60 = bandpower_from_psd(f, pxx, 59.0, 61.0)
    bp_loc = bandpower_from_psd(f, pxx, 50.0, 70.0)
    line60_ratio = np.divide(bp_60, np.maximum(bp_loc, 1e-20))
    # Low frequency content
    bp_low = bandpower_from_psd(f, pxx, 0.5, 10.0)
    bp_0100 = bandpower_from_psd(f, pxx, 0.5, 100.0)
    lowfreq_ratio = np.divide(bp_low, np.maximum(bp_0100, 1e-20))
    # Broadband increase (EMG-ish)
    bp_bb = bandpower_from_psd(f, pxx, 300.0, 3000.0)
    bp_mid = bandpower_from_psd(f, pxx, 30.0, 300.0)
    broadband_ratio = np.divide(bp_bb, np.maximum(bp_mid, 1e-20))
    # Clipping fraction
    eps = 1e-12
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = np.maximum(maxs - mins, eps)
    lo = mins + 0.001 * rng
    hi = maxs - 0.001 * rng
    clip_fraction = np.mean((X <= lo[None, :]) | (X >= hi[None, :]), axis=0)
    return {
        "line60_ratio": line60_ratio.tolist(),
        "lowfreq_ratio": lowfreq_ratio.tolist(),
        "broadband_ratio": broadband_ratio.tolist(),
        "clip_fraction": clip_fraction.tolist(),
    }


def _plot_psd_compare(ax: plt.Axes, f1: np.ndarray, p1: np.ndarray, f2: np.ndarray, p2: np.ndarray, ch_idx: List[int], max_freq: float) -> None:
    colors = plt.cm.tab20.colors
    for k, i in enumerate(range(p1.shape[1])):
        c = colors[k % len(colors)]
        ax.semilogy(f1, p1[:, i], color=c, lw=1.0, alpha=0.9, label=f"raw ch{ch_idx[i]}")
        ax.semilogy(f2, p2[:, i], color=c, lw=1.0, alpha=0.9, ls='--', label=f"post ch{ch_idx[i]}")
    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (units^2/Hz)")
    ax.set_title("PSD before (solid) vs after (dashed)")
    if p1.shape[1] <= 10:
        ax.legend(ncol=2, fontsize=8)


def _choose_channels(C: int, requested: Optional[Sequence[int]], max_count: int) -> List[int]:
    if requested is not None and len(requested) > 0:
        sel = sorted(set(int(i) for i in requested))
        if any(i < 0 or i >= C for i in sel):
            raise ValueError(f"Requested channel indices out of range [0, {C-1}]: {sel}")
        return sel[:max_count]
    if C <= max_count:
        return list(range(C))
    # Evenly spread selection across all channels
    return sorted({int(round(i)) for i in np.linspace(0, C - 1, num=max_count)})


def _plot_snippet(ax: plt.Axes, lfp: LFPTimeSeries, ch_idx: List[int], duration_s: float) -> None:
    fs = float(lfp.fs or 0.0)
    if fs <= 0:
        raise ValueError("LFPTimeSeries.fs must be provided and > 0 for plotting")
    i_stop = min(int(duration_s * fs), lfp.x.shape[0])
    t = np.arange(i_stop, dtype=float) / fs
    for i in ch_idx:
        ax.plot(t, lfp.x[:i_stop, i], lw=0.8, label=f"ch{i}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Raw continuous (first {duration_s:g}s)")
    if len(ch_idx) <= 12:
        ax.legend(ncol=4, fontsize=8)


def _plot_psd(ax: plt.Axes, lfp: LFPTimeSeries, ch_idx: List[int], max_freq: float) -> None:
    fs = float(lfp.fs or 0.0)
    if fs <= 0:
        raise ValueError("LFPTimeSeries.fs must be provided and > 0 for PSD")
    f, pxx = welch_psd(lfp.x[:, ch_idx], fs)
    ax.semilogy(f, pxx)
    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (units^2/Hz)")
    ax.set_title("Power spectral density (Welch)")
    # Mark 60 Hz and harmonics
    for k in range(1, int(max_freq // 60) + 1):
        ax.axvline(60 * k, color="r", ls="--", lw=0.8, alpha=0.6)


def _plot_channel_stds(ax: plt.Axes, stds: np.ndarray, dead_mask: np.ndarray) -> None:
    ax.plot(stds, ".", ms=4, alpha=0.9)
    bad = np.where(dead_mask)[0]
    if bad.size:
        ax.plot(bad, stds[bad], "ro", ms=4, label="flagged")
        ax.legend(fontsize=8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("STD (signal units)")
    ax.set_title("Per-channel standard deviation")


def run_nex_qc(
    *,
    filepath: str,
    channels: Optional[List[int]] = None,
    duration_s: float = 10.0,
    max_plot_channels: int = 16,
    max_psd_freq: float = 1000.0,
    heuristics: Optional[DeadChannelHeuristics] = None,
    outdir: Optional[str] = None,
) -> Tuple[Path, List[Tuple[str, Path]]]:
    """Run QC on a .nex/.nex5 file and write a JSON + plots.

    Args:
        filepath: Path to the .nex/.nex5 file.
        channels: Optional explicit channel indices to plot/consider.
        duration_s: Duration (s) to load from the start for QC.
        max_plot_channels: Cap on the number of plotted channels.
        max_psd_freq: Upper x-limit for PSD plots.
        heuristics: Dead-channel heuristic config.
        outdir: Optional output directory; defaults to results/<basename>/qc.

    Returns:
        (qc_json_path, plots_manifest) where plots_manifest is a list of
        (name, path) tuples.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load a snippet of continuous data
    lfp = load_continuous_from_nex(str(p), duration_s=duration_s, fs_hint=(None if args.fs is None else float(args.fs)))
    fs = float(lfp.fs or 0.0)
    if fs <= 0:
        raise RuntimeError("Sampling rate not found or invalid in loaded data")

    # Choose channels to visualize
    C = lfp.x.shape[1]
    ch_sel = _choose_channels(C, channels, max_plot_channels)

    # Compute dead-channel mask
    h = heuristics or DeadChannelHeuristics()
    stds = compute_channel_std(lfp.x)
    dead_mask = detect_dead_channels(stds, h)

    # Resolve output directory
    base = Path(outdir) if outdir else (Path("results") / p.stem / "qc")
    base.mkdir(parents=True, exist_ok=True)

    plots: List[Tuple[str, Path]] = []

    # Time snippet and PSD
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8))
    _plot_snippet(ax0, lfp, ch_sel, min(duration_s, lfp.x.shape[0] / fs))
    _plot_psd(ax1, lfp, ch_sel, max_psd_freq)
    fig.suptitle(f"NEX QC: {p.name} (fs={fs:g} Hz)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_ts_psd = base / f"nex_qc_{p.stem}_snippet_psd.png"
    fig.savefig(out_ts_psd, dpi=150)
    plt.close(fig)
    plots.append(("snippet_psd", out_ts_psd))

    # Channel std distribution
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
    _plot_channel_stds(ax2, stds, dead_mask)
    fig2.tight_layout()
    out_stds = base / f"nex_qc_{p.stem}_stds.png"
    fig2.savefig(out_stds, dpi=150)
    plt.close(fig2)
    plots.append(("channel_stds", out_stds))

    # Noise scores on plotted channels
    noise_scores = compute_noise_scores(lfp.x, fs, ch_sel)

    # Write JSON summary
    qc = {
        "file": str(p),
        "fs_hz": fs,
        "n_samples": int(lfp.x.shape[0]),
        "n_channels": int(C),
        "channels_plotted": ch_sel,
        "std_per_channel": stds.tolist(),
        "dead_channel_mask": dead_mask.astype(int).tolist(),
        "heuristics": asdict(h),
        "plots": {name: str(path) for name, path in plots},
        "noise_scores": noise_scores,
    }
    qc_path = base / f"nex_qc_{p.stem}.json"
    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)

    return qc_path, plots


def run_nex_compare(
    *,
    raw_file: str,
    post_file: str,
    channels: Optional[List[int]] = None,
    duration_s: float = 10.0,
    max_plot_channels: int = 12,
    max_psd_freq: float = 1000.0,
    outdir: Optional[str] = None,
) -> Tuple[Path, List[Tuple[str, Path]]]:
    """Compare raw vs. post-processed NEX/NEX5 files on the same segment/channels.

    Writes an overlay PSD figure and a JSON with per-channel ratios.
    """
    p_raw = Path(raw_file)
    p_post = Path(post_file)
    if not p_raw.exists():
        raise FileNotFoundError(f"File not found: {raw_file}")
    if not p_post.exists():
        raise FileNotFoundError(f"File not found: {post_file}")

    lfp_raw = load_continuous_from_nex(str(p_raw), duration_s=duration_s, fs_hint=(None if args.fs is None else float(args.fs)))
    lfp_post = load_continuous_from_nex(str(p_post), duration_s=duration_s, fs_hint=(None if args.fs is None else float(args.fs)))
    if not lfp_raw.fs or not lfp_post.fs or float(lfp_raw.fs) <= 0 or float(lfp_post.fs) <= 0:
        raise RuntimeError("Both files must provide valid sampling rates")
    fsr, fsp = float(lfp_raw.fs), float(lfp_post.fs)
    if abs(fsr - fsp) / max(fsr, 1.0) > 1e-6:
        raise RuntimeError(f"Sampling rates differ (raw={fsr}, post={fsp})")

    C = min(lfp_raw.x.shape[1], lfp_post.x.shape[1])
    ch_sel = _choose_channels(C, channels, max_plot_channels)

    f1, p1 = welch_psd(lfp_raw.x[:, ch_sel], fsr)
    f2, p2 = welch_psd(lfp_post.x[:, ch_sel], fsp)

    # Output directory
    base = Path(outdir) if outdir else (Path("results") / f"{p_raw.stem}_vs_{p_post.stem}" / "qc")
    base.mkdir(parents=True, exist_ok=True)

    # PSD overlay
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    _plot_psd_compare(ax, f1, p1, f2, p2, ch_sel, max_psd_freq)
    for k in range(1, int(max_psd_freq // 60) + 1):
        ax.axvline(60 * k, color="r", ls="--", lw=0.8, alpha=0.6)
    fig.tight_layout()
    out_cmp = base / f"nex_qc_psd_compare_{p_raw.stem}_vs_{p_post.stem}.png"
    fig.savefig(out_cmp, dpi=150)
    plt.close(fig)

    # Ratios: notch efficacy and high-pass efficacy
    def _ratio(bp_after: np.ndarray, bp_before: np.ndarray) -> List[float]:
        return np.divide(bp_after, np.maximum(bp_before, 1e-20)).tolist()

    bp60_raw = bandpower_from_psd(f1, p1, 59.0, 61.0)
    bp60_post = bandpower_from_psd(f2, p2, 59.0, 61.0)
    notch_ratio = _ratio(bp60_post, bp60_raw)

    bplow_raw = bandpower_from_psd(f1, p1, 0.5, 10.0)
    bplow_post = bandpower_from_psd(f2, p2, 0.5, 10.0)
    highpass_ratio = _ratio(bplow_post, bplow_raw)

    cmp_json = {
        "raw_file": str(p_raw),
        "post_file": str(p_post),
        "fs_hz": fsr,
        "channels": ch_sel,
        "notch_ratio_60hz": notch_ratio,  # expect << 1 when notch worked
        "lowfreq_ratio_0p5_10hz": highpass_ratio,  # expect << 1 when high-pass worked
        "plot_psd_compare": str(out_cmp),
    }
    cmp_path = base / f"nex_qc_compare_{p_raw.stem}_vs_{p_post.stem}.json"
    with cmp_path.open("w", encoding="utf-8") as f:
        json.dump(cmp_json, f, indent=2)

    return cmp_path, [("psd_compare", out_cmp)]


def _parse_channels(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    parts = [s.strip() for s in arg.split(",") if s.strip()]
    return [int(s) for s in parts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QC for NeuroExplorer (.nex/.nex5) continuous data")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--file", help="Path to .nex or .nex5 file (single-file QC)")
    grp.add_argument("--dir", help="Directory with multiple .nex5 files to aggregate as channels")
    grp.add_argument("--compare", nargs=2, metavar=("RAW", "POST"), help="Compare raw vs post-processed NEX/NEX5")
    ap.add_argument("--dump-header", action="store_true", help="Print and optionally write header summary instead of QC")
    ap.add_argument("--channels", default=None, help="Comma-separated channel indices to plot (e.g., 0,8,16)")
    ap.add_argument("--duration", type=float, default=10.0, help="Seconds of data to load from the start (default: 10s)")
    ap.add_argument("--max-plot-channels", type=int, default=16, help="Max number of channels to plot (default: 16)")
    ap.add_argument("--max-psd-freq", type=float, default=1000.0, help="Upper x-limit for PSD (Hz)")
    ap.add_argument("--min-std-ratio", type=float, default=0.05, help="Dead-channel std ratio threshold vs. median (default: 0.05)")
    ap.add_argument("--min-absolute-std", type=float, default=None, help="Optional absolute std floor for dead-channel flagging")
    ap.add_argument("--fs", type=float, default=None, help="Sampling rate hint (Hz) when file header lacks it")
    ap.add_argument("--outdir", default=None, help="Output directory (default: results/<basename>/qc)")
    args = ap.parse_args()

    ch = _parse_channels(args.channels)
    if args.dump_header:
        if not args.file:
            raise SystemExit("--dump-header requires --file")
        hdr = read_nex_header(args.file)
        # Pretty print streams
        print("Streams:")
        for s in hdr.get("streams", []):
            print(f"  [{s.get('index')}] id={s.get('id')} name={s.get('name')} fs={s.get('sampling_rate')}")
        # Print a compact channel table (first 120 channels max)
        chans = hdr.get("channels", [])
        print(f"Channels (showing up to 120): total={len(chans)}")
        for c in chans[:120]:
            fs = c.get("sampling_rate")
            sid = c.get("stream_id")
            nm = c.get("name")
            print(f"  ch[{c.get('index')}] name={nm} stream_id={sid} fs={fs}")
        if args.outdir:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            p = outdir / (f"nex_header_{Path(args.file).stem}.json")
            with p.open("w", encoding="utf-8") as f:
                json.dump(hdr, f, indent=2)
            print(f"[qc_nex] wrote header JSON: {p}")
        raise SystemExit(0)
    elif args.dir:
        lfp = load_continuous_from_dir(
            args.dir,
            duration_s=float(args.duration),
            fs_hint=(None if args.fs is None else float(args.fs)),
        )
        fs = float(lfp.fs or 0.0)
        if fs <= 0:
            raise RuntimeError("Sampling rate not found or invalid in aggregated data")
        C = lfp.x.shape[1]
        ch_sel = _choose_channels(C, ch, int(args.max_plot_channels))

        h = DeadChannelHeuristics(min_std_ratio=float(args.min_std_ratio), min_absolute_std=(None if args.min_absolute_std is None else float(args.min_absolute_std)))
        stds = compute_channel_std(lfp.x)
        dead_mask = detect_dead_channels(stds, h)
        base = Path(args.outdir) if args.outdir else (Path("results") / Path(args.dir).name / "qc")
        base.mkdir(parents=True, exist_ok=True)

        plots: List[Tuple[str, Path]] = []
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8))
        _plot_snippet(ax0, lfp, ch_sel, min(float(args.duration), lfp.x.shape[0] / fs))
        _plot_psd(ax1, lfp, ch_sel, float(args.max_psd_freq))
        fig.suptitle(f"NEX QC (dir): {Path(args.dir).name} (fs={fs:g} Hz)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_ts_psd = base / f"nex_qc_{Path(args.dir).name}_snippet_psd.png"
        fig.savefig(out_ts_psd, dpi=150)
        plt.close(fig)
        plots.append(("snippet_psd", out_ts_psd))

        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
        _plot_channel_stds(ax2, stds, dead_mask)
        fig2.tight_layout()
        out_stds = base / f"nex_qc_{Path(args.dir).name}_stds.png"
        fig2.savefig(out_stds, dpi=150)
        plt.close(fig2)
        plots.append(("channel_stds", out_stds))

        noise_scores = compute_noise_scores(lfp.x, fs, ch_sel)
        qc = {
            "dir": str(Path(args.dir)),
            "fs_hz": fs,
            "n_samples": int(lfp.x.shape[0]),
            "n_channels": int(C),
            "channels_plotted": ch_sel,
            "std_per_channel": stds.tolist(),
            "dead_channel_mask": dead_mask.astype(int).tolist(),
            "heuristics": asdict(h),
            "plots": {name: str(path) for name, path in plots},
            "noise_scores": noise_scores,
        }
        qc_path = base / f"nex_qc_{Path(args.dir).name}.json"
        with qc_path.open("w", encoding="utf-8") as f:
            json.dump(qc, f, indent=2)
        print(f"[qc_nex] wrote {qc_path}")
        for name, path in plots:
            print(f"[qc_nex] plot {name}: {path}")
    elif args.compare:
        cmp_json, plots = run_nex_compare(
            raw_file=args.compare[0],
            post_file=args.compare[1],
            channels=ch,
            duration_s=float(args.duration),
            max_plot_channels=int(min(args.max_plot_channels, 12)),
            max_psd_freq=float(args.max_psd_freq),
            outdir=args.outdir,
        )
        print(f"[qc_nex] wrote {cmp_json}")
        for name, path in plots:
            print(f"[qc_nex] plot {name}: {path}")
    else:
        heur = DeadChannelHeuristics(
            min_std_ratio=float(args.min_std_ratio),
            min_absolute_std=(None if args.min_absolute_std is None else float(args.min_absolute_std)),
        )
        qc_json, plots = run_nex_qc(
            filepath=args.file,
            channels=ch,
            duration_s=float(args.duration),
            max_plot_channels=int(args.max_plot_channels),
            max_psd_freq=float(args.max_psd_freq),
            heuristics=heur,
            outdir=args.outdir,
        )
        print(f"[qc_nex] wrote {qc_json}")
        for name, path in plots:
            print(f"[qc_nex] plot {name}: {path}")
