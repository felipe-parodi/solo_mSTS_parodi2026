"""CLI and helper to compute spike QC metrics and write keep lists.

Single- or multi-logger sessions
--------------------------------
This module operates on a single session YAML and can process one or both
loggers (e.g., social sessions with ``m1_neural`` and ``m2_neural``) in a
single run. Use ``--logger all`` to compute QC for both and write per-logger
outputs suffixed by the logger tag (``_m1``/``_m2``), plus a small combined
manifest.

It computes core QC metrics (mean FR, ISI violation rate, presence ratio,
FR stability across 5‑min bins by default) and writes a JSON summary plus a
separate keep list JSON (mask and indices) for downstream use.

Examples (CLI):
    # Per‑logger run
    python -m freemat_data.qc.qc_spikes --config configs/session_240531_latesync.yaml --logger m1

    # Both loggers in one pass
    python -m freemat_data.qc.qc_spikes --config configs/session_240531_latesync.yaml --logger all

Defaults
--------
The default output directory is taken from the session YAML
``data_files.output_dir`` if provided, under a ``qc`` subfolder. Otherwise,
outputs go to ``results/<session_id>/qc``. Thresholds are sensible defaults
and can be overridden via CLI flags.

Visualization aims to be robust (per‑figure guards) so one failing panel does
not block other plots from being saved.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from .. import (
    load_session_yaml,
    compute_qc_metrics_bundle,
    compute_isi_metrics,
    SpikesQCMetrics,
    infer_logger,
    load_banks_config,
    map_channels_to_bank,
    excluded_channels_for_subject,
)
from ..loaders.spikes import load_spikes_events_mat, load_spikes_events_npz
from ..types import SessionRef, SpikesEvents
from .qc_plots import (
    plot_fr_heatmap as _plot_fr_heatmap,
    plot_summary_panels as _plot_summary_panels,
    plot_active_units as _plot_active_units,
    plot_population_fr as _plot_population_fr,
    plot_fr_hist as _plot_fr_hist,
    plot_drop_breakdown as _plot_drop_breakdown,
    plot_scatter_panels as _plot_scatter_panels,
    raster_kept_units as _raster_kept_units,
    try_plot as _try_plot,
)


@dataclass
class Thresholds:
    """Thresholds for keep-list policy.

    Attributes:
        min_mean_fr_hz: Minimum mean FR to consider a unit active.
        max_isi_violation_rate: Maximum fraction of ISIs below refractory threshold.
        min_presence_ratio: Minimum fraction of time bins with spikes.
        max_fr_stability_cv: Maximum FR coefficient of variation across long bins.
        presence_bin_s: Bin size (s) for presence ratio.
        fr_window_s: Window size (s) for long FR bins (stability, heatmap).
        isi_threshold_s: ISI refractory threshold in seconds (default 1.5 ms).
    """

    min_mean_fr_hz: float = 0.02
    max_isi_violation_rate: float = 0.01
    min_presence_ratio: float = 0.5
    max_fr_stability_cv: float = 2.0
    presence_bin_s: float = 1.0
    fr_window_s: float = 300.0
    isi_threshold_s: float = 0.0015


def _default_outdir(session: SessionRef) -> Path:
    """Resolve default outdir preferring YAML ``data_files.output_dir``.

    Falls back to ``results/<session_id>/qc`` when not present.
    """
    try:
        base = session.data_files.get("output_dir") if hasattr(session, "data_files") else None
    except Exception:
        base = None
    if base:
        return Path(str(base)) / "qc"
    return Path("results") / session.session_id / "qc"


def _resolve_outdir(session: SessionRef, override: Optional[str]) -> Path:
    return Path(override) if override else _default_outdir(session)


def _available_loggers(session: SessionRef) -> List[str]:
    tags: List[str] = []
    try:
        for k in getattr(session, "data_files", {}).keys():
            if k.endswith("_neural") and isinstance(k, str) and len(k) >= 3 and k[:2] in ("m1", "m2"):
                tag = k.split("_")[0]
                if tag in ("m1", "m2") and tag not in tags:
                    tags.append(tag)
    except Exception:
        pass
    return sorted(tags)


def _load_spikes_for_logger(session: SessionRef, logger_tag: str) -> SpikesEvents:
    key = f"{logger_tag}_neural"
    path = None
    try:
        path = session.data_files.get(key)
    except Exception:
        path = None
    if not path:
        raise FileNotFoundError(f"Session YAML missing '{key}' path for logger {logger_tag}")
    p = str(path)
    if p.lower().endswith(".npz"):
        spk = load_spikes_events_npz(p)
    elif p.lower().endswith(".mat"):
        d = load_spikes_events_mat(p)
        spk = SpikesEvents(
            t=d["spike_t"],
            unit_index=d["spike_unit_index"],
            unit_ids=(d.get("unit_ids") if "unit_ids" in d else np.asarray([], dtype=object)),
            session_id=str(d.get("session_id") or Path(p).stem),
            meta={k: d[k] for k in ("unit_channels", "unit_kind", "event_source", "bin_width_s") if k in d},
        )
    else:
        raise ValueError(f"Unsupported spikes path for {logger_tag}: {p}")
    return spk


def compute_keep_mask(
    *,
    mean_fr_hz: np.ndarray,
    isi_violation_rate: np.ndarray,
    presence_ratio: np.ndarray,
    fr_stability_cv: np.ndarray,
    thr: Thresholds,
) -> np.ndarray:
    """Return boolean mask of units to keep based on threshold criteria."""
    keep = (
        (mean_fr_hz >= float(thr.min_mean_fr_hz))
        & (isi_violation_rate <= float(thr.max_isi_violation_rate))
        & (presence_ratio >= float(thr.min_presence_ratio))
        & (np.isfinite(fr_stability_cv) & (fr_stability_cv <= float(thr.max_fr_stability_cv)))
    )
    return keep.astype(bool, copy=False)


# Plot functions moved to qc_plots.py (W1.4 refactor)
# Baseline detection moved to baseline_detection.py (W1.4 refactor)
# Old names (_plot_fr_heatmap, _plot_summary_panels, etc.) are re-imported
# at the top of this module for backward compatibility.


# ---------------------------------------------------------------------------
# Unified pipeline helpers (W1.1 refactor)
# ---------------------------------------------------------------------------


def _resolve_events_text(session: SessionRef) -> Optional[str]:
    """Resolve events text path from multiple session YAML keys."""
    if not hasattr(session, "data_files"):
        return None
    return (
        session.data_files.get("events_text")
        or session.data_files.get("events_txt")
        or session.data_files.get("session_events")
    )


def _resolve_bank_metadata(
    spk_meta: Optional[dict],
    banks_path: str = "configs/banks.yaml",
) -> Tuple[List[int], Optional[List[str]], Optional[Dict[str, List[bool]]]]:
    """Resolve unit channels, bank labels, and per-subject exclude masks.

    Args:
        spk_meta: Spike metadata dict (may contain 'unit_channels').
        banks_path: Path to banks YAML config.

    Returns:
        (unit_channels_list, bank_by_unit_list, exclude_masks)
    """
    unit_channels_list: List[int] = []
    bank_by_unit_list: Optional[List[str]] = None
    exclude_masks: Optional[Dict[str, List[bool]]] = None
    if not isinstance(spk_meta, dict) or "unit_channels" not in spk_meta:
        return unit_channels_list, bank_by_unit_list, exclude_masks
    unit_channels_np = np.asarray(spk_meta["unit_channels"]).astype(int)
    unit_channels_list = unit_channels_np.tolist()
    if not Path(banks_path).is_file():
        return unit_channels_list, bank_by_unit_list, exclude_masks
    cfg = load_banks_config(banks_path)
    bank_by_unit_list = list(map(str, map_channels_to_bank(unit_channels_np, cfg)))
    excl_hooke = excluded_channels_for_subject("M1", cfg)
    excl_hedy = excluded_channels_for_subject("M2", cfg)
    exclude_masks = {
        "M1": np.isin(unit_channels_np, np.fromiter(excl_hooke, dtype=int)).tolist(),
        "M2": np.isin(unit_channels_np, np.fromiter(excl_hedy, dtype=int)).tolist(),
    }
    return unit_channels_list, bank_by_unit_list, exclude_masks


def _resolve_logger_tags(
    session: SessionRef,
    logger_arg: str,
) -> List[str]:
    """Resolve which logger tags to process based on CLI --logger argument.

    Args:
        session: Loaded session YAML.
        logger_arg: One of 'auto', 'all', 'm1', 'm2'.

    Returns:
        List of logger tags to process (e.g., ['m1'] or ['m1', 'm2']).
    """
    if logger_arg in ("auto", "all"):
        tags = _available_loggers(session)
        if tags:
            return tags if logger_arg == "all" else tags[:1]
        # Fallback: try infer_logger or default to 'm1'
        inferred = infer_logger(session)
        return [inferred or "m1"]
    return [logger_arg]


@dataclass
class PlotConfig:
    """Configuration for QC plot generation.

    Attributes:
        active_threshold_hz: FR threshold for 'active' unit count plots.
        smooth_sigma_bins: Gaussian sigma (bins) for population FR smoothing.
        raster_max_units: Max units to show in raster plot.
        raster_window_s: Time window (s) for windowed raster; None = full.
        close_figs: Close matplotlib figures after saving.
    """

    active_threshold_hz: float = 0.1
    smooth_sigma_bins: float = 0.5
    raster_max_units: int = 50
    raster_window_s: Optional[float] = 600.0
    close_figs: bool = True


def _process_one_logger(
    t: np.ndarray,
    ui: np.ndarray,
    *,
    session: SessionRef,
    session_id: str,
    logger_tag: str,
    thr: Thresholds,
    outdir: Path,
    suffix: str,
    event_source: Optional[str] = None,
    spk_meta: Optional[dict] = None,
    spk_unit_ids: Optional[np.ndarray] = None,
    time_scale: float = 1.0,
    plot_cfg: PlotConfig,
) -> Dict[str, str]:
    """Process QC for one logger's spike data — the unified core pipeline.

    Steps:
        1. Apply time scaling (if != 1.0)
        2. Compute metrics via compute_qc_metrics_bundle
        3. ISI dedup for Unit_rasters sources
        4. Compute keep mask + breakdown
        5. Resolve bank/exclusion metadata
        6. Write QC JSON + keep JSON
        7. Generate all plots
        8. Write summary text

    Args:
        t: Spike times (s).
        ui: Unit indices.
        session: Loaded session YAML (for raster baseline detection).
        session_id: Session identifier string.
        logger_tag: Logger tag (e.g., 'm1', 'm2').
        thr: QC thresholds.
        outdir: Output directory (must exist).
        suffix: Filename suffix (e.g., '_m1' or '').
        event_source: Spike event source ('SpikeData', 'Unit_rasters', etc.).
        spk_meta: Spike metadata dict (for unit_channels).
        spk_unit_ids: Unit ID labels (for kept_unit_ids).
        time_scale: Multiply spike times by this factor (default 1.0).
        plot_cfg: Plot configuration.

    Returns:
        Dict mapping output names to file paths.
    """
    from .spikes import (
        deduplicate_isi_ties as _dedup_isi,
        compute_breakdown as _compute_breakdown,
    )

    if t.size == 0 or ui.size == 0:
        raise ValueError(f"Loaded spikes for {logger_tag} are empty; cannot perform QC")

    # 1. Apply time scaling
    if time_scale != 1.0:
        t = t * time_scale

    t_start = float(np.nanmin(t))
    t_end = float(np.nanmax(t))

    # 2. Compute core metrics
    metrics, bins = compute_qc_metrics_bundle(
        t, ui,
        isi_threshold_s=float(thr.isi_threshold_s),
        presence_bin_s=float(thr.presence_bin_s),
        fr_window_s=float(thr.fr_window_s),
        t_start=t_start,
        t_end=t_end,
    )

    # 3. ISI dedup for Unit_rasters (fixes inflated violation rates from ties)
    if isinstance(event_source, str) and event_source == "Unit_rasters":
        isi_rate_dedup = _dedup_isi(t, ui, isi_threshold_s=float(thr.isi_threshold_s))
        metrics = SpikesQCMetrics(
            mean_fr_hz=metrics.mean_fr_hz,
            isi_violation_rate=isi_rate_dedup,
            presence_ratio=metrics.presence_ratio,
            fr_stability_cv=metrics.fr_stability_cv,
        )

    # 4. Keep mask + breakdown
    keep_mask = compute_keep_mask(
        mean_fr_hz=metrics.mean_fr_hz,
        isi_violation_rate=metrics.isi_violation_rate,
        presence_ratio=metrics.presence_ratio,
        fr_stability_cv=metrics.fr_stability_cv,
        thr=thr,
    )
    kept_indices = np.where(keep_mask)[0].astype(int)

    # Kept unit IDs
    kept_unit_ids: List[str] = []
    if spk_unit_ids is not None and len(spk_unit_ids) >= kept_indices.size:
        kept_unit_ids = [str(spk_unit_ids[i]) for i in kept_indices]

    # Breakdown
    from .spikes import KeepThresholds as _KT
    _kt = _KT(
        min_mean_fr_hz=thr.min_mean_fr_hz,
        max_isi_violation_rate=thr.max_isi_violation_rate,
        min_presence_ratio=thr.min_presence_ratio,
        max_fr_stability_cv=thr.max_fr_stability_cv,
    )
    breakdown, exclusive_counts = _compute_breakdown(metrics, keep_mask, _kt)

    # 5. Bank/exclusion metadata
    unit_channels_list, bank_by_unit_list, exclude_masks = _resolve_bank_metadata(spk_meta)
    # Stash unit_channels on session for raster bank coloring
    if unit_channels_list:
        setattr(session, "_spike_unit_channels", np.asarray(unit_channels_list))

    # 6. Write QC JSON + keep JSON
    sid = session_id
    qc_json = {
        "session_id": sid,
        "logger": logger_tag,
        "n_units": int(np.max(ui) + 1),
        "t_start": t_start,
        "t_end": t_end,
        "dur_s": float(t_end - t_start),
        "thresholds": asdict(thr),
        "plot_params": {
            "active_threshold_hz": float(plot_cfg.active_threshold_hz),
            "smooth_sigma_bins": float(plot_cfg.smooth_sigma_bins),
            "raster_max_units": plot_cfg.raster_max_units,
            "raster_window_s": plot_cfg.raster_window_s,
            "raster_full": True,
        },
        "metrics": {
            "mean_fr_hz": metrics.mean_fr_hz.tolist(),
            "isi_violation_rate": metrics.isi_violation_rate.tolist(),
            "presence_ratio": metrics.presence_ratio.tolist(),
            "fr_stability_cv": np.where(
                np.isfinite(metrics.fr_stability_cv),
                metrics.fr_stability_cv,
                np.nan,
            ).tolist(),
        },
        "keep": {
            "mask": keep_mask.astype(bool).tolist(),
            "indices": kept_indices.tolist(),
            "unit_ids": kept_unit_ids,
            "unit_channels": unit_channels_list,
            "n_kept": int(kept_indices.size),
        },
        "breakdown": breakdown,
        "breakdown_exclusive_primary": exclusive_counts,
        "bins": {
            "bin_s": float(bins.bin_s),
            "centers": bins.centers.astype(float).tolist(),
        },
    }
    if time_scale != 1.0:
        qc_json["time_scale_applied"] = time_scale
    if bank_by_unit_list is not None:
        qc_json["bank_by_unit"] = bank_by_unit_list
    if exclude_masks is not None:
        qc_json["exclude_mask_by_subject"] = exclude_masks

    qc_json_path = outdir / f"qc_spikes_{sid}{suffix}.json"
    keep_json_path = outdir / f"keep_units_{sid}{suffix}.json"
    with qc_json_path.open("w", encoding="utf-8") as f:
        json.dump(qc_json, f, indent=2)
    with keep_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "session_id": sid,
            "logger": logger_tag,
            "indices": kept_indices.tolist(),
            "mask": keep_mask.astype(bool).tolist(),
            "unit_ids": kept_unit_ids,
            "unit_channels": unit_channels_list,
            "n_units": int(np.max(ui) + 1),
            "n_kept": int(kept_indices.size),
        }, f, indent=2)

    # 7. Generate all plots
    close = plot_cfg.close_figs
    subject = logger_tag.upper()
    plots_manifest: Dict[str, str] = {}
    errlog = outdir / f"qc_plot_error{suffix}.txt"
    if errlog.exists():
        errlog.unlink(missing_ok=True)

    heatmap_path = outdir / f"fr_heatmap_{sid}{suffix}.png"
    summary_path = outdir / f"qc_summary_{sid}{suffix}.png"
    active_path = outdir / f"active_units_{sid}{suffix}.png"
    popfr_path = outdir / f"population_fr_{sid}{suffix}.png"
    hist_path = outdir / f"fr_hist_{sid}{suffix}.png"
    breakdown_path = outdir / f"drop_breakdown_{sid}{suffix}.png"
    raster_path = outdir / f"raster_kept_{sid}{suffix}.png"
    rasterfull_path = outdir / f"raster_kept_full_{sid}{suffix}.png"
    scatter_path = outdir / f"qc_scatter_{sid}{suffix}.png"

    _try_plot(
        lambda: _plot_fr_heatmap(bins.fr, bins.centers, metrics.mean_fr_hz, keep_mask, out_path=heatmap_path, close_fig=close, subject=subject),
        name="fr_heatmap", manifest=plots_manifest, path=heatmap_path, errlog=errlog,
    )
    _try_plot(
        lambda: _plot_summary_panels(
            bins.fr, bins.centers, metrics.mean_fr_hz, keep_mask,
            active_threshold_hz=float(plot_cfg.active_threshold_hz),
            smooth_sigma_bins=float(plot_cfg.smooth_sigma_bins),
            breakdown=breakdown, out_path=summary_path, close_fig=close, subject=subject,
        ),
        name="qc_summary", manifest=plots_manifest, path=summary_path, errlog=errlog,
    )
    _try_plot(
        lambda: _plot_active_units(bins.fr, bins.centers, threshold_hz=float(plot_cfg.active_threshold_hz), out_path=active_path, close_fig=close, subject=subject),
        name="active_units", manifest=plots_manifest, path=active_path, errlog=errlog,
    )
    _try_plot(
        lambda: _plot_population_fr(bins.fr, bins.centers, smooth_sigma_bins=float(plot_cfg.smooth_sigma_bins), out_path=popfr_path, close_fig=close, subject=subject),
        name="population_fr", manifest=plots_manifest, path=popfr_path, errlog=errlog,
    )
    _try_plot(
        lambda: _plot_fr_hist(metrics.mean_fr_hz, keep_mask, out_path=hist_path, close_fig=close, subject=subject),
        name="fr_hist", manifest=plots_manifest, path=hist_path, errlog=errlog,
    )
    _try_plot(
        lambda: _plot_drop_breakdown(breakdown, out_path=breakdown_path, breakdown_exclusive=exclusive_counts, close_fig=close, subject=subject),
        name="drop_breakdown", manifest=plots_manifest, path=breakdown_path, errlog=errlog,
    )

    events_text_path = _resolve_events_text(session)
    banks_yaml = "configs/banks.yaml" if Path("configs/banks.yaml").is_file() else None
    _try_plot(
        lambda: _raster_kept_units(
            t, ui, keep_mask, session=session, logger_tag=logger_tag,
            out_path=raster_path, max_units=plot_cfg.raster_max_units,
            banks_yaml=banks_yaml, events_text_path=events_text_path,
            window_s=plot_cfg.raster_window_s, close_fig=close,
        ),
        name="raster_kept", manifest=plots_manifest, path=raster_path, errlog=errlog,
    )
    _try_plot(
        lambda: _raster_kept_units(
            t, ui, keep_mask, session=session, logger_tag=logger_tag,
            out_path=rasterfull_path, max_units=plot_cfg.raster_max_units,
            banks_yaml=banks_yaml, events_text_path=events_text_path,
            window_s=None, close_fig=close,
        ),
        name="raster_kept_full", manifest=plots_manifest, path=rasterfull_path, errlog=errlog,
    )

    # Scatter panels need ISI rate at current threshold
    isi_extra = compute_isi_metrics(t, ui, refractory_threshold_s=float(thr.isi_threshold_s))
    isi_rate_scatter = np.asarray(isi_extra["refractory_violation_rate"], dtype=float)
    _try_plot(
        lambda: _plot_scatter_panels(
            metrics.mean_fr_hz, metrics.presence_ratio, metrics.fr_stability_cv,
            isi_rate_scatter, keep_mask,
            out_path=scatter_path, close_fig=close, subject=subject,
        ),
        name="qc_scatter", manifest=plots_manifest, path=scatter_path, errlog=errlog,
    )

    # Update QC JSON with plots manifest
    qc_json["plots"] = plots_manifest
    with qc_json_path.open("w", encoding="utf-8") as f:
        json.dump(qc_json, f, indent=2)

    # 8. Summary text
    summary_txt = outdir / f"qc_summary_{sid}{suffix}.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"Session: {sid}\n")
        f.write(f"Logger: {logger_tag}\n")
        f.write(f"Units kept: {breakdown['kept']} / {breakdown['total']}\n")
        f.write("Drop breakdown:\n")
        for k in ("low_mean_fr", "high_isi_violation", "low_presence", "high_fr_cv", "cv_nan"):
            f.write(f"  {k}: {breakdown.get(k, 0)}\n")
        f.write("Drop breakdown (exclusive primary):\n")
        for k in ("low_mean_fr", "high_isi_violation", "low_presence", "high_fr_cv", "cv_nan"):
            f.write(f"  {k}: {exclusive_counts.get(k, 0)}\n")
        f.write("Plots:\n")
        for name, path in plots_manifest.items():
            f.write(f"  {name}: {path}\n")

    return {
        "qc_json": str(qc_json_path),
        "keep_json": str(keep_json_path),
        "fr_heatmap": str(heatmap_path),
        "qc_summary": str(summary_path),
        "active_units": str(active_path),
        "population_fr": str(popfr_path),
        "fr_hist": str(hist_path),
        "drop_breakdown": str(breakdown_path),
        "raster_kept": str(raster_path),
        "raster_kept_full": str(rasterfull_path),
        "qc_scatter": str(scatter_path),
    }


def run_qc_pipeline(
    config_path: str,
    *,
    outdir: Optional[str] = None,
    thresholds: Optional[Thresholds] = None,
    logger_tags: Optional[List[str]] = None,
    time_scale: float = 1.0,
    diagnose_timebase: bool = False,
    plot_cfg: Optional[PlotConfig] = None,
) -> Tuple[Path, Dict[str, Dict[str, str]]]:
    """Run QC pipeline for one or more loggers in a session.

    This is the unified entry point replacing the old run_qc_spikes() and
    run_qc_spikes_multi() functions.

    Args:
        config_path: Path to per-session YAML.
        outdir: Output directory override. Defaults from session YAML.
        thresholds: QC thresholds. Defaults to Thresholds().
        logger_tags: Logger tags to process (e.g., ['m1', 'm2']).
            If None, auto-detects from session YAML.
        time_scale: Multiply spike times by this factor (default 1.0).
        diagnose_timebase: Print per-logger timebase diagnostics.
        plot_cfg: Plot configuration. Defaults to PlotConfig().

    Returns:
        (manifest_path, per_logger_outputs) where per_logger_outputs maps
        each logger tag to its output file paths.
    """
    session = load_session_yaml(config_path)
    thr = thresholds or Thresholds()
    pcfg = plot_cfg or PlotConfig()
    base_out = _resolve_outdir(session, outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    if logger_tags is None:
        logger_tags = _resolve_logger_tags(session, "auto")

    use_suffix = len(logger_tags) > 1

    all_outputs: Dict[str, Dict[str, str]] = {}
    for tag in logger_tags:
        spk = _load_spikes_for_logger(session, tag)
        t = np.asarray(spk.t, dtype=float).reshape(-1)
        ui = np.asarray(spk.unit_index, dtype=int).reshape(-1)

        # Extract metadata
        event_source = None
        if isinstance(spk.meta, dict):
            event_source = spk.meta.get("event_source")
        spk_unit_ids = getattr(spk, "unit_ids", None)

        # Optional timebase diagnostics
        if diagnose_timebase:
            _print_timebase_diag(session.session_id, tag, t, ui)

        suffix = f"_{tag}" if use_suffix else ""
        outputs = _process_one_logger(
            t, ui,
            session=session,
            session_id=session.session_id,
            logger_tag=tag,
            thr=thr,
            outdir=base_out,
            suffix=suffix,
            event_source=event_source,
            spk_meta=spk.meta if isinstance(spk.meta, dict) else None,
            spk_unit_ids=spk_unit_ids,
            time_scale=time_scale,
            plot_cfg=pcfg,
        )
        all_outputs[tag] = outputs

    # Write combined manifest
    combo_path = base_out / f"qc_session_{session.session_id}_multi.json"
    with combo_path.open("w", encoding="utf-8") as f:
        json.dump({
            "session_id": session.session_id,
            "loggers": logger_tags,
            "outputs": all_outputs,
        }, f, indent=2)

    return combo_path, all_outputs


def _print_timebase_diag(
    session_id: str,
    tag: str,
    t: np.ndarray,
    ui: np.ndarray,
) -> None:
    """Print timebase diagnostic info for one logger."""
    U = int(ui.max()) + 1 if ui.size else 0
    sample: List[dict] = []
    cnt = 0
    for u in range(U):
        m = (ui == u)
        n = int(np.sum(m))
        if n < 200:
            continue
        tt = np.sort(t[m])
        if tt.size < 3:
            continue
        keep = np.ones(tt.size, dtype=bool)
        keep[1:] = tt[1:] > tt[:-1]
        ttu = tt[keep]
        ties = int(tt.size - ttu.size)
        if ttu.size < 3:
            continue
        isi = np.diff(ttu)
        if isi.size == 0:
            continue
        sample.append({"unit": u, "n_spikes": n, "isi_med": float(np.median(isi)), "ties": ties})
        cnt += 1
        if cnt >= 10:
            break
    print(_fmt_timebase_diag(f"{session_id}:{tag}", sample))


# ---------------------------------------------------------------------------
# Public API — backward-compatible wrappers
# ---------------------------------------------------------------------------


def run_qc_spikes(
    config_path: str,
    *,
    outdir: Optional[str] = None,
    thresholds: Optional[Thresholds] = None,
    plot_active_threshold_hz: float = 0.1,
    plot_smooth_sigma_bins: float = 0.5,
    close_figs: bool = True,
) -> Tuple[Path, Dict[str, str]]:
    """Compute metrics and write outputs for one session (single logger).

    Thin wrapper around :func:`run_qc_pipeline` that preserves the legacy
    return signature ``(qc_json_path, plots_manifest)``.

    Args:
        config_path: Path to per-session YAML.
        outdir: Optional output directory. Defaults to results/<session_id>/qc.
        thresholds: Optional thresholds override.
        plot_active_threshold_hz: FR threshold for 'active' unit count plots.
        plot_smooth_sigma_bins: Gaussian sigma (bins) for pop FR smoothing.
        close_figs: Close matplotlib figures after saving.

    Returns:
        (qc_json_path, plots_manifest) for the single logger processed.
    """
    # Auto-detect single logger
    session = load_session_yaml(config_path)
    tags = _resolve_logger_tags(session, "auto")[:1]

    pcfg = PlotConfig(
        active_threshold_hz=plot_active_threshold_hz,
        smooth_sigma_bins=plot_smooth_sigma_bins,
        close_figs=close_figs,
    )
    _manifest_path, per_logger = run_qc_pipeline(
        config_path,
        outdir=outdir,
        thresholds=thresholds,
        logger_tags=tags,
        plot_cfg=pcfg,
    )
    # Extract the single logger's outputs
    tag = tags[0]
    outputs = per_logger[tag]
    qc_json_path = Path(outputs["qc_json"])
    plots_manifest = {k: v for k, v in outputs.items() if k not in ("qc_json", "keep_json")}
    return qc_json_path, plots_manifest


def run_qc_spikes_multi(
    config_path: str,
    *,
    outdir: Optional[str] = None,
    thresholds: Optional[Thresholds] = None,
    plot_active_threshold_hz: float = 0.1,
    plot_smooth_sigma_bins: float = 0.5,
    logger: Optional[str] = None,
    close_figs: bool = True,
) -> Tuple[Path, Dict[str, str]]:
    """Multi-logger wrapper — delegates to :func:`run_qc_pipeline`.

    Preserves legacy return signature ``(combo_manifest_path, {})``.

    Args:
        config_path: Path to per-session YAML.
        outdir: Optional output directory override.
        thresholds: Optional thresholds override.
        plot_active_threshold_hz: FR threshold for 'active' unit count plots.
        plot_smooth_sigma_bins: Gaussian sigma (bins) for pop FR smoothing.
        logger: Logger selection ('m1', 'm2', 'all', 'auto', or None).
        close_figs: Close matplotlib figures after saving.

    Returns:
        (combo_manifest_path, {}) matching legacy contract.
    """
    logger_arg = logger or "auto"
    session = load_session_yaml(config_path)
    tags = _resolve_logger_tags(session, logger_arg)

    pcfg = PlotConfig(
        active_threshold_hz=plot_active_threshold_hz,
        smooth_sigma_bins=plot_smooth_sigma_bins,
        close_figs=close_figs,
    )
    combo_path, _per_logger = run_qc_pipeline(
        config_path,
        outdir=outdir,
        thresholds=thresholds,
        logger_tags=tags,
        plot_cfg=pcfg,
    )
    return combo_path, {}


def _fmt_timebase_diag(name: str, stats: List[dict]) -> str:
    """Format a concise, human-readable timebase diagnostic for stdout.

    Reports median ISI (s), ties (exact-duplicate timestamps), and suggests
    plausible scale factors to bring median ISI into a physiological range
    (~2 ms to 2 s).
    """
    import math

    if not stats:
        return f"[timebase] {name}: insufficient spikes for diagnosis"
    med_med = float(np.median([s["isi_med"] for s in stats if np.isfinite(s.get("isi_med", np.nan))]))
    ties_total = int(sum(int(s.get("ties", 0)) for s in stats))
    # Candidate scales to test
    candidates = [
        ("as_is", 1.0, med_med),
        ("x1000", 1000.0, med_med * 1000.0),
        ("/1000", 0.001, med_med * 0.001),
        ("x30000", 30000.0, med_med * 30000.0),
        ("/30000", 1.0 / 30000.0, med_med / 30000.0),
        ("x1e6", 1e6, med_med * 1e6),
        ("/1e6", 1e-6, med_med * 1e-6),
    ]
    def plausible(x: float) -> bool:
        # Physiological ISI median range heuristic: ~2 ms – 2 s
        return (not math.isnan(x)) and (0.002 <= x <= 2.0)
    ok = [(n, s, v) for (n, s, v) in candidates if plausible(v)]
    parts = [
        f"median_isi={med_med:.6f}s",
        f"ties={ties_total}",
        "suggest=" + ", ".join([f"{n}(x{s:g} -> {v:.4f}s)" for (n, s, v) in ok]) if ok else "suggest=none",
    ]
    return f"[timebase] {name}: " + " | ".join(parts)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute spike QC metrics and write keep lists for a session")
    ap.add_argument("--config", required=True, help="Path to per-session YAML")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <yaml data_files.output_dir>/qc or results/<session_id>/qc)",
    )
    ap.add_argument("--min-mean-fr", type=float, default=Thresholds.min_mean_fr_hz, help="Min mean FR (Hz)")
    ap.add_argument("--max-isi-violation", type=float, default=Thresholds.max_isi_violation_rate, help="Max ISI violation rate (fraction)")
    ap.add_argument("--min-presence", type=float, default=Thresholds.min_presence_ratio, help="Min presence ratio (fraction)")
    ap.add_argument("--max-fr-cv", type=float, default=Thresholds.max_fr_stability_cv, help="Max FR stability CV (std/mean)")
    ap.add_argument("--presence-bin-s", type=float, default=Thresholds.presence_bin_s, help="Presence ratio bin size (s)")
    ap.add_argument("--fr-window-s", type=float, default=Thresholds.fr_window_s, help="Long bin size for FR heatmap/stability (s)")
    ap.add_argument("--isi-threshold-s", type=float, default=Thresholds.isi_threshold_s, help="ISI refractory threshold (s)")
    ap.add_argument("--no-show", action="store_true", help="Do not show plots after saving")
    ap.add_argument("--plot-active-threshold-hz", type=float, default=0.1, help="Threshold for 'active units over time' plot")
    ap.add_argument("--plot-smooth-sigma-bins", type=float, default=0.5, help="Gaussian sigma (bins) for population FR smoothing in plots")
    ap.add_argument("--logger", choices=["m1", "m2", "all", "auto"], default="auto", help="Which logger to QC (default: auto; 'all' runs both if present)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="Multiply spike times by this factor before QC (use to correct timebase mismatches)")
    ap.add_argument("--diagnose-timebase", action="store_true", help="Print per-logger timebase diagnostics (median ISI, ties, suggested scale)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point — delegates to :func:`run_qc_pipeline`."""
    ap = _build_argparser()
    args = ap.parse_args(argv)

    thr = Thresholds(
        min_mean_fr_hz=float(args.min_mean_fr),
        max_isi_violation_rate=float(args.max_isi_violation),
        min_presence_ratio=float(args.min_presence),
        max_fr_stability_cv=float(args.max_fr_cv),
        presence_bin_s=float(args.presence_bin_s),
        fr_window_s=float(args.fr_window_s),
        isi_threshold_s=float(args.isi_threshold_s),
    )

    session = load_session_yaml(args.config)
    tags = _resolve_logger_tags(session, args.logger)

    pcfg = PlotConfig(
        active_threshold_hz=float(args.plot_active_threshold_hz),
        smooth_sigma_bins=float(args.plot_smooth_sigma_bins),
        close_figs=bool(args.no_show),
    )

    combo_path, per_logger = run_qc_pipeline(
        args.config,
        outdir=args.outdir,
        thresholds=thr,
        logger_tags=tags,
        time_scale=float(args.time_scale),
        diagnose_timebase=bool(args.diagnose_timebase),
        plot_cfg=pcfg,
    )

    print(f"[qc_spikes] wrote {combo_path}")
    for tag, outputs in per_logger.items():
        print(f"[qc_spikes] {tag}: {outputs.get('qc_json', '')}")

    if not args.no_show:
        plt.show()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
