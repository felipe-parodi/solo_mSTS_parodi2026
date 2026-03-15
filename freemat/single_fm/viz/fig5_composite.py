"""Figure 5 composite: 2-row panel assembly for Nature Communications.

Assembles 7 panels (A-G) into a single publication-ready figure using
nested subgridspecs (outer 2x1 for rows, inner grids per row). Each
panel is drawn directly onto composite axes via local ``_draw_*``
helpers (no intermediate Figure objects).

Layout::

    Row 0 (taller): [A: schematic]       [B: CEBRA-SL]       [C: CEBRA-B]
    Row 1:          [D: height decode]   [E: syl decode]      [F: context]    [G: drift]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from freemat.single_fm.viz.pub_data import (
    ALL_SESSIONS,
    HEIGHT_COLORS_FIG5,
    HEIGHT_NAMES,
    METHOD_COLORS,
    MOTIF_NAMES_FIG5,
    SUBJECT_COLORS,
    SUBJECT_MAP,
    setup_nature_style,
)

_LEGEND_KW: Dict[str, Any] = dict(
    facecolor="#D9D9D9", edgecolor="#999999", framealpha=0.95,
)

# Default schematic path (relative to project root)
_SCHEMATIC_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "results" / "single_fm" / "publication_figures"
    / "main" / "fig5" / "fig5_cebra_schematic.png"
)


# ── Panel label helper ──────────────────────────────────────────────────

def _add_scale_bar(
    ax: plt.Axes,
    length: float = 0.15,
    x0: float = 0.05,
    y0: float = 0.05,
    fontsize: int = 6,
) -> None:
    """Add minimal L-shaped scale indicator for embedding axes."""
    kw = dict(transform=ax.transAxes, color="black", linewidth=1.2, clip_on=False)
    ax.plot([x0, x0 + length], [y0, y0], **kw)  # horizontal
    ax.plot([x0, x0], [y0, y0 + length], **kw)  # vertical
    ax.text(x0 + length / 2, y0 - 0.03, "Dim 1", fontsize=fontsize,
            ha="center", va="top", transform=ax.transAxes)
    ax.text(x0 - 0.03, y0 + length / 2, "Dim 2", fontsize=fontsize,
            ha="right", va="center", transform=ax.transAxes, rotation=90)


def _thin_yticks(ax: plt.Axes, nbins: int = 4) -> None:
    """Reduce y-axis ticks to ~nbins for visual clarity."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, integer=False))


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    x_offset: float = -0.01,
    y_pad: float = 0.005,
    force_y: Optional[float] = None,
) -> None:
    """Add bold uppercase panel label (A, B, ...) at top-left of axes.

    Uses the actual axes position (after ``fig.canvas.draw()``) so that
    labels track the constrained box, not the gridspec cell.

    Args:
        ax: Target axes.
        label: Panel letter.
        x_offset: Horizontal offset from axes left edge (fig coords).
        y_pad: Vertical padding above the reference y position.
        force_y: If provided, use this as the y-reference instead of
            the axes top. Useful for row-level label alignment.
    """
    fig = ax.get_figure()
    pos = ax.get_position()
    fig_x = pos.x0 + x_offset
    fig_y = (force_y if force_y is not None else pos.y1) + y_pad
    fig.text(
        fig_x, fig_y, label,
        fontsize=16, fontweight="bold", va="bottom", ha="right",
    )


def _align_row_titles(row_axes: List[plt.Axes], fig: plt.Figure) -> None:
    """Align titles in a row to the highest axes top in figure coords.

    After ``set_box_aspect`` constrains non-square panels, their tops sit
    lower than square panels in the same row.  This helper adjusts each
    title's y-position so all titles in the row land at the same height.

    Args:
        row_axes: Axes objects belonging to one gridspec row.
        fig: Parent figure (used for coordinate computation).
    """
    fig.canvas.draw()
    tops = [ax.get_position().y1 for ax in row_axes]
    max_top = max(tops)
    for ax in row_axes:
        pos = ax.get_position()
        ax_height = pos.y1 - pos.y0
        if ax_height > 0:
            title_y = 1.0 + (max_top - pos.y1) / ax_height
            ax.title.set_position((0.5, title_y))


# ── Panel A: Schematic image ─────────────────────────────────────────────

def _draw_panel_a(
    ax: plt.Axes,
    schematic_path: Optional[Path] = None,
) -> None:
    """Embed CEBRA schematic image in Panel A."""
    path = schematic_path or _SCHEMATIC_PATH
    if path.exists():
        img = mpimg.imread(str(path))
        # Crop transparent/empty margins so content fills the axes
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            rows = np.any(alpha > 0.05, axis=1)
            cols = np.any(alpha > 0.05, axis=0)
            if rows.any() and cols.any():
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                img = img[r0:r1 + 1, c0:c1 + 1]
        ax.imshow(img, aspect="equal")
    else:
        ax.text(
            0.5, 0.5, "Schematic\n(not found)",
            ha="center", va="center", fontsize=11,
            color="gray", fontstyle="italic",
            transform=ax.transAxes,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_facecolor("none")


# ── Panel B: Height 2D embedding (CEBRA-Spatial-lite) ─────────────────────

def _draw_panel_b_height_2d(
    ax: plt.Axes,
    emb_spatial_3d: np.ndarray,
    height_tier: np.ndarray,
    y_motif: Optional[np.ndarray] = None,
) -> None:
    """Draw 2D height embedding scatter (Spatial-lite, dims 1 vs 2)."""
    mask = np.ones(len(height_tier), dtype=bool)
    if y_motif is not None:
        mask = y_motif > 0

    for tier in sorted(HEIGHT_COLORS_FIG5.keys()):
        tier_mask = mask & (height_tier == tier)
        if tier_mask.sum() == 0:
            continue
        ax.scatter(
            emb_spatial_3d[tier_mask, 0],
            emb_spatial_3d[tier_mask, 1],
            c=HEIGHT_COLORS_FIG5[tier], s=0.5, alpha=0.3,
            label=HEIGHT_NAMES[tier], rasterized=True,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("CEBRA-Spatial-lite", fontsize=10, fontweight="bold")
    ax.legend(markerscale=6, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=3, fontsize=6, frameon=False)
    _add_scale_bar(ax)


# ── Panel C: Motif 2D embedding (CEBRA-Behavior) ─────────────────────────

def _draw_panel_c_motif_2d(
    ax: plt.Axes,
    emb_behavior_3d: np.ndarray,
    y_motif: np.ndarray,
    motif_names: Optional[Dict[int, str]] = None,
) -> None:
    """Draw 2D motif embedding scatter (Behavior, dims 1 vs 2)."""
    if motif_names is None:
        motif_names = MOTIF_NAMES_FIG5

    unique_motifs = sorted(set(y_motif[y_motif > 0]))
    cmap = plt.colormaps["tab10"]

    for i, m in enumerate(unique_motifs):
        m_mask = y_motif == m
        label = motif_names.get(m, f"M{m}")
        ax.scatter(
            emb_behavior_3d[m_mask, 0], emb_behavior_3d[m_mask, 1],
            c=[cmap(i % 10)], s=0.5, alpha=0.3, label=label, rasterized=True,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("CEBRA-Behavior", fontsize=10, fontweight="bold")
    if len(unique_motifs) <= 12:
        ax.legend(markerscale=6, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                  ncol=4, fontsize=5, frameon=False)
    _add_scale_bar(ax)


# ── Summary bars helper (mean bar + session dots) ─────────────────────────

def _draw_summary_bars(
    ax: plt.Axes,
    values_by_method: List[Tuple[str, List[float]]],
    sessions: List[str],
    title: str,
    ylabel: str = "Balanced accuracy",
    chance: Optional[float] = None,
    ylim_top: float = 1.0,
    ref_line: Optional[float] = None,
) -> None:
    """Draw mean bars with overlaid per-session dots.

    Args:
        ax: Target axes.
        values_by_method: [(method_name, [val_per_session]), ...].
        sessions: Session keys (for subject coloring of dots).
        title: Panel title.
        ylabel: Y-axis label.
        chance: Draw dashed chance line at this value.
        ylim_top: Upper y-axis limit.
        ref_line: Draw a reference line (e.g. 1.0 for context ratio).
    """
    n_methods = len(values_by_method)
    x = np.arange(n_methods) * 0.5  # tight bar spacing
    width = 0.25
    rng = np.random.default_rng(42)

    for j, (method_name, vals) in enumerate(values_by_method):
        vals_arr = np.array(vals, dtype=float)
        valid = vals_arr[~np.isnan(vals_arr)]
        mean_val = np.nanmean(vals_arr)
        sem_val = np.nanstd(vals_arr) / np.sqrt(len(valid)) if len(valid) > 1 else 0

        color = METHOD_COLORS.get(method_name, "#999999")
        ax.bar(
            x[j], mean_val, width, color=color, alpha=0.75,
            edgecolor="white", linewidth=0.5,
            yerr=sem_val, capsize=3, error_kw={"linewidth": 1.0},
        )

        # Overlay per-session dots with jitter
        for k, (sess, val) in enumerate(zip(sessions, vals)):
            if np.isnan(val):
                continue
            subject = SUBJECT_MAP.get(sess, sess)
            dot_color = SUBJECT_COLORS.get(subject, "#555555")
            jitter = rng.uniform(-0.12, 0.12)
            ax.scatter(
                x[j] + jitter, val,
                c=dot_color, s=18, zorder=5,
                edgecolors="white", linewidths=0.3, alpha=0.85,
            )

    if chance is not None:
        ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Chance ({chance:.0%})", zorder=0)
        ax.legend(fontsize=6, loc="lower right", **_LEGEND_KW)

    if ref_line is not None:
        ax.axhline(ref_line, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in values_by_method], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, ylim_top)
    ax.set_xlim(-0.25, x[-1] + 0.25)
    ax.tick_params(labelsize=7)
    _thin_yticks(ax)


# ── Data extraction helpers ───────────────────────────────────────────────

def _extract_decode_values(
    fig5_results: Dict[str, Any],
    sessions: List[str],
    metric_key: str,
    method_keys: List[Tuple[str, str]],
) -> List[Tuple[str, List[float]]]:
    """Extract per-session accuracy for each method.

    Args:
        fig5_results: Per-session results dict.
        sessions: Session keys.
        metric_key: e.g. "height_3class" or "motif".
        method_keys: [(json_key, display_name), ...].

    Returns:
        [(display_name, [val_per_session]), ...].
    """
    result = []
    for json_key, display_name in method_keys:
        vals = []
        for sess in sessions:
            r = fig5_results[sess]
            decode = r.get("decode_results", {}).get(metric_key, {})
            method_result = decode.get(json_key, {})
            if "knn_ba" in method_result:
                vals.append(method_result["knn_ba"])
            elif "logistic_ba" in method_result:
                vals.append(method_result["logistic_ba"])
            else:
                vals.append(np.nan)
        result.append((display_name, vals))
    return result


def _extract_context_values(
    fig5_results: Dict[str, Any],
    sessions: List[str],
    min_qualifying_motifs: int = 2,
) -> Tuple[List[Tuple[str, List[float]]], List[str]]:
    """Extract per-session context ratios for CEBRA and PCA.

    Returns:
        (values_by_method, filtered_sessions).
    """
    filtered = []
    for sess in sessions:
        r = fig5_results[sess]
        cc = r.get("context_cebra", {})
        motif_set = {sr.get("motif", 0) for sr in cc.get("syllable_results", [])}
        if len(motif_set) >= min_qualifying_motifs:
            filtered.append(sess)

    cebra_vals = []
    pca_vals = []
    for sess in filtered:
        r = fig5_results[sess]
        cebra_vals.append(r.get("context_cebra", {}).get("context_ratio", np.nan))
        pca_vals.append(r.get("context_pca", {}).get("context_ratio", np.nan))

    return [("CEBRA", cebra_vals), ("PCA-8D", pca_vals)], filtered


# ── Panel G: Anticipatory drift ─────────────────────────────────────────

def _draw_panel_g_drift(
    ax: plt.Axes,
    manifold_results: List[Dict[str, Any]],
) -> None:
    """Draw anticipatory drift cosine similarity curves."""
    if not manifold_results:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    all_next = []
    all_current = []
    time_axis = None

    for r in manifold_results:
        ad = r["anticipatory_drift"]
        if not ad["time_axis"]:
            continue
        t = np.array(ad["time_axis"])
        sim_next = np.array(ad["mean_sim_next"])
        sim_current = np.array(ad["mean_sim_current"])
        if time_axis is None:
            time_axis = t
        ax.plot(t, sim_next, color="#E63946", alpha=0.2, linewidth=0.6)
        ax.plot(t, sim_current, color="#457B9D", alpha=0.2, linewidth=0.6)
        all_next.append(sim_next)
        all_current.append(sim_current)

    if all_next and time_axis is not None:
        ax.plot(time_axis, np.mean(all_next, axis=0),
                color="#E63946", linewidth=2, label="Next syllable")
        ax.plot(time_axis, np.mean(all_current, axis=0),
                color="#457B9D", linewidth=2, label="Current syllable")

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Cosine sim.", fontsize=9)
    ax.set_title("Anticipatory\ndrift", fontsize=10, fontweight="bold")
    ax.legend(fontsize=5, loc="lower left", **_LEGEND_KW)
    ax.tick_params(labelsize=7)
    _thin_yticks(ax)


# ── Panel H: Velocity at transitions + AUC annotation ───────────────────

def _draw_panel_h_velocity(
    ax: plt.Axes,
    fig5_results: Dict[str, Any],
    sessions: List[str],
    manifold_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Draw velocity ratio profile overlaying all sessions with AUC annotation.

    Thin per-session lines + bold mean. AUC-ROC from manifold results
    is displayed as a text annotation.
    """
    all_ratios = []
    time_axis = None

    for sess in sessions:
        dyn = fig5_results[sess].get("dynamics_cebra", {})
        if not dyn.get("mean_velocity"):
            continue
        t = np.array(dyn["time_axis"])
        mean_vel = np.array(dyn["mean_velocity"])
        baseline = dyn["baseline_velocity"]
        ratio = mean_vel / baseline if baseline > 0 else mean_vel

        if time_axis is None:
            time_axis = t

        subject = SUBJECT_MAP.get(sess, sess)
        color = SUBJECT_COLORS.get(subject, "#555555")
        ax.plot(t, ratio, color=color, alpha=0.3, linewidth=0.8)
        all_ratios.append(ratio)

    # Bold mean across sessions
    if all_ratios and time_axis is not None:
        mean_ratio = np.mean(all_ratios, axis=0)
        ax.plot(time_axis, mean_ratio, color=METHOD_COLORS["CEBRA"],
                linewidth=2, label=f"Mean (n={len(all_ratios)})")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Transition")

    # AUC annotation from manifold results
    if manifold_results:
        aucs = [r["velocity_prediction"]["auc_roc"]
                for r in manifold_results
                if "velocity_prediction" in r]
        if aucs:
            mean_auc = np.mean(aucs)
            sem_auc = np.std(aucs) / np.sqrt(len(aucs))
            ax.text(
                0.97, 0.05,
                f"AUC = {mean_auc:.2f} \u00b1 {sem_auc:.2f}",
                transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                bbox=dict(facecolor="#D9D9D9", alpha=0.8, edgecolor="#999999",
                          boxstyle="round,pad=0.3"),
            )
    elif all_ratios:
        # Fallback: annotate mean peak ratio
        peak_ratios = [r.max() for r in all_ratios]
        mean_peak = np.mean(peak_ratios)
        ax.text(0.03, 0.93, f"mean peak={mean_peak:.2f}",
                transform=ax.transAxes, fontsize=7, va="top")

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Vel. ratio", fontsize=9)
    ax.set_title("Velocity at transitions", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", **_LEGEND_KW)
    ax.tick_params(labelsize=7)
    _thin_yticks(ax)


# ══════════════════════════════════════════════════════════════════════════
# Main composite assembly — 3-row, 9-panel layout
# ══════════════════════════════════════════════════════════════════════════

def fig5_composite(
    fig5_results: Dict[str, Any],
    fig5_embeddings: Dict[str, np.ndarray],
    manifold_results: List[Dict[str, Any]],
    primary_session: str = "M2_0526",
    figsize: tuple = (7.1, 6.0),
    min_qualifying_motifs: int = 2,
    schematic_path: Optional[Path] = None,
) -> plt.Figure:
    """Assemble Figure 5 as a 2-row, 7-panel composite.

    Layout::

        Row 0: [A]  [B]  [C]
        Row 1: [D]  [E]  [F]  [G]

    Args:
        fig5_results: Per-session results from ``load_fig5_results()``.
        fig5_embeddings: Per-session embeddings from ``load_fig5_embeddings()``.
        manifold_results: From ``load_manifold_results()``.
        primary_session: Session for embedding panels.
        figsize: Figure size in inches (default 7.1 x 6.0, NatComm max).
        min_qualifying_motifs: For context coding panel.
        schematic_path: Path to schematic PNG (default: auto-detected).

    Returns:
        Assembled Figure with 7 panels (A through G).
    """
    setup_nature_style()

    fig = plt.figure(figsize=figsize)

    # Outer grid: 2 rows
    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.2, 1],
        hspace=-0.1,
        left=0.08, right=0.93, top=0.97, bottom=0.05,
    )

    # Row 0: 3 cols — A (wider), B, C (embeddings)
    row0 = outer[0].subgridspec(1, 3, wspace=0.5, width_ratios=[1.5, 1, 1])
    # Row 1: 4 cols — D, E, F (bar plots) + G (drift, equal width)
    row1 = outer[1].subgridspec(1, 4, wspace=0.55,
                                 width_ratios=[1, 1, 1, 1])

    sessions = [s for s in ALL_SESSIONS if s in fig5_results]
    emb = fig5_embeddings.get(primary_session, {})

    # ═══ Row 0: A (schematic) + B (CEBRA-SL) + C (CEBRA-B) ═══

    ax_a = fig.add_subplot(row0[0, 0])
    _draw_panel_a(ax_a, schematic_path)

    ax_b = fig.add_subplot(row0[0, 1])
    ax_b.set_box_aspect(0.85)
    if emb:
        _draw_panel_b_height_2d(
            ax_b, emb["emb_spatial_viz_3d"],
            emb["height_tier"], y_motif=emb.get("y_motif"),
        )

    ax_c = fig.add_subplot(row0[0, 2])
    ax_c.set_box_aspect(0.85)
    if emb:
        _draw_panel_c_motif_2d(
            ax_c, emb["emb_behavior_viz_3d"], emb["y_motif"],
        )

    # ═══ Row 1: D (height) + E (syllable) + F (context) + G (drift) ═══

    # Panel D: Height decoding
    height_values = _extract_decode_values(
        fig5_results, sessions, "height_3class",
        [("CEBRA-SpatialLite-8D kNN", "CEBRA"),
         ("PCA-8D kNN", "PCA-8D"),
         ("Raw logistic", "Raw")],
    )
    height_chances = []
    for sess in sessions:
        decode = fig5_results[sess].get("decode_results", {}).get("height_3class", {})
        first = list(decode.values())[0] if decode else {}
        height_chances.append(first.get("chance", 1 / 3))
    height_chance = np.nanmean(height_chances)

    ax_d = fig.add_subplot(row1[0, 0])
    ax_d.set_box_aspect(1.0)
    _draw_summary_bars(
        ax_d, height_values, sessions,
        title="Height (3-class)", ylabel="Bal. accuracy",
        chance=height_chance, ylim_top=1.0,
    )
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Panel E: Syllable decoding
    syl_values = _extract_decode_values(
        fig5_results, sessions, "motif",
        [("CEBRA-Behavior-8D kNN", "CEBRA"),
         ("PCA-8D kNN", "PCA-8D"),
         ("Raw logistic", "Raw")],
    )
    syl_chances = []
    for sess in sessions:
        decode = fig5_results[sess].get("decode_results", {}).get("motif", {})
        first = list(decode.values())[0] if decode else {}
        syl_chances.append(first.get("chance", 0.1))
    syl_chance = np.nanmean(syl_chances)

    ax_e = fig.add_subplot(row1[0, 1])
    ax_e.set_box_aspect(1.0)
    _draw_summary_bars(
        ax_e, syl_values, sessions,
        title="Syllable", ylabel="Bal. accuracy",
        chance=syl_chance, ylim_top=0.4,
    )
    ax_e.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Panel F: Context-dependent coding
    context_values, context_sessions = _extract_context_values(
        fig5_results, sessions, min_qualifying_motifs,
    )
    all_ctx = [v for _, vals in context_values for v in vals if not np.isnan(v)]
    ctx_ylim = max(all_ctx) + 0.5 if all_ctx else 4.5

    ax_f = fig.add_subplot(row1[0, 2])
    ax_f.set_box_aspect(1.0)
    _draw_summary_bars(
        ax_f, context_values, context_sessions,
        title="Contextual\ncoding", ylabel="Context ratio",
        ref_line=1.0, ylim_top=ctx_ylim,
    )

    # Panel G: Anticipatory drift
    ax_g = fig.add_subplot(row1[0, 3])
    ax_g.set_box_aspect(1.0)
    _draw_panel_g_drift(ax_g, manifold_results)

    # Finalize constrained positions, then place labels + align titles
    fig.canvas.draw()

    # Compute shared label y per row so all labels align horizontally
    row0_top = max(ax.get_position().y1 for ax in [ax_a, ax_b, ax_c])
    row1_top = max(ax.get_position().y1 for ax in [ax_d, ax_e, ax_f, ax_g])

    _add_panel_label(ax_a, "A", x_offset=-0.04, force_y=row0_top)
    _add_panel_label(ax_b, "B", force_y=row0_top)
    _add_panel_label(ax_c, "C", force_y=row0_top)
    _add_panel_label(ax_d, "D", x_offset=-0.04, force_y=row1_top)
    _add_panel_label(ax_e, "E", force_y=row1_top)
    _add_panel_label(ax_f, "F", force_y=row1_top)
    _add_panel_label(ax_g, "G", force_y=row1_top)

    _align_row_titles([ax_a, ax_b, ax_c], fig)
    _align_row_titles([ax_d, ax_e, ax_f, ax_g], fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════
# ED Fig 5 composite — focused 4-panel layout (A-D)
# ══════════════════════════════════════════════════════════════════════════

def ed_fig5_composite(
    fig5_results: Dict[str, Any],
    fig5_embeddings: Dict[str, np.ndarray],
    manifold_results: List[Dict[str, Any]],
    primary_session: str = "M2_0526",
    figsize: tuple = (7.1, 5.5),
    min_qualifying_motifs: int = 2,
) -> plt.Figure:
    """Assemble Extended Data Figure 5 as a 2-row, 4-panel composite.

    Layout::

        Row 0: [A-height]  [A-motif]  [B: context bars]
        Row 1: [C: velocity]          [D-heatmap] [D-fraction]

    Args:
        fig5_results: Per-session results from ``load_fig5_results()``.
        fig5_embeddings: Embeddings dict for primary session.
        manifold_results: From ``load_manifold_results()``.
        primary_session: Session key for CEBRA-Time panel A.
        figsize: Figure size in inches (NatComms max width).
        min_qualifying_motifs: For context coding panel B.

    Returns:
        Assembled Figure with 4 panels (A through D).
    """
    setup_nature_style()

    fig = plt.figure(figsize=figsize)

    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 1],
        hspace=0.45,
        left=0.08, right=0.95, top=0.95, bottom=0.08,
    )

    # Row 0: A (2 embedding scatters) + B (context bars)
    row0 = outer[0].subgridspec(1, 3, wspace=0.4, width_ratios=[1, 1, 1.2])
    # Row 1: C (velocity) + D (heatmap + fraction)
    row1 = outer[1].subgridspec(1, 3, wspace=0.4, width_ratios=[1, 1.2, 0.8])

    sessions = [s for s in ALL_SESSIONS if s in fig5_results]
    emb = fig5_embeddings

    # ═══ Panel A: CEBRA-Time self-supervised (height + motif coloring) ═══
    ax_a1 = fig.add_subplot(row0[0, 0])
    if "emb_time_3d" in emb:
        _draw_panel_b_height_2d(
            ax_a1, emb["emb_time_3d"],
            emb["height_tier"], y_motif=emb.get("y_motif"),
        )
        ax_a1.set_title("CEBRA-Time\n(height, M2 sess. 1)", fontsize=9, fontweight="bold")
    else:
        ax_a1.text(0.5, 0.5, "No Time emb", ha="center", va="center",
                   transform=ax_a1.transAxes, color="gray")

    ax_a2 = fig.add_subplot(row0[0, 1])
    if "emb_time_3d" in emb:
        _draw_panel_c_motif_2d(
            ax_a2, emb["emb_time_3d"],
            emb.get("y_motif", np.array([])),
        )
        ax_a2.set_title("CEBRA-Time\n(syllable, M2 sess. 1)", fontsize=9, fontweight="bold")
    else:
        ax_a2.text(0.5, 0.5, "No Time emb", ha="center", va="center",
                   transform=ax_a2.transAxes, color="gray")

    # ═══ Panel B: Context-dependent coding (per-session bars) ═══
    ax_b = fig.add_subplot(row0[0, 2])
    context_values, context_sessions = _extract_context_values(
        fig5_results, sessions, min_qualifying_motifs,
    )
    all_ctx = [v for _, vals in context_values for v in vals if not np.isnan(v)]
    ctx_ylim = max(all_ctx) + 0.5 if all_ctx else 4.5
    _draw_summary_bars(
        ax_b, context_values, context_sessions,
        title="Context ratio", ylabel="Context ratio",
        ref_line=1.0, ylim_top=ctx_ylim,
    )

    # ═══ Panel C: Velocity ratio at transitions ═══
    ax_c = fig.add_subplot(row1[0, 0])
    _draw_panel_h_velocity(ax_c, fig5_results, sessions, manifold_results)

    # ═══ Panel D: Conjunctive coding (heatmap + fraction bar) ═══
    ax_d1 = fig.add_subplot(row1[0, 1])
    ax_d2 = fig.add_subplot(row1[0, 2])
    _draw_conjunctive_coding(ax_d1, ax_d2, manifold_results)

    # ── Labels ──
    fig.canvas.draw()
    row0_top = max(ax.get_position().y1 for ax in [ax_a1, ax_a2, ax_b])
    row1_top = max(ax.get_position().y1 for ax in [ax_c, ax_d1, ax_d2])

    _add_panel_label(ax_a1, "A", x_offset=-0.04, force_y=row0_top)
    _add_panel_label(ax_b, "B", force_y=row0_top)
    _add_panel_label(ax_c, "C", x_offset=-0.04, force_y=row1_top)
    _add_panel_label(ax_d1, "D", force_y=row1_top)

    return fig


def _draw_conjunctive_coding(
    ax_heat: plt.Axes,
    ax_dots: plt.Axes,
    manifold_results: List[Dict[str, Any]],
) -> None:
    """Draw conjunctive coding heatmap + fraction bar into provided axes.

    Args:
        ax_heat: Axes for eta-squared heatmap.
        ax_dots: Axes for conjunctive fraction bar chart.
        manifold_results: List of per-session manifold analysis dicts.
    """
    if not manifold_results:
        ax_heat.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_heat.transAxes, color="gray")
        return

    # Heatmap: first session with data
    example = manifold_results[0]
    fc = example["factored_conjunctive"]
    eta_motif = np.array(fc["eta_motif"])
    eta_height = np.array(fc["eta_height"])
    n_dims = len(eta_motif)
    eta_matrix = np.array([eta_motif, eta_height])

    ax_heat.imshow(
        eta_matrix, aspect="auto", cmap="YlOrRd", vmin=0,
        vmax=max(eta_matrix.max(), 0.1),
    )
    ax_heat.set_yticks([0, 1])
    ax_heat.set_yticklabels(["Syllable", "Height"], fontsize=8)
    ax_heat.set_xticks(range(n_dims))
    ax_heat.set_xticklabels([f"D{i+1}" for i in range(n_dims)], fontsize=7)
    ax_heat.set_xlabel("CEBRA dim.", fontsize=8)

    for row in range(2):
        for col in range(n_dims):
            val = eta_matrix[row, col]
            color = "white" if val > 0.3 else "black"
            ax_heat.text(col, row, f"{val:.2f}", ha="center", va="center",
                         fontsize=6, color=color)

    _pub_name = {"M2": "M2", "M1": "M1"}
    sess_id = example["session_id"]
    subject = SUBJECT_MAP.get(sess_id, sess_id)
    pub_subj = _pub_name.get(subject, subject)
    ax_heat.set_title(f"η² per dim ({pub_subj})", fontsize=9, fontweight="bold")

    # Fraction bar chart
    sessions_list = []
    conj_fracs = []
    colors = []
    for r in manifold_results:
        sess = r["session_id"]
        sessions_list.append(sess)
        conj_fracs.append(
            r["factored_conjunctive"]["conjunctive_fraction"] * 100
        )
        subject = SUBJECT_MAP.get(sess, sess)
        colors.append(SUBJECT_COLORS.get(subject, "#555555"))

    x = np.arange(len(sessions_list))
    ax_dots.bar(x, conj_fracs, color=colors, alpha=0.8,
                edgecolor="white", width=0.6)
    ax_dots.set_ylim(0, 110)
    ax_dots.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_dots.set_ylabel("Conjunctive\ndims (%)", fontsize=8)
    ax_dots.set_title("Both\nvariables", fontsize=9, fontweight="bold")

    tick_labels = []
    for sess in sessions_list:
        subject = SUBJECT_MAP.get(sess, sess)
        pub_subj = _pub_name.get(subject, subject)
        date = sess.split("_")[1] if "_" in sess else sess
        tick_labels.append(f"{pub_subj}\n{date}")
    ax_dots.set_xticks(x)
    ax_dots.set_xticklabels(tick_labels, fontsize=6)
    ax_dots.tick_params(labelsize=7)
