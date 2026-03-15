"""Smoke tests for Figure 5 composite assembly and style utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from freemat.single_fm.viz.pub_data import save_fig, setup_nature_style


# ── Style tests ─────────────────────────────────────────────────────────

def test_setup_nature_style_sets_svg_fonttype() -> None:
    """setup_nature_style must set svg.fonttype to 'none' for editability."""
    setup_nature_style()
    assert plt.rcParams["svg.fonttype"] == "none"


def test_setup_nature_style_sets_font_family() -> None:
    """setup_nature_style must set font.family to sans-serif."""
    setup_nature_style()
    assert plt.rcParams["font.family"] == ["sans-serif"]


def test_setup_nature_style_removes_top_right_spines() -> None:
    """setup_nature_style must disable top and right spines."""
    setup_nature_style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


# ── save_fig tests ──────────────────────────────────────────────────────

def test_save_fig_creates_png_and_svg(tmp_path: Path) -> None:
    """save_fig must produce both .png and .svg files."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_fig(fig, "test_fig", tmp_path)

    assert (tmp_path / "test_fig.png").exists()
    assert (tmp_path / "test_fig.svg").exists()


def test_save_fig_transparent_default(tmp_path: Path) -> None:
    """save_fig with default transparent=True produces files."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_fig(fig, "test_transparent", tmp_path, transparent=True)

    assert (tmp_path / "test_transparent.png").exists()
    assert (tmp_path / "test_transparent.svg").exists()


# ── Composite smoke test ────────────────────────────────────────────────

_SESSIONS = ["M2_0526", "M2_0605", "M1_0525"]


def _make_session_results(rng: np.random.Generator) -> dict:
    """Create synthetic results for one session."""
    return {
        "decode_results": {
            "height_3class": {
                "CEBRA-SpatialLite-8D kNN": {"knn_ba": rng.uniform(0.4, 0.8), "knn_std": 0.05, "chance": 0.333},
                "PCA-8D kNN": {"knn_ba": rng.uniform(0.4, 0.7), "knn_std": 0.04, "chance": 0.333},
                "Raw logistic": {"logistic_ba": rng.uniform(0.3, 0.6), "logistic_std": 0.06, "chance": 0.333},
            },
            "motif": {
                "CEBRA-Behavior-8D kNN": {"knn_ba": rng.uniform(0.15, 0.3), "knn_std": 0.03, "chance": 0.1},
                "PCA-8D kNN": {"knn_ba": rng.uniform(0.12, 0.25), "knn_std": 0.03, "chance": 0.1},
                "Raw logistic": {"logistic_ba": rng.uniform(0.1, 0.2), "logistic_std": 0.04, "chance": 0.1},
            },
        },
        "dynamics_cebra": {
            "time_axis": list(np.arange(-2, 2, 0.5)),
            "mean_velocity": list(rng.uniform(0.1, 0.3, 8)),
            "sem_velocity": list(rng.uniform(0.01, 0.05, 8)),
            "baseline_velocity": float(rng.uniform(0.1, 0.2)),
            "velocity_ratio": float(rng.uniform(1.2, 2.5)),
            "velocity_null_p": 0.001,
            "n_transitions": 100,
            "status": "ok",
        },
        "dynamics_pca": {"status": "skip"},
        "context_cebra": {
            "context_ratio": float(rng.uniform(1.0, 3.0)),
            "p_value": 0.001,
            "syllable_results": [{"motif": 1}, {"motif": 2}, {"motif": 3}],
        },
        "context_pca": {"context_ratio": float(rng.uniform(0.5, 2.0)), "p_value": 0.01},
    }


def _make_manifold_result(sess: str, rng: np.random.Generator) -> dict:
    """Create synthetic manifold result for one session."""
    return {
        "session_id": sess,
        "factored_conjunctive": {
            "eta_motif": list(rng.uniform(0.05, 0.4, 8)),
            "eta_height": list(rng.uniform(0.03, 0.3, 8)),
            "conjunctive_fraction": 1.0,
        },
        "anticipatory_drift": {
            "time_axis": list(np.arange(-2, 2, 0.5)),
            "mean_sim_next": list(rng.uniform(0.3, 0.7, 8)),
            "mean_sim_current": list(rng.uniform(0.4, 0.8, 8)),
            "pre_ramp": float(rng.uniform(0.01, 0.1)),
            "p_value": 0.01,
        },
        "velocity_prediction": {
            "auc_roc": float(rng.uniform(0.55, 0.7)),
            "p_value": 0.01,
        },
    }


def _make_synthetic_data():
    """Create multi-session synthetic data for composite smoke test."""
    n_bins = 200
    rng = np.random.default_rng(42)

    # Primary session embeddings
    embeddings = {
        "emb_spatial_viz_3d": rng.standard_normal((n_bins, 3)),
        "emb_behavior_viz_3d": rng.standard_normal((n_bins, 3)),
        "emb_time_3d": rng.standard_normal((n_bins, 3)),
        "y_motif": rng.integers(1, 6, size=n_bins),
        "height_tier": rng.integers(0, 3, size=n_bins),
    }

    # Multi-session results
    all_results = {sess: _make_session_results(rng) for sess in _SESSIONS}
    manifold_results = [_make_manifold_result(sess, rng) for sess in _SESSIONS]

    return embeddings, all_results, manifold_results


def test_fig5_composite_returns_figure() -> None:
    """fig5_composite must return a plt.Figure without errors."""
    from freemat.single_fm.viz.fig5_composite import fig5_composite

    embeddings, all_results, manifold_results = _make_synthetic_data()

    fig = fig5_composite(
        fig5_results=all_results,
        fig5_embeddings={"M2_0526": embeddings},
        manifold_results=manifold_results,
        primary_session="M2_0526",
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_fig5_composite_has_panel_labels() -> None:
    """Composite must contain bold A-I panel labels (no J)."""
    from freemat.single_fm.viz.fig5_composite import fig5_composite

    embeddings, all_results, manifold_results = _make_synthetic_data()

    fig = fig5_composite(
        fig5_results=all_results,
        fig5_embeddings={"M2_0526": embeddings},
        manifold_results=manifold_results,
        primary_session="M2_0526",
    )

    all_texts = [t.get_text() for t in fig.texts]
    for ax in fig.get_axes():
        all_texts.extend(t.get_text() for t in ax.texts)
    for label in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        assert label in all_texts, f"Panel label '{label}' not found in figure"
    assert "I" not in all_texts, "Panel I should not exist in 8-panel layout"

    plt.close(fig)


def test_fig5_syllable_ylim() -> None:
    """Syllable decode panel must have ylim capped at 0.4."""
    from freemat.single_fm.viz.fig5_composite import fig5_composite

    embeddings, all_results, manifold_results = _make_synthetic_data()

    fig = fig5_composite(
        fig5_results=all_results,
        fig5_embeddings={"M2_0526": embeddings},
        manifold_results=manifold_results,
        primary_session="M2_0526",
    )

    # Panel E (syllable) is the 5th axes (index 4): A, B, C, D, E
    axes = fig.get_axes()
    # Find the axes with title "Syllable"
    syl_ax = None
    for ax in axes:
        if ax.get_title() == "Syllable":
            syl_ax = ax
            break
    assert syl_ax is not None, "Syllable panel not found"
    assert syl_ax.get_ylim()[1] <= 0.4 + 0.01, (
        f"Syllable ylim top should be ≤0.4, got {syl_ax.get_ylim()[1]}"
    )

    plt.close(fig)
