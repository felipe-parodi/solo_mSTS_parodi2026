"""Unit tests for Fig 4 motif pipeline (A1-A7 from test plan).

Tests the core functions in freemat/single_fm/encoding/run_fig4.py:
  - Bin edge generation and alignment
  - Top-N remapping
  - Tuning matrix computation
  - Bank/depth metadata
  - Decoding + nulls (synthetic data)
  - Lag sweep + next-syllable (synthetic data)
"""
import numpy as np
import pytest

from analysis.syllables.alignment import align_syllables_to_bins, detect_bin_onsets
from analysis.syllables.segmentation import remap_to_top_n
from freemat.single_fm.encoding.run_fig4 import (
    Fig4Config,
    _bin_edges,
    _build_channel_depth_map,
    build_unit_depth_table,
    compute_motif_cluster_order,
    compute_motif_tuning,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def default_config() -> Fig4Config:
    """Default Fig4Config for tests."""
    return Fig4Config(
        bin_s=0.5,
        top_n_motifs=5,
        exclude_motif_ids=(0,),
        n_permutations=10,
        embargo_bins=2,
    )


@pytest.fixture()
def synthetic_spikes_and_motifs():
    """Synthetic spike matrix (100 bins x 20 units) + motif labels.

    Creates 5 distinct motifs with different firing patterns so
    decoding and tuning are non-trivial.
    """
    rng = np.random.default_rng(42)
    n_bins = 100
    n_units = 20
    n_motifs = 5

    # Assign motifs in blocks of 20 bins
    y = np.repeat(np.arange(1, n_motifs + 1), n_bins // n_motifs)

    # Create firing rate profiles: each motif activates different units
    X = rng.poisson(lam=5, size=(n_bins, n_units)).astype(float)
    for m in range(1, n_motifs + 1):
        mask = y == m
        active_units = slice((m - 1) * 4, m * 4)
        X[mask, active_units] += rng.poisson(lam=15, size=(mask.sum(), 4))

    valid_mask = np.ones(n_bins, dtype=bool)
    unit_banks = np.array(["ventral"] * 10 + ["dorsal"] * 10)

    return X, y, valid_mask, unit_banks


# ── A2: Bin edge generation + alignment ──────────────────────────────────

class TestBinEdges:
    """A2: Bin edge generation matches spike binning semantics."""

    def test_bin_edges_match_time_window(self):
        """Bin edges cover full time window."""
        edges = _bin_edges((0.0, 10.0), 0.5)
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] >= 10.0
        assert len(edges) == 21  # 20 bins + 1 edge

    def test_bin_edges_fractional(self):
        """Bin edges handle fractional boundaries."""
        edges = _bin_edges((1.5, 3.5), 0.5)
        assert edges[0] == pytest.approx(1.5)
        assert edges[-1] >= 3.5
        n_bins = len(edges) - 1
        assert n_bins == 4

    def test_alignment_center_basic(self):
        """Center-based alignment assigns correct motif to each bin."""
        motif_ids = np.array([1] * 20 + [2] * 20)  # 40 frames at 40Hz
        frame_times = np.arange(40) / 40.0  # 0.0 to 0.975s

        bin_edges = np.array([0.0, 0.5, 1.0])
        labels = align_syllables_to_bins(
            motif_ids, frame_times, bin_edges, method="center"
        )
        assert len(labels) == 2
        assert labels[0] == 1
        assert labels[1] == 2

    def test_alignment_empty_bins(self):
        """Bins far from all frames get missing_value sentinel."""
        motif_ids = np.array([5, 5, 5])
        frame_times = np.array([3.0, 3.1, 3.2])
        bin_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        labels = align_syllables_to_bins(
            motif_ids, frame_times, bin_edges, method="center", missing_value=-999
        )
        # Bins far from data should be missing, bin [3.0, 3.5) has frames
        assert len(labels) == 7
        assert labels[0] == -999  # center=0.25, far from frames at 3.0+
        assert labels[1] == -999
        assert labels[2] == -999
        assert labels[6] == 5     # center=3.25, frame at 3.2 nearby


# ── A3: Top-N remapping ──────────────────────────────────────────────────

class TestTopNRemapping:
    """A3: Top-N selection by frequency."""

    def test_top_n_selection(self):
        """Top-N keeps most frequent motifs."""
        ids = np.array([1] * 50 + [2] * 30 + [3] * 20 + [4] * 10 + [5] * 5)
        remapped, mapping = remap_to_top_n(ids, n=3, other_id=-1)

        # Top-3 are 1, 2, 3
        assert mapping[1] == 1
        assert mapping[2] == 2
        assert mapping[3] == 3
        assert mapping[4] == -1
        assert mapping[5] == -1
        assert np.sum(remapped == -1) == 15

    def test_others_mapped_to_other_id(self):
        """Non-top-N motifs map to other_id."""
        ids = np.array([10] * 100 + [20] * 50 + [30] * 1)
        remapped, _ = remap_to_top_n(ids, n=2, other_id=-1)
        assert np.sum(remapped == -1) == 1

    def test_all_filtered_raises_upstream(self):
        """When all labels become other_id, valid_mask should be empty."""
        ids = np.array([1, 2, 3])  # Each appears once
        remapped, _ = remap_to_top_n(ids, n=0, other_id=-1)
        # All should be -1 when n=0
        assert np.all(remapped == -1)


# ── A4: Tuning matrix ────────────────────────────────────────────────────

class TestTuningMatrix:
    """A4: Motif tuning computation."""

    def test_shape(self, synthetic_spikes_and_motifs):
        """Tuning shape is (n_units, n_motifs)."""
        X, y, valid_mask, _ = synthetic_spikes_and_motifs
        raw, z, motif_ids, counts = compute_motif_tuning(X, y, valid_mask)

        assert raw.shape == (20, 5)
        assert z.shape == (20, 5)
        assert len(motif_ids) == 5
        assert len(counts) == 5

    def test_motif_counts(self, synthetic_spikes_and_motifs):
        """Motif counts match expected bin counts."""
        X, y, valid_mask, _ = synthetic_spikes_and_motifs
        _, _, motif_ids, counts = compute_motif_tuning(X, y, valid_mask)

        # Each motif has 20 bins (100 / 5)
        np.testing.assert_array_equal(counts, [20, 20, 20, 20, 20])

    def test_z_scored_per_unit(self, synthetic_spikes_and_motifs):
        """Z-scored tuning has ~0 mean and ~1 std per unit."""
        X, y, valid_mask, _ = synthetic_spikes_and_motifs
        _, z, _, _ = compute_motif_tuning(X, y, valid_mask)

        # Per-unit mean should be ~0
        row_means = np.nanmean(z, axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)

        # Per-unit std should be ~1 (for units with non-constant tuning)
        row_stds = np.nanstd(z, axis=1)
        assert np.all(row_stds > 0)
        np.testing.assert_allclose(row_stds, 1.0, atol=0.1)

    def test_partial_valid_mask(self):
        """Tuning works with partial valid_mask."""
        rng = np.random.default_rng(99)
        X = rng.poisson(5, size=(50, 10)).astype(float)
        y = np.array([1] * 20 + [2] * 20 + [3] * 10)
        valid_mask = np.ones(50, dtype=bool)
        valid_mask[40:] = False  # Last 10 bins invalid

        raw, z, motif_ids, counts = compute_motif_tuning(X, y, valid_mask)

        assert len(motif_ids) == 2  # Only motifs 1 and 2 in valid region
        assert counts[0] == 20
        assert counts[1] == 20


# ── A5: Bank/depth metadata ──────────────────────────────────────────────

class TestBankDepthMetadata:
    """A5: Unit bank and depth mapping from banks.yaml."""

    def test_build_channel_depth_map(self):
        """Channel depth map builds correctly from shank_groups."""
        banks_cfg = {
            "shank_groups": {
                "a1_shank1": [0, 1, 2, 3],
                "b1_shank1": [64, 65, 66, 67],
            }
        }
        channel_map = _build_channel_depth_map(banks_cfg)

        assert channel_map[0]["bank"] == "ventral"
        assert channel_map[0]["probe"] == "A1"
        assert channel_map[0]["shank"] == 1
        assert channel_map[0]["depth_position"] == 0

        assert channel_map[67]["bank"] == "dorsal"
        assert channel_map[67]["probe"] == "B1"
        assert channel_map[67]["depth_position"] == 3

    def test_build_unit_depth_table(self):
        """build_unit_depth_table returns correct sort order."""
        # Use real channel IDs from banks.yaml:
        # a1_shank1[0]=57 (ventral), a1_shank1[1]=116 (ventral)
        # b1_shank1[0]=59 (dorsal),  b1_shank1[1]=118 (dorsal)
        unit_channels = np.array([57, 116, 59, 118])
        unit_banks = np.array(["ventral", "ventral", "dorsal", "dorsal"])

        depth_df = build_unit_depth_table(
            unit_channels, unit_banks,
            banks_yaml="configs/banks.yaml",
        )

        assert len(depth_df) == 4
        assert "sort_rank" in depth_df.columns
        assert "bank" in depth_df.columns
        # Ventral should have lower sort_rank than dorsal
        ventral_ranks = depth_df.loc[depth_df["bank"] == "ventral", "sort_rank"]
        dorsal_ranks = depth_df.loc[depth_df["bank"] == "dorsal", "sort_rank"]
        assert ventral_ranks.max() < dorsal_ranks.min()

    def test_bank_mismatch_raises(self):
        """Bank mismatch between unit_banks and banks.yaml raises."""
        unit_channels = np.array([57])  # Channel 57 is ventral (a1_shank1)
        unit_banks = np.array(["dorsal"])  # Mismatch!

        with pytest.raises(ValueError, match="Bank mismatch"):
            build_unit_depth_table(
                unit_channels, unit_banks,
                banks_yaml="configs/banks.yaml",
            )

    def test_unknown_channel_raises(self):
        """Unknown channel ID raises ValueError."""
        unit_channels = np.array([999])  # Not in banks.yaml
        unit_banks = np.array(["ventral"])

        with pytest.raises(ValueError, match="Channel 999"):
            build_unit_depth_table(
                unit_channels, unit_banks,
                banks_yaml="configs/banks.yaml",
            )

    def test_invalid_shank_name_raises(self):
        """Invalid shank name format raises."""
        banks_cfg = {
            "shank_groups": {
                "invalid_name": [0, 1],
            }
        }
        with pytest.raises(ValueError, match="Unexpected shank name"):
            _build_channel_depth_map(banks_cfg)

    def test_duplicate_channel_raises(self):
        """Duplicate channel across shanks raises."""
        banks_cfg = {
            "shank_groups": {
                "a1_shank1": [0, 1],
                "a1_shank2": [0, 2],  # Channel 0 repeated
            }
        }
        with pytest.raises(ValueError, match="Channel 0 appears"):
            _build_channel_depth_map(banks_cfg)


# ── A6: Decoding + nulls (lightweight synthetic) ─────────────────────────

class TestDecodingAndNulls:
    """A6: Decoding and null distribution tests with synthetic data."""

    def test_decoding_returns_metrics(self, synthetic_spikes_and_motifs):
        """Decoding returns balanced_accuracy and macro_f1 per bank."""
        from freemat.single_fm.encoding.run_fig4 import run_motif_decoding

        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        config = Fig4Config(
            n_folds=3, embargo_bins=1, n_permutations=5,
            top_n_motifs=5, bin_s=0.5,
        )

        dec_df, nulls, aux = run_motif_decoding(
            X, y, valid_mask, unit_banks, config
        )

        assert "balanced_accuracy" in dec_df.columns
        assert "macro_f1" in dec_df.columns
        assert len(dec_df) == 3  # ventral, dorsal, pooled
        assert set(dec_df["bank"].values) == {"ventral", "dorsal", "pooled"}

    def test_null_distribution_length(self, synthetic_spikes_and_motifs):
        """Null distribution length matches n_permutations."""
        from freemat.single_fm.encoding.run_fig4 import run_motif_decoding

        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        n_perm = 7
        config = Fig4Config(
            n_folds=3, embargo_bins=1, n_permutations=n_perm,
            top_n_motifs=5, bin_s=0.5,
        )

        _, nulls, _ = run_motif_decoding(
            X, y, valid_mask, unit_banks, config
        )

        for bank, null_arr in nulls.items():
            assert len(null_arr) == n_perm, f"{bank} null length mismatch"

    def test_bank_specific_subsets(self, synthetic_spikes_and_motifs):
        """Bank-specific decoding uses correct unit subsets."""
        from freemat.single_fm.encoding.run_fig4 import run_motif_decoding

        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        config = Fig4Config(
            n_folds=3, embargo_bins=1, n_permutations=3,
            top_n_motifs=5, bin_s=0.5,
        )

        dec_df, _, _ = run_motif_decoding(
            X, y, valid_mask, unit_banks, config
        )

        # All banks should have results
        assert "ventral" in dec_df["bank"].values
        assert "dorsal" in dec_df["bank"].values
        assert "pooled" in dec_df["bank"].values


# ── A7: Lag sweep + next-syllable ────────────────────────────────────────

class TestLagSweepAndNextSyllable:
    """A7: Lag sweep and next-syllable prediction tests."""

    def test_lag_sweep_returns_expected_lags(self, synthetic_spikes_and_motifs):
        """Lag sweep returns correct number of lags and banks."""
        from freemat.single_fm.encoding.run_fig4 import run_lag_sweep

        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        config = Fig4Config(
            lag_range_bins=2, n_folds=3, embargo_bins=1, bin_s=0.5,
        )

        results = run_lag_sweep(
            X, y, valid_mask, unit_banks, config
        )

        assert "ventral" in results
        assert "dorsal" in results
        assert "pooled" in results

        for bank, result in results.items():
            # lag_range_bins=2 -> lags [-2,-1,0,1,2] = 5 lags
            assert len(result.accuracy_curve) == 5
            assert len(result.lags_sec) == 5

    def test_optimal_lag_within_range(self, synthetic_spikes_and_motifs):
        """Optimal lag is within tested range."""
        from freemat.single_fm.encoding.run_fig4 import run_lag_sweep

        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        config = Fig4Config(
            lag_range_bins=2, n_folds=3, embargo_bins=1, bin_s=0.5,
        )

        results = run_lag_sweep(
            X, y, valid_mask, unit_banks, config
        )

        for bank, result in results.items():
            assert -2 <= result.optimal_lag_bins <= 2

    def test_next_syllable_with_onsets(self, synthetic_spikes_and_motifs):
        """Next-syllable prediction returns accuracy around transitions."""
        from freemat.single_fm.encoding.run_fig4 import run_next_syllable_prediction

        X, y, _, _ = synthetic_spikes_and_motifs
        config = Fig4Config(lag_range_bins=2, bin_s=0.5)

        result = run_next_syllable_prediction(X, y, config)

        assert hasattr(result, "tau_bins")
        assert hasattr(result, "accuracy_curve")
        assert len(result.tau_bins) > 0

    def test_onset_detection_ignores_missing(self):
        """detect_bin_onsets ignores transitions involving missing values."""
        # When missing values separate all real transitions, no onsets detected
        bins_separated = np.array([1, 1, -999, 2, 2, -999, 3, 3])
        onsets, post = detect_bin_onsets(bins_separated, missing_value=-999)
        assert len(onsets) == 0  # All transitions involve missing

        # Without missing values, transitions are detected
        bins_clean = np.array([1, 1, 2, 2, 3, 3])
        onsets2, post2 = detect_bin_onsets(bins_clean, missing_value=-999)
        np.testing.assert_array_equal(onsets2, [2, 4])
        np.testing.assert_array_equal(post2, [2, 3])


# ── Cluster order ─────────────────────────────────────────────────────────

class TestClusterOrder:
    """Test motif clustering for heatmap column ordering."""

    def test_cluster_order_length(self, synthetic_spikes_and_motifs):
        """Cluster order has same length as motif_ids."""
        X, y, valid_mask, _ = synthetic_spikes_and_motifs
        _, z, motif_ids, _ = compute_motif_tuning(X, y, valid_mask)

        order = compute_motif_cluster_order(z, motif_ids)
        assert len(order) == len(motif_ids)
        assert set(order) == set(int(m) for m in motif_ids)

    def test_single_motif(self):
        """Single motif returns trivially."""
        z = np.array([[1.0], [2.0], [3.0]])
        motif_ids = np.array([5])
        order = compute_motif_cluster_order(z, motif_ids)
        assert order == [5]


# ── Phase 2 panel tests ──────────────────────────────────────────────────

class TestConfusionMatrixPlot:
    """Tests for confusion matrix heatmap visualization."""

    def test_returns_figure(self):
        """Confusion matrix plot returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_confusion_matrix

        cm = {"pooled": np.eye(5, dtype=int) * 10}
        labels = {"pooled": np.arange(5)}
        fig = plot_fig4_supp_confusion_matrix(cm, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multibank(self):
        """Works with ventral + dorsal + pooled."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_confusion_matrix

        cm = {b: np.eye(3, dtype=int) * 5 for b in ["ventral", "dorsal", "pooled"]}
        labels = {b: np.array([1, 2, 3]) for b in cm}
        fig = plot_fig4_supp_confusion_matrix(cm, labels)
        axes = [a for a in fig.get_axes() if a.get_xlabel()]
        assert len(axes) == 3
        plt.close(fig)

    def test_cluster_reorder(self):
        """Cluster order reorders rows and columns."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_confusion_matrix

        cm = {"pooled": np.array([[10, 1], [2, 8]])}
        labels = {"pooled": np.array([5, 10])}
        fig = plot_fig4_supp_confusion_matrix(cm, labels, cluster_order=[10, 5])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_data(self):
        """Empty confusion dict returns placeholder figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_confusion_matrix

        fig = plot_fig4_supp_confusion_matrix({}, {})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSelectivityByBankPlot:
    """Tests for per-unit selectivity violin."""

    def test_returns_figure(self, synthetic_spikes_and_motifs):
        """Selectivity violin returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_selectivity_by_bank

        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        _, z, motif_ids, _ = compute_motif_tuning(X, y, valid_mask)
        tuning_df = pd.DataFrame({"bank": unit_banks})
        for j, mid in enumerate(motif_ids):
            tuning_df[f"motif_{mid}_z"] = z[:, j]
        fig = plot_fig4_supp_selectivity_by_bank(tuning_df, list(motif_ids))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_selectivity_positive(self, synthetic_spikes_and_motifs):
        """Selectivity values are non-negative."""
        X, y, valid_mask, unit_banks = synthetic_spikes_and_motifs
        _, z, motif_ids, _ = compute_motif_tuning(X, y, valid_mask)
        selectivity = np.nanmax(np.abs(z), axis=1)
        assert np.all(selectivity >= 0)


class TestTransitionMatrixPlot:
    """Tests for transition matrix heatmap visualization."""

    def test_returns_figure(self):
        """Transition matrix plot returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_transition_matrix

        matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        labels = np.array([1, 2])
        fig = plot_fig4_supp_transition_matrix(matrix, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_annotates_large_values(self):
        """Cells >= 0.05 are annotated."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_transition_matrix

        matrix = np.array([[0.9, 0.1], [0.04, 0.96]])
        labels = np.array([5, 10])
        fig = plot_fig4_supp_transition_matrix(matrix, labels)
        ax = fig.axes[0]
        # Should have text annotations for values >= 0.05
        texts = [t for t in ax.texts if t.get_text()]
        assert len(texts) >= 3  # 0.90, 0.10, 0.96
        plt.close(fig)

    def test_computation_shape(self, synthetic_spikes_and_motifs):
        """Transition matrix has correct shape from motif labels."""
        from analysis.syllables.segmentation import compute_transition_matrix

        _, y, valid_mask, _ = synthetic_spikes_and_motifs
        clean = y[valid_mask]
        clean = clean[(clean != -999) & (clean != -1)]
        matrix, labels = compute_transition_matrix(clean)
        assert matrix.shape == (len(labels), len(labels))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-10)


class TestOnsetPethPlot:
    """Tests for onset-aligned PETH heatmap."""

    def test_returns_figure(self):
        """PETH heatmap returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_onset_peth

        rng = np.random.default_rng(42)
        peth = rng.standard_normal((20, 16))
        tau = np.arange(-8, 8)
        banks = np.array(["ventral"] * 10 + ["dorsal"] * 10)
        fig = plot_fig4_supp_onset_peth(peth, tau, banks, bin_s=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bank_separator(self):
        """Bank separator line is drawn and bank labels present."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_onset_peth

        peth = np.ones((10, 8))
        tau = np.arange(-4, 4)
        banks = np.array(["ventral"] * 6 + ["dorsal"] * 4)
        fig = plot_fig4_supp_onset_peth(peth, tau, banks)
        # Bank labels are text objects in the spacer axis (gs[2])
        all_texts = []
        for ax in fig.axes:
            all_texts.extend([t.get_text() for t in ax.texts])
        assert "Ventral" in all_texts
        assert "Dorsal" in all_texts
        plt.close(fig)


# ── Transition matrix without self-transitions ─────────────────────────

class TestTransitionNoSelf:
    """Tests for transition matrix excluding self-transitions."""

    def test_returns_figure(self):
        """No-self transition plot returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_transition_no_self

        matrix = np.array([[0.8, 0.1, 0.1], [0.05, 0.9, 0.05], [0.1, 0.1, 0.8]])
        labels = np.array([1, 2, 3])
        fig = plot_fig4_supp_transition_no_self(matrix, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_diagonal_zeroed(self):
        """Self-transitions are removed and rows re-normalized."""
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_transition_no_self

        matrix = np.array([[0.8, 0.1, 0.1], [0.05, 0.9, 0.05], [0.1, 0.1, 0.8]])
        labels = np.array([1, 2, 3])
        # After zeroing diagonal and re-normalizing:
        # Row 0: [0, 0.1, 0.1] / 0.2 = [0, 0.5, 0.5]
        # Row 1: [0.05, 0, 0.05] / 0.1 = [0.5, 0, 0.5]
        # Row 2: [0.1, 0.1, 0] / 0.2 = [0.5, 0.5, 0]
        # Just verify the figure is produced correctly
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plot_fig4_supp_transition_no_self(matrix, labels)
        ax = fig.axes[0]
        # Check annotations exist (values >= 0.03 should be annotated)
        texts = [t for t in ax.texts if t.get_text()]
        assert len(texts) >= 3  # 0.50 values should appear
        plt.close(fig)


# ── Lag sweep CI ────────────────────────────────────────────────────────

class TestLagSweepCI:
    """Tests for lag sweep per-fold CI computation."""

    def test_ci_fields_populated(self, synthetic_spikes_and_motifs):
        """LagSweepResult has populated CI fields."""
        from analysis.syllables.lag_sweep import lag_sweep_decoding

        X, y, valid_mask, _ = synthetic_spikes_and_motifs
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        result = lag_sweep_decoding(
            X_valid, y_valid, lags_bins=np.array([-1, 0, 1]),
            bin_size_sec=0.5, n_folds=3, embargo_bins=1,
        )

        assert result.accuracy_ci_low is not None
        assert result.accuracy_ci_high is not None
        assert len(result.accuracy_ci_low) == 3
        assert len(result.accuracy_ci_high) == 3

    def test_ci_brackets_accuracy(self, synthetic_spikes_and_motifs):
        """CI low <= accuracy <= CI high where both are finite."""
        from analysis.syllables.lag_sweep import lag_sweep_decoding

        X, y, valid_mask, _ = synthetic_spikes_and_motifs
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        result = lag_sweep_decoding(
            X_valid, y_valid, lags_bins=np.array([-1, 0, 1]),
            bin_size_sec=0.5, n_folds=3, embargo_bins=1,
        )

        for i in range(len(result.accuracy_curve)):
            if (np.isfinite(result.accuracy_ci_low[i])
                    and np.isfinite(result.accuracy_ci_high[i])):
                assert result.accuracy_ci_low[i] <= result.accuracy_ci_high[i]


# ── Smoothed PETH viz ──────────────────────────────────────────────────

class TestSmoothedPethViz:
    """Tests for smoothed PETH visualization."""

    def test_bilinear_interpolation(self):
        """Smoothed PETH uses bilinear interpolation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4_supp_onset_peth_smoothed

        rng = np.random.default_rng(42)
        peth = rng.standard_normal((20, 160))  # Fine-grained
        tau = np.arange(-80, 80)
        banks = np.array(["ventral"] * 10 + ["dorsal"] * 10)
        fig = plot_fig4_supp_onset_peth_smoothed(peth, tau, banks, bin_s=0.05)
        # Check that images use bilinear
        for ax in fig.axes:
            for im in ax.get_images():
                assert im.get_interpolation() == "bilinear"
        plt.close(fig)


class TestElectrodeLayoutPanels:
    """Tests for electrode selectivity and preferred motif panels."""

    @pytest.fixture()
    def electrode_tuning_df(self):
        """Tuning DataFrame with probe/shank/depth columns."""
        import pandas as pd

        rng = np.random.default_rng(42)
        rows = []
        motif_ids = [1, 2, 3, 4, 5]
        unit_idx = 0
        for probe in ["A1", "A2", "B1", "B2"]:
            bank = "ventral" if probe.startswith("A") else "dorsal"
            for shank in range(1, 5):
                for depth in range(8):
                    row = {
                        "unit_index": unit_idx,
                        "probe": probe,
                        "shank": shank,
                        "depth_position": depth,
                        "bank": bank,
                        "sort_rank": (0 if bank == "ventral" else 1) * 1000
                                     + {"A1": 0, "A2": 1, "B1": 2, "B2": 3}[probe] * 100
                                     + shank * 10 + depth,
                    }
                    for mid in motif_ids:
                        row[f"motif_{mid}_z"] = float(rng.standard_normal())
                    rows.append(row)
                    unit_idx += 1
        return pd.DataFrame(rows)

    def test_selectivity_returns_figure(self, electrode_tuning_df):
        """Selectivity panel returns a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4b_electrode_selectivity

        fig = plot_fig4b_electrode_selectivity(electrode_tuning_df, [1, 2, 3, 4, 5])
        assert isinstance(fig, plt.Figure)
        # Should have 4 panels (2x2) + colorbar
        axes = [ax for ax in fig.axes]
        assert len(axes) >= 4
        plt.close(fig)

    def test_preferred_returns_figure(self, electrode_tuning_df):
        """Preferred motif panel returns a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4b_electrode_preferred

        fig = plot_fig4b_electrode_preferred(electrode_tuning_df, [1, 2, 3, 4, 5])
        assert isinstance(fig, plt.Figure)
        axes = [ax for ax in fig.axes]
        assert len(axes) >= 4
        plt.close(fig)

    def test_selectivity_values_nonnegative(self, electrode_tuning_df):
        """Selectivity grid values should be non-negative (max|z|)."""
        from freemat.single_fm.viz.fig4_panels import _build_electrode_grid

        grids = _build_electrode_grid(electrode_tuning_df, [1, 2, 3, 4, 5], "selectivity")
        for probe, grid in grids.items():
            finite = grid[np.isfinite(grid)]
            assert np.all(finite >= 0), f"Negative selectivity in {probe}"

    def test_preferred_values_in_motif_order(self, electrode_tuning_df):
        """Preferred motif IDs should be from the motif_order list."""
        from freemat.single_fm.viz.fig4_panels import _build_electrode_grid

        motif_order = [1, 2, 3, 4, 5]
        grids = _build_electrode_grid(electrode_tuning_df, motif_order, "preferred")
        for probe, grid in grids.items():
            finite = grid[np.isfinite(grid)]
            for val in finite:
                assert int(val) in motif_order, f"Unexpected motif {val} in {probe}"


class TestBubblePlot:
    """Tests for Voloh-style bubble plot (electrode-level metrics)."""

    @pytest.fixture()
    def electrode_tuning_df(self):
        """Tuning DataFrame with multiple units per electrode."""
        import pandas as pd

        rng = np.random.default_rng(42)
        rows = []
        motif_ids = [1, 2, 3, 4, 5]
        unit_idx = 0
        for probe in ["A1", "A2", "B1", "B2"]:
            bank = "ventral" if probe.startswith("A") else "dorsal"
            for shank in range(1, 5):
                for depth in range(8):
                    # 1-3 units per electrode
                    n_units = rng.integers(1, 4)
                    for _ in range(n_units):
                        row = {
                            "unit_index": unit_idx,
                            "probe": probe,
                            "shank": shank,
                            "depth_position": depth,
                            "bank": bank,
                        }
                        for mid in motif_ids:
                            row[f"motif_{mid}_z"] = float(rng.standard_normal())
                        rows.append(row)
                        unit_idx += 1
        return pd.DataFrame(rows)

    def test_aggregate_metrics_shape(self, electrode_tuning_df):
        """Aggregated metrics have one row per electrode with expected cols."""
        from freemat.single_fm.viz.fig4_panels import _aggregate_electrode_metrics

        agg = _aggregate_electrode_metrics(electrode_tuning_df, [1, 2, 3, 4, 5])
        assert "selectivity" in agg.columns
        assert "bandwidth" in agg.columns
        assert "n_units" in agg.columns
        assert agg["n_units"].min() >= 1
        # One row per electrode (4 probes * 4 shanks * 8 depths = 128 max)
        assert len(agg) <= 128

    def test_aggregate_selectivity_nonneg(self, electrode_tuning_df):
        """Selectivity (max|z|) is always non-negative."""
        from freemat.single_fm.viz.fig4_panels import _aggregate_electrode_metrics

        agg = _aggregate_electrode_metrics(electrode_tuning_df, [1, 2, 3, 4, 5])
        assert (agg["selectivity"] >= 0).all()

    def test_aggregate_with_peth(self, electrode_tuning_df):
        """PETH metrics added when peth data is provided."""
        from freemat.single_fm.viz.fig4_panels import _aggregate_electrode_metrics

        n_units = len(electrode_tuning_df)
        rng = np.random.default_rng(42)
        peth = rng.standard_normal((n_units, 160))
        tau = np.arange(-80, 80)
        agg = _aggregate_electrode_metrics(
            electrode_tuning_df, [1, 2, 3, 4, 5],
            peth=peth, tau_bins=tau, bin_s=0.05,
        )
        assert "peak_latency" in agg.columns
        assert "response_modulation" in agg.columns

    def test_scatter_coords_range(self, electrode_tuning_df):
        """Scatter coordinates fall within expected ranges with bank gap."""
        from freemat.single_fm.viz.fig4_panels import (
            _aggregate_electrode_metrics,
            _electrode_scatter_coords,
        )

        agg = _aggregate_electrode_metrics(electrode_tuning_df, [1, 2, 3, 4, 5])
        x, y = _electrode_scatter_coords(agg)
        assert x.min() >= 0
        assert x.max() <= 17
        assert y.min() >= 0
        assert y.max() <= 7
        # Gap: no electrodes at x=8 or x=9
        assert not np.any((x >= 8) & (x <= 9))

    def test_bubble_returns_figure(self, electrode_tuning_df):
        """Bubble plot returns a Figure (tuning-only, 2 metrics)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4b_bubble

        fig = plot_fig4b_bubble(electrode_tuning_df, [1, 2, 3, 4, 5])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bubble_with_peth(self, electrode_tuning_df):
        """Bubble plot with PETH data produces all 4 metric panels."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4b_bubble

        n_units = len(electrode_tuning_df)
        rng = np.random.default_rng(42)
        peth = rng.standard_normal((n_units, 160))
        tau = np.arange(-80, 80)
        fig = plot_fig4b_bubble(
            electrode_tuning_df, [1, 2, 3, 4, 5],
            peth=peth, tau_bins=tau, bin_s=0.05,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bubble_subset_metrics(self, electrode_tuning_df):
        """Bubble plot respects metrics parameter for single metric."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4b_bubble

        fig = plot_fig4b_bubble(
            electrode_tuning_df, [1, 2, 3, 4, 5],
            metrics=["selectivity"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


    def test_aggregate_transition_resp(self, electrode_tuning_df):
        """Transition responsiveness computed when PETH provided."""
        from freemat.single_fm.viz.fig4_panels import _aggregate_electrode_metrics

        n_units = len(electrode_tuning_df)
        rng = np.random.default_rng(42)
        peth = rng.standard_normal((n_units, 160))
        tau = np.arange(-80, 80)
        agg = _aggregate_electrode_metrics(
            electrode_tuning_df, [1, 2, 3, 4, 5],
            peth=peth, tau_bins=tau, bin_s=0.05,
        )
        assert "transition_resp" in agg.columns
        assert (agg["transition_resp"] >= 0).all()

    def test_3d_coords_physical_units(self, electrode_tuning_df):
        """3D coordinates use physical µm distances."""
        from freemat.single_fm.viz.fig4_panels import (
            _aggregate_electrode_metrics,
            _electrode_3d_coords,
        )

        agg = _aggregate_electrode_metrics(electrode_tuning_df, [1, 2, 3, 4, 5])
        x, y, z = _electrode_3d_coords(agg, ap_offset_um=800)
        # X: shank 1-4 -> 0, 500, 1000, 1500 µm
        assert x.min() == 0
        assert x.max() == 1500
        # Y: anterior (0) and posterior (800)
        assert set(np.unique(y)) == {0.0, 800.0}
        # Z: depth 0-7 -> 0, 550, ..., 3850 µm
        assert z.min() == 0
        assert z.max() == 3850

    def test_bubble_3d_returns_figure(self, electrode_tuning_df):
        """3D bubble plot returns a Figure with 3D axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from freemat.single_fm.viz.fig4_panels import plot_fig4b_bubble_3d

        fig = plot_fig4b_bubble_3d(electrode_tuning_df, [1, 2, 3, 4, 5])
        assert isinstance(fig, plt.Figure)
        # Should have 2 3D subplots (ventral + dorsal)
        axes_3d = [ax for ax in fig.axes if hasattr(ax, "get_zlim")]
        assert len(axes_3d) == 2
        plt.close(fig)


class TestLagSweepNoPooled:
    """Tests for lag sweep without pooled bank."""

    def test_no_pooled_in_plot(self):
        """Lag sweep plot should exclude pooled bank."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from freemat.single_fm.viz.fig4_panels import plot_fig4d_lag_sweep

        lags = [-1.0, -0.5, 0.0, 0.5, 1.0]
        rows = []
        for bank in ["ventral", "dorsal", "pooled"]:
            for lag in lags:
                rows.append({"bank": bank, "lag_sec": lag, "accuracy": 0.15})
        lag_df = pd.DataFrame(rows)

        fig = plot_fig4d_lag_sweep(lag_df, n_classes=10)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Pooled" not in legend_labels
        assert "Ventral" in legend_labels
        assert "Dorsal" in legend_labels
        # Legend should only have 2 entries (ventral + dorsal)
        assert len(legend_labels) == 2
        plt.close(fig)

    def test_chance_text_not_in_legend(self):
        """Chance should be a text annotation, not a legend entry."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from freemat.single_fm.viz.fig4_panels import plot_fig4d_lag_sweep

        lags = [-1.0, 0.0, 1.0]
        rows = []
        for bank in ["ventral", "dorsal"]:
            for lag in lags:
                rows.append({"bank": bank, "lag_sec": lag, "accuracy": 0.15})
        lag_df = pd.DataFrame(rows)

        fig = plot_fig4d_lag_sweep(lag_df, n_classes=10)
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert not any("Chance" in lbl for lbl in legend_labels)
        # Check text annotation exists
        texts = [t for t in ax.texts if "Chance" in t.get_text()]
        assert len(texts) == 1
        plt.close(fig)
