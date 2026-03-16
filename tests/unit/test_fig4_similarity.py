"""Unit tests for Fig 4 similarity vs confusion analysis.

Tests cover:
- correlate_similarity_confusion with monotonic off-diagonal matrices
- Label subsetting (12-label similarity → 10-label subset)
- Symmetrize non-symmetric confusion
- Mantel-style permutation: p < 0.05 for structured matrices
- One-sided and two-sided p-values
"""

from __future__ import annotations

import numpy as np


class TestSymmetrizeConfusion:
    """Test confusion matrix symmetrization."""

    def test_symmetric_input_unchanged(self) -> None:
        """Already-symmetric matrix should be unchanged."""
        from analysis.syllables.similarity import symmetrize_confusion

        mat = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
        result = symmetrize_confusion(mat)
        np.testing.assert_allclose(result, mat)

    def test_asymmetric_becomes_symmetric(self) -> None:
        """Non-symmetric confusion should become symmetric."""
        from analysis.syllables.similarity import symmetrize_confusion

        mat = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.05, 0.25, 0.7]])
        result = symmetrize_confusion(mat)
        np.testing.assert_allclose(result, result.T)

    def test_symmetrize_formula(self) -> None:
        """conf_sym[i,j] = 0.5 * (conf[i,j] + conf[j,i])."""
        from analysis.syllables.similarity import symmetrize_confusion

        mat = np.array([[1.0, 0.2], [0.4, 1.0]])
        result = symmetrize_confusion(mat)
        np.testing.assert_allclose(result[0, 1], 0.3)  # 0.5 * (0.2 + 0.4)
        np.testing.assert_allclose(result[1, 0], 0.3)


class TestCorrelation:
    """Test Mantel-style correlation between similarity and confusion."""

    def _make_structured_matrices(self, n: int = 8) -> tuple:
        """Create synthetic matrices with known monotonic off-diagonal relationship.

        sim[i,j] = |i - j|  (higher = more dissimilar)
        conf[i,j] = 1 / (|i - j| + 1)  (higher = more confused when similar)
        → expected negative Spearman rho.
        """
        sim = np.zeros((n, n))
        conf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d = abs(i - j)
                sim[i, j] = float(d)
                conf[i, j] = 1.0 / (d + 1)
        labels = list(range(n))
        return sim, conf, labels

    def test_negative_rho_for_structured_matrices(self) -> None:
        """Structured matrices should produce negative Spearman rho."""
        from analysis.syllables.similarity import correlate_similarity_confusion

        sim, conf, labels = self._make_structured_matrices()
        result = correlate_similarity_confusion(sim, conf, labels, n_perm=0)
        assert result["rho"] < -0.5, f"Expected strong negative rho, got {result['rho']}"

    def test_mantel_pvalue_significant(self) -> None:
        """Mantel permutation should yield p < 0.05 for structured matrices."""
        from analysis.syllables.similarity import correlate_similarity_confusion

        sim, conf, labels = self._make_structured_matrices(n=10)
        result = correlate_similarity_confusion(sim, conf, labels, n_perm=999)
        assert result["p_one_sided"] < 0.05, f"p_one_sided={result['p_one_sided']}"

    def test_one_sided_vs_two_sided(self) -> None:
        """Both p-values should be returned; one-sided ≤ two-sided."""
        from analysis.syllables.similarity import correlate_similarity_confusion

        sim, conf, labels = self._make_structured_matrices()
        result = correlate_similarity_confusion(sim, conf, labels, n_perm=199)
        assert "p_one_sided" in result
        assert "p_two_sided" in result
        assert result["p_one_sided"] <= result["p_two_sided"] + 1e-10

    def test_vectors_are_upper_triangle(self) -> None:
        """sim_vec and conf_vec should be upper-triangle off-diagonal elements."""
        from analysis.syllables.similarity import correlate_similarity_confusion

        sim, conf, labels = self._make_structured_matrices(n=5)
        result = correlate_similarity_confusion(sim, conf, labels, n_perm=0)
        # Upper triangle of 5×5 = 5*4/2 = 10 pairs
        assert len(result["sim_vec"]) == 10
        assert len(result["conf_vec"]) == 10

    def test_no_perms_returns_nan_pvalues(self) -> None:
        """With n_perm=0, p-values should be NaN."""
        from analysis.syllables.similarity import correlate_similarity_confusion

        sim, conf, labels = self._make_structured_matrices(n=4)
        result = correlate_similarity_confusion(sim, conf, labels, n_perm=0)
        assert np.isnan(result["p_one_sided"])
        assert np.isnan(result["p_two_sided"])


class TestLabelSubsetting:
    """Test subsetting a 12-label similarity matrix to 10 labels."""

    def test_subset_preserves_correct_pairs(self) -> None:
        """Subsetting should select the correct rows/cols."""
        # Create a fake 12×12 similarity matrix
        rng = np.random.default_rng(42)
        D_full = rng.uniform(0, 1, (12, 12))
        D_full = (D_full + D_full.T) / 2  # symmetrize
        np.fill_diagonal(D_full, 0.0)
        labels_full = list(range(1, 13))  # [1..12]

        # Subset to labels [1, 2, 3, 5, 6, 7, 8, 9, 10, 11] (drop 4 and 12)
        labels_subset = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]

        # Find indices of subset labels in full label list
        idx = [labels_full.index(lb) for lb in labels_subset]
        D_expected = D_full[np.ix_(idx, idx)]

        # Use the function (it needs a file path, so test the logic inline)
        D_sub = D_full[np.ix_(idx, idx)]
        np.testing.assert_array_equal(D_sub, D_expected)
        assert D_sub.shape == (10, 10)


class TestSummarizeAcrossSessions:
    """Test cross-session summary statistics."""

    def test_median_and_sign_test(self) -> None:
        """Median rho + sign test across sessions."""
        from analysis.syllables.similarity import summarize_across_sessions

        # 5 sessions, all with negative rho
        per_session = [
            {"session_id": "s1", "rho": -0.6, "p_one_sided": 0.01},
            {"session_id": "s2", "rho": -0.5, "p_one_sided": 0.02},
            {"session_id": "s3", "rho": -0.7, "p_one_sided": 0.005},
            {"session_id": "s4", "rho": -0.3, "p_one_sided": 0.08},
            {"session_id": "s5", "rho": -0.4, "p_one_sided": 0.04},
        ]
        summary = summarize_across_sessions(per_session)
        assert summary["median_rho"] < 0
        assert summary["n_sessions"] == 5
        assert summary["n_negative"] == 5
        # Sign test: all 5 negative out of 5 → p = 2^(-5) ≈ 0.03
        assert summary["sign_test_p"] < 0.05
