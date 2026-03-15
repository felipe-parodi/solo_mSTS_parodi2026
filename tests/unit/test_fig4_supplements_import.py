"""Verify fig4 supplement functions are importable from both old and new paths."""

from __future__ import annotations

import pytest

_SUPPLEMENT_NAMES = [
    "plot_fig4_supp_next_syllable",
    "plot_fig4_supp_descriptives",
    "plot_fig4_supp_confusion_matrix",
    "plot_fig4_supp_selectivity_by_bank",
    "plot_fig4_supp_transition_matrix",
    "plot_fig4_supp_transition_no_self",
    "plot_fig4_supp_onset_peth",
    "plot_fig4_supp_onset_peth_smoothed",
]


class TestNewImportPath:
    """All 8 supplement functions importable from fig4_supplements."""

    @pytest.mark.parametrize("name", _SUPPLEMENT_NAMES)
    def test_importable(self, name: str) -> None:
        from freemat.single_fm.viz import fig4_supplements
        assert hasattr(fig4_supplements, name)
        assert callable(getattr(fig4_supplements, name))


class TestBackwardCompat:
    """All 8 supplement functions still importable from fig4_panels."""

    @pytest.mark.parametrize("name", _SUPPLEMENT_NAMES)
    def test_importable(self, name: str) -> None:
        from freemat.single_fm.viz import fig4_panels
        assert hasattr(fig4_panels, name)
        assert callable(getattr(fig4_panels, name))


class TestIdentity:
    """Old and new paths resolve to the same function object."""

    @pytest.mark.parametrize("name", _SUPPLEMENT_NAMES)
    def test_same_object(self, name: str) -> None:
        from freemat.single_fm.viz import fig4_panels, fig4_supplements
        assert getattr(fig4_panels, name) is getattr(fig4_supplements, name)
