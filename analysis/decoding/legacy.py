"""Legacy decoding utilities for backward compatibility.

This module provides functions from the original monolithic decoding module
that are still needed by other modules (e.g., synchrony.py) but haven't been
fully refactored into the modular decoding package.

These functions should be gradually migrated to proper locations:
- prepare_decoding_data → freemat_data/transforms/
- load_neural_and_behavior_data → freemat_data/loaders/

Usage:
    from analysis.decoding.legacy import prepare_decoding_data, load_neural_and_behavior_data
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports to avoid loading heavy dependencies until needed
_legacy_module = None


def _get_legacy_module():
    """Lazy load the legacy module."""
    global _legacy_module
    if _legacy_module is None:
        from analysis import _decoding_legacy
        _legacy_module = _decoding_legacy
    return _legacy_module


def prepare_decoding_data(*args, **kwargs):
    """Prepare aligned neural and behavioral data for SVM decoding.

    See analysis._decoding_legacy.prepare_decoding_data for full docstring.
    """
    return _get_legacy_module().prepare_decoding_data(*args, **kwargs)


def load_neural_and_behavior_data(*args, **kwargs):
    """Load neural and behavioral data for a session.

    See analysis._decoding_legacy.load_neural_and_behavior_data for full docstring.
    """
    return _get_legacy_module().load_neural_and_behavior_data(*args, **kwargs)


def check_unit_quality(*args, **kwargs):
    """Check unit quality metrics."""
    return _get_legacy_module().check_unit_quality(*args, **kwargs)


def run_decoder_with_cv(*args, **kwargs):
    """Run decoder with cross-validation."""
    return _get_legacy_module().run_decoder_with_cv(*args, **kwargs)


def noise_ceiling(*args, **kwargs):
    """Compute noise ceiling estimate."""
    return _get_legacy_module().noise_ceiling(*args, **kwargs)


def bootstrap_confidence_interval(*args, **kwargs):
    """Compute bootstrap confidence interval."""
    return _get_legacy_module().bootstrap_confidence_interval(*args, **kwargs)


__all__ = [
    "prepare_decoding_data",
    "load_neural_and_behavior_data",
    "check_unit_quality",
    "run_decoder_with_cv",
    "noise_ceiling",
    "bootstrap_confidence_interval",
]
