"""Syllable analysis infrastructure for Figure 3.

This subpackage provides analysis of behavioral syllables (motifs/llac) including:
- Bout segmentation from continuous frame-wise labels
- Alignment to neural time bins
- Multi-class syllable decoding
- Transition-aligned temporal dynamics
- Lead/lag analysis (predictive vs reactive coding)
- Bank × agent dissociation analysis

Terminology:
    syllable = motif = llac (Low-Level Action Cluster)
    ~15-20 unique classes derived from 3D pose

Main classes and functions:
    SyllableBout: Dataclass for a single bout (syllable_id, start, end, duration)
    segment_bouts: Extract bouts from continuous labels
    remap_to_top_n: Remap syllables to top-N vocabulary
    align_syllables_to_bins: Assign syllables to neural time bins
    detect_onsets: Find transition times
    decode_syllables: Multi-class syllable decoding
    compute_onset_aligned_accuracy: Accuracy as function of time relative to onset
    lag_sweep_decoding: Decode with shifted neural features

Usage:
    from analysis.syllables import segment_bouts, align_syllables_to_bins

    # Segment continuous labels into bouts
    bouts = segment_bouts(motif_ids, times, min_duration_sec=0.4)

    # Align to neural bins
    syllable_labels = align_syllables_to_bins(motif_ids, times, bin_edges)

    # Remap to top-N vocabulary
    remapped, mapping = remap_to_top_n(syllable_labels, n=10)

    # Decode syllables from spikes
    from analysis.syllables import decode_syllables, SyllableDecodingConfig
    config = SyllableDecodingConfig(top_n_syllables=10)
    result = decode_syllables(X_spk, syllable_labels, config)

    # Lag sweep for lead/lag analysis
    from analysis.syllables import lag_sweep_decoding
    result = lag_sweep_decoding(X_spk, syllable_labels)
"""

from .alignment import (
    align_syllables_to_bins,
    align_syllables_for_two_subjects,
    compute_bin_centers,
    detect_bin_onsets,
    get_onset_aligned_indices,
)
from .decoding import (
    SyllableDecodingConfig,
    SyllableDecodingResult,
    compute_permutation_null,
    decode_model_families,
    decode_syllables,
    decode_syllables_by_bank,
)
from .lag_sweep import (
    LagSweepResult,
    compute_pre_onset_accuracy,
    lag_sweep_by_bank,
    lag_sweep_decoding,
)
from .segmentation import (
    SyllableBout,
    compute_duration_stats,
    compute_transition_matrix,
    compute_transition_rate,
    detect_onsets,
    remap_to_top_n,
    segment_bouts,
)
from .transitions import (
    OnsetAlignedResult,
    compute_next_syllable_accuracy,
    compute_onset_aligned_accuracy,
    compute_onset_jitter_control,
    compute_onset_peth,
)

__all__ = [
    # Segmentation
    "SyllableBout",
    "segment_bouts",
    "detect_onsets",
    "remap_to_top_n",
    "compute_duration_stats",
    "compute_transition_rate",
    "compute_transition_matrix",
    # Alignment
    "align_syllables_to_bins",
    "align_syllables_for_two_subjects",
    "compute_bin_centers",
    "detect_bin_onsets",
    "get_onset_aligned_indices",
    # Decoding
    "SyllableDecodingConfig",
    "SyllableDecodingResult",
    "decode_syllables",
    "decode_syllables_by_bank",
    "decode_model_families",
    "compute_permutation_null",
    # Transitions
    "OnsetAlignedResult",
    "compute_onset_peth",
    "compute_onset_aligned_accuracy",
    "compute_onset_jitter_control",
    "compute_next_syllable_accuracy",
    # Lag sweep
    "LagSweepResult",
    "lag_sweep_decoding",
    "lag_sweep_by_bank",
    "compute_pre_onset_accuracy",
]
