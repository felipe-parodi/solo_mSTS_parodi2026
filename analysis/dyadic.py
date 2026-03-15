"""Utilities for joint social state (dyadic) decoding.

This module provides functions for:
- Behavior pooling and normalization
- Dyadic state classification (parallel, interactive, asynchronous)
- Sample building for dyadic decoding analyses

Example usage:
    from data_utils.dyadic_utils import build_dyadic_samples, classify_dyadic_type

    X, y, label_names = build_dyadic_samples(
        m1_t, m1_rates, m1_bank_mask,
        m2_t, m2_rates,
        m1_segments, m1_labels,
        m2_segments, m2_labels,
        question="dyadic-type",
        min_total_duration=20.0
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Point events and non-states to exclude
EXCLUDE_LABELS = {
    "enters",
    "climb up",
    "climb down",
    "scratch",
    "threat",
    "threatened",
    "experimenter enters",
    "experimenter leaves",
    "start",
}


# Behavior pooling taxonomy
BEHAVIOR_POOL_MAP = {
    # Locomotion
    "travel": "locomotion",
    "pace": "locomotion",
    "chase": "locomotion",
    "chased": "locomotion",

    # Visual attention
    "scan": "visual",
    "inspect": "visual",

    # Social interactions (keep separate for fine-grained analysis)
    "groom give": "groom-give",
    "groom receive": "groom-receive",
    "self-groom": "self-groom",

    # Rare social interactions
    "mount": "social-misc",
    "mounted": "social-misc",
    "solicitation": "social-misc",
    "grab": "social-misc",
    "grabbed": "social-misc",

    # Stationary states (keep separate)
    "hang": "hang",
    "forage": "forage",
    "rest": "rest",

    # Other
    "pause": "rest",
    "approach": "locomotion",
}


def normalize_label(label: str) -> str:
    """Normalize a behavior label and unify common synonyms.

    Applies lowercase/strip, collapses internal whitespace, and maps known
    synonyms to canonical tokens used downstream (e.g., 'self groom' →
    'self-groom', 'groom_give' → 'groom-give', 'social_misc' → 'social-misc').

    Args:
        label: Raw behavior label

    Returns:
        Canonical, normalized label string (may be empty if input is None/empty)
    """
    if label is None:
        return ""
    s = str(label).strip().lower()
    if not s:
        return ""
    # Collapse whitespace
    s = " ".join(s.split())

    # Known synonym normalization to canonical tokens
    syn_map = {
        "self groom": "self-groom",
        "self_groom": "self-groom",
        "self-groom": "self-groom",
        "groom give": "groom-give",
        "groom_give": "groom-give",
        "groom-give": "groom-give",
        "groom receive": "groom-receive",
        "groom_receive": "groom-receive",
        "groom-receive": "groom-receive",
        "social misc": "social-misc",
        "social_misc": "social-misc",
        "social-misc": "social-misc",
        # Maintain climb naming used across modules
        "climb up": "climb up",
        "climb_up": "climb up",
        "climb down": "climb down",
        "climb_down": "climb down",
    }
    if s in syn_map:
        return syn_map[s]
    return s


def pool_behavior(label: str) -> str:
    """Apply behavior pooling map.

    Args:
        label: Normalized behavior label

    Returns:
        Pooled behavior label
    """
    if not label:
        return ""
    return BEHAVIOR_POOL_MAP.get(label, label)


def classify_dyadic_type(m1_state: str, m2_state: str) -> str:
    """Classify dyadic state into parallel, interactive, or asynchronous.

    Args:
        m1_state: M1's pooled behavior
        m2_state: M2's pooled behavior

    Returns:
        Type: "parallel", "interactive", or "asynchronous"

    Notes:
        - Parallel: Both doing same thing (rest__rest, locomotion__locomotion)
        - Interactive: Complementary social behaviors (groom-give__groom-receive, social-misc)
        - Asynchronous: Independent activities (rest__locomotion, forage__visual)
    """
    if not m1_state or not m2_state:
        return ""

    # Parallel: both doing same thing
    if m1_state == m2_state:
        return "parallel"

    # Interactive: reciprocal grooming
    groom_states = {"groom-give", "groom-receive"}
    if m1_state in groom_states and m2_state in groom_states:
        return "interactive"

    # Interactive: social-misc interactions
    if m1_state == "social-misc" or m2_state == "social-misc":
        return "interactive"

    # Otherwise asynchronous
    return "asynchronous"


def build_dyadic_labels(
    m1_segments: np.ndarray,
    m1_labels: np.ndarray,
    m2_segments: np.ndarray,
    m2_labels: np.ndarray,
    t_min: float,
    t_max: float,
    bin_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[int, str]]:
    """Build dyadic state labels with pooling and type classification.

    Args:
        m1_segments: M1 behavioral segments (N, 2)
        m1_labels: M1 behavior labels (N,)
        m2_segments: M2 behavioral segments (M, 2)
        m2_labels: M2 behavior labels (M,)
        t_min: Start time (seconds)
        t_max: End time (seconds)
        bin_s: Bin size (seconds)

    Returns:
        t_bins: Time bin centers (K,)
        dyadic_labels: Integer dyadic state labels (K,), -1 for unlabeled
        dyadic_types: Integer type labels (K,), 0=parallel, 1=interactive, 2=asynchronous, -1=unlabeled
        dyadic_names: List of unique dyadic state names (pooled)
        type_map: Dict mapping type_int -> type_name
    """
    # Create time bins
    t_bins = np.arange(t_min, t_max, bin_s)
    n_bins = len(t_bins)

    # Initialize state arrays
    m1_state = np.full(n_bins, "", dtype=object)
    m2_state = np.full(n_bins, "", dtype=object)

    # Fill M1 states (with pooling)
    for (t0, t1), label in zip(m1_segments, m1_labels):
        label_norm = normalize_label(label)
        if not label_norm or label_norm in EXCLUDE_LABELS:
            continue
        label_pooled = pool_behavior(label_norm)
        if not label_pooled:
            continue
        mask = (t_bins >= t0) & (t_bins < t1)
        m1_state[mask] = label_pooled

    # Fill M2 states (with pooling)
    for (t0, t1), label in zip(m2_segments, m2_labels):
        label_norm = normalize_label(label)
        if not label_norm or label_norm in EXCLUDE_LABELS:
            continue
        label_pooled = pool_behavior(label_norm)
        if not label_pooled:
            continue
        mask = (t_bins >= t0) & (t_bins < t1)
        m2_state[mask] = label_pooled

    # Build dyadic labels and types
    dyadic_strs = []
    dyadic_type_strs = []

    for i in range(n_bins):
        if m1_state[i] and m2_state[i]:
            dyadic_str = f"{m1_state[i]}__{m2_state[i]}"
            dyadic_strs.append(dyadic_str)

            dyadic_type = classify_dyadic_type(m1_state[i], m2_state[i])
            dyadic_type_strs.append(dyadic_type)
        else:
            dyadic_strs.append("")
            dyadic_type_strs.append("")

    # Get unique states and types
    dyadic_names = sorted(set(s for s in dyadic_strs if s))
    dyadic_to_int = {name: i for i, name in enumerate(dyadic_names)}

    type_map = {0: "parallel", 1: "interactive", 2: "asynchronous"}
    type_to_int = {v: k for k, v in type_map.items()}

    # Convert to integer labels
    dyadic_labels = np.array([
        dyadic_to_int[s] if s else -1
        for s in dyadic_strs
    ], dtype=int)

    dyadic_types = np.array([
        type_to_int[t] if t else -1
        for t in dyadic_type_strs
    ], dtype=int)

    return t_bins, dyadic_labels, dyadic_types, dyadic_names, type_map


def compute_dyadic_durations(
    dyadic_labels: np.ndarray,
    dyadic_types: np.ndarray,
    dyadic_names: List[str],
    type_map: Dict[int, str],
    bin_s: float,
) -> Dict[str, Dict[str, float]]:
    """Compute duration per dyadic state.

    Args:
        dyadic_labels: Integer labels (K,)
        dyadic_types: Integer type labels (K,)
        dyadic_names: List of dyadic state names
        type_map: Dict mapping type_int -> type_name
        bin_s: Bin size (seconds)

    Returns:
        Dict mapping dyadic_name -> {duration, type}
    """
    durations = {}

    for i, name in enumerate(dyadic_names):
        mask = (dyadic_labels == i)
        n_samples = int(mask.sum())
        duration = n_samples * bin_s

        # Get type
        type_idx = dyadic_types[mask][0] if mask.any() else -1
        type_name = type_map.get(type_idx, "unknown")

        durations[name] = {
            "duration": duration,
            "type": type_name,
        }

    return durations


def build_dyadic_samples(
    neural_t: np.ndarray,
    neural_rates: np.ndarray,
    unit_mask: np.ndarray,
    m1_segments: np.ndarray,
    m1_labels: np.ndarray,
    m2_segments: np.ndarray,
    m2_labels: np.ndarray,
    question: str,
    min_total_duration: float = 20.0,
    balance_epochs: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build samples for dyadic state decoding.

    Args:
        neural_t: Neural time vector (T,)
        neural_rates: Neural firing rates (T, U)
        unit_mask: Unit inclusion mask (U,)
        m1_segments: M1 behavioral segments (N, 2)
        m1_labels: M1 behavior labels (N,)
        m2_segments: M2 behavioral segments (M, 2)
        m2_labels: M2 behavior labels (M,)
        question: Decoding question type:
            - "dyadic-type": Decode parallel vs interactive vs asynchronous
            - "within-parallel": Decode states within parallel type
            - "within-asynchronous": Decode states within asynchronous type
            - "full-multiclass": Decode all dyadic states
        min_total_duration: Minimum duration per state (seconds)
        balance_epochs: If True, downsample to minority class before extracting samples
        rng: Random number generator

    Returns:
        X: Feature matrix (n_samples, n_units)
        y: Integer labels (n_samples,)
        label_names: List of label names
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build dyadic labels
    t_min = float(neural_t.min())
    t_max = float(neural_t.max())
    bin_s = float(np.median(np.diff(neural_t)))

    t_bins, dyadic_labels, dyadic_types, dyadic_names, type_map = build_dyadic_labels(
        m1_segments, m1_labels,
        m2_segments, m2_labels,
        t_min, t_max, bin_s
    )

    # Compute durations
    durations = compute_dyadic_durations(
        dyadic_labels, dyadic_types, dyadic_names, type_map, bin_s
    )

    # Filter by duration
    valid_states = [
        name for name, stat in durations.items()
        if stat["duration"] >= min_total_duration
    ]

    # Select labels based on question type
    if question == "dyadic-type":
        # Decode type (parallel, interactive, asynchronous)
        label_names = ["parallel", "interactive", "asynchronous"]

        # First, collect all epochs organized by type
        epochs_by_type = {type_name: [] for type_name in label_names}

        for name in valid_states:
            state_idx = dyadic_names.index(name)
            state_type = durations[name]["type"]

            # Get time windows for this state
            mask = (dyadic_labels == state_idx)
            if not mask.any():
                continue

            # Find contiguous epochs
            transitions = np.diff(mask.astype(int))
            starts = np.where(transitions == 1)[0] + 1
            ends = np.where(transitions == -1)[0] + 1

            # Handle edge cases
            if mask[0]:
                starts = np.concatenate([[0], starts])
            if mask[-1]:
                ends = np.concatenate([ends, [len(mask)]])

            # Store epoch info
            for start, end in zip(starts, ends):
                epoch_times = t_bins[start:end]
                epochs_by_type[state_type].append((epoch_times[0], epoch_times[-1]))

        # Balance epochs if requested
        if balance_epochs:
            # Find minority class
            min_epochs = min(len(epochs) for epochs in epochs_by_type.values())
            if min_epochs > 0:
                print(f"        Balancing epochs: {min_epochs} per type (was {[len(e) for e in epochs_by_type.values()]})")
                for type_name in epochs_by_type:
                    if len(epochs_by_type[type_name]) > min_epochs:
                        # Randomly downsample
                        indices = rng.choice(len(epochs_by_type[type_name]), size=min_epochs, replace=False)
                        epochs_by_type[type_name] = [epochs_by_type[type_name][i] for i in indices]

        # Now extract samples from (potentially balanced) epochs
        X_list = []
        y_list = []

        for type_name, type_label in zip(label_names, range(len(label_names))):
            for t_start, t_end in epochs_by_type[type_name]:
                # Find corresponding neural times
                neural_mask = (neural_t >= t_start) & (neural_t <= t_end)
                if not neural_mask.any():
                    continue

                # Mean rate over epoch
                x = np.mean(neural_rates[neural_mask][:, unit_mask], axis=0)
                X_list.append(x)
                y_list.append(type_label)

    elif question == "within-parallel":
        # Decode states within parallel type
        label_names = sorted([
            name for name in valid_states
            if durations[name]["type"] == "parallel"
        ])

        X_list, y_list = _extract_samples_for_states(
            neural_t, neural_rates, unit_mask,
            t_bins, dyadic_labels, dyadic_names,
            label_names, balance_epochs, rng
        )

    elif question == "within-asynchronous":
        # Decode states within asynchronous type
        label_names = sorted([
            name for name in valid_states
            if durations[name]["type"] == "asynchronous"
        ])

        X_list, y_list = _extract_samples_for_states(
            neural_t, neural_rates, unit_mask,
            t_bins, dyadic_labels, dyadic_names,
            label_names, balance_epochs, rng
        )

    elif question == "full-multiclass":
        # Decode all valid dyadic states
        label_names = sorted(valid_states)

        X_list, y_list = _extract_samples_for_states(
            neural_t, neural_rates, unit_mask,
            t_bins, dyadic_labels, dyadic_names,
            label_names, balance_epochs, rng
        )

    else:
        raise ValueError(f"Unknown question type: {question}")

    if not X_list:
        n_units = int(np.sum(unit_mask))
        return (
            np.zeros((0, n_units), dtype=float),
            np.zeros((0,), dtype=int),
            []
        )

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    return X, y, label_names


def _extract_samples_for_states(
    neural_t: np.ndarray,
    neural_rates: np.ndarray,
    unit_mask: np.ndarray,
    t_bins: np.ndarray,
    dyadic_labels: np.ndarray,
    dyadic_names: List[str],
    label_names: List[str],
    balance_epochs: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Helper to extract samples for specific dyadic states.

    Args:
        neural_t: Neural time vector
        neural_rates: Neural firing rates
        unit_mask: Unit inclusion mask
        t_bins: Time bins
        dyadic_labels: Integer dyadic state labels
        dyadic_names: List of all dyadic state names
        label_names: List of states to extract
        balance_epochs: If True, balance epochs to minority class
        rng: Random number generator

    Returns:
        X_list: List of feature vectors
        y_list: List of integer labels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # First collect all epochs by state
    epochs_by_state = {name: [] for name in label_names}

    for name in label_names:
        if name not in dyadic_names:
            continue

        state_idx = dyadic_names.index(name)

        # Get time windows for this state
        mask = (dyadic_labels == state_idx)
        if not mask.any():
            continue

        # Find contiguous epochs
        transitions = np.diff(mask.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        # Handle edge cases
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        # Store epoch info
        for start, end in zip(starts, ends):
            epoch_times = t_bins[start:end]
            epochs_by_state[name].append((epoch_times[0], epoch_times[-1]))

    # Balance epochs if requested
    if balance_epochs:
        # Find minority class
        min_epochs = min(len(epochs) for epochs in epochs_by_state.values() if epochs)
        if min_epochs > 0:
            print(f"        Balancing epochs: {min_epochs} per state")
            for state_name in epochs_by_state:
                if len(epochs_by_state[state_name]) > min_epochs:
                    # Randomly downsample
                    indices = rng.choice(len(epochs_by_state[state_name]), size=min_epochs, replace=False)
                    epochs_by_state[state_name] = [epochs_by_state[state_name][i] for i in indices]

    # Now extract samples from (potentially balanced) epochs
    X_list = []
    y_list = []

    for name in label_names:
        label_idx = label_names.index(name)
        for t_start, t_end in epochs_by_state[name]:
            # Find corresponding neural times
            neural_mask = (neural_t >= t_start) & (neural_t <= t_end)
            if not neural_mask.any():
                continue

            # Mean rate over epoch
            x = np.mean(neural_rates[neural_mask][:, unit_mask], axis=0)
            X_list.append(x)
            y_list.append(label_idx)

    return X_list, y_list
