#!/usr/bin/env python3
"""
spk_synchrony_unified.py
========================
Unified spike-train synchrony analysis combining best features from all versions.

Key features:
1. Consistent data loading using decoding pipeline
2. Vectorized synchrony computation for efficiency
3. Proper behavioral bout extraction and alignment
4. Multiple metrics and statistical testing
5. GPU acceleration support

Core Metrics:

    CCG Peak (Cross-Correlogram):
        CCG(τ) = IFFT(FFT(spike_A) × conj(FFT(spike_B)))
        metric = CCG[τ=0]  (zero-lag cross-correlation)

    JPSTH Coincidence Index:
        CI = Σ_t(A_t × B_t) / (Σ_t A_t × Σ_t B_t)
        Fraction of co-activity occurring at zero lag.

    Rate Correlation:
        r = corr(binned_rate_A, binned_rate_B)  (Pearson correlation)

Statistical Testing:
    Permutation test p-value:
        p = (1 + Σ_i 𝟙[|perm_diff_i| ≥ |obs_diff|]) / (1 + n_perm)

    Effect size (Cohen's d):
        d = (μ₁ - μ₂) / s_pooled
        s_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁ + n₂ - 2)]

Usage:
    python -m freemat.bhvr.spk_synchrony_unified --config synchrony_config.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    logging.warning("statsmodels not found. FDR correction will be skipped.")
    HAS_STATSMODELS = False

# Add parent directories to path for imports
script_dir = Path(__file__).parent.absolute()
analyses_dir = script_dir.parent
neural_preprocessing_dir = analyses_dir.parent
sys.path.append(str(analyses_dir))
sys.path.append(str(neural_preprocessing_dir))

from data_utils import gpu_utils
from data_utils.behavior_utils import clean_behavior
from data_utils.cache_utils import NeuralDataCache
from analysis.decoding.legacy import load_neural_and_behavior_data, prepare_decoding_data

try:
    import neural_preprocessing.analyses.freemat.nature_neuro_style as nns
    HAS_NATURE_STYLE = True
except ImportError:
    logging.warning("`nature_neuro_style.py` not found. Plots will use default styling.")
    HAS_NATURE_STYLE = False

# Import plotting functions from visualization module (refactored location)
try:
    from visualization.synchrony_plots import (
        plot_behavior_synchrony_heatmap,
        create_summary_synchrony_plot,
        plot_synchrony_comparison,
    )
except ImportError:
    logging.warning("visualization.synchrony_plots not found, plotting functions unavailable")

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Vectorized Synchrony Metrics
# -----------------------------------------------------------

def compute_ccg_peak_vectorized(m1_data: "gpu_utils.np.ndarray",
                               m2_data: "gpu_utils.np.ndarray",
                               max_lag_bins: int = 10) -> "gpu_utils.np.ndarray":
    """Vectorized CCG peak computation for all M1-M2 neuron pairs.

    Mathematical Definition:
        Cross-correlation via FFT for spike trains a, b:
            CCG(τ) = IFFT(FFT(a, n=2T-1) × FFT(flip(b), n=2T-1))

        For all pairs simultaneously using broadcasting:
            CCG_all[i,j,:] = IFFT(FFT(M1)[i,:] × FFT(flip(M2))[j,:])

        Output: CCG peak at zero lag (τ=0), i.e., middle of correlation.

    Args:
        m1_data: Spike data for M1, shape (n_units_m1, n_time).
        m2_data: Spike data for M2, shape (n_units_m2, n_time).
        max_lag_bins: Maximum lag to consider (not used in current impl).

    Returns:
        CCG peaks matrix of shape (n_units_m1, n_units_m2).
    """
    xp = gpu_utils.get_array_module(m1_data)
    n_m1, n_time = m1_data.shape
    n_m2 = m2_data.shape[0]
    
    if n_time == 0:
        return xp.full((n_m1, n_m2), xp.nan)
    
    # Check if data has any activity
    m1_sum = xp.sum(m1_data)
    m2_sum = xp.sum(m2_data)

    if m1_sum == 0 or m2_sum == 0:
        return xp.zeros((n_m1, n_m2))
    
    # Precompute FFTs for efficient correlation
    m1_fft = xp.fft.fft(m1_data, n=2*n_time-1, axis=1)
    m2_fft = xp.fft.fft(xp.flip(m2_data, axis=1), n=2*n_time-1, axis=1)  # Flip for correlation

    # Compute all cross-correlations at once using broadcasting
    ccgs = xp.fft.ifft(m1_fft[:, None, :] * m2_fft[None, :, :], axis=2).real

    # Extract zero-lag values (middle of correlation)
    mid = n_time - 1
    ccg_peaks = ccgs[:, :, mid]

    return ccg_peaks

def compute_jpsth_coincidence_vectorized(m1_data: "gpu_utils.np.ndarray",
                                        m2_data: "gpu_utils.np.ndarray") -> "gpu_utils.np.ndarray":
    """Vectorized JPSTH coincidence computation for all M1-M2 neuron pairs.

    Mathematical Definition:
        Joint Peri-Stimulus Time Histogram coincidence index:

            CI[i,j] = Σ_t(A_i[t] × B_j[t]) / (Σ_t A_i[t] × Σ_t B_j[t])

        Numerator: Sum of coincident activity (diagonal of outer product).
        Denominator: Product of total activity (normalizes by firing rates).

        CI ∈ [0, 1] where higher values indicate more synchronized firing.

    Args:
        m1_data: Spike data for M1, shape (n_units_m1, n_time).
        m2_data: Spike data for M2, shape (n_units_m2, n_time).

    Returns:
        Coincidence index matrix of shape (n_units_m1, n_units_m2).
    """
    xp = gpu_utils.get_array_module(m1_data)
    
    if m1_data.shape[1] == 0 or m2_data.shape[1] == 0:
        return xp.full((m1_data.shape[0], m2_data.shape[0]), xp.nan)
    
    # Compute coincidence index: trace(JPSTH) / sum(JPSTH)
    # Using broadcasting to compute for all pairs at once
    jpsth_diag = xp.sum(m1_data[:, None, :] * m2_data[None, :, :], axis=2)
    jpsth_sum = xp.sum(xp.outer(xp.sum(m1_data, axis=1), xp.sum(m2_data, axis=1)))
    
    return jpsth_diag / (jpsth_sum + 1e-12)

def compute_rate_correlation_vectorized(m1_data: "gpu_utils.np.ndarray",
                                       m2_data: "gpu_utils.np.ndarray",
                                       bin_step: int = 5) -> "gpu_utils.np.ndarray":
    """Vectorized rate correlation computation for all M1-M2 neuron pairs.

    Mathematical Definition:
        Pearson correlation on binned (smoothed) spike rates:

            r[i,j] = Σ_k(x_k - x̄)(y_k - ȳ) / √[Σ_k(x_k - x̄)² × Σ_k(y_k - ȳ)²]

        Where:
            - x, y are binned spike counts (bin_step consecutive bins summed)
            - k indexes the coarser time bins
            - r ∈ [-1, 1]

        Binning reduces noise and captures slower rate co-fluctuations.

    Args:
        m1_data: Spike data for M1, shape (n_units_m1, n_time).
        m2_data: Spike data for M2, shape (n_units_m2, n_time).
        bin_step: Number of fine bins to aggregate into each coarse bin.

    Returns:
        Correlation matrix of shape (n_units_m1, n_units_m2).
    """
    xp = gpu_utils.get_array_module(m1_data)
    n_time = m1_data.shape[1]
    
    if n_time < bin_step:
        return xp.full((m1_data.shape[0], m2_data.shape[0]), xp.nan)
    
    # Smooth rates by binning
    n_bins = n_time // bin_step
    if n_bins <= 1:
        return xp.full((m1_data.shape[0], m2_data.shape[0]), xp.nan)
    
    # Reshape to bin data
    m1_binned = m1_data[:, :n_bins*bin_step].reshape(m1_data.shape[0], n_bins, bin_step).sum(axis=2)
    m2_binned = m2_data[:, :n_bins*bin_step].reshape(m2_data.shape[0], n_bins, bin_step).sum(axis=2)
    
    # Compute correlations for all pairs
    correlations = xp.full((m1_data.shape[0], m2_data.shape[0]), xp.nan)
    
    for i in range(m1_data.shape[0]):
        for j in range(m2_data.shape[0]):
            r1, r2 = m1_binned[i], m2_binned[j]
            if r1.std() > 0 and r2.std() > 0:
                correlations[i, j] = xp.corrcoef(r1, r2)[0, 1]
    
    return correlations

SYNCHRONY_METRICS = {
    'ccg_peak': compute_ccg_peak_vectorized,
    'jpsth_coinc': compute_jpsth_coincidence_vectorized,
    'rate_corr': compute_rate_correlation_vectorized
}

# -----------------------------------------------------------
# Data Processing Functions  
# -----------------------------------------------------------

def extract_behavioral_windows_from_labels(y_labels: List[str], 
                                           target_behavior: str, 
                                           min_duration_bins: int) -> List[Tuple[int, int]]:
    """
    Extract contiguous windows where behavior equals target_behavior.
    
    Args:
        y_labels: List of behavior labels aligned to neural data
        target_behavior: Target behavior to extract
        min_duration_bins: Minimum duration in bins
        
    Returns:
        List of (start, end) tuples for valid behavioral windows
    """
    if not y_labels:
        return []
    
    # Create boolean mask
    mask = np.array([label == target_behavior for label in y_labels])
    
    # Find contiguous True periods
    if not mask.any():
        return []
        
    # Find transitions
    transitions = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Filter by minimum duration
    windows = []
    for start, end in zip(starts, ends):
        if end - start >= min_duration_bins:
            windows.append((start, end))
            
    return windows

def compute_synchrony_for_windows(m1_spikes: "gpu_utils.np.ndarray",
                                 m2_spikes: "gpu_utils.np.ndarray", 
                                 windows: List[Tuple[int, int]],
                                 metric_name: str,
                                 **metric_kwargs) -> List[float]:
    """
    Compute synchrony values for specified time windows.
    """
    metric_func = SYNCHRONY_METRICS[metric_name]
    sync_values = []

    for i, (start, end) in enumerate(windows):
        # Extract data for this window
        m1_window = m1_spikes[:, start:end]
        m2_window = m2_spikes[:, start:end]

        # Compute synchrony matrix for all pairs
        sync_matrix = metric_func(m1_window, m2_window, **metric_kwargs)

        # Take mean across all pairs (excluding NaN values)
        if sync_matrix.size > 0:
            mean_sync = float(gpu_utils.np.nanmean(sync_matrix))
            if not gpu_utils.np.isnan(mean_sync):
                sync_values.append(mean_sync)

    return sync_values

def sample_matched_control_windows(control_windows: List[Tuple[int, int]],
                                  target_durations: List[int],
                                  n_samples: int = 1) -> List[Tuple[int, int]]:
    """
    Sample control windows matched to target durations.
    
    Args:
        control_windows: Available control periods
        target_durations: Target durations to match
        n_samples: Number of samples per duration
        
    Returns:
        List of sampled control windows
    """
    matched_windows = []
    
    for target_duration in target_durations:
        # Find control windows long enough
        valid_windows = [(start, end) for start, end in control_windows 
                        if end - start >= target_duration]
        
        if not valid_windows:
            continue
            
        # Sample random windows of target duration
        for _ in range(n_samples):
            window_start, window_end = valid_windows[np.random.randint(len(valid_windows))]
            # Random start within the window
            max_start = window_end - target_duration
            sample_start = np.random.randint(window_start, max_start + 1)
            matched_windows.append((sample_start, sample_start + target_duration))
    
    return matched_windows

def permutation_test(group1_values: List[float],
                    group2_values: List[float],
                    n_permutations: int = 10000,
                    alternative: str = 'two-sided') -> float:
    """Permutation test comparing two groups of synchrony values.

    Mathematical Definition:
        H₀: Both groups come from the same distribution.

        Test statistic: observed_diff = mean(group2) - mean(group1)

        Procedure:
            For i = 1..n_permutations:
                1. Pool all values and shuffle
                2. Split into two groups of original sizes
                3. perm_diff_i = mean(shuffled_group2) - mean(shuffled_group1)

        P-value (two-sided):
            p = (1 + Σ_i 𝟙[|perm_diff_i| ≥ |observed_diff|]) / (1 + n_perm)

        P-value (greater):
            p = (1 + Σ_i 𝟙[perm_diff_i ≥ observed_diff]) / (1 + n_perm)

        The +1 in numerator and denominator ensures p ∈ (0, 1].

    Args:
        group1_values: Synchrony values for group 1 (e.g., rest).
        group2_values: Synchrony values for group 2 (e.g., behavior).
        n_permutations: Number of permutations (default 10000).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        P-value for the permutation test.
    """
    if not group1_values or not group2_values:
        return np.nan
        
    observed_diff = np.mean(group2_values) - np.mean(group1_values)
    all_values = np.array(group1_values + group2_values)
    n_group1 = len(group1_values)
    
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(all_values)
        perm_diff = np.mean(all_values[n_group1:]) - np.mean(all_values[:n_group1])
        
        if alternative == 'greater' and perm_diff >= observed_diff:
            count += 1
        elif alternative == 'less' and perm_diff <= observed_diff:
            count += 1
        elif alternative == 'two-sided' and abs(perm_diff) >= abs(observed_diff):
            count += 1
            
    return (count + 1) / (n_permutations + 1)

# -----------------------------------------------------------
# Analysis Functions
# -----------------------------------------------------------

def analyze_all_behavior_synchrony(m1_spikes: "gpu_utils.np.ndarray",
                                  m2_spikes: "gpu_utils.np.ndarray",
                                  m1_behavior_labels: List[str],
                                  m2_behavior_labels: List[str], 
                                  all_behaviors: List[str],
                                  min_duration_bins: int,
                                  metric_name: str,
                                  n_permutations: int = 1000,
                                  **metric_kwargs) -> Dict:
    """
    Comprehensive synchrony analysis for all behavior combinations.
    """
    # Align behavioral data
    min_length = min(len(m1_behavior_labels), len(m2_behavior_labels), 
                    m1_spikes.shape[1], m2_spikes.shape[1])
    
    # Create joint behavior labels for simultaneous periods
    joint_behavior_pairs = []
    for i in range(min_length):
        if (i < len(m1_behavior_labels) and i < len(m2_behavior_labels)):
            joint_behavior_pairs.append((m1_behavior_labels[i], m2_behavior_labels[i]))
    
    # Initialize results matrix for all behavior pairs
    synchrony_matrix = {}
    window_counts_matrix = {}
    pvalue_matrix = {}
    effect_size_matrix = {}
    
    # Analyze all possible M1-M2 behavior combinations
    for m1_behavior in all_behaviors:
        synchrony_matrix[m1_behavior] = {}
        window_counts_matrix[m1_behavior] = {}
        pvalue_matrix[m1_behavior] = {}
        effect_size_matrix[m1_behavior] = {}
        
        for m2_behavior in all_behaviors:
            # Count occurrences
            m1_count = sum(1 for m1_beh, _ in joint_behavior_pairs if m1_beh == m1_behavior)
            m2_count = sum(1 for _, m2_beh in joint_behavior_pairs if m2_beh == m2_behavior)
            simultaneous_count = sum(1 for m1_beh, m2_beh in joint_behavior_pairs
                                   if m1_beh == m1_behavior and m2_beh == m2_behavior)

            # Define behavioral constraints
            exclusive_pairs = [
                ('groom give', 'groom give'),  # Both can't give simultaneously
                ('groom receive', 'groom receive'),  # Both can't receive simultaneously  
                ('chase', 'chase'),  # Both can't chase simultaneously
                ('chased', 'chased'),  # Both can't be chased simultaneously
                ('grab', 'grab'),  # Both can't grab simultaneously
                ('grabbed', 'grabbed'),  # Both can't be grabbed simultaneously
            ]
            
            # Define complementary pairs (these SHOULD occur together)
            complementary_pairs = [
                ('groom give', 'groom receive'),   # If M1 gives, M2 receives
                ('groom receive', 'groom give'),   # If M1 receives, M2 gives
                ('chase', 'chased'),              # If M1 chases, M2 is chased
                ('chased', 'chase'),              # If M1 is chased, M2 chases
                ('grab', 'grabbed'),              # If M1 grabs, M2 is grabbed
                ('grabbed', 'grab'),              # If M1 is grabbed, M2 grabs
            ]
            
            # Check if this combination is impossible (exclusive)
            if (m1_behavior, m2_behavior) in exclusive_pairs:
                synchrony_matrix[m1_behavior][m2_behavior] = np.nan
                window_counts_matrix[m1_behavior][m2_behavior] = 0
                pvalue_matrix[m1_behavior][m2_behavior] = np.nan
                effect_size_matrix[m1_behavior][m2_behavior] = np.nan
                continue
            
            # Find windows where M1 shows m1_behavior AND M2 shows m2_behavior
            simultaneous_windows = []
            current_start = None
            
            for i, (m1_beh, m2_beh) in enumerate(joint_behavior_pairs):
                if m1_beh == m1_behavior and m2_beh == m2_behavior:
                    if current_start is None:
                        current_start = i
                elif current_start is not None:
                    # End of a window
                    if i - current_start >= min_duration_bins:
                        simultaneous_windows.append((current_start, i))
                    current_start = None
            
            # Handle case where behavior continues to the end
            if current_start is not None:
                if len(joint_behavior_pairs) - current_start >= min_duration_bins:
                    simultaneous_windows.append((current_start, len(joint_behavior_pairs)))

            if len(simultaneous_windows) > 0:
                # Compute synchrony for these windows
                sync_values = compute_synchrony_for_windows(
                    m1_spikes, m2_spikes, simultaneous_windows, metric_name, **metric_kwargs
                )
                
                mean_synchrony = np.mean(sync_values) if sync_values else np.nan
                synchrony_matrix[m1_behavior][m2_behavior] = mean_synchrony
                window_counts_matrix[m1_behavior][m2_behavior] = len(sync_values)
                
                # Statistical testing vs rest periods (if sufficient data)
                if len(sync_values) >= 3:  # Need minimum samples for stats
                    # Find rest windows for comparison
                    rest_windows = []
                    current_start = None
                    
                    for i, (m1_beh, m2_beh) in enumerate(joint_behavior_pairs):
                        if m1_beh == 'rest' and m2_beh == 'rest':
                            if current_start is None:
                                current_start = i
                        elif current_start is not None:
                            if i - current_start >= min_duration_bins:
                                rest_windows.append((current_start, i))
                            current_start = None
                    
                    # Handle case where rest continues to the end
                    if current_start is not None:
                        if len(joint_behavior_pairs) - current_start >= min_duration_bins:
                            rest_windows.append((current_start, len(joint_behavior_pairs)))
                    
                    if len(rest_windows) >= 3:  # Need minimum rest windows
                        # Sample matched rest windows
                        behavior_durations = [end - start for start, end in simultaneous_windows]
                        matched_rest_windows = sample_matched_control_windows(
                            rest_windows, behavior_durations, n_samples=min(len(rest_windows), len(behavior_durations) * 2)
                        )
                        
                        if len(matched_rest_windows) > 0:
                            rest_sync_values = compute_synchrony_for_windows(
                                m1_spikes, m2_spikes, matched_rest_windows, metric_name, **metric_kwargs
                            )
                            
                            if len(rest_sync_values) > 0:
                                # Permutation test
                                p_value = permutation_test(rest_sync_values, sync_values, n_permutations, 'two-sided')
                                
                                # Effect size (Cohen's d)
                                pooled_std = np.sqrt(((len(sync_values) - 1) * np.var(sync_values) + 
                                                    (len(rest_sync_values) - 1) * np.var(rest_sync_values)) / 
                                                   (len(sync_values) + len(rest_sync_values) - 2))
                                effect_size = (np.mean(sync_values) - np.mean(rest_sync_values)) / pooled_std if pooled_std > 0 else 0
                                
                                pvalue_matrix[m1_behavior][m2_behavior] = p_value
                                effect_size_matrix[m1_behavior][m2_behavior] = effect_size
                            else:
                                pvalue_matrix[m1_behavior][m2_behavior] = np.nan
                                effect_size_matrix[m1_behavior][m2_behavior] = np.nan
                        else:
                            pvalue_matrix[m1_behavior][m2_behavior] = np.nan
                            effect_size_matrix[m1_behavior][m2_behavior] = np.nan
                    else:
                        pvalue_matrix[m1_behavior][m2_behavior] = np.nan
                        effect_size_matrix[m1_behavior][m2_behavior] = np.nan
                else:
                    pvalue_matrix[m1_behavior][m2_behavior] = np.nan
                    effect_size_matrix[m1_behavior][m2_behavior] = np.nan
            else:
                synchrony_matrix[m1_behavior][m2_behavior] = np.nan
                window_counts_matrix[m1_behavior][m2_behavior] = 0
                pvalue_matrix[m1_behavior][m2_behavior] = np.nan
                effect_size_matrix[m1_behavior][m2_behavior] = np.nan
    
    return {
        'synchrony_matrix': synchrony_matrix,
        'window_counts': window_counts_matrix,
        'pvalue_matrix': pvalue_matrix,
        'effect_size_matrix': effect_size_matrix,
        'behaviors': all_behaviors,
        'metric': metric_name
    }

def analyze_behavior_vs_rest(m1_spikes: "gpu_utils.np.ndarray",
                            m2_spikes: "gpu_utils.np.ndarray",
                            m1_behavior_labels: List[str],
                            m2_behavior_labels: List[str], 
                            target_behavior: str,
                            min_duration_bins: int,
                            metric_name: str,
                            n_permutations: int = 10000,
                            **metric_kwargs) -> Dict:
    """
    Compare synchrony during target behavior vs rest periods.
    """
    # Find simultaneous target behavior periods
    min_length = min(len(m1_behavior_labels), len(m2_behavior_labels),
                    m1_spikes.shape[1], m2_spikes.shape[1])
    
    # Create joint behavior labels for simultaneous periods
    joint_labels = []
    simultaneous_target_count = 0
    simultaneous_rest_count = 0
    
    for i in range(min_length):
        if (i < len(m1_behavior_labels) and i < len(m2_behavior_labels) and
            m1_behavior_labels[i] == target_behavior and 
            m2_behavior_labels[i] == target_behavior):
            joint_labels.append(target_behavior)
            simultaneous_target_count += 1
        elif (i < len(m1_behavior_labels) and i < len(m2_behavior_labels) and
              m1_behavior_labels[i] == 'rest' and m2_behavior_labels[i] == 'rest'):
            joint_labels.append('rest')
            simultaneous_rest_count += 1
        else:
            joint_labels.append('other')

    # Extract behavioral windows
    behavior_windows = extract_behavioral_windows_from_labels(
        joint_labels, target_behavior, min_duration_bins
    )
    rest_windows = extract_behavioral_windows_from_labels(
        joint_labels, 'rest', min_duration_bins
    )

    logger.info(f"Found {len(behavior_windows)} {target_behavior} windows, {len(rest_windows)} rest windows")

    # Compute synchrony for behavior windows
    behavior_sync_values = compute_synchrony_for_windows(
        m1_spikes, m2_spikes, behavior_windows, metric_name, **metric_kwargs
    )

    # Sample matched rest windows
    behavior_durations = [end - start for start, end in behavior_windows]

    matched_rest_windows = sample_matched_control_windows(
        rest_windows, behavior_durations, n_samples=1
    )

    # Compute synchrony for matched rest windows
    rest_sync_values = compute_synchrony_for_windows(
        m1_spikes, m2_spikes, matched_rest_windows, metric_name, **metric_kwargs
    )

    # Statistical comparison
    p_value = permutation_test(rest_sync_values, behavior_sync_values,
                              n_permutations, 'two-sided')

    logger.info(f"Behavior sync: {len(behavior_sync_values)} values, "
                f"Rest sync: {len(rest_sync_values)} values, p={p_value:.4f}")
    
    return {
        'behavior_sync_mean': np.mean(behavior_sync_values) if behavior_sync_values else np.nan,
        'behavior_sync_std': np.std(behavior_sync_values) if behavior_sync_values else np.nan,
        'rest_sync_mean': np.mean(rest_sync_values) if rest_sync_values else np.nan,
        'rest_sync_std': np.std(rest_sync_values) if rest_sync_values else np.nan,
        'p_value': p_value,
        'n_behavior_windows': len(behavior_sync_values),
        'n_rest_windows': len(rest_sync_values),
        'effect_size': ((np.mean(behavior_sync_values) - np.mean(rest_sync_values)) 
                       if behavior_sync_values and rest_sync_values else np.nan)
    }

# -----------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------

def plot_behavior_synchrony_heatmap(results: Dict, output_dir: Path, 
                                   filename_prefix: str, plot_format: str = 'png') -> None:
    """Plot comprehensive behavior synchrony heatmap."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    synchrony_matrix = results['synchrony_matrix']
    window_counts = results['window_counts']
    pvalue_matrix = results.get('pvalue_matrix', {})
    effect_size_matrix = results.get('effect_size_matrix', {})
    behaviors = results['behaviors']
    metric_name = results['metric']
    
    # Convert to numpy matrices
    n_behaviors = len(behaviors)
    sync_array = np.full((n_behaviors, n_behaviors), np.nan)
    counts_array = np.zeros((n_behaviors, n_behaviors))
    pvalue_array = np.full((n_behaviors, n_behaviors), np.nan)
    effect_size_array = np.full((n_behaviors, n_behaviors), np.nan)
    
    for i, m1_beh in enumerate(behaviors):
        for j, m2_beh in enumerate(behaviors):
            if m1_beh in synchrony_matrix and m2_beh in synchrony_matrix[m1_beh]:
                sync_array[i, j] = synchrony_matrix[m1_beh][m2_beh]
                counts_array[i, j] = window_counts[m1_beh][m2_beh]
                if m1_beh in pvalue_matrix and m2_beh in pvalue_matrix[m1_beh]:
                    pvalue_array[i, j] = pvalue_matrix[m1_beh][m2_beh]
                if m1_beh in effect_size_matrix and m2_beh in effect_size_matrix[m1_beh]:
                    effect_size_array[i, j] = effect_size_matrix[m1_beh][m2_beh]
    
    # Create figure with subplots
    if HAS_NATURE_STYLE:
        nns.apply_nature_style()
        fig, (ax1, ax2) = nns.create_subplots(1, 2, width=12, height=6)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Synchrony values heatmap
    mask = np.isnan(sync_array)
    im1 = ax1.imshow(sync_array, cmap='viridis', aspect='auto')
    
    # Add text annotations for synchrony values
    exclusive_pairs = [
        ('groom give', 'groom give'), ('groom receive', 'groom receive'),  
        ('chase', 'chase'), ('chased', 'chased'),
        ('grab', 'grab'), ('grabbed', 'grabbed'),
    ]
    
    for i in range(n_behaviors):
        for j in range(n_behaviors):
            m1_beh = behaviors[i]
            m2_beh = behaviors[j]
            
            if (m1_beh, m2_beh) in exclusive_pairs:
                # Mark exclusive pairs with X
                ax1.text(j, i, 'X', ha='center', va='center', 
                        color='red', fontsize=12, fontweight='bold')
            elif not mask[i, j] and counts_array[i, j] > 0:
                # Format value with significance markers
                sync_val = sync_array[i, j]
                p_val = pvalue_array[i, j]
                
                # Add significance stars
                if not np.isnan(p_val):
                    if p_val < 0.001:
                        sig_marker = '***'
                    elif p_val < 0.01:
                        sig_marker = '**'
                    elif p_val < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = ''
                    text = f'{sync_val:.1f}{sig_marker}'
                else:
                    text = f'{sync_val:.1f}'
                
                ax1.text(j, i, text, ha='center', va='center', 
                        color='white' if sync_val > np.nanmean(sync_array) else 'black',
                        fontsize=7)
            elif mask[i, j]:
                # Mark NaN values with -
                ax1.text(j, i, '-', ha='center', va='center', 
                        color='gray', fontsize=8)
    
    # Extract bank from filename_prefix
    bank_name = filename_prefix.split('_')[0] if '_' in filename_prefix else 'all'
    ax1.set_title(f'Neural Synchrony Matrix - {bank_name.upper()} Bank\n({metric_name.upper()})')
    ax1.set_xlabel('M2 Behaviors')
    ax1.set_ylabel('M1 Behaviors')
    ax1.set_xticks(range(n_behaviors))
    ax1.set_yticks(range(n_behaviors))
    ax1.set_xticklabels(behaviors, rotation=45, ha='right')
    ax1.set_yticklabels(behaviors)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Synchrony Value')
    
    # Plot 2: Window counts heatmap  
    im2 = ax2.imshow(counts_array, cmap='Blues', aspect='auto')
    
    # Add text annotations for window counts
    for i in range(n_behaviors):
        for j in range(n_behaviors):
            if counts_array[i, j] > 0:
                text = f'{int(counts_array[i, j])}'
                ax2.text(j, i, text, ha='center', va='center',
                        color='white' if counts_array[i, j] > np.mean(counts_array) else 'black',
                        fontsize=8)
    
    ax2.set_title(f'Number of Analysis Windows - {bank_name.upper()} Bank')
    ax2.set_xlabel('M2 Behaviors')  
    ax2.set_ylabel('M1 Behaviors')
    ax2.set_xticks(range(n_behaviors))
    ax2.set_yticks(range(n_behaviors))
    ax2.set_xticklabels(behaviors, rotation=45, ha='right')
    ax2.set_yticklabels(behaviors)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Window Count')
    
    plt.tight_layout()
    plot_path = output_dir / f"{filename_prefix}.{plot_format}"
    
    if HAS_NATURE_STYLE:
        nns.save_figure(str(plot_path))
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved synchrony heatmap to {plot_path}")
    
    # Also create a summary plot with just the most interesting pairs
    create_summary_synchrony_plot(results, output_dir, filename_prefix, plot_format)

def create_summary_synchrony_plot(results: Dict, output_dir: Path, 
                                 filename_prefix: str, plot_format: str = 'png') -> None:
    """Create a focused plot of the most interesting behavior synchrony pairs."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    synchrony_matrix = results['synchrony_matrix']
    window_counts = results['window_counts']
    behaviors = results['behaviors']
    metric_name = results['metric']
    
    # Focus on specific interesting behavior pairs
    interesting_pairs = [
        ('groom give', 'groom receive'),
        ('groom receive', 'groom give'), 
        ('chase', 'chased'),
        ('grab', 'grabbed'),
        ('mount', 'mounted'),
        ('rest', 'rest'),
        ('travel', 'travel'),
        ('self-groom', 'self-groom')
    ]
    
    # Filter pairs that exist in our data
    valid_pairs = []
    sync_values = []
    count_values = []
    
    for m1_beh, m2_beh in interesting_pairs:
        if (m1_beh in behaviors and m2_beh in behaviors and 
            m1_beh in synchrony_matrix and m2_beh in synchrony_matrix[m1_beh]):
            sync_val = synchrony_matrix[m1_beh][m2_beh]
            count_val = window_counts[m1_beh][m2_beh]
            if not np.isnan(sync_val) and count_val > 0:
                valid_pairs.append(f"{m1_beh} → {m2_beh}")
                sync_values.append(sync_val)
                count_values.append(count_val)
    
    if not valid_pairs:
        logger.warning("No valid interesting pairs found for summary plot")
        return
    
    # Create bar plot
    if HAS_NATURE_STYLE:
        nns.apply_nature_style()
        fig, ax = nns.create_figure(width=10, height=6)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(valid_pairs)), sync_values, alpha=0.7)
    
    # Color bars based on synchrony strength
    sync_array = np.array(sync_values)
    norm_values = (sync_array - sync_array.min()) / (sync_array.max() - sync_array.min())
    
    for bar, norm_val in zip(bars, norm_values):
        bar.set_color(plt.cm.viridis(norm_val))
    
    # Add count annotations
    for i, (sync_val, count_val) in enumerate(zip(sync_values, count_values)):
        ax.text(i, sync_val + sync_array.max() * 0.02, f'n={count_val}', 
               ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Behavior Pairs (M1 → M2)')
    ax.set_ylabel(f'Neural Synchrony ({metric_name.upper()})')
    ax.set_title('Key Behavior Pair Synchrony')
    ax.set_xticks(range(len(valid_pairs)))
    ax.set_xticklabels(valid_pairs, rotation=45, ha='right')
    
    if HAS_NATURE_STYLE:
        nns.clean_axes(ax)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = output_dir / f"{filename_prefix}_summary.{plot_format}"
    
    if HAS_NATURE_STYLE:
        nns.save_figure(str(plot_path))
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved synchrony summary plot to {plot_path}")

def plot_synchrony_comparison(results: Dict, behavior_name: str, 
                             output_dir: Path, plot_format: str = 'png') -> None:
    """Plot synchrony comparison between behavior and rest."""
    import matplotlib.pyplot as plt
    
    if HAS_NATURE_STYLE:
        nns.apply_nature_style()
        fig, ax = nns.create_figure(width=4, height=5)
        colors = [nns.NATURE_COLORS['blue'], nns.NATURE_COLORS['grey']]
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ['#2E86AB', '#A23B72']
    
    # Data for plotting
    means = [results['behavior_sync_mean'], results['rest_sync_mean']]
    stds = [results['behavior_sync_std'], results['rest_sync_std']]
    labels = [behavior_name.title(), 'Rest']
    
    # Bar plot
    x_pos = [0, 1]
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Neural Synchrony')
    
    # Add significance annotation
    if results['p_value'] < 0.05:
        significance = '*' if results['p_value'] < 0.05 else 'ns'
        y_max = max(np.nanmax(np.array(means) + np.array(stds)), 0)
        ax.text(0.5, y_max * 1.1, significance, ha='center', va='bottom', fontsize=16)
    
    ax.set_title(f"Neural Synchrony: {behavior_name.title()} vs Rest\\n"
                f"p = {results['p_value']:.3f}, effect size = {results['effect_size']:.3f}")
    
    # Clean up axes
    if HAS_NATURE_STYLE:
        nns.clean_axes(ax)
    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = output_dir / f"synchrony_{behavior_name}_vs_rest.{plot_format}"
    
    if HAS_NATURE_STYLE:
        nns.save_figure(str(plot_path))
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    logger.info(f"Saved synchrony comparison plot to {plot_path}")

# -----------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main(config_path: str, verbose: bool = False) -> None:
    """Main synchrony analysis pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=== Starting Unified Neural Synchrony Analysis ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize GPU utilities
    use_gpu = config.get('analysis_params', {}).get('use_gpu', False)
    gpu_utils.initialize(use_gpu=use_gpu)
    
    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / Path(config_path).name)
    
    # Setup cache
    cache_dir = Path(config.get('cache_dir', 'cache'))
    cache = NeuralDataCache(cache_dir=cache_dir)
    
    # Data loading parameters
    bin_duration_sec = config['analysis_params']['neural_bin_duration_sec']
    min_bout_bins = config['analysis_params']['min_bout_bins']
    min_behavior_duration_sec = min_bout_bins * bin_duration_sec
    
    logger.info(f"Using bin_duration_sec={bin_duration_sec}, min_bout_bins={min_bout_bins}, min_behavior_duration_sec={min_behavior_duration_sec}")
    
    # Analysis parameters
    metrics = config['analysis_params'].get('metrics', ['ccg_peak'])
    n_perm = config['analysis_params']['n_perm']
    
    # Run analysis for each bank type
    all_results = []
    
    for bank_type in config['analysis_params']['bank_types_to_run']:
        logger.info(f"Processing {bank_type} bank...")
        
        # Load neural and behavioral data using decoding pipeline
        # M1 data
        m1_neural, m1_behavior_df, m1_sec_col, m1_beh_col = load_neural_and_behavior_data(
            neural_mat_file=config['data_paths']['m1_neural_mat_file'],
            behavioral_labels_file_path=config['data_paths']['behavioral_labels_file'],
            behavior_key_col_name_template="M1-Behavior",
            use_multiunit=config['analysis_params']['use_multiunit'],
            bank_type=bank_type,
            ventral_deuteron_channels=config['channel_config'].get('m1_ventral', []),
            dorsal_deuteron_channels=config['channel_config'].get('m1_dorsal', []),
            bin_from_raw_spikes=True,
            bin_duration_sec=bin_duration_sec,
            use_gpu=use_gpu
        )

        if m1_neural is None:
            logger.error(f"Failed to load M1 data for {bank_type} bank")
            continue

        # M2 data
        m2_neural, m2_behavior_df, m2_sec_col, m2_beh_col = load_neural_and_behavior_data(
            neural_mat_file=config['data_paths']['m2_neural_mat_file'],
            behavioral_labels_file_path=config['data_paths']['behavioral_labels_file'],
            behavior_key_col_name_template="M2-Behavior",
            use_multiunit=config['analysis_params']['use_multiunit'],
            bank_type=bank_type,
            ventral_deuteron_channels=config['channel_config'].get('m2_ventral', []),
            dorsal_deuteron_channels=config['channel_config'].get('m2_dorsal', []),
            bin_from_raw_spikes=True,
            bin_duration_sec=bin_duration_sec,
            use_gpu=use_gpu
        )

        if m2_neural is None:
            logger.error(f"Failed to load M2 data for {bank_type} bank")
            continue

        # Convert to GPU arrays if needed
        m1_neural = gpu_utils.from_numpy(m1_neural)
        m2_neural = gpu_utils.from_numpy(m2_neural)

        logger.info(f"Loaded M1: {m1_neural.shape}, M2: {m2_neural.shape}")
        
        # Get all available behaviors from the data
        from analysis.decoding.legacy import prepare_decoding_data

        _, m1_sample_labels = prepare_decoding_data(
            neural_data_binned=m1_neural,
            behavior_events_df=m1_behavior_df,
            seconds_col_name=m1_sec_col,
            behavior_col_name=m1_beh_col,
            label_type='concurrent',
            time_lag_seconds=0,
            neural_bin_duration_sec=bin_duration_sec,
            min_behavior_duration_sec=min_behavior_duration_sec,
        )
        
        _, m2_sample_labels = prepare_decoding_data(
            neural_data_binned=m2_neural,
            behavior_events_df=m2_behavior_df,
            seconds_col_name=m2_sec_col,
            behavior_col_name=m2_beh_col,
            label_type='concurrent',
            time_lag_seconds=0,
            neural_bin_duration_sec=bin_duration_sec,
            min_behavior_duration_sec=min_behavior_duration_sec,
        )
        
        # Helper function to get string labels aligned to neural bins (like prepare_decoding_data but without encoding)
        def get_string_labels_aligned(neural_data, behavior_df, sec_col, beh_col, bin_duration, min_duration):
            import numpy as np
            import pandas as pd
            
            # Get the properly aligned data first (this does encoding)
            X_aligned, y_encoded = prepare_decoding_data(
                neural_data_binned=neural_data,
                behavior_events_df=behavior_df,
                seconds_col_name=sec_col,
                behavior_col_name=beh_col,
                label_type='concurrent',
                time_lag_seconds=0,
                neural_bin_duration_sec=bin_duration,
                min_behavior_duration_sec=min_duration,
            )
            
            if X_aligned is None or y_encoded is None or len(y_encoded) == 0:
                return []
            
            # Now manually recreate the alignment to get string labels 
            # (This is inefficient but ensures we get the same alignment as decoding pipeline)
            num_neural_bins = neural_data.shape[1]
            y_labels_at_neural_resolution = [None] * num_neural_bins
            
            if behavior_df is not None and not behavior_df.empty:
                for i, row in behavior_df.iterrows():
                    raw_label = row[beh_col]
                    start_sec_event = row[sec_col]
                    
                    if start_sec_event is None or raw_label is None:
                        continue
                    
                    # Get end time
                    if i < len(behavior_df) - 1:
                        end_sec_event = behavior_df[sec_col].iloc[i + 1]
                        if pd.isna(end_sec_event):
                            end_sec_event = num_neural_bins * bin_duration
                    else:
                        end_sec_event = num_neural_bins * bin_duration
                    
                    # Clean the behavior
                    from data_utils.behavior_utils import clean_behavior
                    cleaned_behaviors = clean_behavior(str(raw_label).lower())
                    primary_behavior = cleaned_behaviors[0] if cleaned_behaviors else None
                    
                    if primary_behavior is None:
                        continue

                    # Validate and convert to float - skip if not numeric
                    if not isinstance(start_sec_event, (int, float, np.integer, np.floating)):
                        continue
                    if not isinstance(end_sec_event, (int, float, np.integer, np.floating)):
                        continue
                    start_sec_num = float(start_sec_event)
                    end_sec_num = float(end_sec_event)

                    if start_sec_num >= end_sec_num:
                        continue
                    
                    # Convert to neural bin indices
                    start_idx = int(np.floor(start_sec_num / bin_duration))
                    end_idx = int(np.ceil(end_sec_num / bin_duration))
                    
                    for bin_k in range(start_idx, end_idx):
                        if 0 <= bin_k < num_neural_bins:
                            y_labels_at_neural_resolution[bin_k] = primary_behavior
            
            # Filter to match the same indices as the decoding pipeline returned
            # (Only return labels for time bins that weren't filtered out)
            valid_labels = []
            bin_idx = 0
            for label in y_labels_at_neural_resolution:
                if label is not None:
                    # Check if this corresponds to a behavior that would be kept by prepare_decoding_data
                    excluded = ["enters", "experimenter enters", "experimenter leaves", "start", "miscellaneous", "misc"]
                    if label not in excluded:
                        valid_labels.append(label)
                bin_idx += 1
                if len(valid_labels) >= len(y_encoded):
                    break
            
            return valid_labels[:len(y_encoded)]  # Match the length of what prepare_decoding_data returned
        
        # Get string labels for comprehensive analysis
        m1_labels = get_string_labels_aligned(m1_neural, m1_behavior_df, m1_sec_col, m1_beh_col, bin_duration_sec, min_behavior_duration_sec)
        m2_labels = get_string_labels_aligned(m2_neural, m2_behavior_df, m2_sec_col, m2_beh_col, bin_duration_sec, min_behavior_duration_sec)
        
        # Get all unique behaviors
        all_behaviors = sorted(list(set(m1_labels + m2_labels)))
        logger.info(f"All unique behaviors found: {all_behaviors}")
        
        # Run comprehensive synchrony analysis for all behavior pairs using all metrics
        for metric_name in metrics:
            logger.info(f"Processing metric: {metric_name}")
            
            # Metric-specific parameters
            metric_kwargs = {}
            if metric_name == 'ccg_peak':
                max_lag_ms = config['analysis_params'].get('max_lag_ms', 100)
                metric_kwargs['max_lag_bins'] = int(max_lag_ms / (bin_duration_sec * 1000))
            elif metric_name == 'rate_corr':
                metric_kwargs['bin_step'] = 5
            
            results = analyze_all_behavior_synchrony(
                m1_neural, m2_neural, m1_labels, m2_labels,
                all_behaviors, min_bout_bins, metric_name, n_perm, **metric_kwargs
            )

            # Add metadata
            results.update({
                'bank': bank_type,
                'metric': metric_name,
                'bin_duration_sec': bin_duration_sec,
                'min_bout_bins': min_bout_bins
            })

            all_results.append(results)

            # Generate heatmap plot
            plot_formats = config.get('plotting_params', {}).get('plot_formats', ['png'])
            for plot_format in plot_formats:
                plot_behavior_synchrony_heatmap(results, output_dir, f"{bank_type}_{metric_name}_synchrony_heatmap", plot_format)

            logger.info(f"Completed comprehensive synchrony analysis for {bank_type} bank using {metric_name}")
        
        # Also run individual behavior vs rest analyses for key behaviors using first metric
        key_behaviors_to_analyze = ['groom give', 'groom receive', 'self-groom']
        metric_name = metrics[0]  # Use first metric for individual analyses
        
        # Metric-specific parameters
        metric_kwargs = {}
        if metric_name == 'ccg_peak':
            max_lag_ms = config['analysis_params'].get('max_lag_ms', 100)
            metric_kwargs['max_lag_bins'] = int(max_lag_ms / (bin_duration_sec * 1000))
        elif metric_name == 'rate_corr':
            metric_kwargs['bin_step'] = 5
        
        for behavior in key_behaviors_to_analyze:
            logger.info(f"Analyzing {behavior} behavior...")

            # Extract behavior labels for both monkeys using the same pipeline as decoding
            # Note: We use prepare_decoding_data but extract the string labels before encoding
            from analysis.decoding.legacy import prepare_decoding_data
            from data_utils.behavior_utils import clean_behavior
            import numpy as np

            # Helper function to get string labels aligned to neural bins (like prepare_decoding_data but without encoding)
            def get_string_labels_aligned(neural_data, behavior_df, sec_col, beh_col, bin_duration, min_duration):
                # Get the properly aligned data first (this does encoding)
                X_aligned, y_encoded = prepare_decoding_data(
                    neural_data_binned=neural_data,
                    behavior_events_df=behavior_df,
                    seconds_col_name=sec_col,
                    behavior_col_name=beh_col,
                    label_type='concurrent',
                    time_lag_seconds=0,
                    neural_bin_duration_sec=bin_duration,
                    min_behavior_duration_sec=min_duration,
                )

                if X_aligned is None or y_encoded is None or len(y_encoded) == 0:
                    return []

                # Now manually recreate the alignment to get string labels
                # (This is inefficient but ensures we get the same alignment as decoding pipeline)
                num_neural_bins = neural_data.shape[1]
                y_labels_at_neural_resolution = [None] * num_neural_bins

                if behavior_df is not None and not behavior_df.empty:
                    for i, row in behavior_df.iterrows():
                        raw_label = row[beh_col]
                        start_sec_event = row[sec_col]

                        if start_sec_event is None or raw_label is None:
                            continue

                        # Get end time
                        if i < len(behavior_df) - 1:
                            end_sec_event = behavior_df[sec_col].iloc[i + 1]
                            if pd.isna(end_sec_event):
                                end_sec_event = num_neural_bins * bin_duration
                        else:
                            end_sec_event = num_neural_bins * bin_duration

                        # Clean the behavior
                        cleaned_behaviors = clean_behavior(str(raw_label).lower())
                        primary_behavior = cleaned_behaviors[0] if cleaned_behaviors else None

                        if primary_behavior is None:
                            continue

                        # Validate and convert to float - skip if not numeric
                        if not isinstance(start_sec_event, (int, float, np.integer, np.floating)):
                            continue
                        if not isinstance(end_sec_event, (int, float, np.integer, np.floating)):
                            continue
                        start_sec_num = float(start_sec_event)
                        end_sec_num = float(end_sec_event)

                        if start_sec_num >= end_sec_num:
                            continue

                        # Convert to neural bin indices
                        start_idx = int(np.floor(start_sec_num / bin_duration))
                        end_idx = int(np.ceil(end_sec_num / bin_duration))

                        for bin_k in range(start_idx, end_idx):
                            if 0 <= bin_k < num_neural_bins:
                                y_labels_at_neural_resolution[bin_k] = primary_behavior

                # Filter to match the same indices as the decoding pipeline returned
                # (Only return labels for time bins that weren't filtered out)
                valid_labels = []
                bin_idx = 0
                for label in y_labels_at_neural_resolution:
                    if label is not None:
                        # Check if this corresponds to a behavior that would be kept by prepare_decoding_data
                        excluded = ["enters", "experimenter enters", "experimenter leaves", "start", "miscellaneous", "misc"]
                        if label not in excluded:
                            valid_labels.append(label)
                    bin_idx += 1
                    if len(valid_labels) >= len(y_encoded):
                        break

                return valid_labels[:len(y_encoded)]  # Match the length of what prepare_decoding_data returned

            m1_labels = get_string_labels_aligned(m1_neural, m1_behavior_df, m1_sec_col, m1_beh_col, bin_duration_sec, min_behavior_duration_sec)
            m2_labels = get_string_labels_aligned(m2_neural, m2_behavior_df, m2_sec_col, m2_beh_col, bin_duration_sec, min_behavior_duration_sec)

            if len(m1_labels) == 0 or len(m2_labels) == 0:
                logger.warning(f"No behavior labels created for {behavior} in {bank_type} bank")
                continue

            # Run behavior vs rest analysis
            results = analyze_behavior_vs_rest(
                m1_neural, m2_neural, m1_labels, m2_labels,
                behavior, min_bout_bins, metric_name, n_perm, **metric_kwargs
            )

            # Add metadata
            results.update({
                'bank': bank_type,
                'behavior': behavior,
                'metric': metric_name,
                'bin_duration_sec': bin_duration_sec,
                'min_bout_bins': min_bout_bins
            })

            all_results.append(results)

            # Generate plot
            plot_formats = config.get('plotting_params', {}).get('plot_formats', ['png'])
            for plot_format in plot_formats:
                plot_synchrony_comparison(results, behavior, output_dir, plot_format)

            logger.info(f"Completed {behavior} analysis for {bank_type} bank")
    
    # Save all results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = output_dir / "synchrony_results_all.feather"
        results_df.to_feather(results_path)
        logger.info(f"Saved all results to {results_path}")
    
    logger.info("=== Unified Neural Synchrony Analysis Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified cross-monkey neural synchrony analysis")
    parser.add_argument('--config', required=True, help='Path to config JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    main(args.config, args.verbose)