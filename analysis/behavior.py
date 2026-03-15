"""
Unified behavioral analysis module for FreeMat.

This module consolidates all behavioral analysis functionality including:
- Behavioral state duration and frequency analysis
- Transition probability calculations
- Exclusivity checking (e.g., groom give/receive, chase/chased)
- Neural-aligned behavioral analysis
- Inter-bout interval analysis
- Statistical comparisons between monkeys

Core Metrics:

    Transition Probability Matrix:
        P(i → j) = count(i → j) / Σ_k count(i → k)

        Where count(i → j) = number of transitions from state i to state j.
        Rows sum to 1: Σ_j P(i,j) = 1 for each row i.

    Transition Entropy:
        H = -Σ_i Σ_j P(i,j) · log₂(P(i,j))

        Measures unpredictability of state transitions.
        Higher H = more random/varied transitions.
        Lower H = more deterministic/structured sequences.

    Behavioral Switch Rate:
        rate = n_switches / (total_time / time_window)

        Where n_switches = count of behavior state changes.

Example:
    Basic behavioral analysis workflow:

    >>> from bhvr.behavior_analyzer import BehaviorAnalyzer
    >>>
    >>> # Initialize with behavioral data
    >>> analyzer = BehaviorAnalyzer()
    >>> analyzer.load_behavioral_data('session_001_behavior.csv')
    >>>
    >>> # Analyze exclusivity patterns
    >>> exclusivity_report = analyzer.analyze_exclusivity(
    >>>     behaviors=['groom give', 'groom receive'],
    >>>     check_complementary=True
    >>> )
    >>>
    >>> # Get transition probabilities
    >>> trans_matrix, behaviors = analyzer.get_transition_matrix('Hooke')
    >>>
    >>> # Calculate state durations
    >>> durations = analyzer.calculate_state_durations('Hooke')
"""

import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.signal import find_peaks
import networkx as nx


class BehaviorAnalyzer:
    """
    Comprehensive behavioral analysis for freely-moving macaque recordings.
    
    This class consolidates all behavioral analysis functionality previously
    spread across multiple modules. It provides methods for analyzing
    behavioral patterns, exclusivity, transitions, and neural alignment.
    
    Attributes:
        hooke_df: DataFrame containing Hooke's behavioral data
        hedy_df: DataFrame containing Hedy's behavioral data
        behaviors: List of unique behaviors across both monkeys
        exclude_behaviors: Set of behaviors to exclude from analysis
        neural_data: Optional neural data for aligned analysis
    """
    
    def __init__(self, 
                 hooke_df: Optional[pl.DataFrame] = None,
                 hedy_df: Optional[pl.DataFrame] = None):
        """
        Initialize the BehaviorAnalyzer.
        
        Args:
            hooke_df: Optional DataFrame with Hooke's behavioral data
            hedy_df: Optional DataFrame with Hedy's behavioral data
        """
        self.hooke_df = hooke_df
        self.hedy_df = hedy_df
        self.exclude_behaviors = {
            "start",
            "experimenter leaves",
            "experimenter enters", 
            "miscellaneous",
            "misc",
            "enters"
        }
        self.behaviors = []
        self.neural_data = {}
        
        if hooke_df is not None and hedy_df is not None:
            self.behaviors = self._get_unique_behaviors()
    
    def load_behavioral_data(self, 
                           behavioral_file_path: str,
                           neural_start_time: float = 0,
                           neural_end_time: float = 7200) -> None:
        """
        Load and align behavioral data from CSV file.
        
        Args:
            behavioral_file_path: Path to behavioral CSV file
            neural_start_time: Start time for alignment (seconds)
            neural_end_time: End time for alignment (seconds)
        """
        try:
            # Import here to avoid circular dependencies
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from data_utils.behavior_utils import load_and_align_behavioral_data
            
            behavior_df = load_and_align_behavioral_data(
                behavioral_file_path,
                neural_start_time,
                neural_end_time,
                verbose=False
            )
            
            if behavior_df is None or len(behavior_df) == 0:
                raise ValueError("Failed to load behavioral data")
            
            # Convert to polars and split by monkey
            pl_df = pl.from_pandas(behavior_df)
            
            # Find columns for each monkey
            hooke_cols = [col for col in pl_df.columns 
                         if 'hooke' in col.lower()]
            hedy_cols = [col for col in pl_df.columns 
                        if 'hedy' in col.lower()]
            
            if hooke_cols:
                self.hooke_df = pl_df.select(hooke_cols)
            if hedy_cols:
                self.hedy_df = pl_df.select(hedy_cols)
                
            self.behaviors = self._get_unique_behaviors()
            
        except Exception as e:
            warnings.warn(f"Error loading behavioral data: {e}")
            raise
    
    @staticmethod
    def clean_behavior(behavior_str: str) -> List[str]:
        """
        Clean and standardize behavior string.
        
        Args:
            behavior_str: Raw behavior string
            
        Returns:
            List of cleaned behavior labels
        """
        if not behavior_str or pd.isna(behavior_str):
            return []
        
        behavior_str = str(behavior_str).lower().strip()
        
        # Handle multiple behaviors separated by '+'
        if '+' in behavior_str:
            behaviors = behavior_str.split('+')
        else:
            behaviors = [behavior_str]
        
        cleaned = []
        for b in behaviors:
            b = b.strip()
            # Standardize common variations
            if 'groom' in b and 'give' in b:
                cleaned.append('groom give')
            elif 'groom' in b and 'receive' in b:
                cleaned.append('groom receive')
            elif 'chase' in b and 'ed' in b:
                cleaned.append('chased')
            elif 'chase' in b:
                cleaned.append('chase')
            elif b:
                cleaned.append(b)
        
        return cleaned
    
    @staticmethod
    def parse_time(time_str: str) -> Optional[datetime]:
        """
        Parse time string to datetime object.
        
        Args:
            time_str: Time string in various formats
            
        Returns:
            datetime object or None if parsing fails
        """
        if not time_str:
            return None
            
        try:
            # Try different time formats
            for fmt in ['%H:%M:%S', '%H:%M:%S.%f', '%M:%S', '%M:%S.%f']:
                try:
                    return datetime.strptime(str(time_str), fmt)
                except ValueError:
                    continue
            return None
        except:
            return None
    
    def _get_unique_behaviors(self) -> List[str]:
        """
        Extract unique behaviors from both monkeys' data.
        
        Returns:
            Sorted list of unique behaviors
        """
        behaviors = set()
        
        for df in [self.hooke_df, self.hedy_df]:
            if df is None:
                continue
                
            behavior_col = None
            for col in df.columns:
                if 'behavior' in col.lower():
                    behavior_col = col
                    break
            
            if behavior_col:
                for behavior in df[behavior_col]:
                    if isinstance(behavior, str):
                        cleaned = self.clean_behavior(behavior)
                        behaviors.update(cleaned)
        
        # Remove excluded behaviors
        behaviors = behaviors - self.exclude_behaviors
        return sorted(list(behaviors))
    
    def analyze_exclusivity(self,
                          behaviors: Optional[List[Tuple[str, str]]] = None,
                          check_complementary: bool = True) -> Dict[str, Any]:
        """
        Analyze behavioral exclusivity patterns.
        
        Checks whether certain behaviors occur exclusively (e.g., both monkeys
        can't "groom give" simultaneously) and whether complementary behaviors
        occur together (e.g., one "groom give" while other "groom receive").
        
        Args:
            behaviors: List of behavior pairs to check for exclusivity.
                      If None, uses default exclusive pairs.
            check_complementary: Whether to also check complementary pairs
            
        Returns:
            Dictionary containing exclusivity analysis results with keys:
                - 'exclusive_violations': List of violations found
                - 'complementary_matches': List of complementary behaviors found
                - 'statistics': Summary statistics
                - 'simultaneous_pairs': All simultaneous behavior pairs
        """
        if behaviors is None:
            behaviors = [
                ('groom give', 'groom give'),
                ('groom receive', 'groom receive'),
                ('chase', 'chase'),
                ('chased', 'chased'),
                ('grab', 'grab'),
                ('grabbed', 'grabbed'),
            ]
        
        results = {
            'exclusive_violations': [],
            'complementary_matches': [],
            'statistics': {},
            'simultaneous_pairs': []
        }
        
        # Get behavior columns
        hooke_col = None
        hedy_col = None
        
        if self.hooke_df is not None:
            for col in self.hooke_df.columns:
                if 'behavior' in col.lower():
                    hooke_col = col
                    break
        
        if self.hedy_df is not None:
            for col in self.hedy_df.columns:
                if 'behavior' in col.lower():
                    hedy_col = col
                    break
        
        if not hooke_col or not hedy_col:
            warnings.warn("Could not find behavior columns")
            return results
        
        # Find simultaneous behaviors
        # Assuming dataframes are aligned by time
        min_len = min(len(self.hooke_df), len(self.hedy_df))
        
        simultaneous_pairs = []
        for i in range(min_len):
            hooke_behaviors = self.clean_behavior(str(self.hooke_df[hooke_col][i]))
            hedy_behaviors = self.clean_behavior(str(self.hedy_df[hedy_col][i]))
            
            for h_beh in hooke_behaviors:
                for he_beh in hedy_behaviors:
                    if h_beh not in self.exclude_behaviors and \
                       he_beh not in self.exclude_behaviors:
                        simultaneous_pairs.append((h_beh, he_beh))
        
        results['simultaneous_pairs'] = simultaneous_pairs
        
        # Check exclusive behaviors
        for behavior_pair in behaviors:
            beh1, beh2 = behavior_pair
            if beh1 == beh2:  # Should be exclusive
                count = sum(1 for p in simultaneous_pairs if p == behavior_pair)
                if count > 0:
                    results['exclusive_violations'].append({
                        'behavior': beh1,
                        'count': count,
                        'percentage': 100 * count / len(simultaneous_pairs)
                    })
        
        # Check complementary behaviors
        if check_complementary:
            complementary_pairs = [
                ('groom give', 'groom receive'),
                ('groom receive', 'groom give'),
                ('chase', 'chased'),
                ('chased', 'chase'),
                ('grab', 'grabbed'),
                ('grabbed', 'grab'),
            ]
            
            for comp_pair in complementary_pairs:
                count = sum(1 for p in simultaneous_pairs if p == comp_pair)
                if count > 0:
                    results['complementary_matches'].append({
                        'pair': comp_pair,
                        'count': count,
                        'percentage': 100 * count / len(simultaneous_pairs)
                    })
        
        # Calculate statistics
        pair_counts = pd.Series(simultaneous_pairs).value_counts()
        results['statistics'] = {
            'total_pairs': len(simultaneous_pairs),
            'unique_pairs': len(pair_counts),
            'top_10_pairs': pair_counts.head(10).to_dict(),
            'violations_found': len(results['exclusive_violations']) > 0
        }
        
        return results
    
    def calculate_state_durations(self, 
                                 monkey: str) -> Dict[str, List[float]]:
        """
        Calculate duration of each behavioral state.
        
        Args:
            monkey: 'Hooke' or 'Hedy'
            
        Returns:
            Dictionary mapping behaviors to list of durations (seconds)
        """
        df = self.hooke_df if monkey.lower() == 'hooke' else self.hedy_df
        
        if df is None:
            return {}
        
        # Find time and behavior columns
        time_col = None
        behavior_col = None
        
        for col in df.columns:
            if 'time' in col.lower() and monkey.lower() in col.lower():
                time_col = col
            elif 'behavior' in col.lower() and monkey.lower() in col.lower():
                behavior_col = col
        
        if not time_col and 'Time' in df.columns:
            time_col = 'Time'
        
        if not behavior_col:
            for col in df.columns:
                if 'behavior' in col.lower():
                    behavior_col = col
                    break
        
        if not time_col or not behavior_col:
            warnings.warn(f"Could not find required columns for {monkey}")
            return {}
        
        durations = {b: [] for b in self.behaviors}
        
        # Parse times and calculate durations
        times = []
        behaviors = []
        
        for t, b in zip(df[time_col], df[behavior_col]):
            parsed_time = self.parse_time(str(t))
            if parsed_time is not None and isinstance(b, str):
                times.append(parsed_time)
                behaviors.append(b)
        
        # Calculate durations between consecutive behaviors
        for i in range(len(times) - 1):
            duration = (times[i + 1] - times[i]).total_seconds()
            current_behaviors = self.clean_behavior(behaviors[i])
            
            for behavior in current_behaviors:
                if behavior in durations:
                    durations[behavior].append(duration)
        
        return durations
    
    def get_behavior_proportions(self, 
                                monkey: str) -> Dict[str, float]:
        """
        Calculate proportion of time spent in each behavior.
        
        Args:
            monkey: 'Hooke' or 'Hedy'
            
        Returns:
            Dictionary mapping behaviors to proportions (0-1)
        """
        durations = self.calculate_state_durations(monkey)
        
        if not durations:
            return {}
        
        total_time = sum(sum(d) for d in durations.values())
        
        if total_time == 0:
            return {}
        
        return {
            b: sum(d) / total_time 
            for b, d in durations.items()
        }
    
    def get_transition_matrix(self,
                             monkey: str,
                             normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Calculate transition probability matrix for behaviors.

        Mathematical Definition:
            Count matrix:
                C[i,j] = number of transitions from state i to state j

            Probability matrix (if normalize=True):
                P[i,j] = C[i,j] / Σ_k C[i,k]

            Properties:
                - Σ_j P[i,j] = 1 for each row i (row-stochastic)
                - P[i,j] ∈ [0, 1]
                - Diagonal P[i,i] = self-transition probability

        Args:
            monkey: 'Hooke' or 'Hedy'
            normalize: If True, normalize rows to probabilities

        Returns:
            Tuple of (transition matrix shape (n_behaviors, n_behaviors),
                      list of behavior names)
        """
        df = self.hooke_df if monkey.lower() == 'hooke' else self.hedy_df
        
        if df is None:
            return np.array([]), []
        
        # Find behavior column
        behavior_col = None
        for col in df.columns:
            if 'behavior' in col.lower():
                behavior_col = col
                break
        
        if not behavior_col:
            return np.array([]), []
        
        # Get all behaviors for this monkey
        monkey_behaviors = set()
        for behavior in df[behavior_col]:
            if isinstance(behavior, str):
                cleaned = self.clean_behavior(behavior)
                monkey_behaviors.update(cleaned)
        
        # Remove excluded behaviors
        behaviors = sorted(
            b for b in monkey_behaviors 
            if b not in self.exclude_behaviors
        )
        
        n_behaviors = len(behaviors)
        behavior_to_idx = {b: i for i, b in enumerate(behaviors)}
        
        # Initialize transition count matrix
        trans_matrix = np.zeros((n_behaviors, n_behaviors))
        
        # Count transitions
        prev_behaviors = []
        for i in range(len(df)):
            curr_behavior = df[behavior_col][i]
            
            if not isinstance(curr_behavior, str):
                prev_behaviors = []
                continue
            
            curr_behaviors = self.clean_behavior(curr_behavior)
            curr_behaviors = [b for b in curr_behaviors 
                             if b not in self.exclude_behaviors]
            
            # Count transitions from previous to current
            if prev_behaviors and curr_behaviors:
                for prev in prev_behaviors:
                    for curr in curr_behaviors:
                        if prev in behavior_to_idx and curr in behavior_to_idx:
                            trans_matrix[behavior_to_idx[prev], 
                                       behavior_to_idx[curr]] += 1
            
            prev_behaviors = curr_behaviors
        
        # Normalize to probabilities
        if normalize:
            row_sums = trans_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            trans_matrix = trans_matrix / row_sums
        
        return trans_matrix, behaviors
    
    def analyze_neural_aligned_exclusivity(self,
                                          m1_neural: np.ndarray,
                                          m2_neural: np.ndarray,
                                          m1_labels: List[str],
                                          m2_labels: List[str]) -> Dict[str, Any]:
        """
        Analyze exclusivity with neural-aligned behavioral labels.
        
        Args:
            m1_neural: Neural data for monkey 1 (bins x neurons)
            m2_neural: Neural data for monkey 2 (bins x neurons)
            m1_labels: Behavioral labels aligned to m1_neural bins
            m2_labels: Behavioral labels aligned to m2_neural bins
            
        Returns:
            Dictionary containing neural-aligned exclusivity results
        """
        results = {
            'aligned_violations': [],
            'aligned_complementary': [],
            'neural_statistics': {}
        }
        
        # Store neural data for potential future use
        self.neural_data = {
            'm1': m1_neural,
            'm2': m2_neural,
            'm1_labels': m1_labels,
            'm2_labels': m2_labels
        }
        
        # Ensure labels are same length
        min_length = min(len(m1_labels), len(m2_labels))
        m1_labels = m1_labels[:min_length]
        m2_labels = m2_labels[:min_length]
        
        # Create joint behavior pairs
        joint_pairs = [(m1_labels[i], m2_labels[i]) 
                      for i in range(min_length)
                      if m1_labels[i] and m2_labels[i]]
        
        # Check exclusive behaviors
        exclusive_pairs = [
            ('groom give', 'groom give'),
            ('groom receive', 'groom receive'),
            ('chase', 'chase'),
            ('chased', 'chased'),
        ]
        
        for pair in exclusive_pairs:
            m1_behavior, m2_behavior = pair
            simultaneous_count = sum(1 for p in joint_pairs if p == pair)
            
            if simultaneous_count > 0:
                m1_count = sum(1 for p in joint_pairs if p[0] == m1_behavior)
                m2_count = sum(1 for p in joint_pairs if p[1] == m2_behavior)
                
                results['aligned_violations'].append({
                    'behavior_pair': pair,
                    'simultaneous_count': simultaneous_count,
                    'm1_total': m1_count,
                    'm2_total': m2_count,
                    'violation_rate': simultaneous_count / len(joint_pairs)
                })
        
        # Check complementary pairs
        complementary_pairs = [
            ('groom give', 'groom receive'),
            ('groom receive', 'groom give'),
            ('chase', 'chased'),
            ('chased', 'chase'),
        ]
        
        for pair in complementary_pairs:
            count = sum(1 for p in joint_pairs if p == pair)
            if count > 0:
                results['aligned_complementary'].append({
                    'pair': pair,
                    'count': count,
                    'rate': count / len(joint_pairs)
                })
        
        # Calculate neural statistics during violations
        if results['aligned_violations']:
            violation_bins = []
            for violation in results['aligned_violations']:
                pair = violation['behavior_pair']
                violation_bins.extend([i for i, p in enumerate(joint_pairs) 
                                     if p == pair])
            
            if violation_bins:
                # Calculate mean firing rates during violations
                m1_violation_fr = np.mean(m1_neural[:, violation_bins], axis=1)
                m2_violation_fr = np.mean(m2_neural[:, violation_bins], axis=1)
                
                results['neural_statistics'] = {
                    'violation_bins': len(violation_bins),
                    'm1_mean_fr_during_violations': float(np.mean(m1_violation_fr)),
                    'm2_mean_fr_during_violations': float(np.mean(m2_violation_fr)),
                    'm1_std_fr_during_violations': float(np.std(m1_violation_fr)),
                    'm2_std_fr_during_violations': float(np.std(m2_violation_fr))
                }
        
        return results
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for both monkeys.

        Computed Metrics:
            Transition Entropy:
                H = -Σ_{p>0} p · log₂(p)

                Where p iterates over all non-zero transition probabilities
                in the flattened transition matrix.

                Interpretation:
                    - H = 0: Completely deterministic transitions
                    - H = log₂(n²): Maximum entropy (uniform over all transitions)

        Returns:
            Dictionary containing summary statistics including:
                - Behavior counts and proportions
                - Mean durations
                - Transition entropy
                - Exclusivity violations
        """
        summary = {
            'Hooke': {},
            'Hedy': {},
            'Joint': {}
        }
        
        # Individual monkey statistics
        for monkey in ['Hooke', 'Hedy']:
            durations = self.calculate_state_durations(monkey)
            proportions = self.get_behavior_proportions(monkey)
            trans_matrix, behaviors = self.get_transition_matrix(monkey)
            
            # Calculate transition entropy
            entropy = 0
            if trans_matrix.size > 0:
                # Flatten and remove zeros for entropy calculation
                probs = trans_matrix.flatten()
                probs = probs[probs > 0]
                if len(probs) > 0:
                    entropy = -np.sum(probs * np.log2(probs))
            
            summary[monkey] = {
                'n_unique_behaviors': len(behaviors),
                'total_time_hours': sum(sum(d) for d in durations.values()) / 3600,
                'most_common_behavior': max(proportions.items(), 
                                           key=lambda x: x[1])[0] if proportions else None,
                'transition_entropy': float(entropy),
                'mean_bout_durations': {
                    b: float(np.mean(d)) if d else 0 
                    for b, d in durations.items()
                }
            }
        
        # Joint analysis
        exclusivity = self.analyze_exclusivity()
        summary['Joint'] = {
            'exclusivity_violations': len(exclusivity['exclusive_violations']),
            'complementary_matches': len(exclusivity['complementary_matches']),
            'total_simultaneous_pairs': exclusivity['statistics'].get('total_pairs', 0),
            'unique_simultaneous_pairs': exclusivity['statistics'].get('unique_pairs', 0)
        }
        
        return summary
    
    def get_behavior_network(self, monkey: str) -> nx.DiGraph:
        """Create directed graph of behavior transitions.
        
        Args:
            monkey: 'Hooke' or 'Hedy'
            
        Returns:
            NetworkX directed graph with behaviors as nodes and
            transition probabilities as edge weights
        """
        trans_probs, behaviors = self.get_transition_matrix(monkey)
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(behaviors)
        
        # Add weighted edges for transitions
        for i, curr_b in enumerate(behaviors):
            for j, next_b in enumerate(behaviors):
                if trans_probs[i, j] > 0.01:  # Only add significant transitions
                    G.add_edge(curr_b, next_b, weight=float(trans_probs[i, j]))
        
        return G
    
    def test_behavior_stationarity(self, monkey: str) -> Dict[str, Dict]:
        """Test if behavioral patterns are stationary over time.
        
        Uses Augmented Dickey-Fuller test to check if behavior
        durations show stationarity (no trend over time).
        
        Args:
            monkey: 'Hooke' or 'Hedy'
            
        Returns:
            Dict mapping behaviors to test results including:
                - adf_statistic: Test statistic
                - p_value: P-value of the test
                - is_stationary: Whether behavior is stationary (p < 0.05)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            warnings.warn("statsmodels not available for stationarity test")
            return {}
        
        durations = self.calculate_state_durations(monkey)
        results = {}
        
        for behavior, dur_list in durations.items():
            if len(dur_list) > 10:  # Need sufficient samples
                adf_stat, p_val, *_ = adfuller(dur_list)
                results[behavior] = {
                    'adf_statistic': float(adf_stat),
                    'p_value': float(p_val),
                    'is_stationary': p_val < 0.05
                }
        
        return results
    
    def get_behavior_frequencies(self, monkey: str) -> Dict[str, int]:
        """Calculate frequency (count) of each behavior.
        
        Args:
            monkey: 'Hooke' or 'Hedy'
            
        Returns:
            Dictionary mapping behaviors to their occurrence counts
        """
        df = self.hooke_df if monkey.lower() == 'hooke' else self.hedy_df
        
        if df is None:
            return {}
        
        # Find behavior column
        behavior_col = None
        for col in df.columns:
            if 'behavior' in col.lower():
                behavior_col = col
                break
        
        if not behavior_col:
            return {}
        
        frequencies = {b: 0 for b in self.behaviors}
        
        for behavior in df[behavior_col]:
            if isinstance(behavior, str):
                cleaned = self.clean_behavior(behavior)
                for b in cleaned:
                    if b in frequencies:
                        frequencies[b] += 1
        
        return frequencies
    
    def get_switch_rate(self,
                       monkey: str,
                       time_window: float = 60.0) -> Tuple[float, Dict[str, float]]:
        """Calculate rate of behavioral switches.

        Mathematical Definition:
            Overall switch rate:
                rate = n_switches / (T_total / T_window)

            Where:
                - n_switches = Σ_t 𝟙[behavior(t) ≠ behavior(t-1)]
                - T_total = total session duration
                - T_window = normalization window (default 60s)

            Per-behavior switch rate:
                rate_b = n_switches_from_b / (T_total / T_window)

            Where n_switches_from_b = count of transitions OUT of behavior b.

        Args:
            monkey: 'Hooke' or 'Hedy'
            time_window: Time window in seconds for rate normalization (default: 60s)

        Returns:
            Tuple of:
                - Overall switch rate (switches per time_window seconds)
                - Dict mapping behaviors to their individual switch rates
        """
        df = self.hooke_df if monkey.lower() == 'hooke' else self.hedy_df
        
        if df is None:
            return 0.0, {}
        
        # Find columns
        time_col = None
        behavior_col = None
        
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
            elif 'behavior' in col.lower():
                behavior_col = col
        
        if not time_col or not behavior_col:
            return 0.0, {}
        
        # Get valid times and behaviors
        times = []
        behaviors = []
        
        for t, b in zip(df[time_col], df[behavior_col]):
            parsed_time = self.parse_time(str(t))
            if parsed_time and isinstance(b, str):
                cleaned = self.clean_behavior(b)
                if cleaned:
                    times.append(parsed_time)
                    behaviors.append(cleaned[0])  # Use first behavior if multiple
        
        if len(behaviors) < 2:
            return 0.0, {}
        
        # Count switches
        total_switches = 0
        behavior_switches = {b: 0 for b in set(behaviors)}
        
        for i in range(len(behaviors) - 1):
            if behaviors[i] != behaviors[i + 1]:
                total_switches += 1
                behavior_switches[behaviors[i]] += 1
        
        # Calculate total time
        total_time = (times[-1] - times[0]).total_seconds() / time_window
        
        if total_time == 0:
            return 0.0, {}
        
        # Calculate rates
        overall_rate = total_switches / total_time
        behavior_rates = {
            b: count / total_time 
            for b, count in behavior_switches.items()
        }
        
        return overall_rate, behavior_rates