"""
Data adapter for loading and preprocessing neural and behavioral data.

This module bridges the SessionManager infrastructure with the data_utils
loading functions, handling both social (2-monkey) and asocial (1-monkey)
recording sessions appropriately.

Example:
    Load data for a social session with both monkeys:
    
    >>> from data.session_manager import SessionManager
    >>> from data.adapters import DataAdapter
    >>> 
    >>> sm = SessionManager('configs/sessions.yaml')
    >>> session = sm.get_session('session_001')
    >>> adapter = DataAdapter()
    >>> data = adapter.load_session_data(session)
    >>> 
    >>> # Social session will have both monkeys' data
    >>> print(data.keys())
    >>> # ['m1_neural', 'm2_neural', 'behavior', 'session_type', ...]
    
    Load data for an asocial session with one monkey:
    
    >>> asocial_session = sm.get_session('session_017')
    >>> asocial_data = adapter.load_session_data(asocial_session)
    >>> # Only one monkey's neural data will be present
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import scipy.io

# Import existing data utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_utils.spike_utils import load_and_prep_neural_data
from data_utils.behavior_utils import load_and_align_behavioral_data, parse_time


class DataAdapter:
    """
    Adapter for loading neural and behavioral data from sessions.
    
    Handles the complexity of social vs asocial sessions, ensuring
    appropriate data is loaded based on session type. Connects the
    SessionManager infrastructure to legacy data loading utilities.
    
    Attributes:
        default_bin_size: Default bin size for neural data in seconds
        default_duration: Default session duration in seconds
        channel_map: Optional mapping of channels to banks
    """
    
    def __init__(self, 
                 default_bin_size: float = 1.0,
                 default_duration: float = 7200.0,
                 channel_map: Optional[Dict[str, List[int]]] = None,
                 channel_excludes: Optional[Dict[str, List[int]]] = None):
        """
        Initialize the data adapter.
        
        Args:
            default_bin_size: Default bin size for spike binning in seconds
            default_duration: Default session duration in seconds (2 hours)
            channel_map: Optional dict mapping bank names to channel indices
                        e.g., {'ventral': [1,2,3], 'dorsal': [33,34,35]}
        """
        self.default_bin_size = default_bin_size
        self.default_duration = default_duration
        self.channel_map = channel_map or {}
        # Expected keys: 'm1', 'm2' mapping to lists of 0-indexed deuteron IDs
        self.channel_excludes = {k.lower(): v for k, v in (channel_excludes or {}).items()}
        
    def load_session_data(self, 
                         session: 'Session',
                         bin_size: Optional[float] = None,
                         load_pose: bool = False) -> Dict[str, Any]:
        """
        Load all data for a session based on its type.
        
        Handles social sessions (2 monkeys) and asocial sessions (1 monkey)
        appropriately, returning different data structures for each.
        
        Args:
            session: Session object from SessionManager
            bin_size: Optional bin size for neural data (uses default if None)
            load_pose: Whether to load 3D pose data if available
            
        Returns:
            Dictionary containing loaded data with keys:
                - 'session_type': 'social' or 'asocial'
                - 'session_id': Session identifier
                - 'monkeys': List of monkey identifiers
                - For social sessions:
                    - 'm1_neural': Neural data for monkey 1
                    - 'm2_neural': Neural data for monkey 2
                    - 'behavior': DataFrame with both monkeys' behaviors
                - For asocial sessions:
                    - '{monkey}_neural': Neural data for single monkey
                    - 'behavior': DataFrame with single monkey's behavior
                - 'pose_3d': Optional 3D pose data if load_pose=True
                
        Raises:
            ValueError: If session type is not recognized
            FileNotFoundError: If required data files are missing
        """
        bin_size = bin_size or self.default_bin_size
        
        data = {
            'session_type': session.type,
            'session_id': session.id,
            'monkeys': session.monkeys,
            'duration_hours': session.duration_hours,
            'session_date': session.config.get('date', None)
        }
        
        if session.type == 'social':
            data.update(self._load_social_session(session, bin_size))
        elif session.type == 'asocial':
            data.update(self._load_asocial_session(session, bin_size))
        else:
            raise ValueError(f"Unknown session type: {session.type}")
            
        # Optionally load 3D pose data
        if load_pose and 'pose_3d' in session.data_files:
            data['pose_3d'] = self._load_pose_data(session.data_files['pose_3d'])
            
        return data
    
    def _load_social_session(self, 
                            session: 'Session', 
                            bin_size: float) -> Dict[str, Any]:
        """
        Load data for a social (2-monkey) session.
        
        Args:
            session: Social session object
            bin_size: Bin size for neural data
            
        Returns:
            Dictionary with both monkeys' neural and behavioral data
        """
        social_data = {}
        
        # Load neural data for both monkeys
        for monkey in ['m1', 'm2']:
            neural_key = f'{monkey}_neural'
            if neural_key in session.data_files:
                neural_path = session.data_files[neural_key]
                if neural_path and neural_path != 'TO_BE_CONFIGURED' and Path(neural_path).exists():
                    block = self._load_neural_data(
                        neural_path, 
                        bin_size,
                        monkey_id=monkey
                    )
                    block['source'] = 'real'
                    social_data[neural_key] = block
                else:
                    if neural_path != 'TO_BE_CONFIGURED':
                        warnings.warn(f"Neural data file not found: {neural_path}")
                    block = self._create_mock_neural_data(
                        bin_size, self.default_duration
                    )
                    block['source'] = 'mock'
                    social_data[neural_key] = block
        
        # Load joint behavioral data
        if 'behavior' in session.data_files:
            behavior_path = session.data_files['behavior']
            if behavior_path and behavior_path != 'TO_BE_CONFIGURED' and Path(behavior_path).exists():
                social_data['behavior'] = self._load_behavioral_data(
                    behavior_path,
                    session_type='social'
                )
                social_data['behavior_source'] = 'real'
            else:
                if behavior_path != 'TO_BE_CONFIGURED' and behavior_path:
                    warnings.warn(f"Behavior file not found: {behavior_path}")
                social_data['behavior'] = self._create_mock_behavior_data(
                    'social', self.default_duration
                )
                social_data['behavior_source'] = 'mock'
                
        # Add metadata for downstream analyses
        social_data['has_partner'] = True
        social_data['can_compute_allocentric'] = True
        social_data['can_compute_inter_brain'] = True
        
        return social_data
    
    def _load_asocial_session(self, 
                              session: 'Session', 
                              bin_size: float) -> Dict[str, Any]:
        """
        Load data for an asocial (1-monkey) session.
        
        Args:
            session: Asocial session object
            bin_size: Bin size for neural data
            
        Returns:
            Dictionary with single monkey's neural and behavioral data
        """
        asocial_data = {}
        
        # Determine which monkey is present
        monkey = session.monkeys[0] if session.monkeys else 'm1'
        
        # Load neural data for single monkey
        neural_key = f'{monkey}_neural'
        if neural_key in session.data_files:
            neural_path = session.data_files[neural_key]
            if neural_path and neural_path != 'TO_BE_CONFIGURED' and Path(neural_path).exists():
                block = self._load_neural_data(
                    neural_path,
                    bin_size,
                    monkey_id=monkey
                )
                block['source'] = 'real'
                asocial_data[neural_key] = block
            else:
                if neural_path != 'TO_BE_CONFIGURED':
                    warnings.warn(f"Neural data file not found: {neural_path}")
                block = self._create_mock_neural_data(
                    bin_size, self.default_duration
                )
                block['source'] = 'mock'
                asocial_data[neural_key] = block
        
        # Load solo behavioral data
        if 'behavior' in session.data_files:
            behavior_path = session.data_files['behavior']
            if behavior_path and behavior_path != 'TO_BE_CONFIGURED' and Path(behavior_path).exists():
                asocial_data['behavior'] = self._load_behavioral_data(
                    behavior_path,
                    session_type='asocial',
                    monkey_id=monkey
                )
                asocial_data['behavior_source'] = 'real'
            else:
                if behavior_path != 'TO_BE_CONFIGURED' and behavior_path:
                    warnings.warn(f"Behavior file not found: {behavior_path}")
                asocial_data['behavior'] = self._create_mock_behavior_data(
                    'asocial', self.default_duration, monkey
                )
                asocial_data['behavior_source'] = 'mock'
                
        # Add metadata for downstream analyses
        asocial_data['has_partner'] = False
        asocial_data['can_compute_allocentric'] = False
        asocial_data['can_compute_inter_brain'] = False
        asocial_data['solo_monkey'] = monkey
        
        return asocial_data
    
    def _load_neural_data(self, 
                         mat_path: str, 
                         bin_size: float,
                         monkey_id: str) -> Dict[str, np.ndarray]:
        """
        Load neural data from .mat file.
        
        Args:
            mat_path: Path to .mat file
            bin_size: Bin size in seconds
            monkey_id: Identifier for the monkey ('m1' or 'm2')
            
        Returns:
            Dictionary containing:
                - 'spike_times': Spike times by unit
                - 'binned_spikes': Binned spike counts
                - 'channel_info': Channel metadata
                - 'bank_assignment': Ventral/dorsal assignment if available
        """
        # Use existing spike loading utility
        # Apply per-monkey channel excludes if present
        excludes = self.channel_excludes.get(monkey_id.lower(), [])
        rasters, unit_metadata = load_and_prep_neural_data(
            mat_file=mat_path,
            bin_from_raw_spikes=True,
            bin_duration_sec=bin_size,
            recording_duration_sec=None,
            channel_map=self.channel_map,
            channels_to_exclude=excludes
        )

        if rasters is None:
            return {}

        # Build per-unit bank labels from unit_metadata['area'] if available
        bank_by_unit = {}
        if isinstance(unit_metadata, list) and len(unit_metadata) == rasters.shape[0]:
            for idx, meta in enumerate(unit_metadata):
                bank_by_unit[idx] = meta.get('area', 'unknown')

        return {
            'binned_spikes': rasters,
            'unit_metadata': unit_metadata or [],
            'bank_by_unit': bank_by_unit
        }
    
    def _load_behavioral_data(self, 
                             csv_path: str,
                             session_type: str,
                             monkey_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load behavioral data from CSV file.
        
        Args:
            csv_path: Path to behavior CSV file
            session_type: 'social' or 'asocial'
            monkey_id: Monkey identifier for asocial sessions
            
        Returns:
            DataFrame with behavioral states and timestamps
        """
        # Use existing behavior loading utility
        behavior_df = load_and_align_behavioral_data(
            behavioral_labels_file=csv_path,
            use_gpu=False,
            behaviors_to_exclude=None,
            min_behavior_duration_sec=0
        )
        
        # Validate columns based on session type
        if session_type == 'social':
            expected_cols = ['Hke Behavior', 'Hdy Behavior']
            for col in expected_cols:
                if col not in behavior_df.columns:
                    warnings.warn(f"Missing expected column: {col}")
            # Convert time strings to seconds from session start for both monkeys
            try:
                for time_col in ['Time-Hooke', 'Time-Hedy']:
                    if time_col in behavior_df.columns:
                        times = behavior_df[time_col].tolist()
                        if len(times) > 0:
                            t0 = parse_time(times[0])
                            behavior_df[time_col] = [
                                (parse_time(t) - t0).total_seconds() if t is not None else 0.0
                                for t in times
                            ]
            except Exception as e:
                warnings.warn(f"Failed to convert behavior times to seconds: {e}")
        else:  # asocial
            # Ensure we have behavior column for the solo monkey
            monkey_col = f'{monkey_id} Behavior' if monkey_id else 'Behavior'
            if monkey_col not in behavior_df.columns:
                # Try to find any behavior column
                behavior_cols = [c for c in behavior_df.columns if 'Behavior' in c]
                if behavior_cols:
                    behavior_df[monkey_col] = behavior_df[behavior_cols[0]]
                    
        return behavior_df
    
    def _load_pose_data(self, pose_path: str) -> Dict[str, np.ndarray]:
        """
        Load 3D pose data.
        
        Args:
            pose_path: Path to pose data file
            
        Returns:
            Dictionary with pose trajectories by subject
        """
        if pose_path.endswith('.mat'):
            pose_data = scipy.io.loadmat(pose_path)
            return {
                'subject': pose_data.get('subject_pose', np.array([])),
                'partner': pose_data.get('partner_pose', np.array([]))
            }
        elif pose_path.endswith('.npy'):
            return np.load(pose_path, allow_pickle=True).item()
        else:
            warnings.warn(f"Unknown pose data format: {pose_path}")
            return {}
    
    def _assign_banks_to_channels(self, 
                                  channel_ids: List[int],
                                  monkey_id: str) -> Dict[int, str]:
        """
        Assign channels to ventral or dorsal banks.
        
        Args:
            channel_ids: List of channel IDs
            monkey_id: Monkey identifier
            
        Returns:
            Dictionary mapping channel ID to bank name
        """
        assignments = {}
        
        for channel_id in channel_ids:
            if channel_id in self.channel_map.get('ventral', []):
                assignments[channel_id] = 'ventral'
            elif channel_id in self.channel_map.get('dorsal', []):
                assignments[channel_id] = 'dorsal'
            else:
                assignments[channel_id] = 'unknown'
                
        return assignments
    
    def _create_mock_neural_data(self, 
                                bin_size: float, 
                                duration: float) -> Dict[str, np.ndarray]:
        """
        Create mock neural data for testing.
        
        Args:
            bin_size: Bin size in seconds
            duration: Session duration in seconds
            
        Returns:
            Dictionary with mock neural data
        """
        n_bins = int(duration / bin_size)
        n_units = 50
        
        return {
            'binned_spikes': np.random.poisson(5, (n_bins, n_units)),
            'spike_times': {},
            'channel_info': {},
            'bank_assignment': {}
        }
    
    def _create_mock_behavior_data(self, 
                                   session_type: str,
                                   duration: float,
                                   monkey_id: str = 'm1') -> pd.DataFrame:
        """
        Create mock behavioral data for testing.
        
        Args:
            session_type: 'social' or 'asocial'
            duration: Session duration in seconds
            monkey_id: Monkey identifier
            
        Returns:
            DataFrame with mock behavioral data
        """
        n_samples = int(duration)
        behaviors = ['rest', 'forage', 'groom', 'travel', 'play']
        
        if session_type == 'social':
            return pd.DataFrame({
                'Time': range(n_samples),
                'Hke Behavior': np.random.choice(behaviors, n_samples),
                'Hdy Behavior': np.random.choice(behaviors, n_samples)
            })
        else:
            return pd.DataFrame({
                'Time': range(n_samples),
                f'{monkey_id} Behavior': np.random.choice(behaviors, n_samples)
            })
