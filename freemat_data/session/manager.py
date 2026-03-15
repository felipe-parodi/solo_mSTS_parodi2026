"""
Session management for 20 sessions of dual-monkey recordings.

This module provides centralized session management for FreeMat's 20 recording
sessions (16 social, 4 asocial). It handles data loading, intelligent caching,
and cross-platform path compatibility.

Example:
    Basic usage for loading and analyzing sessions:

    >>> from data.session_manager import SessionManager
    >>>
    >>> # Initialize with config file
    >>> sm = SessionManager('configs/sessions.yaml')
    >>>
    >>> # Get overview
    >>> print(sm.get_summary())
    >>> # {'total_sessions': 20, 'social_sessions': 16, ...}
    >>>
    >>> # Load specific session
    >>> session_001 = sm.get_session('session_001')
    >>> data = sm.load_session('session_001', modalities=['neural', 'behavior'])
    >>>
    >>> # Iterate over social sessions
    >>> for session in sm.iter_sessions('social'):
    >>>     print(f"Processing {session.id}...")
    >>>     data = sm.load_session(session.id)
    >>>
    >>> # Get drug condition for OT analysis
    >>> drug_info = sm.get_drug_condition('session_001')
    >>> # {'drug': 'oxytocin', 'dosed_subject': 'hooke', 'condition': 'Social'}
"""

import csv
import yaml
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from datetime import datetime
import platform
import warnings


@dataclass
class DrugCondition:
    """Drug condition metadata for a session.

    Attributes:
        date: Session date in YYMMDD format
        condition: Experimental condition ('Social', 'Asocial', 'Baseline')
        drug: Drug administered ('oxytocin', 'saline', 'none')
        dosed_subject: Subject who received drug ('hooke', 'hedy', 'both', 'none')
    """
    date: str
    condition: str
    drug: str
    dosed_subject: str

    @property
    def is_ot(self) -> bool:
        """Return True if this is an oxytocin session."""
        return self.drug == 'oxytocin'

    @property
    def is_saline(self) -> bool:
        """Return True if this is a saline control session."""
        return self.drug == 'saline'

    @property
    def is_social(self) -> bool:
        """Return True if this is a social session."""
        return self.condition == 'Social'

    def get_treatment_label(self, subject: str) -> str:
        """Get treatment label for a specific subject.

        Args:
            subject: Subject name ('hooke' or 'hedy')

        Returns:
            Treatment label: 'OT', 'saline', 'undosed_partner', or 'none'
        """
        subject = subject.lower()
        if self.drug == 'none':
            return 'none'
        if self.drug == 'saline':
            return 'saline'
        if self.drug == 'oxytocin':
            if self.dosed_subject == 'both':
                return 'OT'
            elif self.dosed_subject == subject:
                return 'OT'
            else:
                return 'undosed_partner'
        return 'unknown'


class Session:
    """Represents a single recording session.
    
    Encapsulates all metadata and data paths for a recording session,
    handling both social (2-monkey) and asocial (1-monkey) configurations.
    
    Attributes:
        id: Unique session identifier (e.g., 'session_001')
        type: Session type ('social' or 'asocial')
        duration_hours: Recording duration in hours
        monkeys: List of monkey identifiers present
        data_files: Dict mapping data types to file paths
        metadata: Additional session-specific metadata
    """
    
    def __init__(self, session_id: str, config: Dict[str, Any]) -> None:
        """
        Initialize a Session from configuration.
        
        Args:
            session_id: Unique identifier for the session
            config: Configuration dictionary containing session details
        """
        self.id: str = session_id
        self.config: Dict[str, Any] = config
        self.type: str = config.get('type', 'social')
        self.duration_hours: float = config.get('duration_hours', 2.0)
        self.monkeys: List[str] = config.get('monkeys', [])
        self.data_files: Dict[str, str] = config.get('data_files', {})
        self.metadata: Dict[str, Any] = config.get('metadata', {})
        
    def __repr__(self) -> str:
        """String representation of the session."""
        return f"Session({self.id}, type={self.type}, monkeys={self.monkeys})"
    
    def has_data(self, modality: str) -> bool:
        """Check if session has specific data type.
        
        Args:
            modality: Type of data to check ('neural', 'behavior', or 'pose')
            
        Returns:
            bool: True if the session has this data type configured
        """
        if modality == 'neural':
            return any(k.endswith('_neural') for k in self.data_files.keys())
        elif modality == 'behavior':
            return 'behavior' in self.data_files
        elif modality == 'pose':
            return 'pose_3d' in self.data_files and self.data_files['pose_3d'] is not None
        return False


class SessionManager:
    """
    Manages all recording sessions with intelligent caching.
    
    Central hub for accessing FreeMat's 20 recording sessions. Provides
    lazy loading, intelligent caching, and unified access patterns for
    neural, behavioral, and pose data.
    
    Attributes:
        config_path: Path to YAML configuration file
        config: Loaded configuration dictionary
        sessions: Dict of Session objects by ID
        disk_cache: Optional CacheManager for persistent caching
    
    Example:
        >>> sm = SessionManager('configs/sessions.yaml')
        >>> 
        >>> # Get summary
        >>> print(sm.get_summary())
        >>> 
        >>> # Iterate over social sessions
        >>> for session in sm.iter_sessions('social'):
        >>>     data = sm.load_session(session.id)
    """
    
    def __init__(self, config_path: str = 'configs/sessions.yaml',
                 drug_conditions_path: str = 'configs/drug_conditions.csv') -> None:
        """
        Initialize SessionManager with configuration file.

        Args:
            config_path: Path to sessions YAML configuration
            drug_conditions_path: Path to drug conditions CSV file
        """
        self.config_path: Path = Path(config_path)
        self.drug_conditions_path: Path = Path(drug_conditions_path)
        self.config: Dict[str, Any] = self._load_config()
        self.sessions: Dict[str, Session] = self._load_sessions()
        self._drug_conditions: Dict[str, DrugCondition] = self._load_drug_conditions()
        self._cache: Dict[str, Any] = {}  # Simple memory cache

        # Try to import CacheManager if available
        self.disk_cache: Optional[Any] = None
        try:
            from data.cache_manager import CacheManager
            self.disk_cache = CacheManager(self.config['paths']['cache_dir'])
        except (ImportError, KeyError):
            pass
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML.
        
        Returns:
            Dict containing configuration data
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_sessions(self) -> Dict[str, Session]:
        """Load all session definitions from config.

        Returns:
            Dict mapping session IDs to Session objects
        """
        sessions: Dict[str, Session] = {}

        for session_id, session_config in self.config.get('sessions', {}).items():
            sessions[session_id] = Session(session_id, session_config)

        return sessions

    def _load_drug_conditions(self) -> Dict[str, DrugCondition]:
        """Load drug condition metadata from CSV.

        The CSV file has columns: date, condition, drug, subject
        Maps session dates (YYMMDD) to DrugCondition objects.

        Returns:
            Dict mapping session date strings to DrugCondition objects
        """
        drug_conditions: Dict[str, DrugCondition] = {}

        if not self.drug_conditions_path.exists():
            warnings.warn(f"Drug conditions file not found: {self.drug_conditions_path}")
            return drug_conditions

        with open(self.drug_conditions_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get('date', '')
                condition = row.get('condition', '')
                drug = row.get('drug', 'none')
                subject = row.get('subject', 'none')

                # Normalize values
                drug = drug.lower() if drug else 'none'
                subject = subject.lower() if subject else 'none'

                # Handle special cases in subject field
                if subject in ('hooke_bad', 'hooke'):
                    subject = 'hooke'
                elif subject in ('hedy',):
                    subject = 'hedy'
                elif subject in ('both',):
                    subject = 'both'
                else:
                    subject = 'none'

                drug_conditions[date] = DrugCondition(
                    date=date,
                    condition=condition,
                    drug=drug,
                    dosed_subject=subject
                )

        return drug_conditions

    def get_drug_condition(self, session_id: str) -> Optional[DrugCondition]:
        """Get drug condition for a session.

        Args:
            session_id: Session identifier (e.g., 'session_001')

        Returns:
            DrugCondition object if found, None otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return None

        # Extract date from session config
        session_date = session.config.get('date', '')

        # Try to find matching drug condition by date
        # Session dates are in YYYY-MM-DD format, drug conditions are in YYMMDD
        if session_date:
            # Convert YYYY-MM-DD to YYMMDD
            date_parts = session_date.split('-')
            if len(date_parts) == 3:
                yymmdd = date_parts[0][2:] + date_parts[1] + date_parts[2]
                if yymmdd in self._drug_conditions:
                    return self._drug_conditions[yymmdd]

        # Also try direct lookup with YYMMDD format
        for date_key, condition in self._drug_conditions.items():
            if date_key in session_id or session_date.replace('-', '').endswith(date_key):
                return condition

        return None

    def get_drug_condition_by_date(self, date: str) -> Optional[DrugCondition]:
        """Get drug condition by date string.

        Args:
            date: Date in YYMMDD format (e.g., '240531')

        Returns:
            DrugCondition object if found, None otherwise
        """
        return self._drug_conditions.get(date)

    def iter_sessions_by_drug(
        self,
        drug: str = 'all',
        include_undosed: bool = True
    ) -> Generator[Session, None, None]:
        """Iterate over sessions filtered by drug condition.

        Args:
            drug: 'oxytocin', 'saline', 'none', or 'all'
            include_undosed: If False, skip sessions where drug conditions are unknown

        Yields:
            Session objects matching the drug filter
        """
        for session in self.sessions.values():
            condition = self.get_drug_condition(session.id)

            if condition is None:
                if include_undosed:
                    yield session
                continue

            if drug == 'all':
                yield session
            elif drug == 'oxytocin' and condition.is_ot:
                yield session
            elif drug == 'saline' and condition.is_saline:
                yield session
            elif drug == 'none' and condition.drug == 'none':
                yield session

    def get_sessions_summary_by_drug(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of sessions grouped by drug condition.

        Returns:
            Dict mapping drug condition to session counts and details
        """
        summary: Dict[str, Dict[str, Any]] = {
            'oxytocin': {'count': 0, 'sessions': [], 'dosed_subjects': {}},
            'saline': {'count': 0, 'sessions': []},
            'none': {'count': 0, 'sessions': []},
            'unknown': {'count': 0, 'sessions': []}
        }

        for session in self.sessions.values():
            condition = self.get_drug_condition(session.id)

            if condition is None:
                summary['unknown']['count'] += 1
                summary['unknown']['sessions'].append(session.id)
            elif condition.is_ot:
                summary['oxytocin']['count'] += 1
                summary['oxytocin']['sessions'].append(session.id)
                # Track who was dosed
                subj = condition.dosed_subject
                if subj not in summary['oxytocin']['dosed_subjects']:
                    summary['oxytocin']['dosed_subjects'][subj] = 0
                summary['oxytocin']['dosed_subjects'][subj] += 1
            elif condition.is_saline:
                summary['saline']['count'] += 1
                summary['saline']['sessions'].append(session.id)
            else:
                summary['none']['count'] += 1
                summary['none']['sessions'].append(session.id)

        return summary

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get specific session by ID.
        
        Args:
            session_id: Session identifier (e.g., 'session_001')
            
        Returns:
            Session object if found, None otherwise
        """
        return self.sessions.get(session_id, None)
    
    def iter_sessions(self, 
                     session_type: str = 'all',
                     shuffle: bool = False) -> Generator[Session, None, None]:
        """
        Iterate over sessions of specified type.
        
        Args:
            session_type: 'all', 'social', or 'asocial'
            shuffle: Whether to randomize order
            
        Yields:
            Session objects
        """
        session_list = []
        
        for session in self.sessions.values():
            if session_type == 'all' or session.type == session_type:
                session_list.append(session)
        
        if shuffle:
            import random
            random.shuffle(session_list)
        
        for session in session_list:
            yield session
    
    def load_session(self,
                    session_id: str,
                    modalities: List[str] = ['neural', 'behavior'],
                    use_cache: bool = True,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Load session data with caching.
        
        Args:
            session_id: Session identifier
            modalities: Data types to load
            use_cache: Whether to use caching
            verbose: Print loading messages
            
        Returns:
            Dictionary with requested data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Check memory cache
        cache_key = f"{session_id}_{'_'.join(sorted(modalities))}"
        
        if use_cache and cache_key in self._cache:
            if verbose:
                print(f"  Loading {session_id} from memory cache...")
            return self._cache[cache_key]
        
        # Check disk cache
        if use_cache and self.disk_cache and self.disk_cache.exists(cache_key):
            if verbose:
                print(f"  Loading {session_id} from disk cache...")
            return self.disk_cache.load(cache_key)
        
        # Load data
        if verbose:
            print(f"  Loading {session_id} from files...")
        
        data = {
            'session_id': session_id,
            'metadata': session.metadata,
            'type': session.type
        }
        
        # Load each modality
        if 'neural' in modalities:
            data.update(self._load_neural(session))
        
        if 'behavior' in modalities:
            data['behavior'] = self._load_behavior(session)
        
        if 'pose' in modalities and session.has_data('pose'):
            data['pose'] = self._load_pose(session)
        
        # Cache
        if use_cache:
            self._cache[cache_key] = data
            
            if self.disk_cache:
                self.disk_cache.save(cache_key, data, metadata={
                    'session_id': session_id,
                    'modalities': modalities,
                    'timestamp': datetime.now().isoformat()
                })
        
        return data
    
    def _load_neural(self, session: Session) -> Dict[str, Any]:
        """Load neural data for session."""
        neural_data = {}
        
        # Try to use existing loaders
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from data_utils.spike_utils import load_and_prep_neural_data
            
            for key, path in session.data_files.items():
                if key.endswith('_neural') and path and path != 'TO_BE_CONFIGURED':
                    monkey = key.replace('_neural', '')
                    neural_data[f'{monkey}_neural'] = load_and_prep_neural_data(path)
        except (ImportError, FileNotFoundError) as e:
            warnings.warn(f"Could not load neural data: {e}")
            # Return mock data for testing
            import numpy as np
            for monkey in session.monkeys:
                neural_data[f'{monkey.lower()}_neural'] = np.random.randn(100, 7200)  # 100 neurons, 2 hours
        
        return neural_data
    
    def _load_behavior(self, session: Session) -> Any:
        """Load behavioral data for session."""
        behavior_path = session.data_files.get('behavior', None)
        
        if not behavior_path or behavior_path == 'TO_BE_CONFIGURED':
            # Return mock data for testing
            import pandas as pd
            import numpy as np
            
            behaviors = ['groom', 'travel', 'rest', 'forage']
            n_samples = 7200  # 2 hours at 1Hz
            
            return pd.DataFrame({
                'time': np.arange(n_samples),
                'Hooke-Behavior': np.random.choice(behaviors, n_samples),
                'Hedy-Behavior': np.random.choice(behaviors, n_samples),
                'Seconds-Hooke': np.arange(n_samples),
                'Seconds-Hedy': np.arange(n_samples)
            })
        
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from data_utils.behavior_utils import load_and_align_behavioral_data
            return load_and_align_behavioral_data(behavior_path)
        except (ImportError, FileNotFoundError):
            import pandas as pd
            return pd.read_csv(behavior_path)
    
    def _load_pose(self, session: Session) -> Any:
        """Load 3D pose data for session."""
        pose_path = session.data_files.get('pose_3d', None)
        
        if not pose_path or pose_path == 'TO_BE_CONFIGURED':
            return None
        
        import numpy as np
        if pose_path.endswith('.npy'):
            return np.load(pose_path)
        else:
            warnings.warn(f"Unknown pose format: {pose_path}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all sessions.
        
        Returns:
            Dict containing summary statistics including:
            - total_sessions: Total number of sessions
            - social_sessions: Number of social sessions
            - asocial_sessions: Number of asocial sessions
            - total_hours: Total recording hours
            - data_availability: Counts by data type
            - monkeys: List of unique monkey IDs
        """
        total: int = len(self.sessions)
        social = sum(1 for s in self.sessions.values() if s.type == 'social')
        asocial = total - social
        
        total_hours = sum(s.duration_hours for s in self.sessions.values())
        
        # Data availability
        has_neural = sum(1 for s in self.sessions.values() if s.has_data('neural'))
        has_behavior = sum(1 for s in self.sessions.values() if s.has_data('behavior'))
        has_pose = sum(1 for s in self.sessions.values() if s.has_data('pose'))
        
        return {
            'total_sessions': total,
            'social_sessions': social,
            'asocial_sessions': asocial,
            'total_hours': total_hours,
            'data_availability': {
                'neural': has_neural,
                'behavior': has_behavior,
                'pose': has_pose
            },
            'monkeys': list(set(m for s in self.sessions.values() for m in s.monkeys)),
            'average_duration_hours': total_hours / total if total > 0 else 0
        }
    
    def validate_data_paths(self) -> Dict[str, List[str]]:
        """Check which data files are missing or misconfigured.
        
        Returns:
            Dict with two keys:
            - 'missing': List of configured but missing files
            - 'unconfigured': List of files marked as TO_BE_CONFIGURED
        """
        issues: Dict[str, List[str]] = {
            'missing': [],
            'unconfigured': []
        }
        
        for session in self.sessions.values():
            for file_key, file_path in session.data_files.items():
                if file_path == 'TO_BE_CONFIGURED':
                    issues['unconfigured'].append(f"{session.id}/{file_key}")
                elif file_path and not Path(file_path).exists():
                    issues['missing'].append(f"{session.id}/{file_key}: {file_path}")
        
        return issues
