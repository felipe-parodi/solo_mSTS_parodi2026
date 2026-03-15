"""
Smart caching system for large neural datasets.

This module provides intelligent caching with automatic metadata tracking,
compression, and catalog management for FreeMat's large neural datasets.
Supports both memory and disk caching with automatic cleanup of old entries.

Example:
    Basic caching workflow for neural data:
    
    >>> from data.cache_manager import CacheManager
    >>> cache = CacheManager('./cache')
    >>> 
    >>> # Save processed neural data with metadata
    >>> cache.save('session_001_binned', binned_spikes, metadata={
    >>>     'processing': 'binned_1s',
    >>>     'session': 'session_001',
    >>>     'n_units': 128,
    >>>     'duration_hours': 2.0
    >>> })
    >>> 
    >>> # Load later with automatic decompression
    >>> data = cache.load('session_001_binned')
    >>> 
    >>> # Check cache statistics
    >>> print(cache.get_summary())
    >>> # {'n_items': 15, 'total_size_gb': 2.5, ...}
    >>> 
    >>> # Clean old cache entries
    >>> cache.clear_old(days=30)
"""

import os
import json
import pickle
import hashlib
import joblib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
import numpy as np


class CacheManager:
    """
    Intelligent caching with automatic documentation.
    
    Example:
        cache = CacheManager('./cache')
        
        # Save with metadata
        cache.save('processed_data', data, metadata={
            'processing': 'binned_1s',
            'session': 'session_001'
        })
        
        # Load later
        data = cache.load('processed_data')
    """
    
    def __init__(self, cache_dir: str = './data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.catalog_path = self.cache_dir / '.cache_catalog.json'
        self.catalog = self._load_catalog()
    
    def _load_catalog(self) -> Dict:
        """Load or create cache catalog."""
        if self.catalog_path.exists():
            with open(self.catalog_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_catalog(self):
        """Save cache catalog."""
        with open(self.catalog_path, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)
    
    def _get_data_info(self, data: Any) -> Dict:
        """Extract information about data."""
        info = {'type': type(data).__name__}
        
        if isinstance(data, np.ndarray):
            info['shape'] = list(data.shape)
            info['dtype'] = str(data.dtype)
            info['size_mb'] = data.nbytes / 1e6
        elif isinstance(data, dict):
            info['keys'] = list(data.keys())
            info['size'] = len(data)
        elif hasattr(data, 'shape'):  # DataFrame, etc.
            info['shape'] = list(data.shape)
        elif isinstance(data, (list, tuple)):
            info['length'] = len(data)
        
        return info
    
    def _compute_hash(self, key: str, data: Any) -> str:
        """Compute hash for cache key."""
        # Use key and data type for hash
        hash_input = f"{key}_{type(data).__name__}".encode()
        return hashlib.md5(hash_input).hexdigest()[:12]
    
    def save(self,
             key: str,
             data: Any,
             metadata: Optional[Dict] = None,
             compress: bool = True) -> Path:
        """
        Save data with automatic documentation.
        
        Args:
            key: Unique identifier
            data: Data to cache
            metadata: Additional metadata
            compress: Whether to compress (recommended for arrays)
            
        Returns:
            Path to saved file
        """
        # Create unique filename
        hash_suffix = self._compute_hash(key, data)
        filename = f"{key}_{hash_suffix}.pkl"
        data_path = self.cache_dir / filename
        meta_path = self.cache_dir / f"{key}_{hash_suffix}.meta.json"
        
        # Save data
        if compress and isinstance(data, (np.ndarray, dict)):
            joblib.dump(data, data_path, compress=3)
        else:
            with open(data_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'key': key,
            'filename': filename,
            'created': datetime.now().isoformat(),
            'data_info': self._get_data_info(data),
            'compressed': compress,
            'file_size_mb': data_path.stat().st_size / 1e6
        })
        
        # Save metadata
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Update catalog
        self.catalog[key] = {
            'filename': filename,
            'created': metadata['created'],
            'size_mb': metadata['file_size_mb'],
            'data_type': metadata['data_info']['type']
        }
        self._save_catalog()
        
        return data_path
    
    def load(self, key: str) -> Any:
        """
        Load cached data.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data
        """
        # Find file
        if key not in self.catalog:
            raise KeyError(f"Cache key '{key}' not found")
        
        filename = self.catalog[key]['filename']
        data_path = self.cache_dir / filename
        
        if not data_path.exists():
            # Remove from catalog if file missing
            del self.catalog[key]
            self._save_catalog()
            raise FileNotFoundError(f"Cache file missing: {filename}")
        
        # Load data
        try:
            return joblib.load(data_path)
        except:
            with open(data_path, 'rb') as f:
                return pickle.load(f)
    
    def exists(self, key: str) -> bool:
        """Check if cache key exists."""
        if key not in self.catalog:
            return False
        
        filename = self.catalog[key]['filename']
        return (self.cache_dir / filename).exists()
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata for cached item."""
        if key not in self.catalog:
            return None
        
        filename = self.catalog[key]['filename']
        hash_suffix = filename.replace(f"{key}_", "").replace(".pkl", "")
        meta_path = self.cache_dir / f"{key}_{hash_suffix}.meta.json"
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        return None
    
    def delete(self, key: str):
        """Delete cached item."""
        if key not in self.catalog:
            return
        
        filename = self.catalog[key]['filename']
        data_path = self.cache_dir / filename
        
        # Also delete metadata
        hash_suffix = filename.replace(f"{key}_", "").replace(".pkl", "")
        meta_path = self.cache_dir / f"{key}_{hash_suffix}.meta.json"
        
        if data_path.exists():
            data_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        
        del self.catalog[key]
        self._save_catalog()
    
    def clear_old(self, days: int = 30):
        """Clear cache items older than specified days."""
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        for key, info in list(self.catalog.items()):
            created = datetime.fromisoformat(info['created'])
            if created < cutoff:
                self.delete(key)
                print(f"  Deleted old cache: {key}")
    
    def get_summary(self) -> Dict:
        """Get cache summary statistics."""
        total_size = sum(info.get('size_mb', 0) for info in self.catalog.values())
        
        # Group by data type
        by_type = {}
        for info in self.catalog.values():
            dtype = info.get('data_type', 'unknown')
            by_type[dtype] = by_type.get(dtype, 0) + 1
        
        return {
            'n_items': len(self.catalog),
            'total_size_mb': round(total_size, 2),
            'total_size_gb': round(total_size / 1000, 2),
            'by_type': by_type,
            'cache_dir': str(self.cache_dir)
        }
