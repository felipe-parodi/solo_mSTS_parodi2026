"""
Cross-platform path handling for Windows/Linux compatibility.

This module provides utilities for seamless file path handling across Windows
development environments and Linux compute clusters. It automatically converts
drive letter paths (A:/) to mount paths (/mnt/a/) and vice versa.

Example:
    Basic path normalization across platforms:
    
    >>> from data.path_handler import PathHandler
    >>> ph = PathHandler()
    >>> 
    >>> # On Windows: converts Linux paths
    >>> path = ph.normalize("/mnt/a/data/session.mat")
    >>> # Returns: Path("A:/data/session.mat")
    >>> 
    >>> # On Linux: converts Windows paths  
    >>> path = ph.normalize("A:\\data\\session.mat")
    >>> # Returns: Path("/mnt/a/data/session.mat")
    >>> 
    >>> # Find data root automatically
    >>> root = ph.find_data_root()
    >>> data_file = root / "neural" / "session_001.mat"
"""

import os
import platform
from pathlib import Path
from typing import Union, Optional, Dict


class PathHandler:
    """
    Handle path differences between Windows and Linux systems.
    
    Provides automatic path conversion between Windows drive letters (A:/, B:/)
    and Linux mount points (/mnt/a/, /mnt/b/). Essential for working with
    data stored on network drives accessed from both environments.
    
    Attributes:
        None (all methods are static)
        
    Example:
        >>> ph = PathHandler()
        >>> 
        >>> # Works on both platforms
        >>> mat_file = ph.normalize("A:\\data\\session.mat")
        >>> 
        >>> # Ensure directory exists
        >>> cache_dir = ph.ensure_dir("cache/results")
    """
    
    @staticmethod
    def get_platform() -> str:
        """
        Get current platform name.
        
        Returns:
            str: Platform name ('windows', 'linux', or 'darwin')
        """
        return platform.system().lower()
    
    @staticmethod
    def normalize(path: Union[str, Path]) -> Optional[Path]:
        """
        Convert path to work on current OS.
        
        Handles conversion between Windows drive letters and Linux mount points.
        Preserves None values for optional paths.
        
        Args:
            path: Input path in Windows or Linux format, or None
            
        Returns:
            Path object that works on current OS, or None if input was None
            
        Example:
            >>> # On Windows
            >>> PathHandler.normalize("/mnt/a/data/file.mat")
            >>> # Returns: Path("A:/data/file.mat")
            >>> 
            >>> # On Linux
            >>> PathHandler.normalize("A:\\data\\file.mat")
            >>> # Returns: Path("/mnt/a/data/file.mat")
        """
        if path is None:
            return None
        
        path_str = str(path)
        current_platform = PathHandler.get_platform()
        
        # Handle Windows paths on Linux
        if current_platform == 'linux':
            # Convert Windows paths: A:\ or A:/
            if len(path_str) > 2 and path_str[1] == ':':
                drive = path_str[0].lower()
                rest = path_str[2:].replace('\\', '/').replace('\\', '/')
                path_str = f'/mnt/{drive}{rest}'
        
        # Handle Linux paths on Windows
        elif current_platform == 'windows':
            # Convert /mnt/x/ paths
            if path_str.startswith('/mnt/'):
                parts = path_str.split('/')
                if len(parts) > 2:
                    drive = parts[2].upper()
                    rest = '/'.join(parts[3:])
                    path_str = f'{drive}:/{rest}'
        
        return Path(path_str)
    
    @staticmethod
    def to_universal(path: Union[str, Path]) -> str:
        """
        Convert path to universal format for config files.
        
        Creates a platform-agnostic path string that can be stored in 
        configuration files and used across different operating systems.
        
        Args:
            path: Local path in any format
            
        Returns:
            str: Universal path string with forward slashes
            
        Example:
            >>> PathHandler.to_universal("A:\\data\\session.mat")
            >>> # Returns: "A:/data/session.mat"
            >>> 
            >>> PathHandler.to_universal(Path("/mnt/a/data/session.mat"))
            >>> # Returns: "/mnt/a/data/session.mat"
        """
        path = Path(path)
        
        # Convert to forward slashes
        universal = str(path).replace('\\', '/')
        
        # On Windows, preserve drive letter
        if PathHandler.get_platform() == 'windows' and path.drive:
            universal = f"{path.drive}/{universal[3:]}"
        
        return universal
    
    @staticmethod
    def find_data_root() -> Optional[Path]:
        """
        Find FreeMat data root directory regardless of platform.
        
        Searches common locations for the FreeMat data directory,
        returning the first valid path found.
        
        Returns:
            Optional[Path]: Path to data root if found, None otherwise
            
        Example:
            >>> root = PathHandler.find_data_root()
            >>> if root:
            >>>     session_file = root / "sessions" / "session_001.mat"
        """
        possible_roots = [
            Path.cwd(),
            Path.home() / 'freemat_data',
            Path('/data/freemat'),
            Path('A:/EnclosureProjects/inprep/freemat'),
            Path('/mnt/a/EnclosureProjects/inprep/freemat')
        ]
        
        for root in possible_roots:
            if root.exists():
                return root
        
        return None
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, creating if needed.
        
        Creates all parent directories if they don't exist.
        Safe to call on existing directories.
        
        Args:
            path: Directory path to create
            
        Returns:
            Path: Normalized path to the directory
            
        Example:
            >>> cache_dir = PathHandler.ensure_dir("cache/neural")
            >>> # Creates cache/ and cache/neural/ if needed
        """
        path = PathHandler.normalize(path)
        if path:
            path.mkdir(parents=True, exist_ok=True)
        return path
