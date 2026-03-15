"""Compatibility shim for pathing → paths.

This module exists for backward compatibility. Use freemat_data.paths instead.
"""

from .paths import PathHandler

# Create a default instance for the normalize function
_handler = PathHandler()
normalize = _handler.normalize
