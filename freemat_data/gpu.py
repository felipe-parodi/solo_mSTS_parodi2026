"""GPU/CPU switching utilities.

This module provides transparent GPU/CPU library switching for data science
operations. It wraps the underlying gpu_utils implementation and will be
the canonical location for GPU utilities going forward.

Currently re-exports from data_utils.gpu_utils for backward compatibility.
The implementation will be migrated here in a future refactor.

Usage:
    from freemat_data.gpu import initialize, np, pd, LinearSVC

    initialize(use_gpu=True)
    data = np.random.randn(1000, 100)
"""

# Re-export everything from the existing module
# This will be replaced with direct implementation in future refactor
from data_utils.gpu_utils import (
    initialize,
    to_numpy,
    from_numpy,
    to_pandas,
    from_pandas,
    synchronize,
    IS_GPU_AVAILABLE,
    USE_GPU,
    pd,
    np,
    LinearSVC,
    LogisticRegression,
    RandomForestClassifier,
    PoissonRegressor,
    KFold,
    StandardScaler,
)

__all__ = [
    "initialize",
    "to_numpy",
    "from_numpy",
    "to_pandas",
    "from_pandas",
    "synchronize",
    "IS_GPU_AVAILABLE",
    "USE_GPU",
    "pd",
    "np",
    "LinearSVC",
    "LogisticRegression",
    "RandomForestClassifier",
    "PoissonRegressor",
    "KFold",
    "StandardScaler",
]
