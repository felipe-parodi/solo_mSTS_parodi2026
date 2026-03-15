"""Neural manifold analysis for Figure 4 motif encoding.

Computes PCA and UMAP embeddings of population firing rates,
colored by behavioral motif labels. Quantifies cluster separation
via silhouette score with block-shuffle null distribution.

Block-shuffle permutes labels in contiguous 10 s blocks to preserve
temporal autocorrelation — distinct from the full random permutation
used in motif decoding (analysis/syllables/decoding.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ManifoldConfig:
    """Configuration for manifold analysis.

    Attributes:
        n_pca_components: Number of PCA components to compute.
        n_display_components: Number of components for display (2D/3D).
        zscore_units: Whether to z-score each unit before embedding.
        use_umap: Whether to compute UMAP embedding (requires umap-learn).
        umap_n_neighbors: UMAP n_neighbors parameter.
        umap_min_dist: UMAP min_dist parameter.
        n_perms: Number of block-shuffle permutations for null.
        n_subsample: Number of repeated class-balanced subsamplings.
        max_bins_silhouette: Max bins for O(N^2) silhouette computation.
        block_s: Block size in seconds for block-shuffle permutation.
        bin_s: Bin width in seconds (must match the binning used upstream).
        random_state: Random seed for reproducibility.
    """

    n_pca_components: int = 20
    n_display_components: int = 3
    zscore_units: bool = True
    use_umap: bool = True
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.3
    n_perms: int = 50
    n_subsample: int = 10
    max_bins_silhouette: int = 3000
    block_s: float = 10.0
    bin_s: float = 0.5
    random_state: int = 42


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SeparationResult:
    """Silhouette-based cluster separation result.

    Attributes:
        silhouette_mean: Mean silhouette score (observed, averaged over
            n_subsample class-balanced subsamplings).
        silhouette_ci: 95% CI from repeated subsampling [lower, upper].
        silhouette_per_class: Mean silhouette per class label.
        null_distribution: Array of null silhouette means (n_perms,).
        null_mean: Mean of null distribution.
        null_std: Std of null distribution.
        p_value: (1 + sum(null >= obs)) / (1 + n_perms).
    """

    silhouette_mean: float
    silhouette_ci: tuple[float, float]
    silhouette_per_class: dict[int, float]
    null_distribution: np.ndarray
    null_mean: float
    null_std: float
    p_value: float


@dataclass
class ManifoldResult:
    """Full manifold analysis result.

    Attributes:
        pca_embedding: PCA embedding (n_bins, n_pca_components).
        pca_var_explained: Variance explained per PC (n_pca_components,).
        umap_embedding: UMAP 2D embedding (n_bins, 2) or empty (0, 0).
        has_umap: Whether UMAP was computed.
        motif_labels: Motif labels for each bin (n_bins,).
        unit_banks: Bank labels per unit (n_units,).
        session_id: Session identifier.
        separation: Silhouette separation on PCA display dims (legacy).
        separations: Dict of SeparationResult keyed by embedding name.
    """

    pca_embedding: np.ndarray
    pca_var_explained: np.ndarray
    umap_embedding: np.ndarray
    has_umap: bool
    motif_labels: np.ndarray
    unit_banks: np.ndarray
    session_id: str
    separation: SeparationResult
    separations: dict[str, SeparationResult] | None = None


# ---------------------------------------------------------------------------
# PCA embedding
# ---------------------------------------------------------------------------

def compute_pca_embedding(
    X: np.ndarray,
    n_components: int = 20,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA embedding of population firing rates.

    Args:
        X: Firing rate matrix (n_bins, n_units).
        n_components: Number of PCA components.
        zscore: Whether to z-score each unit before PCA.

    Returns:
        Tuple of (embedding, var_explained):
        - embedding: (n_bins, n_components) PCA scores.
        - var_explained: (n_components,) fraction of variance per PC.
    """
    n_bins, n_units = X.shape
    n_components = min(n_components, n_bins, n_units)

    if zscore:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(X_scaled)
    return embedding, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# UMAP embedding
# ---------------------------------------------------------------------------

def compute_umap_embedding(
    X: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D UMAP embedding.

    Args:
        X: Input matrix (n_bins, n_features). Typically PCA-reduced.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed.

    Returns:
        2D embedding array (n_bins, 2).

    Raises:
        ImportError: If umap-learn is not installed.
    """
    try:
        import umap as umap_mod  # type: ignore[import-untyped]
    except (ImportError, ModuleNotFoundError):
        msg = "umap-learn is required for UMAP embedding. Install with: uv pip install umap-learn"
        raise ImportError(msg) from None

    reducer = umap_mod.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Block-shuffle permutation
# ---------------------------------------------------------------------------

def _block_shuffle_labels(
    labels: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle labels in contiguous blocks.

    Args:
        labels: Original label array (n_bins,).
        block_size: Number of bins per block.
        rng: NumPy random generator.

    Returns:
        Permuted label array with block structure preserved.
    """
    n = len(labels)
    n_blocks = int(np.ceil(n / block_size))
    block_indices = np.arange(n_blocks)
    rng.shuffle(block_indices)

    parts = []
    for idx in block_indices:
        start = idx * block_size
        end = min(start + block_size, n)
        parts.append(labels[start:end])
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Silhouette with class-balanced subsampling
# ---------------------------------------------------------------------------

def _subsample_silhouette(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_subsample: int,
    max_bins: int,
    rng: np.random.Generator,
) -> tuple[float, dict[int, float], list[float]]:
    """Compute silhouette with repeated class-balanced subsampling.

    Args:
        embedding: (n_bins, n_dims) embedding.
        labels: (n_bins,) class labels.
        n_subsample: Number of subsampling repetitions.
        max_bins: Maximum bins for silhouette computation.
        rng: NumPy random generator.

    Returns:
        Tuple of (mean_silhouette, per_class_dict, all_scores).
    """
    classes = sorted(set(labels))
    class_indices = {c: np.where(labels == c)[0] for c in classes}
    min_class_n = min(len(idx) for idx in class_indices.values())

    # Cap per-class count to stay within max_bins
    per_class_cap = max_bins // len(classes)
    per_class_n = min(min_class_n, per_class_cap)

    if per_class_n < 2:
        return 0.0, {c: 0.0 for c in classes}, [0.0]

    all_scores: list[float] = []
    per_class_accum: dict[int, list[float]] = {c: [] for c in classes}

    for _ in range(n_subsample):
        selected = []
        for c in classes:
            chosen = rng.choice(class_indices[c], size=per_class_n, replace=False)
            selected.append(chosen)
        idx = np.concatenate(selected)
        sub_emb = embedding[idx]
        sub_lab = labels[idx]

        score = silhouette_score(sub_emb, sub_lab)
        all_scores.append(float(score))

        samples = silhouette_samples(sub_emb, sub_lab)
        for c in classes:
            c_mask = sub_lab == c
            per_class_accum[c].append(float(np.mean(samples[c_mask])))

    mean_sil = float(np.mean(all_scores))
    per_class_mean = {c: float(np.mean(vals)) for c, vals in per_class_accum.items()}
    return mean_sil, per_class_mean, all_scores


# ---------------------------------------------------------------------------
# Cluster separation
# ---------------------------------------------------------------------------

def compute_cluster_separation(
    embedding: np.ndarray,
    labels: np.ndarray,
    config: ManifoldConfig,
) -> SeparationResult:
    """Compute silhouette cluster separation with block-shuffle null.

    Args:
        embedding: (n_bins, n_dims) embedding for silhouette.
        labels: (n_bins,) motif class labels.
        config: ManifoldConfig with n_perms, n_subsample, etc.

    Returns:
        SeparationResult with observed and null silhouette scores.
    """
    rng = np.random.default_rng(config.random_state)
    block_bins = max(1, int(config.block_s / config.bin_s))

    # Observed
    obs_mean, obs_per_class, obs_scores = _subsample_silhouette(
        embedding, labels, config.n_subsample, config.max_bins_silhouette, rng,
    )
    ci_lower = float(np.percentile(obs_scores, 2.5))
    ci_upper = float(np.percentile(obs_scores, 97.5))

    # Null distribution
    null_means: list[float] = []
    for _ in range(config.n_perms):
        shuffled_labels = _block_shuffle_labels(labels, block_bins, rng)
        null_mean, _, _ = _subsample_silhouette(
            embedding, shuffled_labels, config.n_subsample,
            config.max_bins_silhouette, rng,
        )
        null_means.append(null_mean)

    null_arr = np.array(null_means)
    p_value = float((1 + np.sum(null_arr >= obs_mean)) / (1 + config.n_perms))

    return SeparationResult(
        silhouette_mean=obs_mean,
        silhouette_ci=(ci_lower, ci_upper),
        silhouette_per_class=obs_per_class,
        null_distribution=null_arr,
        null_mean=float(np.mean(null_arr)),
        null_std=float(np.std(null_arr)),
        p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Full manifold pipeline
# ---------------------------------------------------------------------------

def compute_manifold_analysis(
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    unit_banks: np.ndarray,
    session_id: str,
    config: ManifoldConfig,
) -> ManifoldResult:
    """Run full manifold analysis: PCA, optional UMAP, silhouette.

    Args:
        X_valid: Firing rate matrix (n_bins, n_units) for valid bins.
        y_valid: Motif labels (n_bins,).
        unit_banks: Bank labels per unit (n_units,).
        session_id: Session identifier string.
        config: ManifoldConfig with all parameters.

    Returns:
        ManifoldResult with embeddings and separation metrics.
    """
    # PCA
    pca_emb, var_explained = compute_pca_embedding(
        X_valid, n_components=config.n_pca_components, zscore=config.zscore_units,
    )

    # UMAP (optional)
    if config.use_umap:
        umap_emb = compute_umap_embedding(
            pca_emb,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            random_state=config.random_state,
        )
        has_umap = True
    else:
        umap_emb = np.empty((0, 0))
        has_umap = False

    # Silhouette on multiple embeddings
    separations: dict[str, SeparationResult] = {}

    # PCA-3D (display dims)
    display_emb = pca_emb[:, :config.n_display_components]
    separations["pca_3d"] = compute_cluster_separation(display_emb, y_valid, config)

    # PCA-10D
    n_pca10 = min(10, pca_emb.shape[1])
    if n_pca10 > config.n_display_components:
        separations["pca_10d"] = compute_cluster_separation(
            pca_emb[:, :n_pca10], y_valid, config,
        )

    # PCA-full (all computed components)
    if pca_emb.shape[1] > n_pca10:
        separations["pca_full"] = compute_cluster_separation(
            pca_emb, y_valid, config,
        )

    # UMAP (if available)
    if has_umap:
        separations["umap"] = compute_cluster_separation(
            umap_emb, y_valid, config,
        )

    # Primary separation = best available (UMAP > PCA-10D > PCA-3D)
    if "umap" in separations:
        primary_key = "umap"
    elif "pca_10d" in separations:
        primary_key = "pca_10d"
    else:
        primary_key = "pca_3d"
    separation = separations[primary_key]

    return ManifoldResult(
        pca_embedding=pca_emb,
        pca_var_explained=var_explained,
        umap_embedding=umap_emb,
        has_umap=has_umap,
        motif_labels=y_valid,
        unit_banks=unit_banks,
        session_id=session_id,
        separation=separation,
        separations=separations,
    )
