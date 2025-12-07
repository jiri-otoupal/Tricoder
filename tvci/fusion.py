"""Fusion pipeline: concatenate views, PCA, normalize."""
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, List
from multiprocessing import cpu_count


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def pad_to_same_rows(embeddings_list: List[np.ndarray], num_nodes: int) -> List[np.ndarray]:
    """
    Pad each embedding matrix to have num_nodes rows.
    
    Args:
        embeddings_list: list of embedding matrices
        num_nodes: target number of rows
    
    Returns:
        List of padded matrices
    """
    padded = []
    for emb in embeddings_list:
        if emb.shape[0] < num_nodes:
            # Pad with zeros
            padding = np.zeros((num_nodes - emb.shape[0], emb.shape[1]))
            emb_padded = np.vstack([emb, padding])
        else:
            emb_padded = emb[:num_nodes]
        padded.append(emb_padded)
    return padded


def fuse_embeddings(embeddings_list: List[np.ndarray],
                   num_nodes: int,
                   final_dim: int,
                   random_state: int = 42,
                   n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse multiple embedding views using concatenation and PCA.
    
    Args:
        embeddings_list: list of embedding matrices from different views
        num_nodes: number of nodes
        final_dim: final embedding dimensionality
        random_state: random seed
    
    Returns:
        fused_embeddings: final fused embeddings
        pca_components: PCA transformation matrix
        pca_mean: PCA mean vector
    """
    # Pad all matrices to same number of rows
    padded = pad_to_same_rows(embeddings_list, num_nodes)
    
    # Concatenate horizontally
    concatenated = np.hstack(padded)
    
    # Determine actual PCA dimension (can't exceed min(n_samples, n_features))
    num_features = concatenated.shape[1]
    actual_dim = min(final_dim, num_nodes, num_features)
    
    # PCA uses threading internally, no explicit n_jobs parameter
    # Apply PCA
    pca = PCA(n_components=actual_dim, random_state=random_state)
    fused = pca.fit_transform(concatenated)
    
    # Pad if needed to match requested dimension
    if actual_dim < final_dim:
        padding = np.zeros((fused.shape[0], final_dim - actual_dim))
        fused = np.hstack([fused, padding])
        # Pad components
        component_padding = np.zeros((final_dim - actual_dim, pca.components_.shape[1]))
        components = np.vstack([pca.components_, component_padding])
        # Pad mean (just keep as is, PCA handles this)
        mean = pca.mean_
    else:
        components = pca.components_
        mean = pca.mean_
    
    # Normalize each row to unit length
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    fused_normalized = fused / norms
    
    return fused_normalized, components, mean

