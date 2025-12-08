"""Fusion pipeline: concatenate views, PCA, normalize."""
from multiprocessing import cpu_count
from typing import Tuple, List

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


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
                    n_jobs: int = -1,
                    gpu_accelerator=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse multiple embedding views using concatenation and PCA (GPU-accelerated if available).
    
    Args:
        embeddings_list: list of embedding matrices from different views
        num_nodes: number of nodes
        final_dim: final embedding dimensionality
        random_state: random seed
        n_jobs: number of parallel jobs (not used, kept for API consistency)
        gpu_accelerator: Optional GPUAccelerator instance for GPU acceleration
    
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

    # Try GPU acceleration if available
    if gpu_accelerator and gpu_accelerator.use_gpu:
        try:
            fused, components, mean = gpu_accelerator.pca(concatenated, actual_dim, random_state)
        except Exception:
            # Fall back to CPU
            pca = PCA(n_components=actual_dim, random_state=random_state)
            fused = pca.fit_transform(concatenated)
            components = pca.components_
            mean = pca.mean_
    else:
        # CPU path
        pca = PCA(n_components=actual_dim, random_state=random_state)
        fused = pca.fit_transform(concatenated)
        components = pca.components_
        mean = pca.mean_

    # Pad if needed to match requested dimension
    if actual_dim < final_dim:
        padding = np.zeros((fused.shape[0], final_dim - actual_dim))
        fused = np.hstack([fused, padding])
        # Pad components
        component_padding = np.zeros((final_dim - actual_dim, components.shape[1]))
        components = np.vstack([components, component_padding])

    # Normalize each row to unit length (use GPU if available)
    if gpu_accelerator and gpu_accelerator.use_gpu:
        try:
            norms = gpu_accelerator.norm(fused, axis=1, keepdims=True)
            norms = gpu_accelerator.maximum(norms, 1e-10)
            fused_normalized = fused / norms
        except Exception:
            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            fused_normalized = fused / norms
    else:
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        fused_normalized = fused / norms

    return fused_normalized, components, mean


def build_neighbor_graph(edges: List[Tuple[int, int, str, float]], num_nodes: int) -> sparse.csr_matrix:
    """
    Build neighbor graph adjacency matrix for smoothing.
    
    Args:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of nodes
    
    Returns:
        Sparse CSR adjacency matrix
    """
    rows = []
    cols = []

    for src_idx, dst_idx, rel, weight in edges:
        rows.append(src_idx)
        cols.append(dst_idx)

    # Create symmetric matrix (undirected graph)
    rows_sym = rows + cols
    cols_sym = cols + rows

    adj = sparse.csr_matrix((np.ones(len(rows_sym)), (rows_sym, cols_sym)), shape=(num_nodes, num_nodes))
    return adj


def iterative_embedding_smoothing(embeddings: np.ndarray,
                                  edges: List[Tuple[int, int, str, float]],
                                  num_nodes: int,
                                  num_iterations: int = 2,
                                  beta: float = 0.35,
                                  random_state: int = 42,
                                  gpu_accelerator=None) -> np.ndarray:
    """
    Apply iterative embedding smoothing (diffusion) to embeddings.
    Optimized with vectorized operations.
    
    Args:
        embeddings: input embeddings (num_nodes, dim)
        edges: list of edges for neighbor graph
        num_nodes: number of nodes
        num_iterations: number of smoothing iterations (default 2)
        beta: smoothing factor (default 0.35)
        random_state: random seed
    
    Returns:
        Smoothed embeddings
    """
    np.random.seed(random_state)

    # Build neighbor graph
    adj = build_neighbor_graph(edges, num_nodes)

    # Iterative smoothing with vectorized operations
    smoothed = embeddings.copy()

    for iteration in range(num_iterations):
        # Vectorized neighbor averaging using sparse matrix multiplication
        # Use GPU acceleration if available
        if gpu_accelerator and gpu_accelerator.use_gpu:
            try:
                neighbor_sums = gpu_accelerator.sparse_matmul(adj, smoothed)
                degrees = gpu_accelerator.sum(adj, axis=1)
                degrees = gpu_accelerator.maximum(degrees.flatten(), 1.0)
                neighbor_avg = neighbor_sums / degrees[:, np.newaxis]
                smoothed = beta * neighbor_avg + (1 - beta) * smoothed
                norms = gpu_accelerator.norm(smoothed, axis=1, keepdims=True)
                norms = gpu_accelerator.maximum(norms, 1e-10)
                smoothed = smoothed / norms
            except Exception:
                # Fall back to CPU
                neighbor_sums = adj.dot(smoothed)
                degrees = np.array(adj.sum(axis=1)).flatten()
                degrees = np.maximum(degrees, 1.0)
                neighbor_avg = neighbor_sums / degrees[:, np.newaxis]
                smoothed = beta * neighbor_avg + (1 - beta) * smoothed
                norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                smoothed = smoothed / norms
        else:
            # CPU path
            neighbor_sums = adj.dot(smoothed)
            degrees = np.array(adj.sum(axis=1)).flatten()
            degrees = np.maximum(degrees, 1.0)
            neighbor_avg = neighbor_sums / degrees[:, np.newaxis]
            smoothed = beta * neighbor_avg + (1 - beta) * smoothed
            norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            smoothed = smoothed / norms

    return smoothed
