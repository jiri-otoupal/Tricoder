"""Typed view: symbol × type-token matrix, PPMI, and SVD."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Tuple
from multiprocessing import cpu_count


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def build_type_matrix(node_types: Dict[int, Dict[str, int]], 
                     type_to_idx: Dict[str, int],
                     num_nodes: int) -> sparse.csr_matrix:
    """
    Build sparse symbol × type-token matrix.
    
    Args:
        node_types: mapping from node_idx to {type_token: count}
        type_to_idx: mapping from type token to index
        num_nodes: number of nodes
    
    Returns:
        Sparse CSR matrix of shape (num_nodes, num_types)
    """
    num_types = len(type_to_idx)
    rows = []
    cols = []
    data = []
    
    for node_idx, types_dict in node_types.items():
        for type_token, count in types_dict.items():
            if type_token in type_to_idx:
                type_idx = type_to_idx[type_token]
                rows.append(node_idx)
                cols.append(type_idx)
                data.append(float(count))
    
    type_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_types))
    return type_matrix


def compute_ppmi_types(type_matrix: sparse.csr_matrix, k: float = 1.0) -> sparse.csr_matrix:
    """
    Compute PPMI for type matrix.
    
    Args:
        type_matrix: sparse symbol × type matrix
        k: shift parameter
    
    Returns:
        PPMI matrix (sparse)
    """
    cooc = type_matrix.copy()
    cooc.data = np.maximum(cooc.data, 0)
    
    # Compute marginals
    row_sums = np.array(cooc.sum(axis=1)).flatten()
    col_sums = np.array(cooc.sum(axis=0)).flatten()
    total = cooc.sum()
    
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    col_sums = np.maximum(col_sums, 1e-10)
    total = max(total, 1e-10)
    
    # Compute PMI
    rows, cols = cooc.nonzero()
    values = cooc.data
    
    p_ij = values / total
    p_i = row_sums[rows] / total
    p_j = col_sums[cols] / total
    
    pmi = np.log(p_ij / (p_i * p_j + 1e-10) + 1e-10)
    ppmi = np.maximum(pmi, 0.0)
    
    ppmi_matrix = sparse.csr_matrix((ppmi, (rows, cols)), shape=cooc.shape)
    return ppmi_matrix


def compute_typed_view(node_types: Dict[int, Dict[str, int]],
                      type_to_idx: Dict[str, int],
                      num_nodes: int,
                      dim: int,
                      random_state: int = 42,
                      n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute typed view embeddings.
    
    Returns:
        embeddings: node embeddings from typed view
        svd_components: SVD components for reconstruction
    """
    type_matrix = build_type_matrix(node_types, type_to_idx, num_nodes)
    ppmi = compute_ppmi_types(type_matrix)
    embeddings, svd_components = reduce_dimensions_ppmi(ppmi, dim, random_state, n_jobs)
    return embeddings, svd_components


def reduce_dimensions_ppmi(ppmi: sparse.csr_matrix, dim: int, random_state: int = 42,
                           n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce PPMI matrix dimensionality using Truncated SVD.
    
    Args:
        ppmi: PPMI matrix
        dim: target dimensionality
        random_state: random seed
    
    Returns:
        Reduced embeddings matrix and SVD components
    """
    num_features = ppmi.shape[1]
    actual_dim = min(dim, num_features)
    
    # TruncatedSVD doesn't support n_jobs, but sklearn uses threading internally
    svd = TruncatedSVD(n_components=actual_dim, random_state=random_state, n_iter=10)
    embeddings = svd.fit_transform(ppmi)
    
    # Pad embeddings if needed to match requested dimension
    if actual_dim < dim:
        padding = np.zeros((embeddings.shape[0], dim - actual_dim))
        embeddings = np.hstack([embeddings, padding])
        # Pad components similarly
        component_padding = np.zeros((dim - actual_dim, svd.components_.shape[1]))
        components = np.vstack([svd.components_, component_padding])
    else:
        components = svd.components_
    
    return embeddings, components

