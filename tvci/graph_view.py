"""Graph view: adjacency matrix, PPMI, and SVD."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, List
from multiprocessing import cpu_count


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def build_adjacency_matrix(edges: List[Tuple[int, int, str, float]], num_nodes: int) -> sparse.csr_matrix:
    """
    Build weighted adjacency matrix from edges.
    
    Args:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of nodes
    
    Returns:
        Sparse CSR adjacency matrix
    """
    rows = []
    cols = []
    data = []
    
    for src_idx, dst_idx, rel, weight in edges:
        rows.append(src_idx)
        cols.append(dst_idx)
        data.append(weight)
    
    # Create symmetric matrix (undirected graph)
    rows_sym = rows + cols
    cols_sym = cols + rows
    data_sym = data + data
    
    adj = sparse.csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(num_nodes, num_nodes))
    return adj


def compute_ppmi(adj: sparse.csr_matrix, k: float = 1.0) -> sparse.csr_matrix:
    """
    Compute Positive Pointwise Mutual Information (PPMI) matrix.
    
    Args:
        adj: adjacency matrix
        k: shift parameter (typically 1.0)
    
    Returns:
        PPMI matrix (sparse)
    """
    # Convert to cooccurrence matrix (symmetric)
    cooc = adj + adj.T
    cooc.data = np.maximum(cooc.data, 0)  # Ensure non-negative
    
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
    
    # PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    p_ij = values / total
    p_i = row_sums[rows] / total
    p_j = col_sums[cols] / total
    
    pmi = np.log(p_ij / (p_i * p_j + 1e-10) + 1e-10)
    ppmi = np.maximum(pmi, 0.0)  # Positive PMI
    
    # Create PPMI matrix
    ppmi_matrix = sparse.csr_matrix((ppmi, (rows, cols)), shape=cooc.shape)
    
    return ppmi_matrix


def reduce_dimensions_ppmi(ppmi: sparse.csr_matrix, dim: int, random_state: int = 42,
                           n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce PPMI matrix dimensionality using Truncated SVD.
    
    Args:
        ppmi: PPMI matrix
        dim: target dimensionality
        random_state: random seed
        n_jobs: number of parallel jobs (-1 for all cores - 1)
    
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


def compute_graph_view(edges: List[Tuple[int, int, str, float]], num_nodes: int, 
                      dim: int, random_state: int = 42, n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute graph view embeddings.
    
    Returns:
        embeddings: node embeddings from graph view
        svd_components: SVD components for reconstruction
    """
    adj = build_adjacency_matrix(edges, num_nodes)
    ppmi = compute_ppmi(adj)
    embeddings, svd_components = reduce_dimensions_ppmi(ppmi, dim, random_state, n_jobs)
    return embeddings, svd_components

