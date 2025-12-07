"""Typed view: symbol × type-token matrix, PPMI, and SVD."""
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Tuple, List, Set
from multiprocessing import cpu_count
import re


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def parse_composite_type(type_token: str) -> Tuple[List[str], List[str]]:
    """
    Parse composite type into constructor tokens and primitive tokens.
    
    Examples:
        List[int] -> (['List'], ['int'])
        Dict[str, int] -> (['Dict'], ['str', 'int'])
        Optional[T] -> (['Optional'], ['T'])
        List[Dict[str, int]] -> (['List', 'Dict'], ['str', 'int'])
    
    Args:
        type_token: type token string
    
    Returns:
        Tuple of (constructor_tokens, primitive_tokens)
    """
    constructors = []
    primitives = []
    
    # Extract generic type constructors (List, Dict, Optional, Set, Tuple, etc.)
    # Pattern: ConstructorName[content]
    generic_pattern = r'([A-Z][a-zA-Z0-9_]*)\s*\[([^\]]+)\]'
    
    # Find all generic types
    matches = list(re.finditer(generic_pattern, type_token))
    
    if matches:
        # Extract constructors
        for match in matches:
            constructor = match.group(1)
            constructors.append(constructor)
            
            # Recursively parse inner content
            inner_content = match.group(2)
            inner_constructors, inner_primitives = parse_composite_type(inner_content)
            constructors.extend(inner_constructors)
            primitives.extend(inner_primitives)
        
        # Extract remaining primitives (not in generic brackets)
        remaining = type_token
        for match in reversed(matches):
            remaining = remaining[:match.start()] + remaining[match.end():]
        
        # Split by comma and extract primitives
        for part in remaining.split(','):
            part = part.strip()
            if part and not re.match(r'^[A-Z][a-zA-Z0-9_]*\s*\[', part):
                # Check if it's a primitive (lowercase or single letter)
                if part[0].islower() or (len(part) == 1 and part.isalpha()):
                    primitives.append(part)
    else:
        # No generic types, check if it's a primitive or simple type
        parts = [p.strip() for p in type_token.split(',')]
        for part in parts:
            if part:
                # If starts with uppercase and not a known primitive, treat as constructor
                if part[0].isupper() and part not in ['int', 'str', 'float', 'bool', 'None']:
                    constructors.append(part)
                else:
                    primitives.append(part)
    
    return constructors, primitives


def expand_type_semantics(node_types: Dict[int, Dict[str, int]],
                          type_to_idx: Dict[str, int],
                          num_nodes: int) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int]]:
    """
    Expand type tokens into constructors and primitives.
    
    Args:
        node_types: mapping from node_idx to {type_token: count}
        type_to_idx: mapping from type token to index
        num_nodes: number of nodes
    
    Returns:
        Tuple of (expanded_node_types, expanded_type_to_idx)
    """
    expanded_node_types = {}
    expanded_type_to_idx = dict(type_to_idx)
    
    # Start with existing type indices
    current_idx = len(expanded_type_to_idx)
    
    # Process each node's types
    for node_idx, types_dict in node_types.items():
        expanded_types = dict(types_dict)  # Start with original types
        
        for type_token, count in types_dict.items():
            # Parse composite type
            constructors, primitives = parse_composite_type(type_token)
            
            # Add constructor tokens
            for constructor in constructors:
                if constructor not in expanded_type_to_idx:
                    expanded_type_to_idx[constructor] = current_idx
                    current_idx += 1
                
                constructor_idx = expanded_type_to_idx[constructor]
                expanded_types[constructor] = expanded_types.get(constructor, 0) + int(count * 0.5)
            
            # Add primitive tokens
            for primitive in primitives:
                if primitive not in expanded_type_to_idx:
                    expanded_type_to_idx[primitive] = current_idx
                    current_idx += 1
                
                primitive_idx = expanded_type_to_idx[primitive]
                expanded_types[primitive] = expanded_types.get(primitive, 0) + int(count * 0.25)
        
        expanded_node_types[node_idx] = expanded_types
    
    return expanded_node_types, expanded_type_to_idx


def build_type_matrix(node_types: Dict[int, Dict[str, int]], 
                     type_to_idx: Dict[str, int],
                     num_nodes: int,
                     expand_types: bool = True) -> Tuple[sparse.csr_matrix, Dict[str, int]]:
    """
    Build sparse symbol × type-token matrix with optional type expansion.
    
    Args:
        node_types: mapping from node_idx to {type_token: count}
        type_to_idx: mapping from type token to index
        num_nodes: number of nodes (may include subtokens, but node_types only has original nodes)
        expand_types: whether to expand composite types
    
    Returns:
        Tuple of (type_matrix, final_type_to_idx)
    """
    final_node_types = node_types
    final_type_to_idx = type_to_idx
    
    if expand_types:
        # Use original num_nodes (before subtokens) for expansion
        original_num_nodes = max(node_types.keys()) + 1 if node_types else 0
        final_node_types, final_type_to_idx = expand_type_semantics(
            node_types, type_to_idx, original_num_nodes
        )
    
    num_types = len(final_type_to_idx)
    rows = []
    cols = []
    data = []
    
    for node_idx, types_dict in final_node_types.items():
        for type_token, count in types_dict.items():
            if type_token in final_type_to_idx:
                type_idx = final_type_to_idx[type_token]
                rows.append(node_idx)
                cols.append(type_idx)
                data.append(float(count))
    
    # Pad matrix to num_nodes rows (subtokens will have zero rows)
    type_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_types))
    return type_matrix, final_type_to_idx


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
    total = float(cooc.sum())
    
    # Ensure row_sums and col_sums match matrix dimensions exactly
    if len(row_sums) != cooc.shape[0]:
        # Create full-size array and fill with row_sums
        full_row_sums = np.zeros(cooc.shape[0])
        full_row_sums[:len(row_sums)] = row_sums
        row_sums = full_row_sums
    if len(col_sums) != cooc.shape[1]:
        # Create full-size array and fill with col_sums
        full_col_sums = np.zeros(cooc.shape[1])
        full_col_sums[:len(col_sums)] = col_sums
        col_sums = full_col_sums
    
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    col_sums = np.maximum(col_sums, 1e-10)
    total = max(total, 1e-10)
    
    # Compute PMI
    rows, cols = cooc.nonzero()
    values = cooc.data
    
    # Ensure indices are within bounds
    rows = np.clip(rows, 0, len(row_sums) - 1).astype(int)
    cols = np.clip(cols, 0, len(col_sums) - 1).astype(int)
    
    # Ensure all arrays have the same length
    min_len = min(len(values), len(rows), len(cols))
    if min_len < len(values):
        values = values[:min_len]
        rows = rows[:min_len]
        cols = cols[:min_len]
    
    p_ij = values / total
    p_i = row_sums[rows] / total
    p_j = col_sums[cols] / total
    
    # Ensure all have same shape
    assert len(p_ij) == len(p_i) == len(p_j), f"Shape mismatch: p_ij={len(p_ij)}, p_i={len(p_i)}, p_j={len(p_j)}"
    
    pmi = np.log(p_ij / (p_i * p_j + 1e-10) + 1e-10)
    ppmi = np.maximum(pmi, 0.0)
    
    ppmi_matrix = sparse.csr_matrix((ppmi, (rows, cols)), shape=cooc.shape)
    return ppmi_matrix


def compute_typed_view(node_types: Dict[int, Dict[str, int]],
                      type_to_idx: Dict[str, int],
                      num_nodes: int,
                      dim: int,
                      random_state: int = 42,
                      n_jobs: int = -1,
                      expand_types: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Compute typed view embeddings with optional type expansion.
    
    Returns:
        embeddings: node embeddings from typed view
        svd_components: SVD components for reconstruction
        final_type_to_idx: expanded type token mapping
    """
    type_matrix, final_type_to_idx = build_type_matrix(
        node_types, type_to_idx, num_nodes, expand_types=expand_types
    )
    ppmi = compute_ppmi_types(type_matrix)
    embeddings, svd_components = reduce_dimensions_ppmi(ppmi, dim, random_state, n_jobs)
    return embeddings, svd_components, final_type_to_idx


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

