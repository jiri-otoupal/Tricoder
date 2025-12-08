"""Graph view: adjacency matrix, PPMI, and SVD."""
import os
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# Set threading appropriately for multiprocessing
# Each process should use 1 thread to avoid oversubscription when using multiprocessing
# numpy/scipy operations release GIL so threading can help, but with multiprocessing
# we want to avoid thread contention
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def expand_call_graph(edges: List[Tuple[int, int, str, float]], num_nodes: int,
                      max_depth: int = 3) -> List[Tuple[int, int, str, float]]:
    """
    Expand call graph by propagating call edges to depth 2-3.
    
    Args:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of nodes
        max_depth: maximum depth for propagation (2 or 3)
    
    Returns:
        Expanded list of edges including propagated calls
    """
    # Build call graph (only "calls" relations)
    call_graph = defaultdict(list)
    call_edges = []
    other_edges = []

    for src_idx, dst_idx, rel, weight in edges:
        if rel == "calls":
            call_graph[src_idx].append((dst_idx, weight))
            call_edges.append((src_idx, dst_idx, rel, weight))
        else:
            other_edges.append((src_idx, dst_idx, rel, weight))

    # BFS to find transitive calls
    expanded_call_set = set()
    expanded_call_edges = list(call_edges)

    for start_node in call_graph:
        # BFS with depth control
        queue = deque([(start_node, 0, 1.0)])  # (node, depth, cumulative_weight)
        visited = {start_node}

        while queue:
            curr_node, depth, cum_weight = queue.popleft()

            if depth >= max_depth:
                continue

            for next_node, edge_weight in call_graph.get(curr_node, []):
                if next_node == start_node:
                    continue  # Avoid self-loops

                # Compute propagated weight
                propagated_weight = cum_weight * edge_weight

                if depth == 1:
                    propagated_weight *= 0.5  # Depth 2: weight *= 0.5
                elif depth == 2:
                    propagated_weight *= 0.25  # Depth 3: weight *= 0.25

                # Add edge if not already present (avoid cycles)
                edge_key = (start_node, next_node)
                if edge_key not in expanded_call_set and propagated_weight > 1e-6:
                    expanded_call_set.add(edge_key)
                    expanded_call_edges.append((start_node, next_node, "calls", propagated_weight))

                # Continue BFS if not visited at this depth
                if next_node not in visited and depth < max_depth - 1:
                    visited.add(next_node)
                    queue.append((next_node, depth + 1, propagated_weight))

    # Combine all edges
    return other_edges + expanded_call_edges


def add_subtoken_edges(edges: List[Tuple[int, int, str, float]],
                       node_to_idx: Dict[str, int],
                       node_subtokens: Dict[str, List[str]],
                       num_nodes: int) -> Tuple[List[Tuple[int, int, str, float]], int, Dict[str, int]]:
    """
    Add subtoken nodes and edges to the graph.
    
    Args:
        edges: existing edges
        node_to_idx: mapping from node_id to index
        node_subtokens: mapping from node_id to list of normalized subtokens
        num_nodes: current number of nodes
    
    Returns:
        Tuple of (expanded_edges, new_num_nodes, subtoken_to_idx)
    """
    # Create subtoken nodes
    subtoken_to_idx = {}
    new_edges = list(edges)
    current_num_nodes = num_nodes

    # First pass: create all subtoken nodes
    for node_id, subtokens in node_subtokens.items():
        if node_id not in node_to_idx:
            continue

        node_idx = node_to_idx[node_id]

        for subtoken in subtokens:
            if subtoken not in subtoken_to_idx:
                subtoken_to_idx[subtoken] = current_num_nodes
                current_num_nodes += 1

    # Second pass: add edges
    for node_id, subtokens in node_subtokens.items():
        if node_id not in node_to_idx:
            continue

        node_idx = node_to_idx[node_id]

        # Add edges: symbol ↔ subtoken (weight = 1.0)
        for subtoken in subtokens:
            subtoken_idx = subtoken_to_idx[subtoken]
            new_edges.append((node_idx, subtoken_idx, "has_subtoken", 1.0))
            new_edges.append((subtoken_idx, node_idx, "subtoken_of", 1.0))

        # Add edges between subtokens from same symbol (weight = 0.25)
        for i, subtoken1 in enumerate(subtokens):
            for subtoken2 in subtokens[i + 1:]:
                idx1 = subtoken_to_idx[subtoken1]
                idx2 = subtoken_to_idx[subtoken2]
                new_edges.append((idx1, idx2, "co_subtoken", 0.25))
                new_edges.append((idx2, idx1, "co_subtoken", 0.25))

    return new_edges, current_num_nodes, subtoken_to_idx


def add_file_hierarchy_edges(edges: List[Tuple[int, int, str, float]],
                             node_to_idx: Dict[str, int],
                             node_file_info: Dict[str, Tuple[str, str, str]],
                             idx_to_node: Dict[int, str]) -> List[Tuple[int, int, str, float]]:
    """
    Add file hierarchy edges based on file/directory relationships.
    
    Args:
        edges: existing edges
        node_to_idx: mapping from node_id to index
        node_file_info: mapping from node_id to (file_name, directory_path, top_level_package)
        idx_to_node: reverse mapping from index to node_id
    
    Returns:
        Expanded list of edges
    """
    new_edges = list(edges)

    # Group nodes by file, directory, and package
    nodes_by_file = defaultdict(list)
    nodes_by_directory = defaultdict(list)
    nodes_by_package = defaultdict(list)

    for node_id, (file_name, directory_path, top_level_package) in node_file_info.items():
        if node_id not in node_to_idx:
            continue

        node_idx = node_to_idx[node_id]

        if file_name:
            nodes_by_file[file_name].append(node_idx)
        if directory_path:
            nodes_by_directory[directory_path].append(node_idx)
        if top_level_package:
            nodes_by_package[top_level_package].append(node_idx)

    # Add edges for same file (weight += 3.0)
    for file_nodes in nodes_by_file.values():
        for i, node1 in enumerate(file_nodes):
            for node2 in file_nodes[i + 1:]:
                # Find existing edge weight or use 0
                existing_weight = 0.0
                for src, dst, rel, w in edges:
                    if (src == node1 and dst == node2) or (src == node2 and dst == node1):
                        existing_weight = max(existing_weight, w)
                        break

                new_weight = existing_weight + 3.0
                new_edges.append((node1, node2, "same_file", new_weight))
                new_edges.append((node2, node1, "same_file", new_weight))

    # Add edges for same directory (weight += 2.0)
    for dir_nodes in nodes_by_directory.values():
        for i, node1 in enumerate(dir_nodes):
            for node2 in dir_nodes[i + 1:]:
                existing_weight = 0.0
                for src, dst, rel, w in edges:
                    if (src == node1 and dst == node2) or (src == node2 and dst == node1):
                        existing_weight = max(existing_weight, w)
                        break

                new_weight = existing_weight + 2.0
                new_edges.append((node1, node2, "same_directory", new_weight))
                new_edges.append((node2, node1, "same_directory", new_weight))

    # Add edges for same package (weight += 1.0)
    for pkg_nodes in nodes_by_package.values():
        for i, node1 in enumerate(pkg_nodes):
            for node2 in pkg_nodes[i + 1:]:
                existing_weight = 0.0
                for src, dst, rel, w in edges:
                    if (src == node1 and dst == node2) or (src == node2 and dst == node1):
                        existing_weight = max(existing_weight, w)
                        break

                new_weight = existing_weight + 1.0
                new_edges.append((node1, node2, "same_package", new_weight))
                new_edges.append((node2, node1, "same_package", new_weight))

    return new_edges


def _process_file_context(args):
    """Process a single file for context window edges."""
    file_path, node_lines, window_size, edge_weights = args
    new_edges = []

    # Sort by line number
    node_lines.sort(key=lambda x: x[1])

    for i, (node1_idx, line1) in enumerate(node_lines):
        for j in range(i + 1, len(node_lines)):
            node2_idx, line2 = node_lines[j]

            # Check if within window
            if abs(line2 - line1) <= window_size:
                # Find existing edge weight or use 0
                key = (min(node1_idx, node2_idx), max(node1_idx, node2_idx))
                existing_weight = edge_weights.get(key, 0.0)

                new_weight = existing_weight + 1.0
                new_edges.append((node1_idx, node2_idx, "context_window", new_weight))
                new_edges.append((node2_idx, node1_idx, "context_window", new_weight))
            else:
                # Lines are sorted, so we can break early
                break

    return new_edges


def add_context_window_edges(edges: List[Tuple[int, int, str, float]],
                             node_metadata: List[Dict],
                             window_size: int = 5,
                             n_jobs: int = -1) -> List[Tuple[int, int, str, float]]:
    """
    Add edges for symbols appearing within ±W lines in the same file.
    
    Args:
        edges: existing edges
        node_metadata: list of node metadata dictionaries
        window_size: context window size (default 5)
        n_jobs: number of parallel jobs (-1 for all cores - 1)
    
    Returns:
        Expanded list of edges
    """
    new_edges = list(edges)

    # Group nodes by file and line number
    nodes_by_file = defaultdict(list)  # file_path -> [(node_idx, lineno), ...]

    for idx, node_meta in enumerate(node_metadata):
        meta = node_meta.get('meta', {})
        if isinstance(meta, dict):
            file_path = meta.get('file', '')
            lineno = meta.get('lineno', -1)
            if file_path and lineno >= 0:
                nodes_by_file[file_path].append((idx, lineno))

    # Build edge weight map for efficient lookup
    edge_weights = defaultdict(float)
    for edge in edges:
        if len(edge) == 4:
            src, dst, rel, w = edge
            key = (min(src, dst), max(src, dst))
            edge_weights[key] = max(edge_weights[key], w)

    # Parallelize file processing
    if n_jobs == -1:
        n_jobs = _get_num_workers()

    if len(nodes_by_file) > 10 and n_jobs > 1:
        # Parallel processing for multiple files
        args_list = [(file_path, node_lines, window_size, edge_weights)
                     for file_path, node_lines in nodes_by_file.items()]
        chunksize = max(1, len(args_list) // n_jobs)

        with Pool(processes=n_jobs) as pool:
            results = pool.map(_process_file_context, args_list, chunksize=chunksize)

        # Flatten results
        for file_edges in results:
            new_edges.extend(file_edges)
    else:
        # Sequential processing for small cases
        for file_path, node_lines in nodes_by_file.items():
            # Sort by line number
            node_lines.sort(key=lambda x: x[1])

            for i, (node1_idx, line1) in enumerate(node_lines):
                for j in range(i + 1, len(node_lines)):
                    node2_idx, line2 = node_lines[j]

                    # Check if within window
                    if abs(line2 - line1) <= window_size:
                        # Find existing edge weight or use 0
                        key = (min(node1_idx, node2_idx), max(node1_idx, node2_idx))
                        existing_weight = edge_weights.get(key, 0.0)

                        new_weight = existing_weight + 1.0
                        new_edges.append((node1_idx, node2_idx, "context_window", new_weight))
                        new_edges.append((node2_idx, node1_idx, "context_window", new_weight))
                    else:
                        # Lines are sorted, so we can break early
                        break

    return new_edges


def _process_edge_chunk(args):
    """Process a chunk of edges for parallel aggregation."""
    edge_chunk, start_idx = args
    edge_weights = {}
    rows = []
    cols = []
    data = []

    for src_idx, dst_idx, rel, weight in edge_chunk:
        key = (src_idx, dst_idx)
        if key not in edge_weights:
            edge_weights[key] = weight
        else:
            edge_weights[key] = max(edge_weights[key], weight)

    for (src_idx, dst_idx), weight in edge_weights.items():
        rows.append(src_idx)
        cols.append(dst_idx)
        data.append(weight)

    return rows, cols, data


def build_adjacency_matrix(edges: List[Tuple[int, int, str, float]], num_nodes: int,
                           n_jobs: int = -1) -> sparse.csr_matrix:
    """
    Build weighted adjacency matrix from edges with optional parallelization.
    
    Args:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of nodes
        n_jobs: number of parallel jobs (-1 for all cores - 1)
    
    Returns:
        Sparse CSR adjacency matrix
    """
    if n_jobs == -1:
        n_jobs = _get_num_workers()

    # For small edge lists, use sequential processing
    if len(edges) < 10000 or n_jobs == 1:
        rows = []
        cols = []
        data = []

        # Aggregate weights for duplicate edges (same src, dst pair)
        edge_weights = defaultdict(float)
        for src_idx, dst_idx, rel, weight in edges:
            key = (src_idx, dst_idx)
            edge_weights[key] = max(edge_weights[key], weight)

        for (src_idx, dst_idx), weight in edge_weights.items():
            rows.append(src_idx)
            cols.append(dst_idx)
            data.append(weight)
    else:
        # Parallel processing for large edge lists
        chunk_size = max(1, len(edges) // n_jobs)
        chunks = [edges[i:i + chunk_size] for i in range(0, len(edges), chunk_size)]
        args_list = [(chunk, i) for i, chunk in enumerate(chunks)]

        with Pool(processes=n_jobs) as pool:
            results = pool.map(_process_edge_chunk, args_list)

        # Merge results
        edge_weights = defaultdict(float)
        for chunk_rows, chunk_cols, chunk_data in results:
            for i, (src_idx, dst_idx) in enumerate(zip(chunk_rows, chunk_cols)):
                key = (src_idx, dst_idx)
                edge_weights[key] = max(edge_weights[key], chunk_data[i])

        rows = []
        cols = []
        data = []
        for (src_idx, dst_idx), weight in edge_weights.items():
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
                           n_jobs: int = -1, gpu_accelerator=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce PPMI matrix dimensionality using Truncated SVD (GPU-accelerated if available).
    
    Args:
        ppmi: PPMI matrix
        dim: target dimensionality
        random_state: random seed
        n_jobs: number of parallel jobs (not used for SVD, but kept for API consistency)
        gpu_accelerator: Optional GPUAccelerator instance for GPU acceleration
    
    Returns:
        Reduced embeddings matrix and SVD components
    """
    num_features = ppmi.shape[1]
    actual_dim = min(dim, num_features)

    # Try GPU acceleration if available
    if gpu_accelerator and gpu_accelerator.use_gpu:
        try:
            # Convert sparse matrix to dense for GPU SVD (CuPy doesn't support sparse SVD well)
            # Only do this if matrix is reasonably sized
            if ppmi.shape[0] * ppmi.shape[1] < 50_000_000:  # ~50M elements threshold
                ppmi_dense = ppmi.toarray()
                U, S, Vt = gpu_accelerator.svd(ppmi_dense, actual_dim, random_state)
                
                # Transform: U @ diag(S)
                embeddings = U @ np.diag(S)
                components = Vt
            else:
                # Too large, fall back to CPU sparse SVD
                raise ValueError("Matrix too large for GPU dense SVD")
        except Exception as e:
            # Fall back to CPU
            svd = TruncatedSVD(n_components=actual_dim, random_state=random_state, n_iter=5)
            embeddings = svd.fit_transform(ppmi)
            components = svd.components_
    else:
        # CPU path
        svd = TruncatedSVD(n_components=actual_dim, random_state=random_state, n_iter=5)
        embeddings = svd.fit_transform(ppmi)
        components = svd.components_

    # Pad embeddings if needed to match requested dimension
    if actual_dim < dim:
        padding = np.zeros((embeddings.shape[0], dim - actual_dim))
        embeddings = np.hstack([embeddings, padding])
        # Pad components similarly
        component_padding = np.zeros((dim - actual_dim, components.shape[1]))
        components = np.vstack([components, component_padding])

    return embeddings, components


def compute_graph_view(edges: List[Tuple[int, int, str, float]], num_nodes: int,
                       dim: int, random_state: int = 42, n_jobs: int = -1,
                       node_to_idx: Dict[str, int] = None,
                       node_subtokens: Dict[str, List[str]] = None,
                       node_file_info: Dict[str, Tuple[str, str, str]] = None,
                       node_metadata: List[Dict] = None,
                       idx_to_node: Dict[int, str] = None,
                       expand_calls: bool = True,
                       add_subtokens: bool = True,
                       add_hierarchy: bool = True,
                       add_context: bool = True,
                       context_window: int = 5,
                       max_depth: int = 3,
                       gpu_accelerator=None) -> Tuple[
    np.ndarray, np.ndarray, int, Dict[str, int], List[Tuple[int, int, str, float]]]:
    """
    Compute graph view embeddings with all enhancements.
    
    Args:
        max_depth: maximum depth for call graph expansion (default 3, use 2 for faster)
    
    Returns:
        embeddings: node embeddings from graph view
        svd_components: SVD components for reconstruction
        final_num_nodes: final number of nodes (after adding subtokens)
        subtoken_to_idx: mapping from subtoken to index (if subtokens added)
        expanded_edges: expanded edge list including all enhancements
    """
    expanded_edges = list(edges)
    current_num_nodes = num_nodes
    subtoken_to_idx = {}

    # Step 1: Expand call graph (depth 2-3)
    if expand_calls:
        expanded_edges = expand_call_graph(expanded_edges, current_num_nodes, max_depth=max_depth)

    # Step 2: Add context window co-occurrence (must be before PPMI)
    if add_context and node_metadata is not None:
        expanded_edges = add_context_window_edges(expanded_edges, node_metadata, window_size=context_window,
                                                  n_jobs=n_jobs)

    # Step 3: Add subtoken nodes and edges
    if add_subtokens and node_to_idx is not None and node_subtokens is not None:
        expanded_edges, current_num_nodes, subtoken_to_idx = add_subtoken_edges(
            expanded_edges, node_to_idx, node_subtokens, current_num_nodes
        )

    # Step 4: Add file hierarchy edges
    if add_hierarchy and node_to_idx is not None and node_file_info is not None and idx_to_node is not None:
        expanded_edges = add_file_hierarchy_edges(
            expanded_edges, node_to_idx, node_file_info, idx_to_node
        )

    # Step 5: Build adjacency matrix and compute PPMI (with parallelization)
    adj = build_adjacency_matrix(expanded_edges, current_num_nodes, n_jobs=n_jobs)
    ppmi = compute_ppmi(adj)
    embeddings, svd_components = reduce_dimensions_ppmi(ppmi, dim, random_state, n_jobs, gpu_accelerator)

    return embeddings, svd_components, current_num_nodes, subtoken_to_idx, expanded_edges
