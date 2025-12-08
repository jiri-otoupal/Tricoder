"""Graph view: adjacency matrix, PPMI, and SVD."""
import os
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .graph_config import (
    CALL_GRAPH_MAX_NODES_SMALL, CALL_GRAPH_MAX_NODES_MEDIUM, CALL_GRAPH_MAX_NODES_LARGE,
    CALL_GRAPH_EDGE_MULTIPLIER_SMALL, CALL_GRAPH_EDGE_MULTIPLIER_MEDIUM, CALL_GRAPH_EDGE_MULTIPLIER_LARGE,
    CALL_GRAPH_MAX_NEIGHBORS_SMALL, CALL_GRAPH_MAX_NEIGHBORS_MEDIUM, CALL_GRAPH_MAX_NEIGHBORS_LARGE,
    CALL_GRAPH_MAX_QUEUE_SMALL, CALL_GRAPH_MAX_QUEUE_MEDIUM, CALL_GRAPH_MAX_QUEUE_LARGE,
    CALL_GRAPH_EDGE_SEARCH_LIMIT,
    CONTEXT_WINDOW_MAX_NODES_PER_FILE, CONTEXT_WINDOW_MAX_PAIRS_PER_FILE,
    HIERARCHY_MAX_NODES_PER_GROUP, HIERARCHY_MAX_PAIRS_PER_GROUP,
    SIZE_THRESHOLD_SMALL, SIZE_THRESHOLD_MEDIUM
)

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

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
                      max_depth: int = 3, progress_callback=None) -> List[Tuple[int, int, str, float]]:
    """
    Expand call graph by propagating call edges to depth 2-3.
    Optimized with early termination and efficient data structures.
    
    Args:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of nodes
        max_depth: maximum depth for propagation (2 or 3)
    
    Returns:
        Expanded list of edges including propagated calls
    """
    # Build call graph (only "calls" relations) - use list for faster iteration
    call_graph = defaultdict(list)
    call_edges = []
    other_edges = []

    for src_idx, dst_idx, rel, weight in edges:
        if rel == "calls":
            call_graph[src_idx].append((dst_idx, weight))
            call_edges.append((src_idx, dst_idx, rel, weight))
        else:
            other_edges.append((src_idx, dst_idx, rel, weight))

    if not call_graph:
        # Update progress to complete if no call graph
        if progress_callback:
            progress_callback(1, 1)
        return edges

    # Pre-allocate expanded edges list (estimate size)
    expanded_call_edges = []
    expanded_call_edges.extend(call_edges)
    expanded_call_set = {(src, dst) for src, dst, _, _ in call_edges}  # Pre-populate with existing edges

    # Optimize: limit processing for large graphs
    # For very large graphs, only expand from nodes with most connections (top 50% by degree)
    nodes_to_process = list(call_graph.keys())
    
    # Limit expansion if graph is very large (prevents exponential explosion)
    if num_nodes < SIZE_THRESHOLD_SMALL:
        max_nodes_to_expand = min(len(nodes_to_process), CALL_GRAPH_MAX_NODES_SMALL)
    elif num_nodes < SIZE_THRESHOLD_MEDIUM:
        max_nodes_to_expand = min(len(nodes_to_process), max(CALL_GRAPH_MAX_NODES_MEDIUM, num_nodes // 20))
    else:
        max_nodes_to_expand = min(len(nodes_to_process), max(CALL_GRAPH_MAX_NODES_LARGE, num_nodes // 10))
    
    if len(nodes_to_process) > max_nodes_to_expand:
        # Sort by number of outgoing calls and take top nodes
        node_degrees = [(node, len(call_graph[node])) for node in nodes_to_process]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        nodes_to_process = [node for node, _ in node_degrees[:max_nodes_to_expand]]
    
    # Ensure we have nodes to process (should always have at least 1 if call_graph exists)
    if len(nodes_to_process) == 0:
        # This shouldn't happen, but handle it gracefully
        if progress_callback:
            progress_callback(1, 1)
        return other_edges + expanded_call_edges
    
    # Use depth multipliers for faster computation
    depth_multipliers = np.array([1.0, 0.5, 0.25], dtype=np.float64)  # depth 0, 1, 2
    
    # Limit edge generation based on codebase size
    if num_nodes < SIZE_THRESHOLD_SMALL:
        max_new_edges = min(len(call_edges) * CALL_GRAPH_EDGE_MULTIPLIER_SMALL, 1000)
    elif num_nodes < SIZE_THRESHOLD_MEDIUM:
        max_new_edges = len(call_edges) * CALL_GRAPH_EDGE_MULTIPLIER_MEDIUM
    else:
        max_new_edges = len(call_edges) * CALL_GRAPH_EDGE_MULTIPLIER_LARGE
    edges_generated = 0
    
    # Convert call graph to CSR-like structure for Numba
    # Build arrays: neighbor_indices, neighbor_weights, neighbor_offsets, neighbor_counts
    max_node_idx = max(call_graph.keys()) if call_graph else 0
    neighbor_indices_list = []
    neighbor_weights_list = []
    neighbor_offsets = np.zeros(max_node_idx + 2, dtype=np.int64)
    neighbor_counts = np.zeros(max_node_idx + 1, dtype=np.int64)
    
    current_offset = 0
    for node_idx in range(max_node_idx + 1):
        neighbor_offsets[node_idx] = current_offset
        if node_idx in call_graph:
            neighbors = call_graph[node_idx]
            neighbor_counts[node_idx] = len(neighbors)
            for next_node, weight in neighbors:
                neighbor_indices_list.append(next_node)
                neighbor_weights_list.append(weight)
                current_offset += 1
        else:
            neighbor_counts[node_idx] = 0
    
    neighbor_offsets[max_node_idx + 1] = current_offset  # Sentinel
    
    if len(neighbor_indices_list) > 0:
        neighbor_indices = np.array(neighbor_indices_list, dtype=np.int64)
        neighbor_weights = np.array(neighbor_weights_list, dtype=np.float64)
    else:
        neighbor_indices = np.array([], dtype=np.int64)
        neighbor_weights = np.array([], dtype=np.float64)
    
    # Pre-build existing edges arrays for fast lookup
    existing_edges_srcs = np.zeros(len(expanded_call_set), dtype=np.int64)
    existing_edges_dsts = np.zeros(len(expanded_call_set), dtype=np.int64)
    existing_edges_idx = 0
    for src, dst in expanded_call_set:
        existing_edges_srcs[existing_edges_idx] = src
        existing_edges_dsts[existing_edges_idx] = dst
        existing_edges_idx += 1
    
    # Update progress at start (0/nodes)
    if progress_callback and len(nodes_to_process) > 0:
        progress_callback(0, len(nodes_to_process))
    
    # Use Numba-optimized BFS if available
    if NUMBA_AVAILABLE and len(nodes_to_process) > 0:
        try:
            # Set limits based on codebase size
            if num_nodes < SIZE_THRESHOLD_SMALL:
                max_neighbors_per_node = CALL_GRAPH_MAX_NEIGHBORS_SMALL
                max_queue_size = CALL_GRAPH_MAX_QUEUE_SMALL
            elif num_nodes < SIZE_THRESHOLD_MEDIUM:
                max_neighbors_per_node = CALL_GRAPH_MAX_NEIGHBORS_MEDIUM
                max_queue_size = CALL_GRAPH_MAX_QUEUE_MEDIUM
            else:
                max_neighbors_per_node = CALL_GRAPH_MAX_NEIGHBORS_LARGE
                max_queue_size = CALL_GRAPH_MAX_QUEUE_LARGE
            all_srcs = []
            all_dsts = []
            all_weights = []
            
            for idx, start_node in enumerate(nodes_to_process):
                if edges_generated >= max_new_edges:
                    # Update progress before breaking
                    if progress_callback:
                        progress_callback(len(nodes_to_process), len(nodes_to_process))
                    break
                
                # Update progress
                if progress_callback:
                    progress_callback(idx + 1, len(nodes_to_process))
                
                srcs, dsts, edge_weights_array, num_new = _expand_call_graph_bfs_numba(
                    start_node,
                    neighbor_indices,
                    neighbor_weights,
                    neighbor_offsets,
                    neighbor_counts,
                    max_depth,
                    depth_multipliers,
                    max_neighbors_per_node,
                    max_queue_size,
                    max_new_edges - edges_generated,
                    existing_edges_srcs,
                    existing_edges_dsts,
                    existing_edges_idx
                )
                
                if num_new > 0:
                    # Add new edges
                    for i in range(num_new):
                        src = int(srcs[i])
                        dst = int(dsts[i])
                        weight = float(edge_weights_array[i])
                        edge_key = (src, dst)
                        
                        if edge_key not in expanded_call_set:
                            expanded_call_set.add(edge_key)
                            expanded_call_edges.append((src, dst, "calls", weight))
                            edges_generated += 1
                            
                            # Update existing edges arrays for next BFS
                            if existing_edges_idx < len(existing_edges_srcs):
                                existing_edges_srcs[existing_edges_idx] = src
                                existing_edges_dsts[existing_edges_idx] = dst
                                existing_edges_idx += 1
        except Exception as e:
            # Fallback to Python if numba fails
            import warnings
            warnings.warn(f"Numba BFS failed: {e}. Falling back to Python.")
            # Python fallback (original code) - use same limits as Numba path
            for start_node in nodes_to_process:
                if edges_generated >= max_new_edges:
                    break
                    
                queue = [(start_node, 0, 1.0)]
                visited_at_depth = {start_node: 0}
                # Use same queue size limit as Numba path
                
                queue_idx = 0
                while queue_idx < len(queue) and queue_idx < max_queue_size:
                    curr_node, depth, cum_weight = queue[queue_idx]
                    queue_idx += 1

                    if depth >= max_depth:
                        continue

                    neighbors = call_graph.get(curr_node, [])
                    if not neighbors:
                        continue
                    
                    depth_mult = depth_multipliers[depth] if depth < len(depth_multipliers) else 0.0
                    # Use same neighbor limit as Numba path
                    neighbors_to_process = neighbors[:max_neighbors_per_node] if len(neighbors) > max_neighbors_per_node else neighbors
                    
                    for next_node, edge_weight in neighbors_to_process:
                        if next_node == start_node:
                            continue
                        
                        propagated_weight = cum_weight * edge_weight * depth_mult
                        if propagated_weight <= 1e-6:
                            continue
                        
                        edge_key = (start_node, next_node)
                        if edge_key not in expanded_call_set:
                            expanded_call_set.add(edge_key)
                            expanded_call_edges.append((start_node, next_node, "calls", propagated_weight))
                            edges_generated += 1
                            if edges_generated >= max_new_edges:
                                break
                        
                        prev_depth = visited_at_depth.get(next_node, max_depth + 1)
                        if depth + 1 < prev_depth and depth + 1 < max_depth and len(queue) < max_queue_size:
                            visited_at_depth[next_node] = depth + 1
                            queue.append((next_node, depth + 1, propagated_weight))
                    
                    if edges_generated >= max_new_edges:
                        break
    else:
        # Python fallback (original code) - set limits based on codebase size
        if num_nodes < SIZE_THRESHOLD_SMALL:
            max_neighbors_per_node = CALL_GRAPH_MAX_NEIGHBORS_SMALL
            max_queue_size = CALL_GRAPH_MAX_QUEUE_SMALL
        elif num_nodes < SIZE_THRESHOLD_MEDIUM:
            max_neighbors_per_node = CALL_GRAPH_MAX_NEIGHBORS_MEDIUM
            max_queue_size = CALL_GRAPH_MAX_QUEUE_MEDIUM
        else:
            max_neighbors_per_node = CALL_GRAPH_MAX_NEIGHBORS_LARGE
            max_queue_size = CALL_GRAPH_MAX_QUEUE_LARGE
        
        for idx, start_node in enumerate(nodes_to_process):
            # Update progress
            if progress_callback:
                progress_callback(idx + 1, len(nodes_to_process))
            if edges_generated >= max_new_edges:
                break
                
            queue = [(start_node, 0, 1.0)]
            visited_at_depth = {start_node: 0}
            
            queue_idx = 0
            while queue_idx < len(queue) and queue_idx < max_queue_size:
                curr_node, depth, cum_weight = queue[queue_idx]
                queue_idx += 1
                
                if depth >= max_depth:
                    continue
                
                neighbors = call_graph.get(curr_node, [])
                if not neighbors:
                    continue
                
                depth_mult = depth_multipliers[depth] if depth < len(depth_multipliers) else 0.0
                neighbors_to_process = neighbors[:max_neighbors_per_node] if len(neighbors) > max_neighbors_per_node else neighbors
                
                for next_node, edge_weight in neighbors_to_process:
                    if next_node == start_node:
                        continue
                    
                    propagated_weight = cum_weight * edge_weight * depth_mult
                    if propagated_weight <= 1e-6:
                        continue
                    
                    edge_key = (start_node, next_node)
                    if edge_key not in expanded_call_set:
                        expanded_call_set.add(edge_key)
                        expanded_call_edges.append((start_node, next_node, "calls", propagated_weight))
                        edges_generated += 1
                        if edges_generated >= max_new_edges:
                            break
                    
                    prev_depth = visited_at_depth.get(next_node, max_depth + 1)
                    if depth + 1 < prev_depth and depth + 1 < max_depth and len(queue) < max_queue_size:
                        visited_at_depth[next_node] = depth + 1
                        queue.append((next_node, depth + 1, propagated_weight))
                
                if edges_generated >= max_new_edges:
                    break
    
    # Update progress to complete at the end (ensure it's marked complete)
    if progress_callback and len(nodes_to_process) > 0:
        progress_callback(len(nodes_to_process), len(nodes_to_process))

    # Combine all edges
    return other_edges + expanded_call_edges


def add_subtoken_edges(edges: List[Tuple[int, int, str, float]],
                       node_to_idx: Dict[str, int],
                       node_subtokens: Dict[str, List[str]],
                       num_nodes: int,
                       progress_callback=None) -> Tuple[List[Tuple[int, int, str, float]], int, Dict[str, int]]:
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
    node_items = list(node_subtokens.items())
    total_nodes = len([nid for nid, _ in node_items if nid in node_to_idx])
    
    # Update progress at start
    if progress_callback:
        progress_callback(0, total_nodes)
    
    processed = 0
    for node_id, subtokens in node_items:
        if node_id not in node_to_idx:
            continue

        # Update progress
        processed += 1
        if progress_callback:
            progress_callback(processed, total_nodes)

        node_idx = node_to_idx[node_id]

        # Add edges: symbol ↔ subtoken (weight = 1.0)
        for subtoken in subtokens:
            subtoken_idx = subtoken_to_idx[subtoken]
            new_edges.append((node_idx, subtoken_idx, "has_subtoken", 1.0))
            new_edges.append((subtoken_idx, node_idx, "subtoken_of", 1.0))

        # Add edges between subtokens from same symbol (weight = 0.25)
        # Use Numba optimization if available and we have multiple subtokens
        if NUMBA_AVAILABLE and len(subtokens) > 2:
            try:
                # Convert subtoken strings to indices
                subtoken_indices_list = [subtoken_to_idx[st] for st in subtokens]
                subtoken_indices = np.array(subtoken_indices_list, dtype=np.int64)
                
                # Use Numba-optimized function
                srcs, dsts, weights = _add_co_subtoken_edges_numba(subtoken_indices, 0.25)
                
                # Add edges to result
                for i in range(len(srcs)):
                    new_edges.append((int(srcs[i]), int(dsts[i]), "co_subtoken", float(weights[i])))
            except Exception:
                # Fallback to Python if numba fails
                for i, subtoken1 in enumerate(subtokens):
                    for subtoken2 in subtokens[i + 1:]:
                        idx1 = subtoken_to_idx[subtoken1]
                        idx2 = subtoken_to_idx[subtoken2]
                        new_edges.append((idx1, idx2, "co_subtoken", 0.25))
                        new_edges.append((idx2, idx1, "co_subtoken", 0.25))
        else:
            # Python fallback (original code)
            for i, subtoken1 in enumerate(subtokens):
                for subtoken2 in subtokens[i + 1:]:
                    idx1 = subtoken_to_idx[subtoken1]
                    idx2 = subtoken_to_idx[subtoken2]
                    new_edges.append((idx1, idx2, "co_subtoken", 0.25))
                    new_edges.append((idx2, idx1, "co_subtoken", 0.25))

    return new_edges, current_num_nodes, subtoken_to_idx


@jit(nopython=True, cache=True)
def _generate_pairs_numba(node_indices: np.ndarray, max_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized function to generate all pairs from a list of node indices.
    Returns (src_indices, dst_indices) for unique pairs only.
    """
    n = len(node_indices)
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    
    # Calculate number of pairs: n*(n-1)/2
    num_pairs = min(n * (n - 1) // 2, max_pairs)
    srcs = np.zeros(num_pairs, dtype=np.int64)
    dsts = np.zeros(num_pairs, dtype=np.int64)
    
    pair_idx = 0
    for i in range(n):
        if pair_idx >= max_pairs:
            break
        node1 = node_indices[i]
        for j in range(i + 1, n):
            if pair_idx >= max_pairs:
                break
            srcs[pair_idx] = node1
            dsts[pair_idx] = node_indices[j]
            pair_idx += 1
    
    return srcs[:pair_idx], dsts[:pair_idx]


def add_file_hierarchy_edges(edges: List[Tuple[int, int, str, float]],
                             node_to_idx: Dict[str, int],
                             node_file_info: Dict[str, Tuple[str, str, str]],
                             idx_to_node: Dict[int, str],
                             progress_callback=None) -> List[Tuple[int, int, str, float]]:
    """
    Add file hierarchy edges based on file/directory relationships.
    Optimized with Numba and efficient edge weight lookup.
    """
    new_edges = list(edges)

    # Build edge weight lookup dictionary ONCE (O(M) instead of O(N²*M))
    edge_weights = defaultdict(float)
    for src, dst, rel, w in edges:
        key = (min(src, dst), max(src, dst))
        edge_weights[key] = max(edge_weights[key], w)

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

    # Limit nodes per group to prevent explosion
    max_nodes_per_group = HIERARCHY_MAX_NODES_PER_GROUP
    
    # Process file edges
    file_edges_batch = []
    for file_nodes in nodes_by_file.values():
        if len(file_nodes) < 2:
            continue
        # Limit nodes
        if len(file_nodes) > max_nodes_per_group:
            file_nodes = file_nodes[:max_nodes_per_group]
        
        # Use Numba to generate pairs
        node_array = np.array(file_nodes, dtype=np.int64)
        max_pairs = min(len(file_nodes) * (len(file_nodes) - 1) // 2, HIERARCHY_MAX_PAIRS_PER_GROUP)
        srcs, dsts = _generate_pairs_numba(node_array, max_pairs)
        
        # Batch create edges
        for i in range(len(srcs)):
            src, dst = int(srcs[i]), int(dsts[i])
            key = (src, dst)
            existing_weight = edge_weights.get(key, 0.0)
            new_weight = existing_weight + 3.0
            file_edges_batch.append((src, dst, "same_file", new_weight))
            file_edges_batch.append((dst, src, "same_file", new_weight))

    new_edges.extend(file_edges_batch)

    # Process directory edges
    dir_list = list(nodes_by_directory.items())
    total_groups = len(dir_list) + len(nodes_by_package)
    
    if progress_callback and total_groups > 0:
        progress_callback(0, total_groups)
    
    dir_edges_batch = []
    for dir_idx, (dir_path, dir_nodes) in enumerate(dir_list):
        if progress_callback:
            progress_callback(dir_idx + 1, total_groups)
        
        if len(dir_nodes) < 2:
            continue
        # Limit nodes
        if len(dir_nodes) > max_nodes_per_group:
            dir_nodes = dir_nodes[:max_nodes_per_group]
        
        # Use Numba to generate pairs
        node_array = np.array(dir_nodes, dtype=np.int64)
        max_pairs = min(len(dir_nodes) * (len(dir_nodes) - 1) // 2, 500)
        srcs, dsts = _generate_pairs_numba(node_array, max_pairs)
        
        # Batch create edges
        for i in range(len(srcs)):
            src, dst = int(srcs[i]), int(dsts[i])
            key = (src, dst)
            existing_weight = edge_weights.get(key, 0.0)
            new_weight = existing_weight + 2.0
            dir_edges_batch.append((src, dst, "same_directory", new_weight))
            dir_edges_batch.append((dst, src, "same_directory", new_weight))

    new_edges.extend(dir_edges_batch)

    # Process package edges
    pkg_list = list(nodes_by_package.items())
    pkg_edges_batch = []
    
    for pkg_idx, (pkg_name, pkg_nodes) in enumerate(pkg_list):
        if progress_callback:
            progress_callback(len(dir_list) + pkg_idx + 1, total_groups)
        
        if len(pkg_nodes) < 2:
            continue
        # Limit nodes
        if len(pkg_nodes) > max_nodes_per_group:
            pkg_nodes = pkg_nodes[:max_nodes_per_group]
        
        # Use Numba to generate pairs
        node_array = np.array(pkg_nodes, dtype=np.int64)
        max_pairs = min(len(pkg_nodes) * (len(pkg_nodes) - 1) // 2, 500)
        srcs, dsts = _generate_pairs_numba(node_array, max_pairs)
        
        # Batch create edges
        for i in range(len(srcs)):
            src, dst = int(srcs[i]), int(dsts[i])
            key = (src, dst)
            existing_weight = edge_weights.get(key, 0.0)
            new_weight = existing_weight + 1.0
            pkg_edges_batch.append((src, dst, "same_package", new_weight))
            pkg_edges_batch.append((dst, src, "same_package", new_weight))
    
    new_edges.extend(pkg_edges_batch)

    return new_edges


@jit(nopython=True, cache=True)
def _add_co_subtoken_edges_numba(
    subtoken_indices: np.ndarray,  # Array of subtoken indices for a symbol
    weight: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized function to add edges between subtokens from the same symbol.
    
    Args:
        subtoken_indices: array of subtoken node indices
        weight: edge weight (typically 0.25)
    
    Returns:
        Tuple of (src_indices, dst_indices, weights) arrays
    """
    n = len(subtoken_indices)
    if n < 2:
        # Need at least 2 subtokens to create edges
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    
    # Calculate number of edges: n*(n-1) (all pairs, both directions)
    num_edges = n * (n - 1)
    srcs = np.zeros(num_edges, dtype=np.int64)
    dsts = np.zeros(num_edges, dtype=np.int64)
    weights = np.full(num_edges, weight, dtype=np.float64)
    
    edge_idx = 0
    for i in range(n):
        subtoken1 = subtoken_indices[i]
        for j in range(i + 1, n):
            subtoken2 = subtoken_indices[j]
            # Add both directions
            srcs[edge_idx] = subtoken1
            dsts[edge_idx] = subtoken2
            edge_idx += 1
            
            srcs[edge_idx] = subtoken2
            dsts[edge_idx] = subtoken1
            edge_idx += 1
    
    return srcs[:edge_idx], dsts[:edge_idx], weights[:edge_idx]


@jit(nopython=True, cache=True)
def _expand_call_graph_bfs_numba(
    start_node: int,
    neighbor_indices: np.ndarray,  # Flattened array of neighbor node indices
    neighbor_weights: np.ndarray,  # Flattened array of neighbor edge weights
    neighbor_offsets: np.ndarray,  # CSR-like offsets: neighbor_offsets[i] is start index for node i
    neighbor_counts: np.ndarray,   # Number of neighbors per node
    max_depth: int,
    depth_multipliers: np.ndarray,
    max_neighbors_per_node: int,
    max_queue_size: int,
    max_new_edges: int,
    existing_edges_srcs: np.ndarray,  # Array of existing edge source nodes
    existing_edges_dsts: np.ndarray,  # Array of existing edge destination nodes
    existing_edges_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Numba-optimized BFS for call graph expansion from a single start node.
    
    Returns:
        Tuple of (src_indices, dst_indices, weights, num_edges_generated)
    """
    # Pre-allocate result arrays
    max_result_edges = max_new_edges
    srcs = np.zeros(max_result_edges, dtype=np.int64)
    dsts = np.zeros(max_result_edges, dtype=np.int64)
    weights = np.zeros(max_result_edges, dtype=np.float64)
    edge_count = 0
    
    # Queue: (node, depth, cumulative_weight)
    queue_nodes = np.zeros(max_queue_size, dtype=np.int64)
    queue_depths = np.zeros(max_queue_size, dtype=np.int64)
    queue_weights = np.zeros(max_queue_size, dtype=np.float64)
    
    # Visited at depth tracking (use -1 for unvisited)
    visited_depths = np.full(neighbor_offsets.shape[0], -1, dtype=np.int64)
    
    # Initialize queue
    queue_size = 1
    queue_nodes[0] = start_node
    queue_depths[0] = 0
    queue_weights[0] = 1.0
    visited_depths[start_node] = 0
    
    queue_idx = 0
    
    while queue_idx < queue_size and queue_idx < max_queue_size and edge_count < max_new_edges:
        curr_node = queue_nodes[queue_idx]
        depth = queue_depths[queue_idx]
        cum_weight = queue_weights[queue_idx]
        queue_idx += 1
        
        if depth >= max_depth:
            continue
        
        # Get neighbors for current node
        node_start_idx = neighbor_offsets[curr_node]
        node_neighbor_count = neighbor_counts[curr_node]
        
        if node_neighbor_count == 0:
            continue
        
        # Limit neighbors processed
        neighbors_to_check = min(node_neighbor_count, max_neighbors_per_node)
        
        # Apply depth multiplier
        depth_mult = depth_multipliers[depth] if depth < len(depth_multipliers) else 0.0
        
        for i in range(neighbors_to_check):
            neighbor_idx = node_start_idx + i
            next_node = neighbor_indices[neighbor_idx]
            edge_weight = neighbor_weights[neighbor_idx]
            
            if next_node == start_node:  # Skip self-loops
                continue
            
            # Compute propagated weight
            propagated_weight = cum_weight * edge_weight * depth_mult
            
            # Skip if weight too small
            if propagated_weight <= 1e-6:
                continue
            
            # Check if edge already exists (optimized: limit search to recent edges for performance)
            edge_exists = False
            # Only check last N edges if there are many (most duplicates are recent)
            search_limit = min(existing_edges_size, CALL_GRAPH_EDGE_SEARCH_LIMIT)
            start_idx = max(0, existing_edges_size - search_limit)
            for j in range(start_idx, existing_edges_size):
                if existing_edges_srcs[j] == start_node and existing_edges_dsts[j] == next_node:
                    edge_exists = True
                    break
            
            if not edge_exists and edge_count < max_result_edges:
                # Add new edge with computed weight
                srcs[edge_count] = start_node
                dsts[edge_count] = next_node
                weights[edge_count] = propagated_weight
                edge_count += 1
            
            # Continue BFS if not visited at this depth or visited at deeper depth
            prev_depth = visited_depths[next_node]
            if (prev_depth == -1 or depth + 1 < prev_depth) and depth + 1 < max_depth and queue_size < max_queue_size:
                visited_depths[next_node] = depth + 1
                queue_nodes[queue_size] = next_node
                queue_depths[queue_size] = depth + 1
                queue_weights[queue_size] = propagated_weight
                queue_size += 1
    
    return srcs[:edge_count], dsts[:edge_count], weights[:edge_count], edge_count


@jit(nopython=True, cache=True)
def _process_file_context_numba(node_indices: np.ndarray, line_numbers: np.ndarray,
                                 window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized function to process a single file for context window edges.
    Uses sliding window algorithm: O(N*W) where W is average window size.
    Returns only unique edges (one direction) - caller will add reverse.
    
    Returns:
        Tuple of (src_indices, dst_indices) arrays - unique pairs only
    """
    n_nodes = len(node_indices)
    if n_nodes < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    
    # Estimate max edges: n_nodes * window_size (upper bound)
    # Use aggressive limit to prevent explosion
    max_pairs = min(n_nodes * window_size, CONTEXT_WINDOW_MAX_PAIRS_PER_FILE)
    
    # Pre-allocate arrays
    srcs = np.zeros(max_pairs, dtype=np.int64)
    dsts = np.zeros(max_pairs, dtype=np.int64)
    pair_count = 0
    
    # Sliding window: for each node i, find all nodes j > i within window
    for i in range(n_nodes):
        if pair_count >= max_pairs:
            break
            
        node1_idx = node_indices[i]
        line1 = line_numbers[i]
        
        # Find right boundary: first node beyond window
        # Since lines are sorted, we can use linear scan (fast for small windows)
        j = i + 1
        while j < n_nodes and pair_count < max_pairs:
            line2 = line_numbers[j]
            line_diff = line2 - line1
            
            if line_diff > window_size:
                break  # Beyond window, stop
            
            # Within window - add edge (only one direction, caller adds reverse)
            srcs[pair_count] = node1_idx
            dsts[pair_count] = node_indices[j]
            pair_count += 1
            j += 1
    
    return srcs[:pair_count], dsts[:pair_count]


def _process_file_context(args):
    """Process a single file for context window edges - optimized with Numba."""
    file_path, node_lines, window_size = args
    
    if not node_lines or len(node_lines) < 2:
        return []
    
    # Sort by line number (required for sliding window algorithm)
    node_lines.sort(key=lambda x: x[1])
    
    # Convert to NumPy arrays - do this efficiently
    n = len(node_lines)
    node_indices = np.empty(n, dtype=np.int64)
    line_numbers = np.empty(n, dtype=np.int64)
    for i, (idx, line) in enumerate(node_lines):
        node_indices[i] = idx
        line_numbers[i] = line
    
    # Process with Numba (always use if available)
    if NUMBA_AVAILABLE:
        try:
            srcs, dsts = _process_file_context_numba(node_indices, line_numbers, window_size)
            
            # Batch create edges (both directions) - pre-allocate list
            n_edges = len(srcs)
            if n_edges == 0:
                return []
            
            edges = [None] * (n_edges * 2)
            for i in range(n_edges):
                edges[i * 2] = (int(srcs[i]), int(dsts[i]), "context_window", 1.0)
                edges[i * 2 + 1] = (int(dsts[i]), int(srcs[i]), "context_window", 1.0)
            
            return edges
        except Exception:
            # Fallback to Python if numba fails
            pass
    
    # Python fallback (should rarely be used)
    edges = []
    for i in range(n):
        node1_idx = int(node_indices[i])
        line1 = int(line_numbers[i])
        for j in range(i + 1, n):
            node2_idx = int(node_indices[j])
            line2 = int(line_numbers[j])
            if line2 - line1 > window_size:
                break
            edges.append((node1_idx, node2_idx, "context_window", 1.0))
            edges.append((node2_idx, node1_idx, "context_window", 1.0))
    
    return edges




@jit(nopython=True, cache=True)
def _compute_context_similarity_boost(len1: float, len2: float, 
                                      similarity_threshold: float,
                                      existing_weight: float) -> float:
    """
    Numba-optimized function to compute context similarity boost.
    
    Returns:
        Boosted weight if similarity >= threshold, else existing_weight
    """
    if len1 <= 0.0 or len2 <= 0.0:
        return existing_weight
    
    max_len = max(len1, len2)
    min_len = min(len1, len2)
    similarity = min_len / max_len
    
    if similarity >= similarity_threshold:
        boosted_weight = existing_weight * (1.0 + similarity * 0.2)
        return boosted_weight
    
    return existing_weight


@jit(nopython=True, cache=True, parallel=True)
def _compute_boosted_weights_numba(edge_srcs: np.ndarray, edge_dsts: np.ndarray, 
                                    edge_weights: np.ndarray,
                                    context_lengths: np.ndarray,
                                    similarity_threshold: float) -> np.ndarray:
    """
    Numba-optimized function to compute boosted weights for all edges.
    
    Args:
        edge_srcs: source node indices
        edge_dsts: destination node indices
        edge_weights: existing edge weights
        context_lengths: array mapping node_idx -> context_length
        similarity_threshold: minimum similarity threshold
    
    Returns:
        Array of boosted weights
    """
    n_edges = len(edge_srcs)
    boosted = np.zeros(n_edges, dtype=np.float64)
    
    for i in prange(n_edges):
        src_idx = int(edge_srcs[i])
        dst_idx = int(edge_dsts[i])
        
        if src_idx < len(context_lengths) and dst_idx < len(context_lengths):
            len1 = context_lengths[src_idx]
            len2 = context_lengths[dst_idx]
            
            if len1 > 0.0 and len2 > 0.0:
                boosted[i] = _compute_context_similarity_boost(
                    len1, len2, similarity_threshold, edge_weights[i]
                )
            else:
                boosted[i] = edge_weights[i]
        else:
            boosted[i] = edge_weights[i]
    
    return boosted


def add_context_length_edges(edges: List[Tuple[int, int, str, float]],
                              node_metadata: List[Dict],
                              similarity_threshold: float = 0.3) -> List[Tuple[int, int, str, float]]:
    """
    Add edges for symbols with similar context lengths (scope sizes).
    Symbols with similar indentation-based scopes are more semantically related.
    Optimized with Numba JIT compilation for faster computation.
    
    Args:
        edges: existing edges
        node_metadata: list of node metadata dictionaries
        similarity_threshold: minimum similarity ratio for context length (0.0-1.0)
    
    Returns:
        Expanded list of edges with context-length-based connections
    """
    new_edges = list(edges)
    
    # Build context length map: node_idx -> context_length (end_lineno - lineno)
    context_lengths = {}
    for idx, node_meta in enumerate(node_metadata):
        meta = node_meta.get('meta', {})
        if isinstance(meta, dict):
            lineno = meta.get('lineno', -1)
            end_lineno = meta.get('end_lineno', None)
            if lineno >= 0 and end_lineno is not None and end_lineno > lineno:
                context_lengths[idx] = end_lineno - lineno
            elif lineno >= 0:
                # Single-line symbol (variable, etc.) - context length = 1
                context_lengths[idx] = 1
    
    if not context_lengths:
        return new_edges
    
    # Build edge weight map and index map for efficient lookup
    edge_weights = defaultdict(float)
    edge_indices = {}  # Map (src, dst) -> list of indices in new_edges
    for j, edge in enumerate(edges):
        if len(edge) == 4:
            src, dst, rel, w = edge
            key = (min(src, dst), max(src, dst))
            edge_weights[key] = max(edge_weights[key], w)
            # Store indices for both directions
            if (src, dst) not in edge_indices:
                edge_indices[(src, dst)] = []
            edge_indices[(src, dst)].append(j)
            if (dst, src) not in edge_indices:
                edge_indices[(dst, src)] = []
            edge_indices[(dst, src)].append(j)
    
    # Extract edges that have context length info and convert to NumPy arrays
    edge_data = []
    edge_positions = []
    for j, edge in enumerate(edges):
        if len(edge) == 4:
            src, dst, rel, w = edge
            if src in context_lengths and dst in context_lengths:
                edge_data.append((src, dst, w))
                edge_positions.append(j)
    
    if not edge_data:
        return new_edges
    
    # Convert to NumPy arrays for numba
    max_node_idx = max(max(context_lengths.keys()), 
                      max(max(src, dst) for src, dst, _ in edge_data))
    context_array = np.zeros(int(max_node_idx + 1), dtype=np.float64)
    for idx, length in context_lengths.items():
        context_array[int(idx)] = float(length)
    
    edge_srcs = np.array([src for src, _, _ in edge_data], dtype=np.int64)
    edge_dsts = np.array([dst for _, dst, _ in edge_data], dtype=np.int64)
    edge_ws = np.array([w for _, _, w in edge_data], dtype=np.float64)
    
    # Use numba-optimized computation if available
    if NUMBA_AVAILABLE:
        try:
            boosted_weights = _compute_boosted_weights_numba(
                edge_srcs, edge_dsts, edge_ws, context_array, similarity_threshold
            )
            
            # Update edges with boosted weights
            for i, j in enumerate(edge_positions):
                if boosted_weights[i] > edge_ws[i]:  # Only update if boosted
                    src, dst, rel, old_w = new_edges[j]
                    new_edges[j] = (src, dst, rel, float(boosted_weights[i]))
                    # Update reverse edges if they exist
                    reverse_indices = edge_indices.get((dst, src), [])
                    for rev_idx in reverse_indices:
                        if rev_idx != j:  # Don't update the same edge twice
                            rev_src, rev_dst, rev_rel, rev_old_w = new_edges[rev_idx]
                            new_edges[rev_idx] = (rev_src, rev_dst, rev_rel, float(boosted_weights[i]))
        except Exception:
            # Fallback to Python if numba fails
            pass
    
    # Fallback to Python implementation if numba not available or failed
    if not NUMBA_AVAILABLE:
        processed_pairs = set()
        for edge in edges:
            if len(edge) == 4:
                src, dst, rel, w = edge
                if src in context_lengths and dst in context_lengths:
                    pair = (min(src, dst), max(src, dst))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        len1 = context_lengths[src]
                        len2 = context_lengths[dst]
                        
                        # Calculate similarity: min(len1, len2) / max(len1, len2)
                        max_len = max(len1, len2)
                        min_len = min(len1, len2)
                        similarity = min_len / max_len if max_len > 0 else 0.0
                        
                        if similarity >= similarity_threshold:
                            # Boost existing edge weight by context similarity
                            key = (min(src, dst), max(src, dst))
                            existing_weight = edge_weights.get(key, w)
                            boosted_weight = existing_weight * (1.0 + similarity * 0.2)
                            
                            # Update edge weights in new_edges
                            indices = edge_indices.get((src, dst), [])
                            for edge_idx in indices:
                                e_src, e_dst, e_rel, e_w = new_edges[edge_idx]
                                new_edges[edge_idx] = (e_src, e_dst, e_rel, max(e_w, boosted_weight))

    return new_edges


def add_context_window_edges(edges: List[Tuple[int, int, str, float]],
                             node_metadata: List[Dict],
                             window_size: int = 5,
                             n_jobs: int = -1,
                             progress_callback=None) -> List[Tuple[int, int, str, float]]:
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

    # Filter files: skip files with < 2 nodes, limit nodes per file
    max_nodes_per_file = CONTEXT_WINDOW_MAX_NODES_PER_FILE
    
    filtered_files = {}
    for file_path, node_lines in nodes_by_file.items():
        if len(node_lines) < 2:
            continue
        if len(node_lines) <= max_nodes_per_file:
            filtered_files[file_path] = node_lines
        else:
            # Sample large files: take evenly spaced nodes
            sorted_nodes = sorted(node_lines, key=lambda x: x[1])
            step = len(sorted_nodes) // max_nodes_per_file
            filtered_files[file_path] = sorted_nodes[::max(1, step)][:max_nodes_per_file]
    
    if not filtered_files:
        return new_edges
    
    # Always use multiprocessing if we have multiple files and workers available
    if n_jobs == -1:
        n_jobs = _get_num_workers()
    
    num_files = len(filtered_files)
    use_multiprocessing = (num_files > 1 and n_jobs > 1)
    
    if use_multiprocessing:
        # Parallel processing: process all files in parallel
        args_list = [(file_path, node_lines, window_size)
                     for file_path, node_lines in filtered_files.items()]
        chunksize = max(1, len(args_list) // (n_jobs * 2))  # Smaller chunks for better load balancing
        
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_process_file_context, args_list, chunksize=chunksize)
        
        # Flatten results efficiently
        for file_edges in results:
            if file_edges:
                new_edges.extend(file_edges)
    else:
        # Sequential processing with progress callback
        file_list = list(filtered_files.items())
        for file_idx, (file_path, node_lines) in enumerate(file_list):
            if progress_callback:
                progress_callback(file_idx + 1, len(file_list))
            
            file_edges = _process_file_context((file_path, node_lines, window_size))
            if file_edges:
                new_edges.extend(file_edges)

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
        # Aggregate weights for duplicate edges (same src, dst pair)
        edge_weights = defaultdict(float)
        for src_idx, dst_idx, rel, weight in edges:
            key = (src_idx, dst_idx)
            edge_weights[key] = max(edge_weights[key], weight)

        rows = []
        cols = []
        data = []
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
                       gpu_accelerator=None,
                       progress_tasks=None) -> Tuple[
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
        # Create progress callback for call graph expansion
        call_progress_cb = None
        if progress_tasks and 'call_graph' in progress_tasks:
            task = progress_tasks['call_graph']
            def call_progress_cb(current, total):
                progress_tasks['progress'].update(task, completed=current, total=total)
        expanded_edges = expand_call_graph(expanded_edges, current_num_nodes, max_depth=max_depth,
                                          progress_callback=call_progress_cb)

    # Step 2: Add context window co-occurrence (must be before PPMI)
    if add_context and node_metadata is not None:
        # Create progress callback for context window edges
        context_progress_cb = None
        if progress_tasks and 'context_window' in progress_tasks:
            task = progress_tasks['context_window']
            def context_progress_cb(current, total):
                progress_tasks['progress'].update(task, completed=current, total=total)
        expanded_edges = add_context_window_edges(expanded_edges, node_metadata, window_size=context_window,
                                                  n_jobs=n_jobs, progress_callback=context_progress_cb)
        # Add context length edges (scope similarity - symbols with similar indentation-based scopes)
        expanded_edges = add_context_length_edges(expanded_edges, node_metadata, similarity_threshold=0.3)

    # Step 3: Add subtoken nodes and edges
    if add_subtokens and node_to_idx is not None and node_subtokens is not None:
        # Create progress callback for subtoken edges
        subtoken_progress_cb = None
        if progress_tasks and 'subtoken' in progress_tasks:
            task = progress_tasks['subtoken']
            def subtoken_progress_cb(current, total):
                progress_tasks['progress'].update(task, completed=current, total=total)
        expanded_edges, current_num_nodes, subtoken_to_idx = add_subtoken_edges(
            expanded_edges, node_to_idx, node_subtokens, current_num_nodes,
            progress_callback=subtoken_progress_cb
        )

    # Step 4: Add file hierarchy edges
    if add_hierarchy and node_to_idx is not None and node_file_info is not None and idx_to_node is not None:
        # Create progress callback for hierarchy edges
        hierarchy_progress_cb = None
        if progress_tasks and 'hierarchy' in progress_tasks:
            task = progress_tasks['hierarchy']
            def hierarchy_progress_cb(current, total):
                progress_tasks['progress'].update(task, completed=current, total=total)
        expanded_edges = add_file_hierarchy_edges(
            expanded_edges, node_to_idx, node_file_info, idx_to_node,
            progress_callback=hierarchy_progress_cb
        )

    # Step 5: Build adjacency matrix and compute PPMI (with parallelization)
    adj = build_adjacency_matrix(expanded_edges, current_num_nodes, n_jobs=n_jobs)
    ppmi = compute_ppmi(adj)
    embeddings, svd_components = reduce_dimensions_ppmi(ppmi, dim, random_state, n_jobs, gpu_accelerator)

    return embeddings, svd_components, current_num_nodes, subtoken_to_idx, expanded_edges
