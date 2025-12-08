"""Context view: Node2Vec-style random walks and Word2Vec."""
import random
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Set, Dict

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from numba import jit, prange

from .graph_config import RANDOM_WALK_MAX_NEIGHBORS, RANDOM_WALK_MIN_NODES_FOR_PARALLEL


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


@jit(nopython=True)
def _weighted_choice_numba(probs):
    """Weighted random choice using cumulative probabilities (Numba-compatible)."""
    r = np.random.random()
    cumsum = 0.0
    for i in range(len(probs)):
        cumsum += probs[i]
        if r <= cumsum:
            return i
    return len(probs) - 1


@jit(nopython=True)
def _generate_walk_step_numba(neighbor_indices, neighbor_weights, num_neighbors,
                               prev_node, prev_neighbors_set, p, q):
    """Generate next step in walk using Node2Vec probabilities (Numba-optimized)."""
    max_check = min(num_neighbors, RANDOM_WALK_MAX_NEIGHBORS)
    if max_check == 0:
        return -1
    
    if prev_node == -1:
        return neighbor_indices[np.random.randint(0, max_check)]
    
    probs = np.zeros(max_check, dtype=np.float64)
    nodes = np.zeros(max_check, dtype=np.int32)
    
    for i in range(max_check):
        neighbor_idx = neighbor_indices[i]
        weight = neighbor_weights[i]
        nodes[i] = neighbor_idx
        
        if neighbor_idx == prev_node:
            probs[i] = weight / p
        elif neighbor_idx < len(prev_neighbors_set) and prev_neighbors_set[neighbor_idx] > 0:
            probs[i] = weight
        else:
            probs[i] = weight / q
        probs[i] = max(probs[i], 1e-10)
    
    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs = probs / probs_sum
        idx = _weighted_choice_numba(probs)
        return nodes[idx]
    else:
        return neighbor_indices[np.random.randint(0, max_check)]


@jit(nopython=True)
def _generate_walks_sequential_numba(nodes_to_process, neighbor_indices_array, neighbor_weights_array,
                                     neighbor_counts, prev_neighbors_array, num_walks, walk_length,
                                     p, q, random_seed, max_degree, num_nodes):
    """Generate all walks using Numba (sequential with progress tracking)."""
    np.random.seed(random_seed)
    num_start_nodes = len(nodes_to_process)
    total_walks = num_walks * num_start_nodes
    
    max_walks = total_walks
    walks_data = np.zeros(max_walks * walk_length, dtype=np.int32)
    walk_lengths = np.zeros(max_walks, dtype=np.int32)
    
    walk_data_idx = 0
    for walk_idx in range(total_walks):
        start_node_idx = walk_idx % num_start_nodes
        start_node = nodes_to_process[start_node_idx]
        
        walk_start = walk_data_idx
        walks_data[walk_data_idx] = start_node
        walk_data_idx += 1
        walk_len = 1
        prev = -1
        
        for step in range(walk_length - 1):
            curr = walks_data[walk_start + walk_len - 1]
            if curr >= num_nodes or neighbor_counts[curr] == 0:
                break
            
            neighbor_start = curr * max_degree
            neighbor_count = neighbor_counts[curr]
            
            if prev >= 0 and prev < num_nodes:
                prev_base = prev * num_nodes
            else:
                prev_base = -1
            
            next_node = _generate_walk_step_numba(
                neighbor_indices_array[neighbor_start:neighbor_start + neighbor_count],
                neighbor_weights_array[neighbor_start:neighbor_start + neighbor_count],
                neighbor_count,
                prev,
                prev_neighbors_array[prev_base:prev_base + num_nodes] if prev_base >= 0 else np.zeros(num_nodes, dtype=np.int32),
                p, q
            )
            
            if next_node == -1:
                break
            
            walks_data[walk_data_idx] = next_node
            walk_data_idx += 1
            walk_len += 1
            prev = curr
        
        walk_lengths[walk_idx] = walk_len
    
    return walks_data, walk_lengths, total_walks


def _generate_walks_for_node(args):
    """Generate walks for a single node (helper for multiprocessing)."""
    start_node, adj_list, adj_sets, num_walks, walk_length, p, q, seed_offset = args
    # Set seed for this worker
    random.seed(42 + seed_offset + start_node)
    np.random.seed(42 + seed_offset + start_node)

    walks = []
    for walk_idx in range(num_walks):
        if not adj_list[start_node]:
            continue

        walk = [start_node]

        for _ in range(walk_length - 1):
            curr = walk[-1]
            neighbors = adj_list[curr]

            if not neighbors:
                break

            # Limit neighbors to check (prevents slow probability calculation)
            neighbors_to_check = neighbors[:RANDOM_WALK_MAX_NEIGHBORS] if len(neighbors) > RANDOM_WALK_MAX_NEIGHBORS else neighbors

            if len(walk) == 1:
                next_node = random.choice(neighbors_to_check)[0]
            else:
                prev = walk[-2]
                prev_neighbors_set = adj_sets[prev]  # Fast set lookup instead of list iteration
                
                probs = []
                nodes = []

                for neighbor, weight in neighbors_to_check:
                    nodes.append(neighbor)
                    if neighbor == prev:
                        prob = weight / p
                    elif neighbor in prev_neighbors_set:
                        prob = weight
                    else:
                        prob = weight / q
                    probs.append(max(prob, 1e-10))

                probs = np.array(probs, dtype=np.float64)
                probs = probs / probs.sum()
                next_node = np.random.choice(nodes, p=probs)

            walk.append(next_node)

        walks.append([str(node) for node in walk])

    return walks


def generate_random_walks(edges: List[Tuple[int, int, str, float]],
                          num_nodes: int,
                          num_walks: int = 10,
                          walk_length: int = 80,
                          p: float = 1.0,
                          q: float = 1.0,
                          random_state: int = 42,
                          n_jobs: int = -1,
                          progress_callback=None) -> List[List[int]]:
    """
    Generate Node2Vec-style random walks with multiprocessing support.
    
    Args:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of nodes
        num_walks: number of walks per node
        walk_length: length of each walk
        p: return parameter (1/p controls likelihood of returning)
        q: in-out parameter (1/q controls likelihood of going further)
        random_state: random seed
        n_jobs: number of parallel jobs (-1 for all cores - 1)
    
    Returns:
        List of walks (each walk is a list of node indices)
    """
    if progress_callback:
        estimated_total = min(num_nodes, max(100, len(edges) // 10))
        progress_callback(0, estimated_total)
        progress_callback(0, estimated_total)
    
    adj_list: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    adj_sets: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    
    total_edges = len(edges)
    callback_interval = max(1, total_edges // 20) if total_edges > 100 else 1
    for idx, (src_idx, dst_idx, rel, weight) in enumerate(edges):
        adj_list[src_idx].append((dst_idx, weight))
        adj_list[dst_idx].append((src_idx, weight))
        adj_sets[src_idx].add(dst_idx)
        adj_sets[dst_idx].add(src_idx)
        if progress_callback and (idx + 1) % callback_interval == 0:
            progress_callback(0, estimated_total)

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = _get_num_workers()
    n_jobs = max(1, min(n_jobs, num_nodes))

    # Prepare arguments for parallel processing
    nodes_to_process = [node for node in range(num_nodes) if adj_list[node]]
    total_nodes_to_process = len(nodes_to_process)
    
    # Update progress with correct total now that we know it
    # This ensures the progress bar shows the correct total
    if progress_callback:
        progress_callback(0, total_nodes_to_process)

    # Use parallel processing if we have enough nodes (overhead not worth it for small cases)
    use_parallel = (n_jobs > 1 and len(nodes_to_process) >= RANDOM_WALK_MIN_NODES_FOR_PARALLEL)
    
    # Convert adjacency structures to flat NumPy arrays for Numba
    max_degree = max(len(adj_list[i]) for i in range(num_nodes)) if num_nodes > 0 else 0
    max_degree = min(max_degree, RANDOM_WALK_MAX_NEIGHBORS)
    
    neighbor_indices_array = np.zeros(num_nodes * max_degree, dtype=np.int32)
    neighbor_weights_array = np.zeros(num_nodes * max_degree, dtype=np.float64)
    neighbor_counts = np.zeros(num_nodes, dtype=np.int32)
    
    for node_id in range(num_nodes):
        neighbors = adj_list[node_id]
        count = min(len(neighbors), RANDOM_WALK_MAX_NEIGHBORS)
        neighbor_counts[node_id] = count
        
        base_idx = node_id * max_degree
        for i in range(count):
            neighbor_indices_array[base_idx + i] = neighbors[i][0]
            neighbor_weights_array[base_idx + i] = neighbors[i][1]
    
    # Convert sets to flat array for fast lookup
    prev_neighbors_array = np.zeros(num_nodes * num_nodes, dtype=np.int32)
    for node_id in range(num_nodes):
        if node_id in adj_sets:
            neighbors_set = adj_sets[node_id]
            base_idx = node_id * num_nodes
            for n in neighbors_set:
                if n < num_nodes:
                    prev_neighbors_array[base_idx + n] = 1
    
    nodes_to_process_arr = np.array(nodes_to_process, dtype=np.int32)
    
    if not use_parallel:
        # Sequential processing with Numba
        walks_data, walk_lengths, total_walks = _generate_walks_sequential_numba(
            nodes_to_process_arr, neighbor_indices_array, neighbor_weights_array,
            neighbor_counts, prev_neighbors_array, num_walks, walk_length, p, q, random_state, max_degree, num_nodes
        )
        
        walks = []
        walk_idx = 0
        for i in range(total_walks):
            walk_len = int(walk_lengths[i])
            walk = walks_data[walk_idx:walk_idx + walk_len]
            walks.append([str(node) for node in walk])
            walk_idx += walk_len
        
        if progress_callback:
            for i in range(total_nodes_to_process):
                progress_callback(i + 1, total_nodes_to_process)
    else:
        # Parallel processing with progress tracking
        args_list = [
            (node, adj_list, adj_sets, num_walks, walk_length, p, q, random_state + i)
            for i, node in enumerate(nodes_to_process)
        ]

        # Parallel processing - use standard Pool which works on all platforms
        # Use larger chunksize for better load balancing
        chunksize = max(1, len(args_list) // (n_jobs * 4))
        
        # Use imap_unordered for progress tracking
        walks = []
        with Pool(processes=n_jobs) as pool:
            results = pool.imap_unordered(_generate_walks_for_node, args_list, chunksize=chunksize)
            completed = 0
            for result in results:
                walks.extend(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_nodes_to_process)

    # Update progress to complete
    if progress_callback:
        progress_callback(total_nodes_to_process, total_nodes_to_process)
    
    return walks


def train_word2vec(walks: List[List[str]], dim: int, window: int = 7,
                   negative: int = 3, epochs: int = 3, random_state: int = 42,
                   n_jobs: int = -1, batch_words: int = 10000) -> KeyedVectors:
    """
    Train Word2Vec SkipGram model on random walks with multiprocessing.
    Optimized defaults: window=7 (was 10), negative=3 (was 5) for faster training.
    
    Args:
        walks: list of walks (each walk is a list of node ID strings)
        dim: embedding dimensionality
        window: context window size (reduced default: 7)
        negative: number of negative samples (reduced default: 3)
        epochs: number of training epochs
        random_state: random seed
        n_jobs: number of parallel workers (-1 for all cores - 1)
        batch_words: words per batch (larger = faster but more memory)
    
    Returns:
        Trained KeyedVectors model
    """
    if n_jobs == -1:
        n_jobs = _get_num_workers()

    # For gensim 4.x, use workers parameter
    workers = max(1, n_jobs)
    
    # Word2Vec training happens here - this is the actual learning step
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=1,
        workers=workers,
        sg=1,  # SkipGram
        negative=negative,
        epochs=epochs,
        seed=random_state,
        batch_words=batch_words  # Larger batches for faster training
    )

    return model.wv


def compute_context_view(edges: List[Tuple[int, int, str, float]],
                         num_nodes: int,
                         dim: int,
                         num_walks: int = 10,
                         walk_length: int = 80,
                         random_state: int = 42,
                         n_jobs: int = -1,
                         progress_callback=None,
                         word2vec_progress_callback=None) -> Tuple[np.ndarray, KeyedVectors]:
    """
    Compute context view embeddings using Node2Vec + Word2Vec with multiprocessing.
    
    Args:
        progress_callback: callback for random walk generation progress (current, total)
        word2vec_progress_callback: callback to signal Word2Vec training start/end
        console: Rich console for logging (optional)
    
    Returns:
        embeddings: node embeddings from context view
        keyed_vectors: Word2Vec KeyedVectors model
    """
    walks = generate_random_walks(edges, num_nodes, num_walks, walk_length,
                                  random_state=random_state, n_jobs=n_jobs,
                                  progress_callback=progress_callback)
    
    # Signal Word2Vec training is starting
    if word2vec_progress_callback:
        word2vec_progress_callback(True)  # True = start
    
    # Word2Vec training happens here - this is the actual learning step
    # Note: gensim Word2Vec doesn't support progress callbacks, so training happens synchronously
    kv = train_word2vec(walks, dim, random_state=random_state, n_jobs=n_jobs)
    
    # Signal Word2Vec training is complete
    if word2vec_progress_callback:
        word2vec_progress_callback(False)  # False = complete

    # Extract embeddings for all nodes
    embeddings = np.zeros((num_nodes, dim))
    for i in range(num_nodes):
        node_str = str(i)
        if node_str in kv:
            embeddings[i] = kv[node_str]
        else:
            # Initialize with small random values if node not seen
            np.random.seed(random_state + i)
            embeddings[i] = np.random.normal(0, 0.01, dim)

    return embeddings, kv
