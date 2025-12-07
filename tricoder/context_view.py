"""Context view: Node2Vec-style random walks and Word2Vec."""
import random
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def _generate_walks_for_node(args):
    """Generate walks for a single node (helper for multiprocessing)."""
    start_node, adj_list, num_walks, walk_length, p, q, seed_offset = args
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

            if len(walk) == 1:
                next_node = random.choice(neighbors)[0]
            else:
                prev = walk[-2]
                probs = []
                nodes = []

                for neighbor, weight in neighbors:
                    nodes.append(neighbor)
                    if neighbor == prev:
                        prob = weight / p
                    elif neighbor in adj_list[prev]:
                        prob = weight
                    else:
                        prob = weight / q
                    probs.append(max(prob, 1e-10))

                probs = np.array(probs)
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
                          n_jobs: int = -1) -> List[List[int]]:
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
    # Build adjacency list
    adj_list = {i: [] for i in range(num_nodes)}
    for src_idx, dst_idx, rel, weight in edges:
        adj_list[src_idx].append((dst_idx, weight))
        adj_list[dst_idx].append((src_idx, weight))

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = _get_num_workers()
    n_jobs = max(1, min(n_jobs, num_nodes))

    # Prepare arguments for parallel processing
    nodes_to_process = [node for node in range(num_nodes) if adj_list[node]]

    if n_jobs == 1 or len(nodes_to_process) < n_jobs:
        # Sequential processing for small cases
        random.seed(random_state)
        np.random.seed(random_state)
        walks = []
        for _ in range(num_walks):
            for start_node in nodes_to_process:
                walk = [start_node]
                for _ in range(walk_length - 1):
                    curr = walk[-1]
                    neighbors = adj_list[curr]
                    if not neighbors:
                        break

                    if len(walk) == 1:
                        next_node = random.choice(neighbors)[0]
                    else:
                        prev = walk[-2]
                        probs = []
                        nodes = []
                        for neighbor, weight in neighbors:
                            nodes.append(neighbor)
                            if neighbor == prev:
                                prob = weight / p
                            elif neighbor in adj_list[prev]:
                                prob = weight
                            else:
                                prob = weight / q
                            probs.append(max(prob, 1e-10))
                        probs = np.array(probs)
                        probs = probs / probs.sum()
                        next_node = np.random.choice(nodes, p=probs)
                    walk.append(next_node)
                walks.append([str(node) for node in walk])
    else:
        # Parallel processing
        args_list = [
            (node, adj_list, num_walks, walk_length, p, q, random_state + i)
            for i, node in enumerate(nodes_to_process)
        ]

        # Use spawn context on Windows to avoid pickling issues
        import platform
        if platform.system() == 'Windows':
            from multiprocessing import get_context
            ctx = get_context('spawn')
            with ctx.Pool(processes=n_jobs) as pool:
                results = pool.map(_generate_walks_for_node, args_list)
        else:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(_generate_walks_for_node, args_list)

        # Flatten results
        walks = []
        for result in results:
            walks.extend(result)

    return walks


def train_word2vec(walks: List[List[str]], dim: int, window: int = 10,
                   negative: int = 5, epochs: int = 5, random_state: int = 42,
                   n_jobs: int = -1) -> KeyedVectors:
    """
    Train Word2Vec SkipGram model on random walks with multiprocessing.
    
    Args:
        walks: list of walks (each walk is a list of node ID strings)
        dim: embedding dimensionality
        window: context window size
        negative: number of negative samples
        epochs: number of training epochs
        random_state: random seed
        n_jobs: number of parallel workers (-1 for all cores - 1)
    
    Returns:
        Trained KeyedVectors model
    """
    if n_jobs == -1:
        n_jobs = _get_num_workers()

    # For gensim 4.x, use 'workers' parameter correctly
    # On Windows with spawn, limit workers to avoid issues
    import platform
    if platform.system() == 'Windows' and n_jobs > 1:
        # On Windows, use fewer workers to avoid multiprocessing issues
        workers = min(n_jobs, 4)
    else:
        workers = n_jobs
    
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        min_count=1,
        workers=workers,
        sg=1,  # SkipGram
        negative=negative,
        epochs=epochs,
        seed=random_state
    )

    return model.wv


def compute_context_view(edges: List[Tuple[int, int, str, float]],
                         num_nodes: int,
                         dim: int,
                         num_walks: int = 10,
                         walk_length: int = 80,
                         random_state: int = 42,
                         n_jobs: int = -1) -> Tuple[np.ndarray, KeyedVectors]:
    """
    Compute context view embeddings using Node2Vec + Word2Vec with multiprocessing.
    
    Returns:
        embeddings: node embeddings from context view
        keyed_vectors: Word2Vec KeyedVectors model
    """
    walks = generate_random_walks(edges, num_nodes, num_walks, walk_length,
                                  random_state=random_state, n_jobs=n_jobs)
    kv = train_word2vec(walks, dim, random_state=random_state, n_jobs=n_jobs)

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
