"""Temperature calibration using held-out edges."""
import numpy as np
from typing import List, Tuple
import random
from multiprocessing import Pool, cpu_count
from functools import partial


def _get_num_workers() -> int:
    """Get number of workers (all cores - 1, minimum 1)."""
    return max(1, cpu_count() - 1)


def _evaluate_single_tau(args):
    """Evaluate a single tau value (helper for multiprocessing)."""
    tau, embeddings, positive_pairs, negative_pairs = args
    return tau, evaluate_tau(embeddings, positive_pairs, negative_pairs, tau)


def split_edges(edges: List[Tuple[int, int, str, float]], 
               train_ratio: float = 0.8,
               random_state: int = 42) -> Tuple[List[Tuple[int, int, str, float]], 
                                                List[Tuple[int, int, str, float]]]:
    """
    Split edges into training and validation sets.
    
    Args:
        edges: list of edges
        train_ratio: proportion of edges for training
        random_state: random seed
    
    Returns:
        train_edges, val_edges
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    shuffled = edges.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_edges = shuffled[:split_idx]
    val_edges = shuffled[split_idx:]
    
    return train_edges, val_edges


def sample_negatives(num_nodes: int, positive_pairs: List[Tuple[int, int]], 
                    num_negatives: int = 5, random_state: int = 42) -> List[Tuple[int, int]]:
    """
    Sample negative pairs (non-edges).
    
    Args:
        num_nodes: number of nodes
        positive_pairs: list of (src_idx, dst_idx) positive pairs
        num_negatives: number of negatives per positive
        random_state: random seed
    
    Returns:
        List of negative pairs
    """
    np.random.seed(random_state)
    positive_set = set(positive_pairs)
    negative_pairs = []
    
    for src, dst in positive_pairs:
        for _ in range(num_negatives):
            # Sample random node
            neg_dst = np.random.randint(0, num_nodes)
            # Ensure it's not a positive edge
            if (src, neg_dst) not in positive_set and src != neg_dst:
                negative_pairs.append((src, neg_dst))
    
    return negative_pairs


def compute_calibrated_score(emb_u: np.ndarray, emb_v: np.ndarray, tau: float) -> float:
    """
    Compute calibrated similarity score.
    
    Args:
        emb_u: embedding vector for node u
        emb_v: embedding vector for node v
        tau: temperature parameter
    
    Returns:
        Calibrated score
    """
    dot_product = np.dot(emb_u, emb_v)
    return dot_product / tau


def softmax_scores(scores: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities from scores.
    
    Args:
        scores: array of scores
    
    Returns:
        Softmax probabilities
    """
    # Numerical stability: subtract max
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()


def evaluate_tau(embeddings: np.ndarray,
                positive_pairs: List[Tuple[int, int]],
                negative_pairs: List[Tuple[int, int]],
                tau: float) -> float:
    """
    Evaluate temperature tau on validation pairs.
    
    Returns:
        Average probability assigned to positive pairs
    """
    pos_scores = []
    for src, dst in positive_pairs:
        score = compute_calibrated_score(embeddings[src], embeddings[dst], tau)
        pos_scores.append(score)
    
    neg_scores = []
    for src, dst in negative_pairs:
        score = compute_calibrated_score(embeddings[src], embeddings[dst], tau)
        neg_scores.append(score)
    
    # For each positive, compute softmax over positive + negatives
    total_prob = 0.0
    for i, (src, dst) in enumerate(positive_pairs):
        # Get corresponding negatives
        start_idx = i * len(negative_pairs) // len(positive_pairs)
        end_idx = (i + 1) * len(negative_pairs) // len(positive_pairs)
        if end_idx == start_idx:
            end_idx = start_idx + 1
        
        relevant_negatives = negative_pairs[start_idx:end_idx]
        all_scores = [pos_scores[i]]
        for neg_src, neg_dst in relevant_negatives:
            neg_score = compute_calibrated_score(embeddings[neg_src], embeddings[neg_dst], tau)
            all_scores.append(neg_score)
        
        probs = softmax_scores(np.array(all_scores))
        total_prob += probs[0]  # Probability of positive
    
    return total_prob / len(positive_pairs)


def learn_temperature(embeddings: np.ndarray,
                     val_edges: List[Tuple[int, int, str, float]],
                     num_nodes: int,
                     num_negatives: int = 5,
                     tau_candidates: np.ndarray = None,
                     random_state: int = 42,
                     n_jobs: int = -1) -> float:
    """
    Learn optimal temperature parameter via parallel grid search.
    
    Args:
        embeddings: fused embeddings
        val_edges: validation edges
        num_nodes: number of nodes
        num_negatives: number of negatives per positive
        tau_candidates: candidate tau values (if None, use logspace)
        random_state: random seed
        n_jobs: number of parallel jobs (-1 for all cores - 1)
    
    Returns:
        Optimal temperature tau
    """
    # Handle empty validation set
    if not val_edges:
        return 1.0
    
    if tau_candidates is None:
        tau_candidates = np.logspace(-2, 2, num=50)
    
    positive_pairs = [(src, dst) for src, dst, _, _ in val_edges]
    negative_pairs = sample_negatives(num_nodes, positive_pairs, num_negatives, random_state)
    
    # Handle case where no negatives could be sampled
    if not negative_pairs:
        return 1.0
    
    if n_jobs == -1:
        n_jobs = _get_num_workers()
    
    # Parallel evaluation of tau candidates
    if n_jobs > 1 and len(tau_candidates) > 4:
        args_list = [(tau, embeddings, positive_pairs, negative_pairs) for tau in tau_candidates]
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_evaluate_single_tau, args_list)
        
        best_tau = 1.0
        best_score = -np.inf
        for tau, score in results:
            if score > best_score:
                best_score = score
                best_tau = tau
    else:
        # Sequential evaluation for small cases
        best_tau = 1.0
        best_score = -np.inf
        for tau in tau_candidates:
            score = evaluate_tau(embeddings, positive_pairs, negative_pairs, tau)
            if score > best_score:
                best_score = score
                best_tau = tau
    
    return best_tau

