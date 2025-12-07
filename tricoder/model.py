"""SymbolModel: main model class for loading and querying."""
import json
import os
from typing import List, Dict

import numpy as np
from annoy import AnnoyIndex
from gensim.models.keyedvectors import KeyedVectors


class SymbolModel:
    """Main model class for TriVector Code Intelligence."""

    def __init__(self):
        self.embeddings = None
        self.tau = None
        self.node_map = None
        self.node_metadata = None
        self.pca_components = None
        self.pca_mean = None
        self.svd_components_graph = None
        self.svd_components_types = None
        self.word2vec_kv = None
        self.ann_index = None
        self.type_token_map = None
        self.embedding_dim = None
        self.idx_to_node = None
        self.metadata_lookup = None
        self.mean_norm = None
        self.subtoken_to_idx = None
        self.node_subtokens = None
        self.node_types = None
        self.alpha = 0.05  # Length penalty coefficient

    def load(self, model_dir: str):
        """
        Load model from directory.
        
        Args:
            model_dir: path to model directory
        """
        # Load embeddings
        self.embeddings = np.load(os.path.join(model_dir, 'embeddings.npy'))
        self.embedding_dim = self.embeddings.shape[1]

        # Load temperature
        self.tau = float(np.load(os.path.join(model_dir, 'tau.npy')))

        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            self.node_map = metadata['node_map']
            self.node_metadata = metadata['node_metadata']

        # Create reverse mapping for efficient lookup
        self.idx_to_node = {idx: node_id for node_id, idx in self.node_map.items()}

        # Create metadata lookup
        self.metadata_lookup = {nm['id']: nm for nm in self.node_metadata}

        # Load PCA components
        self.pca_components = np.load(os.path.join(model_dir, 'fusion_pca_components.npy'))
        self.pca_mean = np.load(os.path.join(model_dir, 'fusion_pca_mean.npy'))

        # Load SVD components
        self.svd_components_graph = np.load(os.path.join(model_dir, 'svd_components.npy'))

        # Load type SVD components if available
        types_svd_path = os.path.join(model_dir, 'svd_components_types.npy')
        if os.path.exists(types_svd_path):
            self.svd_components_types = np.load(types_svd_path)

        # Load Word2Vec
        w2v_path = os.path.join(model_dir, 'word2vec.kv')
        if os.path.exists(w2v_path):
            self.word2vec_kv = KeyedVectors.load(w2v_path, mmap='r')

        # Load type token map if available
        type_map_path = os.path.join(model_dir, 'type_token_map.json')
        if os.path.exists(type_map_path):
            with open(type_map_path, 'r') as f:
                self.type_token_map = json.load(f)

        # Load mean_norm for length penalty
        mean_norm_path = os.path.join(model_dir, 'mean_norm.npy')
        if os.path.exists(mean_norm_path):
            self.mean_norm = float(np.load(mean_norm_path))
        else:
            # Compute from embeddings if not saved
            if self.embeddings is not None:
                self.mean_norm = float(np.mean(np.linalg.norm(self.embeddings, axis=1)))
            else:
                self.mean_norm = 1.0

        # Load subtoken mapping if available
        subtoken_map_path = os.path.join(model_dir, 'subtoken_map.json')
        if os.path.exists(subtoken_map_path):
            with open(subtoken_map_path, 'r') as f:
                self.subtoken_to_idx = json.load(f)
        else:
            self.subtoken_to_idx = {}

        # Load node subtokens if available
        node_subtokens_path = os.path.join(model_dir, 'node_subtokens.json')
        if os.path.exists(node_subtokens_path):
            with open(node_subtokens_path, 'r') as f:
                self.node_subtokens = json.load(f)
        else:
            self.node_subtokens = {}

        # Load node types if available
        node_types_path = os.path.join(model_dir, 'node_types.json')
        if os.path.exists(node_types_path):
            with open(node_types_path, 'r') as f:
                self.node_types = json.load(f)
        else:
            self.node_types = {}

        # Load ANN index
        self.ann_index = AnnoyIndex(self.embedding_dim, 'angular')
        self.ann_index.load(os.path.join(model_dir, 'ann_index.ann'))

    def expand_query_vector(self, node_id: str) -> np.ndarray:
        """
        Expand query vector with subtokens and type tokens.
        
        Args:
            node_id: symbol ID to query
        
        Returns:
            Expanded and normalized query vector
        """
        if node_id not in self.node_map:
            return None

        node_idx = self.node_map[node_id]
        base_vector = self.embeddings[node_idx]

        vectors_to_average = [base_vector]  # Start with symbol vector (weight 1.0)
        weights = [1.0]

        # Add subtoken vectors (weight 0.6)
        if self.node_subtokens and node_id in self.node_subtokens:
            subtoken_vectors = []
            for subtoken in self.node_subtokens[node_id]:
                if self.subtoken_to_idx and subtoken in self.subtoken_to_idx:
                    subtoken_idx = self.subtoken_to_idx[subtoken]
                    if subtoken_idx < len(self.embeddings):
                        subtoken_vectors.append(self.embeddings[subtoken_idx])

            if subtoken_vectors:
                subtoken_avg = np.mean(subtoken_vectors, axis=0)
                vectors_to_average.append(subtoken_avg)
                weights.append(0.6)

        # Add type token vectors (weight 0.4)
        if self.node_types and node_id in self.node_types:
            type_vectors = []
            for type_token, count in self.node_types[node_id].items():
                if self.type_token_map and type_token in self.type_token_map:
                    type_idx = self.type_token_map[type_token]
                    # Type tokens might be in a separate space, skip for now
                    # In a full implementation, we'd need to map type embeddings

            # For now, we'll skip type expansion in query if types aren't directly mapped
            # This would require storing type embeddings separately

        # Weighted average
        if len(vectors_to_average) > 1:
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()  # Normalize weights

            expanded = np.zeros_like(base_vector)
            for vec, weight in zip(vectors_to_average, weights_array):
                expanded += weight * vec
        else:
            expanded = base_vector

        # Normalize
        norm = np.linalg.norm(expanded)
        if norm > 1e-10:
            expanded = expanded / norm

        return expanded

    def compute_hybrid_score(self, query_vec: np.ndarray, candidate_vec: np.ndarray,
                             candidate_norm_before_normalization: float = None) -> float:
        """
        Compute hybrid similarity score with length penalty.
        
        Args:
            query_vec: query embedding vector (normalized)
            candidate_vec: candidate embedding vector (normalized)
            candidate_norm_before_normalization: norm before normalization (for penalty)
        
        Returns:
            Hybrid similarity score
        """
        # Cosine similarity
        cosine_sim = np.dot(query_vec, candidate_vec)

        # Length penalty
        if candidate_norm_before_normalization is not None and self.mean_norm is not None:
            length_penalty = max(0, candidate_norm_before_normalization - self.mean_norm)
            score = cosine_sim - self.alpha * length_penalty
        else:
            score = cosine_sim

        return score

    def query(self, node_id: str, top_k: int = 10) -> List[Dict]:
        """
        Query for similar symbols with query expansion and hybrid scoring.
        
        Args:
            node_id: symbol ID to query
            top_k: number of results to return
        
        Returns:
            List of result dictionaries with symbol, score, distance, meta
        """
        if node_id not in self.node_map:
            return []

        node_idx = self.node_map[node_id]

        # Expand query vector
        query_vector = self.expand_query_vector(node_id)
        if query_vector is None:
            query_vector = self.embeddings[node_idx]

        # ANN search
        indices, distances = self.ann_index.get_nns_by_vector(
            query_vector, top_k + 1, include_distances=True
        )

        results = []
        for idx, dist in zip(indices, distances):
            # Skip self
            if idx == node_idx:
                continue

            # Find node_id for this index
            node_id_result = self.idx_to_node.get(idx)
            if node_id_result is None:
                continue

            # Compute hybrid score
            candidate_vec = self.embeddings[idx]
            hybrid_score = self.compute_hybrid_score(query_vector, candidate_vec)

            # Compute calibrated score (for probability)
            calibrated_score = hybrid_score / self.tau if self.tau else hybrid_score

            # Get metadata
            meta = self.metadata_lookup.get(node_id_result)

            results.append({
                'symbol': node_id_result,
                'score': float(calibrated_score),
                'hybrid_score': float(hybrid_score),
                'distance': float(dist),
                'meta': meta
            })

            if len(results) >= top_k:
                break

        # Sort by hybrid score (descending)
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return results
