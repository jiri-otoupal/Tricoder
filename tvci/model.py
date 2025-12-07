"""SymbolModel: main model class for loading and querying."""
import numpy as np
import json
import os
from typing import List, Dict, Optional
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
        
        # Load ANN index
        self.ann_index = AnnoyIndex(self.embedding_dim, 'angular')
        self.ann_index.load(os.path.join(model_dir, 'ann_index.ann'))
    
    def query(self, node_id: str, top_k: int = 10) -> List[Dict]:
        """
        Query for similar symbols.
        
        Args:
            node_id: symbol ID to query
            top_k: number of results to return
        
        Returns:
            List of result dictionaries with symbol, score, distance, meta
        """
        if node_id not in self.node_map:
            return []
        
        node_idx = self.node_map[node_id]
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
            
            # Compute calibrated score
            score = np.dot(query_vector, self.embeddings[idx]) / self.tau
            
            # Get metadata
            meta = self.metadata_lookup.get(node_id_result)
            
            results.append({
                'symbol': node_id_result,
                'score': float(score),
                'distance': float(dist),
                'meta': meta
            })
            
            if len(results) >= top_k:
                break
        
        return results

