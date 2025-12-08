"""Multi-stage retrieval pipeline with hybrid indexing for TriCoder."""
import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from .subtoken_utils import extract_subtokens


class LexicalIndex:
    """Inverted index for lexical retrieval over symbol names, subtokens, and type tokens."""
    
    def __init__(self):
        self.token_to_symbols: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_to_tokens: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_metadata: Dict[str, Dict] = {}
    
    def add_symbol(self, symbol_id: str, name: str, subtokens: List[str], 
                   type_tokens: List[str], metadata: Dict):
        """Add a symbol to the lexical index."""
        self.symbol_metadata[symbol_id] = metadata
        
        # Index name (full and normalized)
        name_lower = name.lower()
        self.token_to_symbols[name_lower].add(symbol_id)
        self.symbol_to_tokens[symbol_id].add(name_lower)
        
        # Index subtokens
        for subtoken in subtokens:
            if len(subtoken) > 1:  # Skip single characters
                self.token_to_symbols[subtoken].add(symbol_id)
                self.symbol_to_tokens[symbol_id].add(subtoken)
        
        # Index type tokens
        for type_token in type_tokens:
            type_lower = type_token.lower()
            self.token_to_symbols[type_lower].add(symbol_id)
            self.symbol_to_tokens[symbol_id].add(type_lower)
    
    def search(self, query_tokens: List[str], top_k: int = 1000) -> List[Tuple[str, float]]:
        """
        Search for symbols matching query tokens.
        
        Returns:
            List of (symbol_id, lexical_score) tuples, sorted by score descending
        """
        if not query_tokens:
            return []
        
        # Count matches per symbol
        symbol_scores: Dict[str, float] = defaultdict(float)
        
        for token in query_tokens:
            token_lower = token.lower()
            matching_symbols = self.token_to_symbols.get(token_lower, set())
            
            for symbol_id in matching_symbols:
                # Score based on match type
                symbol_tokens = self.symbol_to_tokens.get(symbol_id, set())
                
                # Exact name match gets highest score
                meta = self.symbol_metadata.get(symbol_id, {})
                name = meta.get('name', '').lower()
                if token_lower == name:
                    symbol_scores[symbol_id] += 10.0
                elif name.startswith(token_lower):
                    symbol_scores[symbol_id] += 5.0
                elif token_lower in name:
                    symbol_scores[symbol_id] += 3.0
                elif token_lower in symbol_tokens:
                    symbol_scores[symbol_id] += 1.0
        
        # Normalize scores by query length
        if query_tokens:
            for symbol_id in symbol_scores:
                symbol_scores[symbol_id] /= len(query_tokens)
        
        # Sort by score and return top_k
        sorted_results = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def save(self, filepath: str):
        """Save lexical index to disk."""
        data = {
            'token_to_symbols': {k: list(v) for k, v in self.token_to_symbols.items()},
            'symbol_to_tokens': {k: list(v) for k, v in self.symbol_to_tokens.items()},
            'symbol_metadata': self.symbol_metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load lexical index from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.token_to_symbols = {k: set(v) for k, v in data.get('token_to_symbols', {}).items()}
        self.symbol_to_tokens = {k: set(v) for k, v in data.get('symbol_to_tokens', {}).items()}
        self.symbol_metadata = data.get('symbol_metadata', {})


class DenseRetriever:
    """FAISS-based dense retrieval using HNSW index."""
    
    def __init__(self, embedding_dim: int, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        
        # FAISS is the default - warn if not available
        if use_faiss and not FAISS_AVAILABLE:
            import warnings
            warnings.warn(
                "FAISS requested but not available. Install with: pip install faiss-cpu (or faiss-gpu). "
                "Falling back to Annoy.",
                UserWarning
            )
        
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.idx_to_node_id: Dict[int, str] = {}
        self.node_id_to_idx: Dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None
        
        if not self.use_faiss:
            # Fallback to Annoy if FAISS not available
            from annoy import AnnoyIndex
            self.annoy_index = AnnoyIndex(embedding_dim, 'angular')
            self.annoy_built = False
    
    def build_index(self, embeddings: np.ndarray, node_ids: List[str]):
        """
        Build FAISS HNSW index or Annoy fallback.
        
        Args:
            embeddings: normalized embedding matrix (n_nodes, embedding_dim)
            node_ids: list of node IDs corresponding to embeddings
        """
        self.embeddings = embeddings
        num_nodes = len(node_ids)
        
        # Build mapping
        for idx, node_id in enumerate(node_ids):
            self.idx_to_node_id[idx] = node_id
            self.node_id_to_idx[node_id] = idx
        
        if self.use_faiss:
            # Build FAISS HNSW index
            # M=32, efConstruction=200 for good quality/speed tradeoff
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64  # efSearch for query time
            
            # Convert to float32 and normalize embeddings for cosine similarity
            embeddings_f32 = embeddings.astype('float32')
            faiss.normalize_L2(embeddings_f32)
            index.add(embeddings_f32)
            
            self.index = index
        else:
            # Fallback to Annoy
            for idx in range(num_nodes):
                self.annoy_index.add_item(idx, embeddings[idx])
            self.annoy_index.build(50)  # 50 trees for good quality
            self.annoy_built = True
    
    def search(self, query_vector: np.ndarray, top_k: int = 100, 
               candidate_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_vector: normalized query embedding (embedding_dim,)
            candidate_ids: optional set of candidate symbol IDs to restrict search
            
        Returns:
            List of (symbol_id, distance) tuples, sorted by distance ascending
        """
        if self.use_faiss and self.index is not None:
            # FAISS search
            query_norm = query_vector.copy().astype('float32')
            faiss.normalize_L2(query_norm.reshape(1, -1))
            
            # Increase efSearch for better recall
            if hasattr(self.index.hnsw, 'efSearch'):
                original_ef = self.index.hnsw.efSearch
                self.index.hnsw.efSearch = min(128, top_k * 4)
            
            distances, indices = self.index.search(query_norm, top_k * 2)  # Get more for filtering
            
            # Restore efSearch
            if hasattr(self.index.hnsw, 'efSearch'):
                self.index.hnsw.efSearch = original_ef
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.idx_to_node_id):
                    continue
                node_id = self.idx_to_node_id.get(idx)
                if node_id is None:
                    continue
                if candidate_ids and node_id not in candidate_ids:
                    continue
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1.0 - dist
                results.append((node_id, similarity))
            
            return results[:top_k]
        else:
            # Annoy fallback
            if not self.annoy_built:
                return []
            
            indices, distances = self.annoy_index.get_nns_by_vector(
                query_vector, top_k * 2, include_distances=True
            )
            
            results = []
            for idx, dist in zip(indices, distances):
                node_id = self.idx_to_node_id.get(idx)
                if node_id is None:
                    continue
                if candidate_ids and node_id not in candidate_ids:
                    continue
                similarity = 1.0 - dist
                results.append((node_id, similarity))
            
            return results[:top_k]
    
    def save(self, filepath_prefix: str):
        """Save index to disk."""
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, f"{filepath_prefix}.faiss")
            # Save mappings
            with open(f"{filepath_prefix}.mapping.json", 'w') as f:
                json.dump({
                    'idx_to_node_id': self.idx_to_node_id,
                    'node_id_to_idx': self.node_id_to_idx,
                    'embedding_dim': self.embedding_dim
                }, f, indent=2)
        else:
            # Annoy fallback
            self.annoy_index.save(f"{filepath_prefix}.annoy")
            with open(f"{filepath_prefix}.mapping.json", 'w') as f:
                json.dump({
                    'idx_to_node_id': self.idx_to_node_id,
                    'node_id_to_idx': self.node_id_to_idx,
                    'embedding_dim': self.embedding_dim
                }, f, indent=2)
    
    def load(self, filepath_prefix: str, embeddings: Optional[np.ndarray] = None):
        """Load index from disk. Always prefers FAISS over Annoy."""
        mapping_path = f"{filepath_prefix}.mapping.json"
        if not os.path.exists(mapping_path):
            return False
        
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        self.idx_to_node_id = {int(k): v for k, v in mapping_data['idx_to_node_id'].items()}
        self.node_id_to_idx = {k: int(v) for k, v in mapping_data['node_id_to_idx'].items()}
        self.embedding_dim = mapping_data['embedding_dim']
        
        # Always try FAISS first (preferred)
        faiss_path = f"{filepath_prefix}.faiss"
        if os.path.exists(faiss_path):
            if FAISS_AVAILABLE:
                self.use_faiss = True
                self.index = faiss.read_index(faiss_path)
                if embeddings is not None:
                    self.embeddings = embeddings
                return True
            else:
                # FAISS file exists but library not available - warn but continue
                import warnings
                warnings.warn(f"FAISS index found at {faiss_path} but FAISS library not available. "
                            f"Install with: pip install faiss-cpu (or faiss-gpu)")
        
        # Fallback to Annoy only if FAISS not available
        annoy_path = f"{filepath_prefix}.annoy"
        if os.path.exists(annoy_path):
            self.use_faiss = False
            from annoy import AnnoyIndex
            self.annoy_index = AnnoyIndex(self.embedding_dim, 'angular')
            self.annoy_index.load(annoy_path)
            self.annoy_built = True
            if embeddings is not None:
                self.embeddings = embeddings
            return True
        
        return False


class TypeEmbeddingRetriever:
    """Type-aware retrieval using type token embeddings."""
    
    def __init__(self, embeddings: np.ndarray, node_types: Dict[str, Dict[str, int]],
                 type_token_map: Dict[str, int], node_map: Dict[str, int]):
        """
        Initialize type embedding retriever.
        
        Args:
            embeddings: full embedding matrix
            node_types: mapping from node_id to {type_token: count}
            type_token_map: mapping from type_token to embedding index
            node_map: mapping from node_id to embedding index
        """
        self.embeddings = embeddings
        self.node_types = node_types
        self.type_token_map = type_token_map
        self.node_map = node_map
        self.type_embeddings: Dict[str, np.ndarray] = {}
        self._compute_type_embeddings()
    
    def _compute_type_embeddings(self):
        """Pre-compute type embeddings for each symbol."""
        for node_id, type_counts in self.node_types.items():
            if node_id not in self.node_map:
                continue
            
            type_vectors = []
            total_count = sum(type_counts.values())
            
            if total_count == 0:
                continue
            
            for type_token, count in type_counts.items():
                if type_token in self.type_token_map:
                    type_idx = self.type_token_map[type_token]
                    if type_idx < len(self.embeddings):
                        # Weight by count
                        weight = count / total_count
                        type_vectors.append(self.embeddings[type_idx] * weight)
            
            if type_vectors:
                # Average weighted type embeddings
                type_embedding = np.mean(type_vectors, axis=0)
                # Normalize
                norm = np.linalg.norm(type_embedding)
                if norm > 1e-10:
                    type_embedding = type_embedding / norm
                self.type_embeddings[node_id] = type_embedding
    
    def compute_type_similarity(self, query_node_id: str, candidate_node_id: str) -> float:
        """Compute type similarity between query and candidate."""
        query_type_emb = self.type_embeddings.get(query_node_id)
        candidate_type_emb = self.type_embeddings.get(candidate_node_id)
        
        if query_type_emb is None or candidate_type_emb is None:
            return 0.0
        
        return float(np.dot(query_type_emb, candidate_type_emb))


class HybridScorer:
    """Hybrid scoring combining dense, lexical, type, and namespace signals."""
    
    def __init__(self, 
                 dense_weight: float = 0.5,
                 lexical_weight: float = 0.2,
                 type_weight: float = 0.15,
                 namespace_weight: float = 0.15):
        """
        Initialize hybrid scorer with weights.
        
        Default weights optimized for code search:
        - Dense: 0.5 (semantic similarity is primary)
        - Lexical: 0.2 (exact matches matter)
        - Type: 0.15 (type compatibility matters)
        - Namespace: 0.15 (context matters)
        """
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.type_weight = type_weight
        self.namespace_weight = namespace_weight
    
    def score(self, dense_score: float, lexical_score: float, 
              type_score: float, namespace_score: float) -> float:
        """Compute hybrid score from component scores."""
        return (self.dense_weight * dense_score +
                self.lexical_weight * lexical_score +
                self.type_weight * type_score +
                self.namespace_weight * namespace_score)


class NamespaceRouter:
    """Contextual routing based on file hierarchy."""
    
    def __init__(self, node_file_info: Dict[str, Tuple[str, str, str]]):
        """
        Initialize namespace router.
        
        Args:
            node_file_info: mapping from node_id to (file_name, directory_path, top_level_package)
        """
        self.node_file_info = node_file_info
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Build hierarchy indices for fast lookup."""
        self.nodes_by_file: Dict[str, Set[str]] = defaultdict(set)
        self.nodes_by_directory: Dict[str, Set[str]] = defaultdict(set)
        self.nodes_by_package: Dict[str, Set[str]] = defaultdict(set)
        
        for node_id, (file_name, directory_path, package) in self.node_file_info.items():
            if file_name:
                self.nodes_by_file[file_name].add(node_id)
            if directory_path:
                self.nodes_by_directory[directory_path].add(node_id)
            if package:
                self.nodes_by_package[package].add(node_id)
    
    def compute_namespace_score(self, query_node_id: str, candidate_node_id: str) -> float:
        """
        Compute namespace proximity score (0.0 to 1.0).
        
        Returns:
            Score based on file hierarchy proximity
        """
        query_info = self.node_file_info.get(query_node_id)
        candidate_info = self.node_file_info.get(candidate_node_id)
        
        if not query_info or not candidate_info:
            return 0.0
        
        query_file, query_dir, query_pkg = query_info
        candidate_file, candidate_dir, candidate_pkg = candidate_info
        
        # Same file: highest score
        if query_file == candidate_file and query_file:
            return 1.0
        
        # Same directory: high score
        if query_dir == candidate_dir and query_dir:
            return 0.7
        
        # Same package: medium score
        if query_pkg == candidate_pkg and query_pkg:
            return 0.4
        
        # Shared directory path components: low score
        if query_dir and candidate_dir:
            query_parts = query_dir.split('/')
            candidate_parts = candidate_dir.split('/')
            common_depth = 0
            for qp, cp in zip(query_parts, candidate_parts):
                if qp == cp:
                    common_depth += 1
                else:
                    break
            if common_depth > 0:
                return 0.2 * (common_depth / max(len(query_parts), len(candidate_parts)))
        
        return 0.0


class LightweightReranker:
    """Simple cosine-based re-ranker."""
    
    def __init__(self, embeddings: np.ndarray, node_map: Dict[str, int]):
        self.embeddings = embeddings
        self.node_map = node_map
    
    def rerank(self, candidates: List[Tuple[str, float]], query_vector: np.ndarray,
               top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using cosine similarity.
        
        Args:
            candidates: list of (node_id, score) tuples
            query_vector: normalized query embedding
            top_k: number of top results to return
            
        Returns:
            Re-ranked list of (node_id, score) tuples
        """
        reranked = []
        for node_id, _ in candidates:
            if node_id not in self.node_map:
                continue
            idx = self.node_map[node_id]
            candidate_vec = self.embeddings[idx]
            cosine_sim = float(np.dot(query_vector, candidate_vec))
            reranked.append((node_id, cosine_sim))
        
        # Sort by cosine similarity
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


class CrossEncoderReranker:
    """Cross-encoder for final re-ranking using (query, candidate) pairs."""
    
    def __init__(self, embeddings: np.ndarray, node_map: Dict[str, int],
                 metadata_lookup: Dict[str, Dict]):
        """
        Initialize cross-encoder.
        
        Args:
            embeddings: embedding matrix
            node_map: node_id to index mapping
            metadata_lookup: node_id to metadata mapping
        """
        self.embeddings = embeddings
        self.node_map = node_map
        self.metadata_lookup = metadata_lookup
    
    def _get_symbol_snippet(self, node_id: str) -> str:
        """Get text snippet for a symbol (name + kind + type info)."""
        meta = self.metadata_lookup.get(node_id, {})
        name = meta.get('name', '')
        kind = meta.get('kind', '')
        
        # Add type information if available
        meta_dict = meta.get('meta', {})
        type_info = ''
        if isinstance(meta_dict, dict):
            typing = meta_dict.get('typing', [])
            if typing:
                type_info = ' '.join(str(t) for t in typing[:3])  # Limit to 3 types
        
        return f"{kind} {name} {type_info}".strip()
    
    def rerank(self, candidates: List[Tuple[str, float]], query_node_id: str,
               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using cross-encoder scoring.
        
        Args:
            candidates: list of (node_id, score) tuples
            query_node_id: query symbol ID
            top_k: number of top results to return
            
        Returns:
            Re-ranked list of (node_id, score) tuples
        """
        if query_node_id not in self.node_map:
            return candidates[:top_k]
        
        query_idx = self.node_map[query_node_id]
        query_vec = self.embeddings[query_idx]
        query_snippet = self._get_symbol_snippet(query_node_id)
        
        reranked = []
        for node_id, base_score in candidates:
            if node_id not in self.node_map:
                continue
            
            candidate_idx = self.node_map[node_id]
            candidate_vec = self.embeddings[candidate_idx]
            candidate_snippet = self._get_symbol_snippet(node_id)
            
            # Cross-encoder score: combination of embedding similarity and text matching
            embedding_sim = float(np.dot(query_vec, candidate_vec))
            
            # Text matching score (simple word overlap)
            query_words = set(query_snippet.lower().split())
            candidate_words = set(candidate_snippet.lower().split())
            if query_words and candidate_words:
                text_overlap = len(query_words & candidate_words) / len(query_words | candidate_words)
            else:
                text_overlap = 0.0
            
            # Combined score: 70% embedding, 30% text
            cross_score = 0.7 * embedding_sim + 0.3 * text_overlap
            
            # Combine with base score (weighted average)
            final_score = 0.6 * cross_score + 0.4 * base_score
            reranked.append((node_id, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


class MultiStageRetrievalPipeline:
    """Four-stage retrieval cascade with hybrid indexing."""
    
    def __init__(self,
                 lexical_index: LexicalIndex,
                 dense_retriever: DenseRetriever,
                 type_retriever: Optional[TypeEmbeddingRetriever],
                 namespace_router: Optional[NamespaceRouter],
                 hybrid_scorer: HybridScorer,
                 lightweight_reranker: LightweightReranker,
                 cross_encoder: CrossEncoderReranker,
                 stage1_top_k: int = 500,
                 stage2_top_k: int = 200,
                 stage3_top_k: int = 50,
                 stage4_top_k: int = 10):
        """
        Initialize multi-stage retrieval pipeline.
        
        Args:
            stage1_top_k: lexical filter candidate count
            stage2_top_k: dense retrieval candidate count
            stage3_top_k: lightweight reranker candidate count
            stage4_top_k: final cross-encoder output count
        """
        self.lexical_index = lexical_index
        self.dense_retriever = dense_retriever
        self.type_retriever = type_retriever
        self.namespace_router = namespace_router
        self.hybrid_scorer = hybrid_scorer
        self.lightweight_reranker = lightweight_reranker
        self.cross_encoder = cross_encoder
        
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self.stage3_top_k = stage3_top_k
        self.stage4_top_k = stage4_top_k
    
    def retrieve(self, query_node_id: str, query_vector: np.ndarray,
                 query_text: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Execute four-stage retrieval pipeline.
        
        Args:
            query_node_id: query symbol ID
            query_vector: normalized query embedding vector
            query_text: optional query text for lexical matching
            
        Returns:
            List of (symbol_id, final_score) tuples
        """
        # Stage 1: Lexical Filter
        if query_text:
            query_tokens = extract_subtokens(query_text, normalize=True)[1]  # Get normalized
        else:
            # Extract from query node metadata if available
            query_tokens = []
            if hasattr(self.lexical_index, 'symbol_to_tokens'):
                query_tokens = list(self.lexical_index.symbol_to_tokens.get(query_node_id, set()))
        
        lexical_candidates = self.lexical_index.search(query_tokens, top_k=self.stage1_top_k)
        lexical_candidate_ids = {node_id for node_id, _ in lexical_candidates}
        lexical_scores = {node_id: score for node_id, score in lexical_candidates}
        
        # If no lexical candidates, use all symbols (fallback)
        if not lexical_candidate_ids:
            lexical_candidate_ids = None
        
        # Stage 2: Dense ANN Search
        dense_results = self.dense_retriever.search(
            query_vector, 
            top_k=self.stage2_top_k,
            candidate_ids=lexical_candidate_ids
        )
        dense_scores = {node_id: score for node_id, score in dense_results}
        
        # Combine lexical and dense scores with hybrid scoring
        combined_candidates = []
        all_candidate_ids = set(dense_scores.keys())
        if lexical_candidate_ids:
            all_candidate_ids.update(lexical_candidate_ids)
        
        for node_id in all_candidate_ids:
            dense_score = dense_scores.get(node_id, 0.0)
            lexical_score = lexical_scores.get(node_id, 0.0)
            
            # Normalize scores to [0, 1]
            dense_score_norm = max(0.0, min(1.0, dense_score))
            lexical_score_norm = max(0.0, min(1.0, lexical_score / 10.0))  # Normalize lexical (max ~10)
            
            # Get type and namespace scores
            type_score = 0.0
            if self.type_retriever:
                type_score = self.type_retriever.compute_type_similarity(query_node_id, node_id)
            
            namespace_score = 0.0
            if self.namespace_router:
                namespace_score = self.namespace_router.compute_namespace_score(query_node_id, node_id)
            
            # Hybrid score
            hybrid_score = self.hybrid_scorer.score(
                dense_score_norm, lexical_score_norm, type_score, namespace_score
            )
            
            combined_candidates.append((node_id, hybrid_score))
        
        # Sort by hybrid score
        combined_candidates.sort(key=lambda x: x[1], reverse=True)
        stage2_candidates = combined_candidates[:self.stage2_top_k]
        
        # Stage 3: Lightweight Re-Ranker
        stage3_candidates = self.lightweight_reranker.rerank(
            stage2_candidates, query_vector, top_k=self.stage3_top_k
        )
        
        # Stage 4: Cross-Encoder Re-Ranker
        final_results = self.cross_encoder.rerank(
            stage3_candidates, query_node_id, top_k=self.stage4_top_k
        )
        
        return final_results

