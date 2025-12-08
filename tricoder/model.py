"""SymbolModel: main model class for loading and querying."""
import json
import os
from typing import List, Dict, Set, Optional

import numpy as np
from annoy import AnnoyIndex
from gensim.models.keyedvectors import KeyedVectors

# Default excluded keywords: Python keywords, builtins, and common library names
# These don't provide value for code intelligence as they're language constructs
# rather than user-defined code patterns
DEFAULT_EXCLUDED_KEYWORDS: Set[str] = {
    # Python keywords
    'import', 'from', 'as', 'def', 'class', 'if', 'else', 'elif', 'for', 'while',
    'try', 'except', 'finally', 'with', 'return', 'pass', 'break', 'continue',
    'yield', 'lambda', 'del', 'global', 'nonlocal', 'assert', 'raise', 'and',
    'or', 'not', 'in', 'is', 'None', 'True', 'False',
    
    # Common builtin functions/types
    'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
    'set', 'frozenset', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
    'delattr', 'dir', 'vars', 'locals', 'globals', 'eval', 'exec', 'compile',
    'open', 'file', 'range', 'enumerate', 'zip', 'map', 'filter', 'reduce',
    'sorted', 'reversed', 'iter', 'next', 'all', 'any', 'sum', 'max', 'min',
    'abs', 'round', 'divmod', 'pow', 'bin', 'hex', 'oct', 'ord', 'chr',
    'repr', 'ascii', 'format', 'hash', 'id', 'slice', 'super', 'property',
    'staticmethod', 'classmethod', 'object', 'Exception', 'BaseException',
    
    # Common standard library module names
    'os', 'sys', 'json', 're', 'datetime', 'time', 'random', 'math', 'collections',
    'itertools', 'functools', 'operator', 'string', 'textwrap', 'unicodedata',
    'stringprep', 'readline', 'rlcompleter', 'struct', 'codecs', 'types', 'copy',
    'pprint', 'reprlib', 'enum', 'numbers', 'cmath', 'decimal', 'fractions',
    'statistics', 'array', 'bisect', 'heapq', 'weakref', 'gc', 'inspect',
    'site', 'fpectl', 'atexit', 'traceback', 'future', 'importlib', 'pkgutil',
    'modulefinder', 'runpy', 'pickle', 'copyreg', 'shelve', 'marshal', 'dbm',
    'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile', 'csv',
    'configparser', 'netrc', 'xdrlib', 'plistlib', 'hashlib', 'hmac', 'secrets',
    'io', 'argparse', 'getopt', 'logging', 'getpass', 'curses', 'platform',
    'errno', 'ctypes', 'threading', 'multiprocessing', 'concurrent', 'subprocess',
    'sched', 'queue', 'select', 'selectors', 'asyncio', 'socket', 'ssl', 'email',
    'urllib', 'http', 'html', 'xml', 'webbrowser', 'tkinter', 'turtle', 'cmd',
    'shlex', 'configparser', 'fileinput', 'linecache', 'shutil', 'tempfile',
    'glob', 'fnmatch', 'linecache', 'shutil', 'macpath', 'pathlib', 'stat',
    'filecmp', 'mmap', 'codecs', 'unicodedata', 'stringprep', 'readline',
    'rlcompleter', 'ast', 'symtable', 'symbol', 'token', 'tokenize', 'keyword',
    'parser', 'dis', 'pickletools', 'doctest', 'unittest', 'test', 'lib2to3',
    'typing', 'pydoc', 'doctest', 'unittest', 'test', 'lib2to3', 'distutils',
    'ensurepip', 'venv', 'zipapp', 'faulthandler', 'pdb', 'profile', 'pstats',
    'timeit', 'trace', 'tracemalloc', 'gc', 'inspect', 'site', 'fpectl',
    'warnings', 'contextlib', 'abc', 'atexit', 'traceback', 'future', '__future__',
    'importlib', 'pkgutil', 'modulefinder', 'runpy', 'zipimport', 'pkgutil',
    'modulefinder', 'runpy', 'zipimport', 'pkgutil', 'modulefinder', 'runpy',
    
    # Common dunder methods (though these might be useful, excluding common ones)
    '__init__', '__main__', '__name__', '__file__', '__doc__', '__package__',
    '__builtins__', '__dict__', '__class__', '__module__', '__qualname__',
    
    # Common variable names that aren't useful
    'self', 'cls', 'args', 'kwargs', 'data', 'result', 'value', 'item',
    'key', 'val', 'obj', 'instance', 'cls', 'self', 'other', 'x', 'y', 'z',
    'i', 'j', 'k', 'n', 'm', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
}


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
        self.ann_index = None  # Legacy Annoy index (kept for backward compatibility)
        self.type_token_map = None
        self.embedding_dim = None
        self.idx_to_node = None
        self.metadata_lookup = None
        self.mean_norm = None
        self.subtoken_to_idx = None
        self.node_subtokens = None
        self.node_types = None
        self.alpha = 0.05  # Length penalty coefficient
        
        # Multi-stage retrieval pipeline components
        self.lexical_index = None
        self.dense_retriever = None
        self.type_retriever = None
        self.namespace_router = None
        self.retrieval_pipeline = None
        self.use_multi_stage = False  # Flag to enable/disable multi-stage retrieval

    def load(self, model_dir: str):
        """
        Load model from directory.
        
        Args:
            model_dir: path to model directory
        """
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load embeddings
        embeddings_path = os.path.join(model_dir, 'embeddings.npy')
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                f"This usually means training was interrupted or failed before completion.\n"
                f"Please retrain the model."
            )
        self.embeddings = np.load(embeddings_path)
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

        # Load ANN index (legacy, for backward compatibility)
        ann_index_path = os.path.join(model_dir, 'ann_index.ann')
        if os.path.exists(ann_index_path):
            self.ann_index = AnnoyIndex(self.embedding_dim, 'angular')
            self.ann_index.load(ann_index_path)
        
        # Load multi-stage retrieval pipeline components
        self._load_retrieval_pipeline(model_dir)
    
    def _load_retrieval_pipeline(self, model_dir: str):
        """Load multi-stage retrieval pipeline components."""
        try:
            from .retrieval import (
                LexicalIndex, DenseRetriever, TypeEmbeddingRetriever,
                NamespaceRouter, HybridScorer, LightweightReranker,
                CrossEncoderReranker, MultiStageRetrievalPipeline
            )
            from .data_loader import get_file_hierarchy
            
            # Load lexical index
            lexical_index_path = os.path.join(model_dir, 'lexical_index.json')
            if os.path.exists(lexical_index_path):
                self.lexical_index = LexicalIndex()
                self.lexical_index.load(lexical_index_path)
            
            # Load dense retriever (FAISS preferred, Annoy fallback)
            dense_index_prefix = os.path.join(model_dir, 'dense_index')
            if os.path.exists(f"{dense_index_prefix}.mapping.json"):
                self.dense_retriever = DenseRetriever(embedding_dim=self.embedding_dim)  # use_faiss=True is default
                if not self.dense_retriever.load(dense_index_prefix, embeddings=self.embeddings):
                    self.dense_retriever = None
            
            # Build type retriever if types available
            if self.node_types and self.type_token_map:
                self.type_retriever = TypeEmbeddingRetriever(
                    self.embeddings, self.node_types, self.type_token_map, self.node_map
                )
            
            # Build namespace router
            node_file_info = {}
            for node_meta in self.node_metadata:
                node_id = node_meta['id']
                meta = node_meta.get('meta', {})
                if isinstance(meta, dict):
                    file_path = meta.get('file', '')
                    if file_path:
                        file_name, directory_path, top_level_package = get_file_hierarchy(file_path)
                        node_file_info[node_id] = (file_name, directory_path, top_level_package)
            
            if node_file_info:
                self.namespace_router = NamespaceRouter(node_file_info)
            
            # Initialize pipeline components
            if self.lexical_index and self.dense_retriever:
                hybrid_scorer = HybridScorer(
                    dense_weight=0.5,
                    lexical_weight=0.2,
                    type_weight=0.15,
                    namespace_weight=0.15
                )
                
                lightweight_reranker = LightweightReranker(self.embeddings, self.node_map)
                cross_encoder = CrossEncoderReranker(
                    self.embeddings, self.node_map, self.metadata_lookup
                )
                
                self.retrieval_pipeline = MultiStageRetrievalPipeline(
                    lexical_index=self.lexical_index,
                    dense_retriever=self.dense_retriever,
                    type_retriever=self.type_retriever,
                    namespace_router=self.namespace_router,
                    hybrid_scorer=hybrid_scorer,
                    lightweight_reranker=lightweight_reranker,
                    cross_encoder=cross_encoder,
                    stage1_top_k=500,
                    stage2_top_k=200,
                    stage3_top_k=50,
                    stage4_top_k=10
                )
                self.use_multi_stage = True
        except Exception as e:
            # Fallback to legacy retrieval if pipeline fails to load
            import warnings
            warnings.warn(f"Failed to load multi-stage retrieval pipeline: {e}. Using legacy retrieval.")
            self.use_multi_stage = False

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

    def _get_context_length(self, node_id: str) -> Optional[int]:
        """Get context length (scope size) for a symbol."""
        if not self.metadata_lookup:
            return None
        meta = self.metadata_lookup.get(node_id)
        if not meta:
            return None
        meta_dict = meta.get('meta', {})
        if not isinstance(meta_dict, dict):
            return None
        lineno = meta_dict.get('lineno', -1)
        end_lineno = meta_dict.get('end_lineno', None)
        if lineno >= 0 and end_lineno is not None and end_lineno > lineno:
            return end_lineno - lineno
        elif lineno >= 0:
            return 1  # Single-line symbol
        return None

    def _compute_context_similarity(self, len1: Optional[int], len2: Optional[int]) -> float:
        """
        Compute context length similarity score (0.0 to 1.0).
        Returns 0.0 if either length is None or if lengths are very different.
        """
        if len1 is None or len2 is None:
            return 0.0
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Similarity ratio: min/max (1.0 for identical, decreasing as difference increases)
        max_len = max(len1, len2)
        min_len = min(len1, len2)
        similarity = min_len / max_len if max_len > 0 else 0.0
        
        # Only return meaningful similarity (threshold 0.2)
        return similarity if similarity >= 0.2 else 0.0

    def compute_hybrid_score(self, query_vec: np.ndarray, candidate_vec: np.ndarray,
                             candidate_norm_before_normalization: float = None,
                             query_node_id: str = None,
                             candidate_node_id: str = None) -> float:
        """
        Compute hybrid similarity score with length penalty and context similarity bonus.
        
        Args:
            query_vec: query embedding vector (normalized)
            candidate_vec: candidate embedding vector (normalized)
            candidate_norm_before_normalization: norm before normalization (for penalty)
            query_node_id: query symbol ID (for context length comparison)
            candidate_node_id: candidate symbol ID (for context length comparison)
        
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

        # Context length similarity bonus (if both node IDs provided)
        if query_node_id and candidate_node_id:
            query_len = self._get_context_length(query_node_id)
            candidate_len = self._get_context_length(candidate_node_id)
            context_sim = self._compute_context_similarity(query_len, candidate_len)
            
            # Add bonus: up to 0.15 boost for similar context lengths
            # This helps symbols with similar scope sizes rank higher
            context_bonus = context_sim * 0.15
            score = score + context_bonus

        return score

    def query(self, node_id: str, top_k: int = 10, use_multi_stage: Optional[bool] = None) -> List[Dict]:
        """
        Query for similar symbols with query expansion and hybrid scoring.
        Uses multi-stage retrieval pipeline if available, otherwise falls back to legacy ANN search.
        
        Args:
            node_id: symbol ID to query
            top_k: number of results to return
            use_multi_stage: whether to use multi-stage retrieval (None = auto-detect)
        
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
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 1e-10:
            query_vector = query_vector / query_norm

        # Use multi-stage retrieval if available and enabled
        if use_multi_stage is None:
            use_multi_stage = self.use_multi_stage
        
        if use_multi_stage and self.retrieval_pipeline:
            return self._query_multi_stage(node_id, query_vector, top_k)
        else:
            return self._query_legacy(node_id, query_vector, node_idx, top_k)
    
    def _query_multi_stage(self, node_id: str, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        """Query using multi-stage retrieval pipeline."""
        # Get query text from metadata
        query_meta = self.metadata_lookup.get(node_id, {})
        query_name = query_meta.get('name', '')
        
        # Execute multi-stage retrieval
        pipeline_results = self.retrieval_pipeline.retrieve(
            query_node_id=node_id,
            query_vector=query_vector,
            query_text=query_name
        )
        
        # Convert to result format
        results = []
        for rank, (result_node_id, final_score) in enumerate(pipeline_results):
            if result_node_id == node_id:
                continue  # Skip self
            
            result_idx = self.node_map.get(result_node_id)
            if result_idx is None:
                continue
            
            # Compute distance for compatibility
            candidate_vec = self.embeddings[result_idx]
            candidate_norm = np.linalg.norm(candidate_vec)
            if candidate_norm > 1e-10:
                candidate_vec = candidate_vec / candidate_norm
            
            distance = 1.0 - float(np.dot(query_vector, candidate_vec))
            
            # Get metadata
            meta = self.metadata_lookup.get(result_node_id)
            
            # Apply temperature calibration
            calibrated_score = final_score / self.tau if self.tau else final_score
            
            results.append({
                'symbol': result_node_id,
                'score': float(calibrated_score),
                'hybrid_score': float(final_score),
                'distance': float(distance),
                'meta': meta
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _query_legacy(self, node_id: str, query_vector: np.ndarray, node_idx: int, top_k: int) -> List[Dict]:
        """Legacy query using Annoy index."""
        if self.ann_index is None:
            return []

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

            # Compute hybrid score with context length similarity
            candidate_vec = self.embeddings[idx]
            hybrid_score = self.compute_hybrid_score(
                query_vector, candidate_vec,
                query_node_id=node_id,
                candidate_node_id=node_id_result
            )

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

    def search_by_keywords(self, keywords: str, top_k: int = 10, 
                          excluded_keywords: Set[str] = None, case_sensitive: bool = False) -> List[Dict]:
        """
        Search for symbols by keywords (name matching).
        
        Args:
            keywords: space-separated keywords or quoted string to search for
            top_k: number of results to return
            excluded_keywords: set of keywords to exclude (defaults to DEFAULT_EXCLUDED_KEYWORDS)
            case_sensitive: if True, perform case-sensitive matching (default: False, case-insensitive)
        
        Returns:
            List of matching symbol dictionaries with symbol, score, meta
        """
        if not self.metadata_lookup:
            return []
        
        # Use default excluded keywords if not provided
        if excluded_keywords is None:
            excluded_keywords = DEFAULT_EXCLUDED_KEYWORDS
        
        # Normalize keywords based on case sensitivity
        if case_sensitive:
            keywords_normalized = keywords.strip()
            keyword_words = keywords_normalized.split()
        else:
            keywords_normalized = keywords.lower().strip()
            keyword_words = keywords_normalized.split()
        
        # Filter out excluded keywords from search query
        filtered_keyword_words = [w for w in keyword_words if w not in excluded_keywords]
        
        # If all keywords were filtered out, return empty results
        if not filtered_keyword_words:
            return []
        
        # Rebuild keywords string from filtered words
        filtered_keywords_normalized = ' '.join(filtered_keyword_words)
        
        # Get type tokens for this symbol (if available)
        def get_type_tokens(node_id: str) -> List[str]:
            """Get all type tokens for a symbol, including expanded primitives."""
            if not self.node_types or node_id not in self.node_types:
                return []
            
            type_tokens = []
            for type_token, count in self.node_types[node_id].items():
                # Add the full type token (case-sensitive or not based on flag)
                if case_sensitive:
                    type_tokens.append(type_token)
                else:
                    type_tokens.append(type_token.lower())
                
                # Also parse composite types to extract primitives (e.g., "List[bool]" -> ["bool"])
                # This allows matching "bool" when searching for "bool variable"
                if '[' in type_token and ']' in type_token:
                    # Extract content between brackets
                    start = type_token.find('[')
                    end = type_token.rfind(']')
                    if start < end:
                        inner = type_token[start+1:end].strip()
                        # Split by comma and add individual types
                        for part in inner.split(','):
                            part = part.strip()
                            if not case_sensitive:
                                part = part.lower()
                            if part and part not in type_tokens:
                                type_tokens.append(part)
            
            return type_tokens
        
        # Find matching symbols
        matches = []
        for node_id, meta in self.metadata_lookup.items():
            if not meta:
                continue
            
            # Get name and kind, normalize based on case sensitivity
            name_original = meta.get('name', '')
            kind_original = meta.get('kind', '')
            
            if case_sensitive:
                name = name_original
                kind = kind_original
            else:
                name = name_original.lower()
                kind = kind_original.lower()
            
            # Skip symbols whose names are in excluded keywords (always case-insensitive for excluded)
            if name.lower() in excluded_keywords:
                continue
            
            # Get type tokens for this symbol
            type_tokens = get_type_tokens(node_id)
            type_tokens_str = ' '.join(type_tokens)  # Combined string for matching
            
            # Calculate a simple relevance score
            score = 0.0
            
            # Check exact phrase match first (highest priority)
            if filtered_keywords_normalized == name:
                score = 1.0  # Exact name match
            elif name.startswith(filtered_keywords_normalized):
                score = 0.8  # Name starts with keywords
            elif filtered_keywords_normalized in name:
                score = 0.6  # Keywords contained in name
            # For multi-word queries, check if all words appear in name, kind, or types
            elif len(filtered_keyword_words) > 1:
                # Check if all words appear in the name
                all_words_in_name = all(word in name for word in filtered_keyword_words)
                if all_words_in_name:
                    # Count how many words match
                    matching_words = sum(1 for word in filtered_keyword_words if word in name)
                    score = 0.5 + (0.2 * matching_words / len(filtered_keyword_words))  # 0.5-0.7 range
                # Check if words match name + type (e.g., "bool variable")
                else:
                    # Try to match some words in name/kind and some in types
                    name_kind_matches = sum(1 for word in filtered_keyword_words if word in name or word in kind)
                    type_matches = sum(1 for word in filtered_keyword_words if word in type_tokens_str)
                    
                    if name_kind_matches > 0 and type_matches > 0:
                        # Combined match: name/kind + type (e.g., "bool variable")
                        score = 0.4 + (0.2 * (name_kind_matches + type_matches) / len(filtered_keyword_words))
                    elif all(word in kind for word in filtered_keyword_words):
                        score = 0.3  # All words in kind
                    elif all(word in type_tokens_str for word in filtered_keyword_words):
                        score = 0.35  # All words in types
            # Single word queries
            elif len(filtered_keyword_words) == 1:
                word = filtered_keyword_words[0]
                if word == name:
                    score = 1.0  # Exact name match
                elif name.startswith(word):
                    score = 0.8  # Name starts with word
                elif word in name:
                    score = 0.6  # Word contained in name
                elif word == kind:
                    score = 0.4  # Kind match
                elif word in kind:
                    score = 0.2  # Word in kind
                elif word in type_tokens_str:
                    score = 0.3  # Word in type tokens
            
            if score > 0:
                matches.append({
                    'symbol': node_id,
                    'score': score,
                    'meta': meta
                })
        
        # Sort by relevance score (descending)
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:top_k]