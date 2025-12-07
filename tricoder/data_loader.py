"""Data loading utilities for TriVector Code Intelligence."""
import json
from collections import defaultdict
from typing import Dict, List, Tuple

from .subtoken_utils import extract_subtokens, get_file_hierarchy


def load_nodes(nodes_path: str) -> Tuple[
    Dict[str, int], List[Dict], Dict[str, List[str]], Dict[str, Tuple[str, str, str]]]:
    """
    Load nodes from JSONL file.
    
    Returns:
        node_to_idx: mapping from node ID to index
        node_metadata: list of node metadata dictionaries
        node_subtokens: mapping from node_id to list of normalized subtokens
        node_file_info: mapping from node_id to (file_name, directory_path, top_level_package)
    """
    node_to_idx = {}
    node_metadata = []
    node_subtokens = {}
    node_file_info = {}

    with open(nodes_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            node = json.loads(line)
            node_id = node['id']
            if node_id not in node_to_idx:
                idx = len(node_to_idx)
                node_to_idx[node_id] = idx

                # Extract name and metadata
                name = node.get('name', '')
                meta = node.get('meta', {})

                # Extract subtokens
                raw_subtokens, normalized_subtokens = extract_subtokens(name, normalize=True)
                node_subtokens[node_id] = normalized_subtokens

                # Extract file hierarchy
                file_path = meta.get('file', '') if isinstance(meta, dict) else ''
                file_name, directory_path, top_level_package = get_file_hierarchy(file_path)
                node_file_info[node_id] = (file_name, directory_path, top_level_package)

                # Store metadata with subtoken info for debugging
                meta_with_subtokens = meta.copy() if isinstance(meta, dict) else {}
                meta_with_subtokens['_raw_subtokens'] = raw_subtokens
                meta_with_subtokens['_normalized_subtokens'] = normalized_subtokens

                node_metadata.append({
                    'id': node_id,
                    'kind': node.get('kind', 'unknown'),
                    'name': name,
                    'meta': meta_with_subtokens
                })

    return node_to_idx, node_metadata, node_subtokens, node_file_info


def load_edges(edges_path: str, node_to_idx: Dict[str, int]) -> Tuple[List[Tuple[int, int, str, float]], int]:
    """
    Load edges from JSONL file.
    
    Returns:
        edges: list of (src_idx, dst_idx, relation, weight) tuples
        num_nodes: number of unique nodes
    """
    edges = []
    seen_nodes = set()

    with open(edges_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            edge = json.loads(line)
            src_id = edge['src']
            dst_id = edge['dst']

            if src_id in node_to_idx and dst_id in node_to_idx:
                src_idx = node_to_idx[src_id]
                dst_idx = node_to_idx[dst_id]
                rel = edge.get('rel', 'unknown')
                weight = float(edge.get('weight', 1.0))
                edges.append((src_idx, dst_idx, rel, weight))
                seen_nodes.add(src_idx)
                seen_nodes.add(dst_idx)

    num_nodes = len(node_to_idx)
    return edges, num_nodes


def load_types(types_path: str, node_to_idx: Dict[str, int]) -> Tuple[
    Dict[int, Dict[str, int]], Dict[str, int]]:
    """
    Load type tokens from JSONL file.
    
    Returns:
        node_types: mapping from node_idx to {type_token: count}
        type_to_idx: mapping from type token to index
    """
    node_types = defaultdict(lambda: defaultdict(int))
    type_to_idx = {}

    with open(types_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            symbol_id = entry['symbol']
            type_token = entry['type_token']
            count = int(entry.get('count', 1))

            if symbol_id in node_to_idx:
                node_idx = node_to_idx[symbol_id]
                node_types[node_idx][type_token] += count

                if type_token not in type_to_idx:
                    type_to_idx[type_token] = len(type_to_idx)

    return dict(node_types), type_to_idx


def build_node_location_map(node_metadata: List[Dict]) -> Dict[int, Tuple[str, int]]:
    """
    Build mapping from node_idx to (file_path, line_number) for context window co-occurrence.
    
    Args:
        node_metadata: list of node metadata dictionaries
    
    Returns:
        Mapping from node_idx to (file_path, line_number)
    """
    location_map = {}
    for idx, node_meta in enumerate(node_metadata):
        meta = node_meta.get('meta', {})
        if isinstance(meta, dict):
            file_path = meta.get('file', '')
            lineno = meta.get('lineno', -1)
            if file_path and lineno >= 0:
                location_map[idx] = (file_path, lineno)
    return location_map
