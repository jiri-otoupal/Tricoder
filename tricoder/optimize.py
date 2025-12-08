"""Optimization utilities for filtering nodes and edges."""
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from .model import DEFAULT_EXCLUDED_KEYWORDS


def is_generic_name(name: str, kind: str) -> bool:
    """Check if a symbol name is too generic to be useful."""
    name_lower = name.lower().strip()
    
    # Single character names (except for classes/functions which might be intentionally short)
    if len(name_lower) <= 1 and kind in ['var', 'import']:
        return True
    
    # Very short names for variables
    if len(name_lower) <= 2 and kind == 'var':
        return True
    
    # Common generic names
    generic_names = {
        'var', 'variable', 'val', 'value', 'item', 'obj', 'object', 'data', 'result',
        'temp', 'tmp', 'arg', 'args', 'kwarg', 'kwargs', 'param', 'params', 'elem',
        'element', 'entry', 'record', 'row', 'col', 'column', 'idx', 'index', 'i', 'j', 'k',
        'x', 'y', 'z', 'n', 'm', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'a', 'b', 'c', 'd',
        'e', 'f', 'g', 'h', 'l', 'o', 'helper', 'util', 'utils', 'func', 'fn', 'cb', 'callback'
    }
    
    if name_lower in generic_names:
        return True
    
    return False


def optimize_nodes_and_edges(
    nodes_path: str,
    edges_path: str,
    types_path: str = None,
    output_nodes: str = None,
    output_edges: str = None,
    output_types: str = None,
    min_edge_weight: float = 0.3,
    remove_isolated: bool = True,
    remove_generic_names: bool = True,
    excluded_keywords: Set[str] = None
) -> Tuple[int, int, int]:
    """
    Optimize nodes and edges by filtering out low-value entries.
    
    Args:
        nodes_path: Path to input nodes.jsonl
        edges_path: Path to input edges.jsonl
        types_path: Path to input types.jsonl (optional)
        output_nodes: Path to output nodes.jsonl (default: overwrites input)
        output_edges: Path to output edges.jsonl (default: overwrites input)
        output_types: Path to output types.jsonl (default: overwrites input)
        min_edge_weight: Minimum edge weight to keep
        remove_isolated: Whether to remove nodes with no edges
        remove_generic_names: Whether to remove nodes with generic names
        excluded_keywords: Additional keywords to exclude
    
    Returns:
        Tuple of (nodes_removed, edges_removed, types_removed)
    """
    if excluded_keywords is None:
        excluded_keywords = DEFAULT_EXCLUDED_KEYWORDS
    
    # Load all data
    nodes = []
    node_ids = set()
    
    with open(nodes_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            node = json.loads(line)
            nodes.append(node)
            node_ids.add(node['id'])
    
    edges = []
    with open(edges_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            edge = json.loads(line)
            edges.append(edge)
    
    types = []
    if types_path:
        try:
            with open(types_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    type_entry = json.loads(line)
                    types.append(type_entry)
        except FileNotFoundError:
            types = []
    
    original_node_count = len(nodes)
    original_edge_count = len(edges)
    original_type_count = len(types)
    
    # Build edge statistics
    node_edge_count = defaultdict(int)
    node_in_degree = defaultdict(int)
    node_out_degree = defaultdict(int)
    
    for edge in edges:
        src = edge['src']
        dst = edge['dst']
        weight = float(edge.get('weight', 1.0))
        
        if src in node_ids and dst in node_ids:
            node_edge_count[src] += 1
            node_edge_count[dst] += 1
            node_out_degree[src] += 1
            node_in_degree[dst] += 1
    
    # Filter nodes
    nodes_to_keep = set()
    nodes_to_remove = set()
    
    for node in nodes:
        node_id = node['id']
        name = node.get('name', '')
        kind = node.get('kind', '')
        
        # Check if node should be removed
        should_remove = False
        
        # Remove if name is in excluded keywords
        if name.lower() in excluded_keywords:
            should_remove = True
        
        # Remove generic names
        if remove_generic_names and is_generic_name(name, kind):
            should_remove = True
        
        # Remove isolated nodes (no edges)
        if remove_isolated and node_edge_count[node_id] == 0:
            # Keep file nodes and important structural nodes
            if kind not in ['file', 'class', 'function']:
                should_remove = True
        
        if should_remove:
            nodes_to_remove.add(node_id)
        else:
            nodes_to_keep.add(node_id)
    
    # Filter edges
    edges_to_keep = []
    for edge in edges:
        src = edge['src']
        dst = edge['dst']
        weight = float(edge.get('weight', 1.0))
        
        # Remove if either node is removed
        if src in nodes_to_remove or dst in nodes_to_remove:
            continue
        
        # Remove low-weight edges
        if weight < min_edge_weight:
            continue
        
        edges_to_keep.append(edge)
    
    # Filter types (only keep types for nodes that are kept)
    types_to_keep = []
    for type_entry in types:
        symbol_id = type_entry.get('symbol', '')
        if symbol_id in nodes_to_keep:
            types_to_keep.append(type_entry)
    
    # Filter nodes (only keep nodes that are kept)
    filtered_nodes = [node for node in nodes if node['id'] in nodes_to_keep]
    
    # Write optimized files
    output_nodes_path = output_nodes or nodes_path
    output_edges_path = output_edges or edges_path
    output_types_path = output_types or (types_path if types_path else None)
    
    with open(output_nodes_path, 'w') as f:
        for node in filtered_nodes:
            f.write(json.dumps(node) + '\n')
    
    with open(output_edges_path, 'w') as f:
        for edge in edges_to_keep:
            f.write(json.dumps(edge) + '\n')
    
    if output_types_path and types_to_keep:
        with open(output_types_path, 'w') as f:
            for type_entry in types_to_keep:
                f.write(json.dumps(type_entry) + '\n')
    
    nodes_removed = original_node_count - len(filtered_nodes)
    edges_removed = original_edge_count - len(edges_to_keep)
    types_removed = original_type_count - len(types_to_keep)
    
    # Calculate detailed statistics
    stats = {
        'original': {
            'nodes': original_node_count,
            'edges': original_edge_count,
            'types': original_type_count
        },
        'final': {
            'nodes': len(filtered_nodes),
            'edges': len(edges_to_keep),
            'types': len(types_to_keep)
        },
        'removed': {
            'nodes': nodes_removed,
            'edges': edges_removed,
            'types': types_removed
        },
        'removal_reasons': {
            'excluded_keywords': 0,
            'generic_names': 0,
            'isolated': 0,
            'low_weight_edges': 0,
            'orphaned_edges': 0  # Edges removed because nodes were removed
        },
        'by_kind': defaultdict(lambda: {'original': 0, 'removed': 0, 'final': 0})
    }
    
    # Count removal reasons
    for node in nodes:
        node_id = node['id']
        name = node.get('name', '')
        kind = node.get('kind', '')
        
        stats['by_kind'][kind]['original'] += 1
        
        if node_id in nodes_to_remove:
            stats['by_kind'][kind]['removed'] += 1
            if name.lower() in excluded_keywords:
                stats['removal_reasons']['excluded_keywords'] += 1
            if remove_generic_names and is_generic_name(name, kind):
                stats['removal_reasons']['generic_names'] += 1
            if remove_isolated and node_edge_count[node_id] == 0:
                stats['removal_reasons']['isolated'] += 1
        else:
            stats['by_kind'][kind]['final'] += 1
    
    # Count edge removal reasons
    orphaned_edges = 0
    low_weight_edges = 0
    for edge in edges:
        src = edge['src']
        dst = edge['dst']
        weight = float(edge.get('weight', 1.0))
        
        if src in nodes_to_remove or dst in nodes_to_remove:
            orphaned_edges += 1
        elif weight < min_edge_weight:
            low_weight_edges += 1
    
    stats['removal_reasons']['orphaned_edges'] = orphaned_edges
    stats['removal_reasons']['low_weight_edges'] = low_weight_edges
    
    return nodes_removed, edges_removed, types_removed, stats

