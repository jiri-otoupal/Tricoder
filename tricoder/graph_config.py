"""
Performance configuration constants for graph operations.

These constants control limits to prevent exponential explosion and ensure
reasonable training times while maintaining quality.
"""

# ============================================================================
# Call Graph Expansion Limits
# ============================================================================

# Node expansion limits based on codebase size
CALL_GRAPH_MAX_NODES_SMALL = 50      # For codebases < 1000 nodes
CALL_GRAPH_MAX_NODES_MEDIUM = 50     # For codebases < 5000 nodes (min 50, or 5% of nodes)
CALL_GRAPH_MAX_NODES_LARGE = 100     # For codebases >= 5000 nodes (min 100, or 10% of nodes)

# Edge generation multipliers (max edges = original_edges * multiplier)
CALL_GRAPH_EDGE_MULTIPLIER_SMALL = 5    # For codebases < 1000 nodes
CALL_GRAPH_EDGE_MULTIPLIER_MEDIUM = 7   # For codebases < 5000 nodes
CALL_GRAPH_EDGE_MULTIPLIER_LARGE = 10   # For codebases >= 5000 nodes

# BFS traversal limits
CALL_GRAPH_MAX_NEIGHBORS_SMALL = 20     # Max neighbors per node for small codebases
CALL_GRAPH_MAX_NEIGHBORS_MEDIUM = 30    # Max neighbors per node for medium codebases
CALL_GRAPH_MAX_NEIGHBORS_LARGE = 50     # Max neighbors per node for large codebases

CALL_GRAPH_MAX_QUEUE_SMALL = 200        # Max BFS queue size for small codebases
CALL_GRAPH_MAX_QUEUE_MEDIUM = 500       # Max BFS queue size for medium codebases
CALL_GRAPH_MAX_QUEUE_LARGE = 1000       # Max BFS queue size for large codebases

# Edge existence check optimization
CALL_GRAPH_EDGE_SEARCH_LIMIT = 50       # Only check last N edges for duplicates (most are recent)

# ============================================================================
# Context Window Edge Limits
# ============================================================================

CONTEXT_WINDOW_MAX_NODES_PER_FILE = 200  # Max nodes to process per file (prevents O(n²) explosion)
CONTEXT_WINDOW_MAX_PAIRS_PER_FILE = 500  # Max pairs to generate per file

# ============================================================================
# File Hierarchy Edge Limits
# ============================================================================

HIERARCHY_MAX_NODES_PER_GROUP = 50      # Max nodes per file/directory/package group
HIERARCHY_MAX_PAIRS_PER_GROUP = 500     # Max pairs to generate per group

# ============================================================================
# Random Walk Limits
# ============================================================================

RANDOM_WALK_MAX_NEIGHBORS = 50           # Max neighbors to consider per step (prevents slow probability calc)
RANDOM_WALK_MIN_NODES_FOR_PARALLEL = 100 # Minimum nodes to use parallel processing

# ============================================================================
# Codebase Size Thresholds
# ============================================================================

# Thresholds for determining codebase size categories
SIZE_THRESHOLD_SMALL = 1000              # Nodes < this = small codebase
SIZE_THRESHOLD_MEDIUM = 5000             # Nodes < this = medium codebase, else = large

