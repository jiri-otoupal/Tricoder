"""Training pipeline for TriVector Code Intelligence."""
import json
import os
import time
from typing import Optional

import click
import numpy as np
from annoy import AnnoyIndex
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from .calibration import split_edges, learn_temperature
from .context_view import compute_context_view
from .data_loader import load_nodes, load_edges, load_types
from .fusion import fuse_embeddings, iterative_embedding_smoothing
from .graph_view import compute_graph_view
from .typed_view import compute_typed_view

console = Console()


def estimate_training_time(num_nodes: int, num_edges: int, num_types: Optional[int],
                           graph_dim: int, context_dim: int, typed_dim: int,
                           num_walks: int, walk_length: int, final_dim: int,
                           n_jobs: int = 1) -> str:
    """
    Estimate training time based on data size and parameters.
    
    Returns:
        Estimated time as formatted string
    """
    # More realistic time estimates based on actual performance
    # Accounts for parallelization efficiency and overhead

    # Parallelization efficiency factor (diminishing returns with more cores)
    # More cores help but not linearly due to overhead
    efficiency = min(0.85, 0.3 + 0.55 * (n_jobs / max(n_jobs, 8)))  # Cap efficiency at ~85%
    effective_cores = n_jobs * efficiency

    # Graph view: PPMI + SVD
    # PPMI computation: sparse matrix operations, scales with edges
    # SVD: matrix decomposition, scales with nodes and dimensions
    # Account for subtoken expansion (roughly doubles nodes)
    expanded_nodes = num_nodes * 2.5  # Account for subtokens and expansion
    ppmi_time = (expanded_nodes * num_edges ** 0.5) / (5000 * effective_cores)
    svd_time = (expanded_nodes * graph_dim ** 2) / (20000 * effective_cores)
    graph_time = ppmi_time + svd_time

    # Context view: Random walks + Word2Vec
    # Random walks: parallelized per node, but has overhead
    total_walks = expanded_nodes * num_walks
    walk_time = (total_walks * walk_length) / (8000 * effective_cores)  # More realistic

    # Word2Vec: training is CPU-intensive, benefits from parallelization
    # window=10, epochs=5, negative=5
    w2v_time = (total_walks * walk_length * 10 * 5) / (150000 * effective_cores)
    context_time = walk_time + w2v_time

    # Typed view: PPMI + SVD (if available)
    typed_time = 0
    if num_types:
        # Account for type expansion
        expanded_types = num_types * 1.3  # Type expansion adds ~30% more types
        typed_ppmi_time = (expanded_nodes * expanded_types ** 0.5) / (5000 * effective_cores)
        typed_svd_time = (expanded_nodes * typed_dim ** 2) / (20000 * effective_cores)
        typed_time = typed_ppmi_time + typed_svd_time

    # Fusion: PCA
    # PCA is memory-bound and benefits less from parallelization
    total_input_dim = graph_dim + context_dim + (typed_dim if num_types else 0)
    fusion_time = (expanded_nodes * total_input_dim * final_dim) / (30000 * effective_cores)

    # Embedding smoothing: iterative neighbor averaging
    smoothing_time = (expanded_nodes * num_edges * 2) / (10000 * effective_cores)  # 2 iterations

    # Temperature calibration: grid search with parallel evaluation
    tau_candidates = 50  # default
    val_edges_est = int(num_edges * 0.2)  # ~20% for validation
    calibration_time = (tau_candidates * val_edges_est) / (5000 * effective_cores)

    # ANN index building (single-threaded mostly)
    ann_time = (num_nodes * final_dim * 10) / 200000  # 10 trees, less parallelizable

    # I/O overhead (saving files)
    io_time = 1.5

    # Data loading overhead
    load_time = 0.5

    total_seconds = (graph_time + context_time + typed_time + fusion_time +
                     smoothing_time + calibration_time + ann_time + io_time + load_time)

    # Format time estimate
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = total_seconds / 3600
        minutes = (total_seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"


def compute_default_dimensions(num_nodes: int, num_edges: int, num_types: Optional[int] = None) -> dict:
    """
    Compute default dimensions based on data characteristics.
    
    Returns:
        Dictionary with graph_dim, context_dim, typed_dim, final_dim
    """
    # Graph dimension: based on number of nodes
    # Use log-based heuristic for better scaling: log2(nodes) * 8, clamped to reasonable range
    # Small codebases (<100 nodes): ~32-48 dims
    # Medium codebases (100-1000 nodes): ~48-64 dims  
    # Large codebases (>1000 nodes): ~64-128 dims
    if num_nodes < 50:
        graph_dim = 32
    elif num_nodes < 200:
        graph_dim = max(32, min(64, int(np.log2(num_nodes) * 8)))
    elif num_nodes < 1000:
        graph_dim = max(48, min(96, int(np.log2(num_nodes) * 7)))
    else:
        graph_dim = max(64, min(128, int(np.log2(num_nodes) * 6)))

    # Context dimension: match graph dimension for balanced fusion
    context_dim = graph_dim

    # Typed dimension: based on number of type tokens if available
    if num_types:
        if num_types < 20:
            typed_dim = 16
        elif num_types < 50:
            typed_dim = max(16, min(32, int(np.sqrt(num_types) * 2)))
        else:
            typed_dim = max(32, min(64, int(np.sqrt(num_types) * 2)))
    else:
        typed_dim = 32  # Default when no types

    # Final dimension: based on total input dimensions and number of nodes
    # Should be less than number of nodes (for PCA to work)
    # Use 50-70% of total input dimensions, but respect node count limit
    total_input_dim = graph_dim + context_dim + (typed_dim if num_types else 0)
    final_dim_candidate = int(total_input_dim * 0.6)  # 60% of input dims
    final_dim = max(32, min(num_nodes - 1, min(256, final_dim_candidate)))

    return {
        'graph_dim': graph_dim,
        'context_dim': context_dim,
        'typed_dim': typed_dim,
        'final_dim': final_dim
    }


def train_model(nodes_path: str,
                edges_path: str,
                types_path: Optional[str],
                output_dir: str,
                graph_dim: Optional[int] = None,
                context_dim: Optional[int] = None,
                typed_dim: Optional[int] = None,
                final_dim: Optional[int] = None,
                num_walks: int = 10,
                walk_length: int = 80,
                train_ratio: float = 0.8,
                random_state: int = 42):
    """
    Train TriVector Code Intelligence model.
    
    Args:
        nodes_path: path to nodes.jsonl
        edges_path: path to edges.jsonl
        types_path: path to types.jsonl (optional)
        output_dir: output directory for model
        graph_dim: dimensionality for graph view
        context_dim: dimensionality for context view
        typed_dim: dimensionality for typed view
        final_dim: final fused embedding dimensionality
        num_walks: number of random walks per node
        walk_length: length of each random walk
        train_ratio: ratio of edges for training (rest for calibration)
        random_state: random seed
    """
    # Set random seeds
    np.random.seed(random_state)

    # Record start time
    start_time = time.time()

    console.print("\n[bold cyan]TriVector Code Intelligence - Training Pipeline[/bold cyan]\n")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
    ) as progress:
        # Load data first to compute dimensions
        task1 = progress.add_task("[cyan]Loading data to compute dimensions...", total=None)
        node_to_idx, node_metadata, node_subtokens, node_file_info = load_nodes(nodes_path)
        edges, num_nodes = load_edges(edges_path, node_to_idx)
        progress.update(task1, completed=True)

        if num_nodes == 0:
            raise ValueError("No nodes found in input file")

        if len(edges) == 0:
            raise ValueError("No edges found in input file")

        # Load types if available
        node_types = None
        type_to_idx = None
        num_types = None
        if types_path and os.path.exists(types_path):
            task2 = progress.add_task("[cyan]Loading types...", total=None)
            node_types, type_to_idx = load_types(types_path, node_to_idx)
            num_types = len(type_to_idx)
            progress.update(task2, completed=True)

        # Compute default dimensions if not provided
        defaults = compute_default_dimensions(num_nodes, len(edges), num_types)

        if graph_dim is None:
            graph_dim = defaults['graph_dim']
            graph_dim_source = "[dim](computed)[/dim]"
        else:
            graph_dim_source = ""

        if context_dim is None:
            context_dim = defaults['context_dim']
            context_dim_source = "[dim](computed)[/dim]"
        else:
            context_dim_source = ""

        if typed_dim is None:
            typed_dim = defaults['typed_dim']
            typed_dim_source = "[dim](computed)[/dim]"
        else:
            typed_dim_source = ""

        if final_dim is None:
            final_dim = defaults['final_dim']
            final_dim_source = "[dim](computed)[/dim]"
        else:
            final_dim_source = ""

    # Get number of workers for time estimation
    from multiprocessing import cpu_count
    n_jobs_est = max(1, cpu_count() - 1)

    # Estimate training time
    estimated_time = estimate_training_time(
        num_nodes, len(edges), num_types,
        graph_dim, context_dim, typed_dim,
        num_walks, walk_length, final_dim,
        n_jobs_est
    )

    # Display configuration
    config_table = Table(box=box.ROUNDED, show_header=False, title="Configuration")
    config_table.add_column("Parameter", style="cyan", width=25)
    config_table.add_column("Value", style="white", width=15)
    config_table.add_column("Source", style="dim", width=12)
    config_table.add_row("Graph Dimension", str(graph_dim), graph_dim_source)
    config_table.add_row("Context Dimension", str(context_dim), context_dim_source)
    config_table.add_row("Typed Dimension", str(typed_dim), typed_dim_source)
    config_table.add_row("Final Dimension", str(final_dim), final_dim_source)
    config_table.add_row("Random Walks", str(num_walks), "")
    config_table.add_row("Walk Length", str(walk_length), "")
    config_table.add_row("Train Ratio", f"{train_ratio:.2f}", "")
    config_table.add_row("Random State", str(random_state), "")
    config_table.add_row("", "", "")  # Separator
    config_table.add_row("Data: Nodes", str(num_nodes), "")
    config_table.add_row("Data: Edges", str(len(edges)), "")
    if num_types:
        config_table.add_row("Data: Type Tokens", str(num_types), "")
    config_table.add_row("", "", "")  # Separator
    config_table.add_row("[bold]Estimated Time[/bold]", f"[bold green]{estimated_time}[/bold green]", "")
    config_table.add_row("Workers", str(n_jobs_est), "")
    console.print(config_table)
    console.print()

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
    ) as progress:

        # Split edges for calibration
        task3 = progress.add_task("[cyan]Splitting edges...", total=None)
        train_edges, val_edges = split_edges(edges, train_ratio, random_state)
        progress.update(task3, completed=True)

        # Get number of workers (all cores - 1, but use all cores on Windows for better performance)
        from multiprocessing import cpu_count
        import platform
        if platform.system() == 'Windows':
            # On Windows, use all cores since spawn method handles it well
            n_jobs = cpu_count()
        else:
            n_jobs = max(1, cpu_count() - 1)

        # Create idx_to_node mapping
        idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

        # Step 1: Compute graph view with all enhancements
        # This task runs completely and sequentially before moving to the next one
        # This ensures full parallelization can be used for graph view operations
        task4 = progress.add_task(
            f"[cyan]Step 1/5: Computing graph view (PPMI + SVD + enhancements) [{n_jobs} workers]...",
            total=None)
        X_graph, svd_components_graph, final_num_nodes, subtoken_to_idx, expanded_edges = compute_graph_view(
            train_edges, num_nodes, graph_dim, random_state, n_jobs=n_jobs,
            node_to_idx=node_to_idx,
            node_subtokens=node_subtokens,
            node_file_info=node_file_info,
            node_metadata=node_metadata,
            idx_to_node=idx_to_node,
            expand_calls=True,
            add_subtokens=True,
            add_hierarchy=True,
            add_context=True,
            context_window=5
        )
        progress.update(task4, completed=True)
        # Graph view is now complete - all resources released before next task

        # Step 2: Compute context view using expanded edges (includes subtokens)
        # This task runs completely after graph view finishes
        # This ensures full parallelization can be used for context view operations
        task5 = progress.add_task(
            f"[cyan]Step 2/5: Computing context view (Node2Vec + Word2Vec) [{n_jobs} workers]...", total=None)
        # Use expanded_edges which includes subtokens and all enhancements
        X_w2v, word2vec_kv = compute_context_view(
            expanded_edges, final_num_nodes, context_dim, num_walks, walk_length, random_state, n_jobs=n_jobs
        )
        progress.update(task5, completed=True)
        # Context view is now complete - all resources released before next task

        # Step 3: Compute typed view if available (with type expansion)
        # This task runs completely after context view finishes
        X_types = None
        svd_components_types = None
        final_type_to_idx = None
        if node_types is not None and type_to_idx is not None:
            task6 = progress.add_task(
                f"[cyan]Step 3/5: Computing typed view (PPMI + SVD + type expansion) [{n_jobs} workers]...",
                total=None)
            X_types, svd_components_types, final_type_to_idx = compute_typed_view(
                node_types, type_to_idx, final_num_nodes, typed_dim, random_state, n_jobs=n_jobs,
                expand_types=True
            )
            progress.update(task6, completed=True)
            # Typed view is now complete - all resources released before next task

        # Step 4: Fuse embeddings
        # This task runs completely after typed view finishes (if available)
        task7 = progress.add_task(
            f"[cyan]Step 4/5: Fusing embeddings (PCA + Normalize) [{n_jobs} workers]...", total=None)
        embeddings_list = [X_graph, X_w2v]
        if X_types is not None:
            embeddings_list.append(X_types)

        E, pca_components, pca_mean = fuse_embeddings(
            embeddings_list, final_num_nodes, final_dim, random_state, n_jobs=n_jobs
        )

        # Store embeddings before normalization for mean_norm computation
        E_before_norm = E.copy()

        # Compute mean_norm for length penalty
        mean_norm = float(np.mean(np.linalg.norm(E_before_norm, axis=1)))

        progress.update(task7, completed=True)

        # Apply iterative embedding smoothing (diffusion)
        # This runs after fusion completes
        task7b = progress.add_task(f"[cyan]Applying embedding smoothing (diffusion) [{n_jobs} workers]...",
                                   total=None)
        E = iterative_embedding_smoothing(
            E, expanded_edges, final_num_nodes,
            num_iterations=2, beta=0.35, random_state=random_state
        )
        progress.update(task7b, completed=True)
        # Smoothing is now complete - all resources released before next task

        # Step 5: Learn temperature with improved negative sampling
        # This task runs completely after smoothing finishes
        task8 = progress.add_task(
            f"[cyan]Step 5/5: Learning temperature parameter (improved negative sampling) [{n_jobs} workers]...",
            total=None)
        tau = learn_temperature(
            E, val_edges, final_num_nodes, random_state=random_state, n_jobs=n_jobs,
            node_metadata=node_metadata,
            node_file_info=node_file_info,
            idx_to_node=idx_to_node
        )
        progress.update(task8, completed=True)

        # Build ANN index (only for original nodes, not subtokens)
        task9 = progress.add_task("[cyan]Building ANN index...", total=None)
        ann_index = AnnoyIndex(final_dim, 'angular')
        # Only index original nodes (not subtokens)
        for i in range(num_nodes):
            ann_index.add_item(i, E[i])
        ann_index.build(10)  # 10 trees
        progress.update(task9, completed=True)

        # Save model
        task10 = progress.add_task("[cyan]Saving model...", total=None)
        os.makedirs(output_dir, exist_ok=True)

        # Save embeddings (only original nodes for query, but keep full for future use)
        # Save full embeddings including subtokens
        np.save(os.path.join(output_dir, 'embeddings.npy'), E)

        # Save temperature
        np.save(os.path.join(output_dir, 'tau.npy'), np.array(tau))

        # Save mean_norm for length penalty
        np.save(os.path.join(output_dir, 'mean_norm.npy'), np.array(mean_norm))

        # Save metadata
        metadata = {
            'node_map': node_to_idx,
            'node_metadata': node_metadata,
            'embedding_dim': final_dim,
            'num_nodes': num_nodes,
            'final_num_nodes': final_num_nodes  # Includes subtokens
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save subtoken mapping
        if subtoken_to_idx:
            with open(os.path.join(output_dir, 'subtoken_map.json'), 'w') as f:
                json.dump(subtoken_to_idx, f, indent=2)

        # Save node subtokens
        if node_subtokens:
            with open(os.path.join(output_dir, 'node_subtokens.json'), 'w') as f:
                json.dump(node_subtokens, f, indent=2)

        # Save node types (for query expansion)
        if node_types is not None:
            # Convert node_types from idx-based to node_id-based
            node_types_by_id = {}
            for node_idx, types_dict in node_types.items():
                node_id = idx_to_node.get(node_idx)
                if node_id:
                    # Convert counts to int for JSON serialization
                    node_types_by_id[node_id] = {k: int(v) for k, v in types_dict.items()}
            if node_types_by_id:
                with open(os.path.join(output_dir, 'node_types.json'), 'w') as f:
                    json.dump(node_types_by_id, f, indent=2)

        # Save PCA components
        np.save(os.path.join(output_dir, 'fusion_pca_components.npy'), pca_components)
        np.save(os.path.join(output_dir, 'fusion_pca_mean.npy'), pca_mean)

        # Save SVD components
        np.save(os.path.join(output_dir, 'svd_components.npy'), svd_components_graph)

        if svd_components_types is not None:
            np.save(os.path.join(output_dir, 'svd_components_types.npy'), svd_components_types)

        # Save Word2Vec
        word2vec_kv.save(os.path.join(output_dir, 'word2vec.kv'))

        # Save ANN index
        ann_index.save(os.path.join(output_dir, 'ann_index.ann'))

        # Save type token map (use final expanded version)
        if final_type_to_idx is not None:
            with open(os.path.join(output_dir, 'type_token_map.json'), 'w') as f:
                json.dump(final_type_to_idx, f, indent=2)
        elif type_to_idx is not None:
            with open(os.path.join(output_dir, 'type_token_map.json'), 'w') as f:
                json.dump(type_to_idx, f, indent=2)

        progress.update(task10, completed=True)

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format elapsed time
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        time_str = f"{hours}h {minutes}m {seconds:.2f}s"

    # Display statistics
    console.print("\n[bold green]✓ Training Complete![/bold green]\n")

    stats_table = Table(box=box.ROUNDED, title="Training Statistics")
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Total Nodes", str(num_nodes))
    stats_table.add_row("Total Nodes (with subtokens)", str(final_num_nodes))
    stats_table.add_row("Total Edges", str(len(edges)))
    stats_table.add_row("Training Edges", str(len(train_edges)))
    stats_table.add_row("Validation Edges", str(len(val_edges)))
    if final_type_to_idx:
        stats_table.add_row("Type Tokens (expanded)", str(len(final_type_to_idx)))
    elif type_to_idx:
        stats_table.add_row("Type Tokens", str(len(type_to_idx)))
    if subtoken_to_idx:
        stats_table.add_row("Subtoken Nodes", str(len(subtoken_to_idx)))
    stats_table.add_row("Graph View Dim", f"{X_graph.shape[1]}")
    stats_table.add_row("Context View Dim", f"{X_w2v.shape[1]}")
    if X_types is not None:
        stats_table.add_row("Typed View Dim", f"{X_types.shape[1]}")
    stats_table.add_row("Final Embedding Dim", f"{E.shape[1]}")
    stats_table.add_row("Temperature (τ)", f"{tau:.6f}")
    stats_table.add_row("Mean Norm", f"{mean_norm:.6f}")
    stats_table.add_row("Model Directory", output_dir)
    stats_table.add_row("", "")  # Separator
    stats_table.add_row("[bold]Total Training Time[/bold]", f"[bold green]{time_str}[/bold green]")

    console.print(stats_table)
    console.print()


@click.command()
@click.option('--nodes', '-n', required=True, type=click.Path(exists=True),
              help='Path to nodes.jsonl file containing symbol definitions.\n'
                   'Each line should be a JSON object with: id, kind, name, meta.')
@click.option('--edges', '-e', required=True, type=click.Path(exists=True),
              help='Path to edges.jsonl file containing symbol relationships.\n'
                   'Each line should be a JSON object with: src, dst, rel, weight.')
@click.option('--types', '-t', default=None, type=click.Path(exists=True),
              help='[Optional] Path to types.jsonl file containing type token information.\n'
                   'Each line should be a JSON object with: symbol, type_token, count.')
@click.option('--out', '-o', required=True, type=click.Path(),
              help='Output directory where the trained model will be saved.\n'
                   'Directory will be created if it does not exist.')
@click.option('--graph-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the graph view embeddings.\n'
                   'Range: 8-512. Higher values capture more graph structure but increase computation.\n'
                   'If not specified, computed automatically based on number of nodes.\n'
                   'Recommended: 32-128 for small codebases, 64-256 for large ones.')
@click.option('--context-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the context view embeddings (Node2Vec + Word2Vec).\n'
                   'Range: 8-512. Higher values capture more semantic context.\n'
                   'If not specified, computed automatically to match graph-dim.\n'
                   'Recommended: 32-128. Should match graph-dim for balanced fusion.')
@click.option('--typed-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the typed view embeddings.\n'
                   'Range: 8-512. Only used if --types file is provided.\n'
                   'If not specified, computed automatically based on number of type tokens.\n'
                   'Recommended: 32-128. Should match other view dimensions.')
@click.option('--final-dim', default=None, type=int, show_default=False,
              help='Final dimensionality of fused embeddings after PCA reduction.\n'
                   'Range: 16-512. This is the output embedding size used for queries.\n'
                   'If not specified, computed automatically based on input dimensions and node count.\n'
                   'Recommended: 64-256. Higher values preserve more information but increase memory.')
@click.option('--num-walks', default=10, type=int, show_default=True,
              help='Number of random walks to generate per node for context view.\n'
                   'Range: 5-100. More walks improve context quality but increase training time.\n'
                   'Recommended: 10-20 for most codebases.')
@click.option('--walk-length', default=80, type=int, show_default=True,
              help='Length of each random walk in the context view.\n'
                   'Range: 20-200. Longer walks capture more distant relationships.\n'
                   'Recommended: 40-100. Shorter for small graphs, longer for large ones.')
@click.option('--train-ratio', default=0.8, type=float, show_default=True,
              help='Fraction of edges used for training (rest used for temperature calibration).\n'
                   'Range: 0.5-0.95. Higher values use more data for training but less for calibration.\n'
                   'Recommended: 0.8. Lower values (0.7-0.75) improve temperature estimation.')
@click.option('--random-state', default=42, type=int, show_default=True,
              help='Random seed for reproducibility.\n'
                   'Range: Any integer. Use the same seed to get identical results.\n'
                   'Recommended: 42 (classic choice) or any fixed integer for reproducibility.')
def main(nodes, edges, types, out, graph_dim, context_dim, typed_dim, final_dim,
         num_walks, walk_length, train_ratio, random_state):
    """
    Train TriVector Code Intelligence (TVI) model.
    
    TVI learns symbol-level semantics from codebases using three complementary views:
    
    \b
    1. Graph View: Structural relationships via PPMI and SVD
    2. Context View: Semantic context via Node2Vec random walks and Word2Vec
    3. Typed View: Type information via type-token co-occurrence (optional)
    
    The views are fused using PCA and normalized to produce final embeddings.
    A temperature parameter is learned for calibrated similarity scores.
    
    \b
    Example:
        python train_tvci.py --nodes nodes.jsonl --edges edges.jsonl --out model_dir
    """
    train_model(
        nodes_path=nodes,
        edges_path=edges,
        types_path=types,
        output_dir=out,
        graph_dim=graph_dim,
        context_dim=context_dim,
        typed_dim=typed_dim,
        final_dim=final_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        train_ratio=train_ratio,
        random_state=random_state
    )


if __name__ == '__main__':
    main()
