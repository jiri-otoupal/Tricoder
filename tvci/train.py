"""Training pipeline for TriVector Code Intelligence."""
import click
import os
import json
import numpy as np
from annoy import AnnoyIndex
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from .data_loader import load_nodes, load_edges, load_types
from .graph_view import compute_graph_view
from .context_view import compute_context_view
from .typed_view import compute_typed_view
from .fusion import fuse_embeddings
from .calibration import split_edges, learn_temperature

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
    # Base time estimates (in seconds) per operation
    # These are rough estimates based on typical performance
    
    # Graph view: PPMI + SVD
    # PPMI: O(n^2) for sparse matrix, SVD: O(n * d^2)
    graph_time = (num_nodes ** 1.5) / (1000 * n_jobs) + (num_nodes * graph_dim ** 2) / (50000 * n_jobs)
    
    # Context view: Random walks + Word2Vec
    # Random walks: O(nodes * num_walks * walk_length)
    # Word2Vec: O(walks * window * epochs)
    total_walks = num_nodes * num_walks
    walk_time = (total_walks * walk_length) / (100000 * n_jobs)
    w2v_time = (total_walks * walk_length * 10 * 5) / (500000 * n_jobs)  # window=10, epochs=5
    context_time = walk_time + w2v_time
    
    # Typed view: PPMI + SVD (if available)
    typed_time = 0
    if num_types:
        typed_time = (num_nodes ** 1.5) / (1000 * n_jobs) + (num_nodes * typed_dim ** 2) / (50000 * n_jobs)
    
    # Fusion: PCA
    total_input_dim = graph_dim + context_dim + (typed_dim if num_types else 0)
    fusion_time = (num_nodes * total_input_dim * final_dim) / (100000 * n_jobs)
    
    # Temperature calibration: grid search
    tau_candidates = 50  # default
    val_edges_est = int(num_edges * 0.2)  # ~20% for validation
    calibration_time = (tau_candidates * val_edges_est) / (10000 * n_jobs)
    
    # ANN index building
    ann_time = (num_nodes * final_dim * 10) / (500000 * n_jobs)  # 10 trees
    
    # I/O overhead
    io_time = 2.0
    
    total_seconds = graph_time + context_time + typed_time + fusion_time + calibration_time + ann_time + io_time
    
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
        node_to_idx, node_metadata = load_nodes(nodes_path)
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
        
        # Get number of workers (all cores - 1)
        from multiprocessing import cpu_count
        n_jobs = max(1, cpu_count() - 1)
        
        # Compute graph view
        task4 = progress.add_task(f"[cyan]Computing graph view (PPMI + SVD) [{n_jobs} workers]...", total=None)
        X_graph, svd_components_graph = compute_graph_view(
            train_edges, num_nodes, graph_dim, random_state, n_jobs=n_jobs
        )
        progress.update(task4, completed=True)
        
        # Compute context view
        task5 = progress.add_task(f"[cyan]Computing context view (Node2Vec + Word2Vec) [{n_jobs} workers]...", total=None)
        X_w2v, word2vec_kv = compute_context_view(
            train_edges, num_nodes, context_dim, num_walks, walk_length, random_state, n_jobs=n_jobs
        )
        progress.update(task5, completed=True)
        
        # Compute typed view if available
        X_types = None
        svd_components_types = None
        if node_types is not None and type_to_idx is not None:
            task6 = progress.add_task(f"[cyan]Computing typed view (PPMI + SVD) [{n_jobs} workers]...", total=None)
            X_types, svd_components_types = compute_typed_view(
                node_types, type_to_idx, num_nodes, typed_dim, random_state, n_jobs=n_jobs
            )
            progress.update(task6, completed=True)
        
        # Fuse embeddings
        task7 = progress.add_task(f"[cyan]Fusing embeddings (PCA + Normalize) [{n_jobs} workers]...", total=None)
        embeddings_list = [X_graph, X_w2v]
        if X_types is not None:
            embeddings_list.append(X_types)
        
        E, pca_components, pca_mean = fuse_embeddings(
            embeddings_list, num_nodes, final_dim, random_state, n_jobs=n_jobs
        )
        progress.update(task7, completed=True)
        
        # Learn temperature
        task8 = progress.add_task(f"[cyan]Learning temperature parameter [{n_jobs} workers]...", total=None)
        tau = learn_temperature(E, val_edges, num_nodes, random_state=random_state, n_jobs=n_jobs)
        progress.update(task8, completed=True)
        
        # Build ANN index
        task9 = progress.add_task("[cyan]Building ANN index...", total=None)
        ann_index = AnnoyIndex(final_dim, 'angular')
        for i in range(num_nodes):
            ann_index.add_item(i, E[i])
        ann_index.build(10)  # 10 trees
        progress.update(task9, completed=True)
        
        # Save model
        task10 = progress.add_task("[cyan]Saving model...", total=None)
        os.makedirs(output_dir, exist_ok=True)
    
        # Save embeddings
        np.save(os.path.join(output_dir, 'embeddings.npy'), E)
        
        # Save temperature
        np.save(os.path.join(output_dir, 'tau.npy'), np.array(tau))
        
        # Save metadata
        idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}
        metadata = {
            'node_map': node_to_idx,
            'node_metadata': node_metadata,
            'embedding_dim': final_dim,
            'num_nodes': num_nodes
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
        
        # Save type token map if available
        if type_to_idx is not None:
            with open(os.path.join(output_dir, 'type_token_map.json'), 'w') as f:
                json.dump(type_to_idx, f, indent=2)
        
        progress.update(task10, completed=True)
    
    # Display statistics
    console.print("\n[bold green]✓ Training Complete![/bold green]\n")
    
    stats_table = Table(box=box.ROUNDED, title="Training Statistics")
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Value", style="white")
    stats_table.add_row("Total Nodes", str(num_nodes))
    stats_table.add_row("Total Edges", str(len(edges)))
    stats_table.add_row("Training Edges", str(len(train_edges)))
    stats_table.add_row("Validation Edges", str(len(val_edges)))
    if type_to_idx:
        stats_table.add_row("Type Tokens", str(len(type_to_idx)))
    stats_table.add_row("Graph View Dim", f"{X_graph.shape[1]}")
    stats_table.add_row("Context View Dim", f"{X_w2v.shape[1]}")
    if X_types is not None:
        stats_table.add_row("Typed View Dim", f"{X_types.shape[1]}")
    stats_table.add_row("Final Embedding Dim", f"{E.shape[1]}")
    stats_table.add_row("Temperature (τ)", f"{tau:.6f}")
    stats_table.add_row("Model Directory", output_dir)
    
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

