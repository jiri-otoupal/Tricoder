#!/usr/bin/env python3
"""Benchmark training components to get accurate time estimates."""
import time
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

from tvci.data_loader import load_nodes, load_edges, load_types
from tvci.graph_view import compute_graph_view
from tvci.context_view import compute_context_view
from tvci.typed_view import compute_typed_view
from tvci.fusion import fuse_embeddings
from tvci.calibration import split_edges, learn_temperature
from multiprocessing import cpu_count

console = Console()


def benchmark_component(name, func, *args, **kwargs):
    """Benchmark a single component."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return elapsed, result


def main():
    """Run benchmark on actual data."""
    console.print("[bold cyan]Training Component Benchmark[/bold cyan]\n")
    
    # Load data
    nodes_path = 'nodes.jsonl'
    edges_path = 'edges.jsonl'
    types_path = 'types.jsonl'
    
    if not Path(nodes_path).exists() or not Path(edges_path).exists():
        console.print("[bold red]Error: nodes.jsonl and edges.jsonl must exist[/bold red]")
        return
    
    console.print("Loading data...")
    node_to_idx, node_metadata = load_nodes(nodes_path)
    edges, num_nodes = load_edges(edges_path, node_to_idx)
    
    node_types = None
    type_to_idx = None
    num_types = None
    if Path(types_path).exists():
        node_types, type_to_idx = load_types(types_path, node_to_idx)
        num_types = len(type_to_idx)
    
    console.print(f"Loaded: {num_nodes} nodes, {len(edges)} edges")
    if num_types:
        console.print(f"         {num_types} type tokens")
    console.print()
    
    # Split edges
    console.print("Splitting edges...")
    split_time, (train_edges, val_edges) = benchmark_component(
        "split_edges", split_edges, edges, 0.8, 42
    )
    console.print(f"  Time: {split_time:.3f}s\n")
    
    # Get workers
    n_jobs = max(1, cpu_count() - 1)
    console.print(f"Using {n_jobs} workers\n")
    
    # Compute dimensions (use defaults)
    from tvci.train import compute_default_dimensions
    defaults = compute_default_dimensions(num_nodes, len(edges), num_types)
    graph_dim = defaults['graph_dim']
    context_dim = defaults['context_dim']
    typed_dim = defaults['typed_dim']
    final_dim = defaults['final_dim']
    
    console.print(f"Dimensions: graph={graph_dim}, context={context_dim}, "
                 f"typed={typed_dim}, final={final_dim}\n")
    
    # Benchmark graph view
    console.print("Benchmarking graph view...")
    graph_time, (X_graph, svd_graph) = benchmark_component(
        "graph_view", compute_graph_view,
        train_edges, num_nodes, graph_dim, 42, n_jobs
    )
    console.print(f"  Time: {graph_time:.3f}s\n")
    
    # Benchmark context view
    console.print("Benchmarking context view...")
    context_time, (X_w2v, kv) = benchmark_component(
        "context_view", compute_context_view,
        train_edges, num_nodes, context_dim, 10, 80, 42, n_jobs
    )
    console.print(f"  Time: {context_time:.3f}s\n")
    
    # Benchmark typed view (if available)
    typed_time = 0
    X_types = None
    svd_types = None
    if node_types and type_to_idx:
        console.print("Benchmarking typed view...")
        typed_time, (X_types, svd_types) = benchmark_component(
            "typed_view", compute_typed_view,
            node_types, type_to_idx, num_nodes, typed_dim, 42, n_jobs
        )
        console.print(f"  Time: {typed_time:.3f}s\n")
    
    # Benchmark fusion
    console.print("Benchmarking fusion...")
    embeddings_list = [X_graph, X_w2v]
    if X_types is not None:
        embeddings_list.append(X_types)
    fusion_time, (E, pca_comp, pca_mean) = benchmark_component(
        "fusion", fuse_embeddings,
        embeddings_list, num_nodes, final_dim, 42, n_jobs
    )
    console.print(f"  Time: {fusion_time:.3f}s\n")
    
    # Benchmark temperature calibration
    console.print("Benchmarking temperature calibration...")
    calib_time, tau = benchmark_component(
        "calibration", learn_temperature,
        E, val_edges, num_nodes, 5, None, 42, n_jobs
    )
    console.print(f"  Time: {calib_time:.3f}s\n")
    
    # Total time
    total_time = split_time + graph_time + context_time + typed_time + fusion_time + calib_time
    
    # Display results
    console.print("[bold cyan]Benchmark Results[/bold cyan]\n")
    
    results_table = Table(box=box.ROUNDED, title="Component Timings")
    results_table.add_column("Component", style="cyan", width=30)
    results_table.add_column("Time", style="white", width=15)
    results_table.add_column("% of Total", style="yellow", width=15)
    
    components = [
        ("Split Edges", split_time),
        ("Graph View", graph_time),
        ("Context View", context_time),
    ]
    
    if typed_time > 0:
        components.append(("Typed View", typed_time))
    
    components.extend([
        ("Fusion", fusion_time),
        ("Calibration", calib_time),
    ])
    
    for name, t in components:
        percentage = (t / total_time * 100) if total_time > 0 else 0
        results_table.add_row(name, f"{t:.3f}s", f"{percentage:.1f}%")
    
    results_table.add_row("", "", "")
    results_table.add_row("[bold]TOTAL[/bold]", f"[bold green]{total_time:.3f}s[/bold green]", "100.0%")
    
    console.print(results_table)
    console.print()
    
    # Calculate coefficients for estimation
    console.print("[bold cyan]Estimation Coefficients[/bold cyan]\n")
    
    # Graph view: O(nodes^1.5) + O(nodes * dim^2)
    graph_base = graph_time / ((num_nodes ** 1.5) / (1000 * n_jobs) + (num_nodes * graph_dim ** 2) / (50000 * n_jobs))
    
    # Context view: walks + word2vec
    total_walks = num_nodes * 10
    walk_base = context_time / ((total_walks * 80) / (100000 * n_jobs) + (total_walks * 80 * 10 * 5) / (500000 * n_jobs))
    
    # Fusion: O(nodes * input_dim * final_dim)
    total_input_dim = graph_dim + context_dim + (typed_dim if num_types else 0)
    fusion_base = fusion_time / ((num_nodes * total_input_dim * final_dim) / (100000 * n_jobs))
    
    # Calibration: O(tau_candidates * val_edges)
    val_edges_count = len(val_edges)
    calib_base = calib_time / ((50 * val_edges_count) / (10000 * n_jobs))
    
    coeff_table = Table(box=box.ROUNDED, title="Coefficients")
    coeff_table.add_column("Component", style="cyan", width=25)
    coeff_table.add_column("Coefficient", style="white", width=20)
    
    coeff_table.add_row("Graph View", f"{graph_base:.4f}")
    coeff_table.add_row("Context View", f"{walk_base:.4f}")
    if typed_time > 0:
        typed_base = typed_time / ((num_nodes ** 1.5) / (1000 * n_jobs) + (num_nodes * typed_dim ** 2) / (50000 * n_jobs))
        coeff_table.add_row("Typed View", f"{typed_base:.4f}")
    coeff_table.add_row("Fusion", f"{fusion_base:.4f}")
    coeff_table.add_row("Calibration", f"{calib_base:.4f}")
    
    console.print(coeff_table)
    console.print()
    
    # Save benchmark data
    benchmark_data = {
        'num_nodes': num_nodes,
        'num_edges': len(edges),
        'num_types': num_types,
        'graph_dim': graph_dim,
        'context_dim': context_dim,
        'typed_dim': typed_dim,
        'final_dim': final_dim,
        'num_walks': 10,
        'walk_length': 80,
        'n_jobs': n_jobs,
        'timings': {
            'split': split_time,
            'graph': graph_time,
            'context': context_time,
            'typed': typed_time,
            'fusion': fusion_time,
            'calibration': calib_time,
            'total': total_time
        },
        'coefficients': {
            'graph': graph_base,
            'context': walk_base,
            'typed': typed_base if typed_time > 0 else None,
            'fusion': fusion_base,
            'calibration': calib_base
        }
    }
    
    with open('benchmark_data.json', 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    console.print(f"[green]✓ Benchmark data saved to benchmark_data.json[/green]")
    console.print("\n[bold yellow]Update estimate_training_time() with these coefficients[/bold yellow]")


if __name__ == '__main__':
    main()

