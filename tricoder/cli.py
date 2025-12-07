#!/usr/bin/env python3
"""Command-line interface for TriCoder."""
import os

import click
from rich.console import Console

from .model import SymbolModel
from .train import train_model

console = Console()


@click.group()
def cli():
    """TriCoder - TriVector Code Intelligence for semantic code analysis."""
    pass


@cli.command(name='train')
@click.option('--nodes', '-n', default='nodes.jsonl', type=click.Path(),
              help='Path to nodes.jsonl file containing symbol definitions (default: nodes.jsonl).')
@click.option('--edges', '-e', default='edges.jsonl', type=click.Path(),
              help='Path to edges.jsonl file containing symbol relationships (default: edges.jsonl).')
@click.option('--types', '-t', default='types.jsonl', type=click.Path(),
              help='[Optional] Path to types.jsonl file containing type token information (default: types.jsonl).')
@click.option('--out', '-o', required=True, type=click.Path(),
              help='Output directory where the trained model will be saved.')
@click.option('--graph-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the graph view embeddings.')
@click.option('--context-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the context view embeddings.')
@click.option('--typed-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the typed view embeddings.')
@click.option('--final-dim', default=None, type=int, show_default=False,
              help='Final dimensionality of fused embeddings after PCA reduction.')
@click.option('--num-walks', default=10, type=int, show_default=True,
              help='Number of random walks to generate per node for context view.')
@click.option('--walk-length', default=80, type=int, show_default=True,
              help='Length of each random walk in the context view.')
@click.option('--train-ratio', default=0.8, type=float, show_default=True,
              help='Fraction of edges used for training (rest used for temperature calibration).')
@click.option('--random-state', default=42, type=int, show_default=True,
              help='Random seed for reproducibility.')
def train(nodes, edges, types, out, graph_dim, context_dim, typed_dim, final_dim,
          num_walks, walk_length, train_ratio, random_state):
    """Train TriCoder model on codebase symbols and relationships."""
    # Handle optional types file - only use if it exists
    types_path = types if types and os.path.exists(types) else None

    train_model(
        nodes_path=nodes,
        edges_path=edges,
        types_path=types_path,
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


@cli.command(name='query')
@click.option('--model-dir', '-m', required=True, help='Path to model directory')
@click.option('--symbol', '-s', help='Symbol ID to query')
@click.option('--top-k', '-k', default=10, help='Number of results to return')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def query(model_dir, symbol, top_k, interactive):
    """Query the TriCoder model for similar symbols."""
    console.print(f"[bold green]Loading model from {model_dir}...[/bold green]")

    try:
        model = SymbolModel()
        model.load(model_dir)
        console.print("[bold green]✓ Model loaded successfully[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    if interactive:
        interactive_mode(model)
    elif symbol:
        display_results(model, symbol, top_k)
    else:
        console.print("[bold yellow]Please provide --symbol or use --interactive mode[/bold yellow]")


def display_results(model, symbol_id, top_k):
    """Display query results in a formatted table."""
    results = model.query(symbol_id, top_k)

    if not results:
        console.print(f"[bold yellow]No results found for symbol: {symbol_id}[/bold yellow]")
        return

    # Get query symbol info
    query_meta = None
    if model.metadata_lookup:
        query_meta = model.metadata_lookup.get(symbol_id)

    console.print(f"\n[bold cyan]Query:[/bold cyan] {symbol_id}")
    if query_meta:
        console.print(f"  [dim]Kind:[/dim] {query_meta.get('kind', 'unknown')}")
        console.print(f"  [dim]Name:[/dim] {query_meta.get('name', 'unknown')}")
        if query_meta.get('meta', {}).get('file'):
            console.print(f"  [dim]File:[/dim] {query_meta['meta']['file']}")

    console.print(f"\n[bold cyan]Top {len(results)} Similar Symbols:[/bold cyan]\n")

    for idx, result in enumerate(results, 1):
        meta = result.get('meta', {})
        meta_dict = meta.get('meta', {}) if isinstance(meta.get('meta'), dict) else {}
        file_path = meta_dict.get('file', '') if meta_dict.get('file') else ''

        console.print(f"[dim]{idx}.[/dim] [cyan]{result['symbol']:15}[/cyan] "
                      f"[green]Score: {result['score']:8.4f}[/green] "
                      f"[yellow]Dist: {result['distance']:6.4f}[/yellow] "
                      f"[blue]{meta.get('kind', 'unknown'):10}[/blue] "
                      f"[white]{meta.get('name', ''):30}[/white]")
        if file_path:
            console.print(f"     [dim]→ {file_path}[/dim]")

    console.print()


def interactive_mode(model):
    """Interactive query mode."""
    console.print("[bold green]Entering interactive mode. Type 'quit' or 'exit' to quit.[/bold green]\n")

    while True:
        try:
            symbol_id = click.prompt("\n[bold cyan]Enter symbol ID[/bold cyan]", type=str)

            if symbol_id.lower() in ['quit', 'exit', 'q']:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break

            top_k = click.prompt("Number of results", default=10, type=int)

            display_results(model, symbol_id, top_k)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@cli.command(name='extract')
@click.option('--input-dir', '--root', '-r', default='.',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Input directory to scan for Python files.')
@click.option('--include-dirs', '-i', multiple=True,
              help='Include only these subdirectories (can be specified multiple times).')
@click.option('--exclude-dirs', '-e', multiple=True,
              default=['.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'],
              help='Exclude these directories (can be specified multiple times).')
@click.option('--output-nodes', '-n', default='nodes.jsonl',
              help='Output file for nodes (default: nodes.jsonl)')
@click.option('--output-edges', '-d', default='edges.jsonl',
              help='Output file for edges (default: edges.jsonl)')
@click.option('--output-types', '-t', default='types.jsonl',
              help='Output file for types (default: types.jsonl)')
@click.option('--no-gitignore', is_flag=True, default=False,
              help='Disable .gitignore filtering (enabled by default)')
def extract(input_dir, include_dirs, exclude_dirs, output_nodes, output_edges, output_types, no_gitignore):
    """Extract symbols and relationships from Python codebase."""
    from .extract import extract_from_directory

    extract_from_directory(
        root_dir=input_dir,
        include_dirs=list(include_dirs) if include_dirs else [],
        exclude_dirs=list(exclude_dirs) if exclude_dirs else [],
        output_nodes=output_nodes,
        output_edges=output_edges,
        output_types=output_types,
        use_gitignore=not no_gitignore
    )


if __name__ == '__main__':
    cli()
