#!/usr/bin/env python3
"""Command-line interface for TriCoder."""
import os
import shlex

import click
from rich.console import Console

from .git_tracker import (
    get_git_commit_hash, get_git_commit_timestamp, get_changed_files_for_retraining,
    save_training_metadata, extract_files_from_jsonl, get_all_python_files
)
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
@click.option('--fast', is_flag=True, default=False,
              help='Enable fast mode: reduces walk parameters by half for faster training (slightly lower quality).')
def train(nodes, edges, types, out, graph_dim, context_dim, typed_dim, final_dim,
          num_walks, walk_length, train_ratio, random_state, fast):
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
        random_state=random_state,
        fast_mode=fast
    )

    # Save git metadata after training
    commit_hash = get_git_commit_hash()
    commit_timestamp = get_git_commit_timestamp()
    files_trained = extract_files_from_jsonl(nodes)
    files_trained.update(extract_files_from_jsonl(edges))
    if types_path and os.path.exists(types_path):
        files_trained.update(extract_files_from_jsonl(types_path))

    save_training_metadata(out, commit_hash, commit_timestamp, files_trained)
    console.print(f"[dim]Saved training metadata (commit: {commit_hash[:8] if commit_hash else 'N/A'})[/dim]")


@cli.command(name='query')
@click.option('--model-dir', '-m', required=True, help='Path to model directory')
@click.option('--symbol', '-s', help='Symbol ID to query')
@click.option('--keywords', '-w', help='Keywords to search for (use quotes for multi-word: "my function")')
@click.option('--top-k', '-k', default=10, help='Number of results to return')
@click.option('--exclude-keywords', '--exclude', multiple=True,
              help='Additional keywords to exclude from search (can be specified multiple times). '
                   'These are appended to the default excluded keywords (Python builtins, etc.)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def query(model_dir, symbol, keywords, top_k, exclude_keywords, interactive):
    """Query the TriCoder model for similar symbols."""
    console.print(f"[bold green]Loading model from {model_dir}...[/bold green]")

    try:
        model = SymbolModel()
        model.load(model_dir)
        console.print("[bold green]✓ Model loaded successfully[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        return

    # Build excluded keywords set (default + user-provided)
    excluded_keywords_set = None
    if exclude_keywords:
        from .model import DEFAULT_EXCLUDED_KEYWORDS
        excluded_keywords_set = DEFAULT_EXCLUDED_KEYWORDS | {kw.lower() for kw in exclude_keywords}
        console.print(f"[dim]Excluding {len(excluded_keywords_set)} keywords "
                     f"({len(exclude_keywords)} user-added)[/dim]\n")

    if interactive:
        interactive_mode(model, excluded_keywords_set)
    elif symbol:
        display_results(model, symbol, top_k)
    elif keywords:
        # Parse keywords (handle quoted strings)
        keywords_parsed = parse_keywords(keywords)
        search_and_display_results(model, keywords_parsed, top_k, excluded_keywords_set)
    else:
        console.print("[bold yellow]Please provide --symbol, --keywords, or use --interactive mode[/bold yellow]")


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


def parse_keywords(keywords_str: str) -> str:
    """
    Parse keywords string, handling quoted strings.
    
    Args:
        keywords_str: Input string that may contain quoted keywords
    
    Returns:
        Parsed keywords string
    """
    try:
        # Use shlex to properly parse quoted strings
        parts = shlex.split(keywords_str)
        return ' '.join(parts)
    except ValueError:
        # If parsing fails, return as-is (handles unclosed quotes)
        return keywords_str.strip()


def search_and_display_results(model, keywords: str, top_k: int, excluded_keywords: set = None):
    """Search for symbols by keywords and display results."""
    from .model import DEFAULT_EXCLUDED_KEYWORDS
    
    # Use provided excluded keywords or default
    if excluded_keywords is None:
        excluded_keywords = DEFAULT_EXCLUDED_KEYWORDS
    
    # Check if any keywords are excluded
    keyword_words = keywords.lower().split()
    excluded_found = [w for w in keyword_words if w in excluded_keywords]
    
    matches = model.search_by_keywords(keywords, top_k, excluded_keywords=excluded_keywords)
    
    # Show warning if excluded keywords were filtered
    if excluded_found:
        console.print(f"[yellow]Note: Filtered out excluded keywords: {', '.join(excluded_found)}[/yellow]")
        console.print("[dim]These are Python builtins/keywords that don't provide useful search results.[/dim]\n")
    
    if not matches:
        if excluded_found and len(excluded_found) == len(keyword_words):
            console.print(f"[bold yellow]All keywords were filtered out (Python builtins/keywords).[/bold yellow]")
            console.print(f"[dim]Try searching for user-defined code patterns instead of language constructs.[/dim]")
        else:
            console.print(f"[bold yellow]No symbols found matching keywords: {keywords}[/bold yellow]")
        return
    
    console.print(f"\n[bold cyan]Search Results for:[/bold cyan] \"{keywords}\"")
    console.print(f"[bold cyan]Found {len(matches)} matching symbol(s):[/bold cyan]\n")
    
    for idx, match in enumerate(matches, 1):
        meta = match.get('meta', {})
        meta_dict = meta.get('meta', {}) if isinstance(meta.get('meta'), dict) else {}
        file_path = meta_dict.get('file', '') if meta_dict.get('file') else ''
        
        console.print(f"[dim]{idx}.[/dim] [cyan]{match['symbol']:15}[/cyan] "
                      f"[green]Relevance: {match['score']:6.4f}[/green] "
                      f"[blue]{meta.get('kind', 'unknown'):10}[/blue] "
                      f"[white]{meta.get('name', ''):30}[/white]")
        if file_path:
            console.print(f"     [dim]→ {file_path}[/dim]")
    
    console.print()
    
    # If there are matches, ask if user wants to query the first one
    if matches:
        first_match = matches[0]
        console.print(f"[dim]Tip: Query similar symbols with: --symbol {first_match['symbol']}[/dim]\n")


def interactive_mode(model, excluded_keywords: set = None):
    """Interactive query mode."""
    from .model import DEFAULT_EXCLUDED_KEYWORDS
    
    # Use provided excluded keywords or default
    if excluded_keywords is None:
        excluded_keywords = DEFAULT_EXCLUDED_KEYWORDS
    
    console.print("[bold green]Entering interactive mode. Type 'quit' or 'exit' to quit.[/bold green]")
    console.print("[dim]You can search by symbol ID or keywords (use quotes for multi-word)[/dim]")
    console.print(f"[dim]Excluding {len(excluded_keywords)} keywords (Python builtins, etc.)[/dim]\n")

    while True:
        try:
            query_input = click.prompt("\n[bold cyan]Enter symbol ID or keywords[/bold cyan]", type=str)

            if query_input.lower() in ['quit', 'exit', 'q']:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break

            top_k = click.prompt("Number of results", default=10, type=int)

            # Check if it looks like a symbol ID (starts with 'sym_') or try as keywords
            if query_input.startswith('sym_') and query_input in model.node_map:
                display_results(model, query_input, top_k)
            else:
                # Try as keywords
                keywords_parsed = parse_keywords(query_input)
                search_and_display_results(model, keywords_parsed, top_k, excluded_keywords)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@cli.command(name='extract')
@click.option('--input-dir', '--root', '-r', default='.',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Input directory to scan for files.')
@click.option('--include-dirs', '-i', multiple=True,
              help='Include only these subdirectories (can be specified multiple times).')
@click.option('--exclude-dirs', '-e', multiple=True,
              default=['.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'],
              help='Exclude these directories (can be specified multiple times).')
@click.option('--extensions', '--ext', default='py',
              help='Comma-separated list of file extensions to process (e.g., "py,js,ts"). Default: py')
@click.option('--exclude-keywords', '--exclude', multiple=True,
              help='Symbol names to exclude from extraction (can be specified multiple times). '
                   'These are appended to the default excluded keywords (Python builtins, etc.).')
@click.option('--output-nodes', '-n', default='nodes.jsonl',
              help='Output file for nodes (default: nodes.jsonl)')
@click.option('--output-edges', '-d', default='edges.jsonl',
              help='Output file for edges (default: edges.jsonl)')
@click.option('--output-types', '-t', default='types.jsonl',
              help='Output file for types (default: types.jsonl)')
@click.option('--no-gitignore', is_flag=True, default=False,
              help='Disable .gitignore filtering (enabled by default)')
def extract(input_dir, include_dirs, exclude_dirs, extensions, exclude_keywords, output_nodes, output_edges, output_types, no_gitignore):
    """Extract symbols and relationships from codebase."""
    from .extract import extract_from_directory
    from .model import DEFAULT_EXCLUDED_KEYWORDS

    # Parse extensions: split by comma, strip whitespace, remove dots if present
    ext_list = [ext.strip().lstrip('.') for ext in extensions.split(',') if ext.strip()]
    if not ext_list:
        ext_list = ['py']  # Default to Python if empty

    # Build excluded keywords set (default + user-provided)
    excluded_keywords_set = None
    if exclude_keywords:
        excluded_keywords_set = DEFAULT_EXCLUDED_KEYWORDS | {kw.lower() for kw in exclude_keywords}
        console.print(f"[dim]Excluding {len(excluded_keywords_set)} symbol names "
                     f"({len(exclude_keywords)} user-added)[/dim]\n")
    else:
        excluded_keywords_set = DEFAULT_EXCLUDED_KEYWORDS

    extract_from_directory(
        root_dir=input_dir,
        include_dirs=list(include_dirs) if include_dirs else [],
        exclude_dirs=list(exclude_dirs) if exclude_dirs else [],
        extensions=ext_list,
        excluded_keywords=excluded_keywords_set,
        output_nodes=output_nodes,
        output_edges=output_edges,
        output_types=output_types,
        use_gitignore=not no_gitignore
    )


@cli.command(name='retrain')
@click.option('--model-dir', '-m', required=True, type=click.Path(exists=True),
              help='Path to existing model directory')
@click.option('--codebase-dir', '-c', default='.', type=click.Path(exists=True, file_okay=False),
              help='Path to codebase root directory (default: current directory)')
@click.option('--output-nodes', '-n', default='nodes_retrain.jsonl',
              help='Temporary output file for nodes (default: nodes_retrain.jsonl)')
@click.option('--output-edges', '-d', default='edges_retrain.jsonl',
              help='Temporary output file for edges (default: edges_retrain.jsonl)')
@click.option('--output-types', '-t', default='types_retrain.jsonl',
              help='Temporary output file for types (default: types_retrain.jsonl)')
@click.option('--graph-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the graph view embeddings (uses model default if not specified).')
@click.option('--context-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the context view embeddings (uses model default if not specified).')
@click.option('--typed-dim', default=None, type=int, show_default=False,
              help='Dimensionality for the typed view embeddings (uses model default if not specified).')
@click.option('--final-dim', default=None, type=int, show_default=False,
              help='Final dimensionality of fused embeddings (uses model default if not specified).')
@click.option('--num-walks', default=10, type=int, show_default=True,
              help='Number of random walks per node for context view.')
@click.option('--walk-length', default=80, type=int, show_default=True,
              help='Length of each random walk in the context view.')
@click.option('--train-ratio', default=0.8, type=float, show_default=True,
              help='Fraction of edges used for training.')
@click.option('--random-state', default=42, type=int, show_default=True,
              help='Random seed for reproducibility.')
@click.option('--force', is_flag=True, default=False,
              help='Force full retraining even if no files changed.')
def retrain(model_dir, codebase_dir, output_nodes, output_edges, output_types,
            graph_dim, context_dim, typed_dim, final_dim, num_walks, walk_length,
            train_ratio, random_state, force):
    """Retrain TriCoder model incrementally on changed files only."""
    from .git_tracker import load_training_metadata
    from .extract import extract_from_directory
    import json

    console.print("[bold cyan]TriCoder Incremental Retraining[/bold cyan]\n")

    # Load previous training metadata
    metadata = load_training_metadata(model_dir)
    if not metadata and not force:
        console.print(
            "[bold yellow]No previous training metadata found. Use 'train' command for initial training.[/bold yellow]")
        return

    if metadata:
        console.print(
            f"[dim]Previous training: commit {metadata.get('commit_hash', 'N/A')[:8] if metadata.get('commit_hash') else 'N/A'}[/dim]")
        console.print(f"[dim]Training timestamp: {metadata.get('training_timestamp', 'N/A')}[/dim]\n")

    # Get changed files
    if force:
        console.print("[yellow]Force flag set - retraining on all files[/yellow]\n")
        changed_files = get_all_python_files(codebase_dir)
    else:
        changed_files = get_changed_files_for_retraining(model_dir, codebase_dir)

    if not changed_files:
        console.print("[bold green]✓ No files changed since last training. Model is up to date![/bold green]")
        return

    console.print(f"[cyan]Found {len(changed_files)} changed file(s):[/cyan]")
    for f in sorted(list(changed_files))[:10]:  # Show first 10
        console.print(f"  [dim]- {f}[/dim]")
    if len(changed_files) > 10:
        console.print(f"  [dim]... and {len(changed_files) - 10} more[/dim]")
    console.print()

    # Extract symbols from changed files only
    console.print("[cyan]Extracting symbols from changed files...[/cyan]")
    extract_from_directory(
        root_dir=codebase_dir,
        include_dirs=[],
        exclude_dirs=['.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache'],
        output_nodes=output_nodes,
        output_edges=output_edges,
        output_types=output_types,
        use_gitignore=True
    )

    # Filter extracted data to only include changed files
    console.print("[cyan]Filtering extracted data to changed files...[/cyan]")
    filtered_nodes = output_nodes + '.filtered'
    filtered_edges = output_edges + '.filtered'
    filtered_types = output_types + '.filtered'

    # Filter nodes
    node_count = 0
    with open(filtered_nodes, 'w') as out_f:
        with open(output_nodes, 'r') as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    file_path = data.get('meta', {}).get('file', '') if isinstance(data.get('meta'),
                                                                                   dict) else ''
                    # Normalize path for comparison
                    normalized_path = file_path.replace('\\', '/')
                    if normalized_path in changed_files or any(
                            normalized_path.endswith('/' + f) for f in changed_files):
                        out_f.write(line)
                        node_count += 1
                except json.JSONDecodeError:
                    continue

    # Filter edges (include if either endpoint is in changed files)
    edge_count = 0
    node_ids_from_changed = set()
    with open(filtered_nodes, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    node_ids_from_changed.add(data.get('id'))
                except json.JSONDecodeError:
                    continue

    with open(filtered_edges, 'w') as out_f:
        with open(output_edges, 'r') as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    src = data.get('src', '')
                    dst = data.get('dst', '')
                    # Include edge if either endpoint is from changed files
                    if src in node_ids_from_changed or dst in node_ids_from_changed:
                        out_f.write(line)
                        edge_count += 1
                except json.JSONDecodeError:
                    continue

    # Filter types
    type_count = 0
    if os.path.exists(output_types):
        with open(filtered_types, 'w') as out_f:
            with open(output_types, 'r') as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        symbol_id = data.get('symbol', '')
                        if symbol_id in node_ids_from_changed:
                            out_f.write(line)
                            type_count += 1
                    except json.JSONDecodeError:
                        continue

    console.print(
        f"[green]✓ Extracted {node_count} nodes, {edge_count} edges, {type_count} type tokens from changed files[/green]\n")

    if node_count == 0:
        console.print("[bold yellow]No nodes found in changed files. Nothing to retrain.[/bold yellow]")
        # Cleanup
        for f in [output_nodes, output_edges, output_types, filtered_nodes, filtered_edges, filtered_types]:
            if os.path.exists(f):
                os.remove(f)
        return

    # Retrain the model
    console.print("[cyan]Retraining model...[/cyan]\n")
    train_model(
        nodes_path=filtered_nodes,
        edges_path=filtered_edges,
        types_path=filtered_types if os.path.exists(filtered_types) else None,
        output_dir=model_dir,
        graph_dim=graph_dim,
        context_dim=context_dim,
        typed_dim=typed_dim,
        final_dim=final_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        train_ratio=train_ratio,
        random_state=random_state
    )

    # Save updated git metadata
    commit_hash = get_git_commit_hash(codebase_dir)
    commit_timestamp = get_git_commit_timestamp(codebase_dir)
    all_files = extract_files_from_jsonl(filtered_nodes)
    all_files.update(extract_files_from_jsonl(filtered_edges))
    if os.path.exists(filtered_types):
        all_files.update(extract_files_from_jsonl(filtered_types))

    save_training_metadata(model_dir, commit_hash, commit_timestamp, all_files)
    console.print(
        f"[dim]Updated training metadata (commit: {commit_hash[:8] if commit_hash else 'N/A'})[/dim]")

    # Cleanup temporary files
    console.print("\n[cyan]Cleaning up temporary files...[/cyan]")
    for f in [output_nodes, output_edges, output_types, filtered_nodes, filtered_edges, filtered_types]:
        if os.path.exists(f):
            os.remove(f)

    console.print("[bold green]✓ Incremental retraining complete![/bold green]")


if __name__ == '__main__':
    cli()
