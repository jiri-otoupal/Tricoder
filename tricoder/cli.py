#!/usr/bin/env python3
"""Command-line interface for TriCoder."""
import os
import shlex
from typing import List

import click
from rich import box
from rich.console import Console
from rich.prompt import Prompt

from tricoder.git_tracker import get_git_commit_hash, get_git_commit_timestamp, extract_files_from_jsonl, \
    save_training_metadata, get_changed_files_for_retraining, get_all_python_files
from tricoder.model import SymbolModel
from tricoder.train import train_model

console = Console()


def get_tricoder_dir() -> str:
    """Get the default .tricoder directory path and ensure it exists."""
    cwd = os.getcwd()
    tricoder_dir = os.path.join(cwd, '.tricoder')
    os.makedirs(tricoder_dir, exist_ok=True)
    return tricoder_dir


def get_model_dir() -> str:
    """Get the default model directory path and ensure it exists."""
    tricoder_dir = get_tricoder_dir()
    model_dir = os.path.join(tricoder_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def is_valid_model_dir(path: str) -> bool:
    """Check if a directory contains a valid model."""
    embeddings_path = os.path.join(path, 'embeddings.npy')
    metadata_path = os.path.join(path, 'metadata.json')
    return os.path.exists(embeddings_path) and os.path.exists(metadata_path)


def discover_models(root_dir: str) -> List[str]:
    """
    Recursively discover all model directories starting from root_dir.
    
    Args:
        root_dir: Root directory to search for models
        
    Returns:
        List of model directory paths (sorted, with root_dir first if it's a model)
    """
    models = []
    
    if not os.path.exists(root_dir):
        return models
    
    # Check if root_dir itself is a model
    if is_valid_model_dir(root_dir):
        models.append(root_dir)
    
    # Recursively search subdirectories
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for d in dirs:
            subdir = os.path.join(root, d)
            if is_valid_model_dir(subdir):
                models.append(subdir)
    
    # Sort: root_dir first if it's a model, then alphabetically
    def sort_key(path):
        if path == root_dir:
            return (0, path)
        return (1, path)
    
    return sorted(models, key=sort_key)


@click.group()
def cli():
    """TriCoder - TriVector Code Intelligence for semantic code analysis."""
    pass


@cli.command(name='train')
@click.option('--nodes', '-n', default=None, type=click.Path(),
              help='Path to nodes.jsonl file (default: .tricoder/nodes.jsonl).')
@click.option('--edges', '-e', default=None, type=click.Path(),
              help='Path to edges.jsonl file (default: .tricoder/edges.jsonl).')
@click.option('--types', '-t', default=None, type=click.Path(),
              help='[Optional] Path to types.jsonl file (default: .tricoder/types.jsonl).')
@click.option('--out', '-o', default=None, type=click.Path(),
              help='Output directory for trained model (default: .tricoder/model).')
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
@click.option('--use-gpu', is_flag=True, default=False,
              help='Enable GPU acceleration: CUDA (NVIDIA) via CuPy, or MPS (Mac) via PyTorch. Falls back to CPU if GPU unavailable.')
def train(nodes, edges, types, out, graph_dim, context_dim, typed_dim, final_dim,
          num_walks, walk_length, train_ratio, random_state, fast, use_gpu):
    """Train TriCoder model on codebase symbols and relationships."""
    # Use default .tricoder directory if paths not specified
    tricoder_dir = get_tricoder_dir()
    nodes_path = nodes if nodes else os.path.join(tricoder_dir, 'nodes.jsonl')
    edges_path = edges if edges else os.path.join(tricoder_dir, 'edges.jsonl')
    types_path = types if types else os.path.join(tricoder_dir, 'types.jsonl')
    output_dir = out if out else get_model_dir()
    
    # Handle optional types file - only use if it exists
    if not os.path.exists(types_path):
        types_path = None

    train_model(
        nodes_path=nodes_path,
        edges_path=edges_path,
        types_path=types_path,
        output_dir=output_dir,
        graph_dim=graph_dim,
        context_dim=context_dim,
        typed_dim=typed_dim,
        final_dim=final_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        train_ratio=train_ratio,
        random_state=random_state,
        fast_mode=fast,
        use_gpu=use_gpu
    )

    # Save git metadata after training
    commit_hash = get_git_commit_hash()
    commit_timestamp = get_git_commit_timestamp()
    files_trained = extract_files_from_jsonl(nodes_path)
    files_trained.update(extract_files_from_jsonl(edges_path))
    if types_path and os.path.exists(types_path):
        files_trained.update(extract_files_from_jsonl(types_path))

    save_training_metadata(output_dir, commit_hash, commit_timestamp, files_trained)
    console.print(f"[dim]Saved training metadata (commit: {commit_hash[:8] if commit_hash else 'N/A'})[/dim]")


@cli.command(name='query')
@click.option('--model-dir', '-m', default=None, help='Path to model directory (default: discovers models in .tricoder/model)')
@click.option('--symbol', '-s', help='Symbol ID to query')
@click.option('--keywords', '-w', help='Keywords to search for (use quotes for multi-word: "my function")')
@click.option('--top-k', '-k', default=10, help='Number of results to return')
@click.option('--exclude-keywords', '--exclude', multiple=True,
              help='Additional keywords to exclude from search (can be specified multiple times). '
                   'These are appended to the default excluded keywords (Python builtins, etc.)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--no-recursive', '--no-discover', is_flag=True, default=False,
              help='Disable recursive model discovery. Only use the base model directory directly.')
@click.option('--full', '-f', is_flag=True, default=False,
              help='Show full code context (from start to end line) for each result')
def query(model_dir, symbol, keywords, top_k, exclude_keywords, interactive, no_recursive, full):
    """Query the TriCoder model for similar symbols."""
    # Discover models if not specified
    if model_dir is None:
        base_model_dir = get_model_dir()
        if no_recursive:
            # Only check the base directory itself, don't search recursively
            if is_valid_model_dir(base_model_dir):
                discovered_models = [base_model_dir]
            else:
                discovered_models = []
        else:
            discovered_models = discover_models(base_model_dir)
        
        if not discovered_models:
            console.print(f"[bold red]No models found in {base_model_dir}[/bold red]")
            console.print(f"[yellow]Please train a model first using: tricoder train[/yellow]")
            return
        
        # Print one-liner summary at the beginning
        if len(discovered_models) > 1:
            console.print(f"[dim]Found {len(discovered_models)} models[/dim]")
        
        if len(discovered_models) == 1:
            # Only one model found, use it automatically
            model_dir = discovered_models[0]
            console.print(f"[dim]Found 1 model: {os.path.relpath(model_dir)}[/dim]")
        else:
            # Multiple models found, ask user to select
            console.print(f"[bold cyan]Found {len(discovered_models)} models:[/bold cyan]\n")
            
            # Display models in a tree-like structure
            try:
                base_path = os.path.commonpath([base_model_dir] + discovered_models)
            except ValueError:
                # Fallback if paths are on different drives (Windows) or can't find common path
                base_path = base_model_dir
            
            for idx, model_path in enumerate(discovered_models, 1):
                try:
                    rel_path = os.path.relpath(model_path, base_path)
                    # Normalize path separators for display
                    rel_path = rel_path.replace('\\', '/')
                    if rel_path == '.':
                        rel_path = 'model'
                except ValueError:
                    rel_path = os.path.basename(model_path) or model_path
                
                # Count symbols in model
                num_nodes = 0
                try:
                    import json
                    metadata_path = os.path.join(model_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            num_nodes = metadata.get('num_nodes', 0)
                except Exception:
                    pass
                
                if num_nodes > 0:
                    console.print(f"  [cyan]{idx}.[/cyan] [white]{rel_path}[/white] [dim]({num_nodes:,} symbols)[/dim]")
                else:
                    console.print(f"  [cyan]{idx}.[/cyan] [white]{rel_path}[/white]")
            
            console.print()
            try:
                choice = Prompt.ask(
                    "[bold cyan]Select model to query[/bold cyan]",
                    default="1",
                    choices=[str(i) for i in range(1, len(discovered_models) + 1)]
                )
                model_dir = discovered_models[int(choice) - 1]
            except (ValueError, IndexError, KeyboardInterrupt):
                console.print("[bold yellow]Cancelled[/bold yellow]")
                return
    
    console.print(f"[bold green]Loading model from {os.path.relpath(model_dir)}...[/bold green]")

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
        interactive_mode(model, excluded_keywords_set, show_full=full)
    elif symbol:
        display_results(model, symbol, top_k, show_full=full)
    elif keywords:
        # Parse keywords (handle quoted strings)
        keywords_parsed = parse_keywords(keywords)
        search_and_display_results(model, keywords_parsed, top_k, excluded_keywords_set, show_full=full)
    else:
        console.print("[bold yellow]Please provide --symbol, --keywords, or use --interactive mode[/bold yellow]")


def get_code_snippet(file_path: str, start_line: int, end_line: int = None) -> str:
    """
    Extract code snippet from file between start_line and end_line.
    
    Args:
        file_path: path to the file
        start_line: starting line number (1-indexed)
        end_line: ending line number (1-indexed), if None uses start_line
    
    Returns:
        Code snippet as string, or error message if file cannot be read
    """
    if not file_path or not os.path.exists(file_path):
        return "[dim]File not found[/dim]"
    
    if start_line is None or start_line < 1:
        return "[dim]Invalid line number[/dim]"
    
    if end_line is None:
        end_line = start_line
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Adjust for 1-indexed line numbers
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        if start_idx >= len(lines):
            return "[dim]Line number out of range[/dim]"
        
        snippet_lines = lines[start_idx:end_idx]
        return ''.join(snippet_lines).rstrip()
    except Exception as e:
        return f"[dim]Error reading file: {e}[/dim]"


def display_results(model, symbol_id, top_k, show_full: bool = False):
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
        query_meta_dict = query_meta.get('meta', {})
        if isinstance(query_meta_dict, dict):
            query_file = query_meta_dict.get('file', '')
            query_lineno = query_meta_dict.get('lineno', None)
            if query_file:
                if query_lineno is not None and query_lineno >= 0:
                    console.print(f"  [dim]File:[/dim] {query_file}:{query_lineno}")
                else:
                    console.print(f"  [dim]File:[/dim] {query_file}")

    console.print(f"\n[bold cyan]Top {len(results)} Similar Symbols:[/bold cyan]\n")

    for idx, result in enumerate(results, 1):
        meta = result.get('meta', {})
        meta_dict = meta.get('meta', {}) if isinstance(meta.get('meta'), dict) else {}
        file_path = meta_dict.get('file', '') if meta_dict.get('file') else ''
        lineno = meta_dict.get('lineno', None)
        end_lineno = meta_dict.get('end_lineno', None)

        console.print(f"[dim]{idx}.[/dim] [cyan]{result['symbol']:15}[/cyan] "
                      f"[green]Score: {result['score']:8.4f}[/green] "
                      f"[yellow]Dist: {result['distance']:6.4f}[/yellow] "
                      f"[blue]{meta.get('kind', 'unknown'):10}[/blue] "
                      f"[white]{meta.get('name', ''):30}[/white]")
        if file_path:
            if lineno is not None and lineno >= 0:
                if end_lineno is not None and end_lineno > lineno:
                    console.print(f"     [dim]→ {file_path}:{lineno}-{end_lineno}[/dim]")
                else:
                    console.print(f"     [dim]→ {file_path}:{lineno}[/dim]")
            else:
                console.print(f"     [dim]→ {file_path}[/dim]")
        
        # Show full code context if requested
        if show_full and file_path and lineno is not None:
            snippet = get_code_snippet(file_path, lineno, end_lineno)
            if snippet and not snippet.startswith("[dim]"):
                console.print(f"     [dim]Code:[/dim]")
                # Indent the code snippet
                for line in snippet.split('\n'):
                    console.print(f"     [dim]│[/dim] {line}")
            elif snippet.startswith("[dim]"):
                console.print(f"     {snippet}")

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


def search_and_display_results(model, keywords: str, top_k: int, excluded_keywords: set = None, show_full: bool = False):
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
        lineno = meta_dict.get('lineno', None)
        end_lineno = meta_dict.get('end_lineno', None)
        
        console.print(f"[dim]{idx}.[/dim] [cyan]{match['symbol']:15}[/cyan] "
                      f"[green]Relevance: {match['score']:6.4f}[/green] "
                      f"[blue]{meta.get('kind', 'unknown'):10}[/blue] "
                      f"[white]{meta.get('name', ''):30}[/white]")
        if file_path:
            if lineno is not None and lineno >= 0:
                if end_lineno is not None and end_lineno > lineno:
                    console.print(f"     [dim]→ {file_path}:{lineno}-{end_lineno}[/dim]")
                else:
                    console.print(f"     [dim]→ {file_path}:{lineno}[/dim]")
            else:
                console.print(f"     [dim]→ {file_path}[/dim]")
        
        # Show full code context if requested
        if show_full and file_path and lineno is not None:
            snippet = get_code_snippet(file_path, lineno, end_lineno)
            if snippet and not snippet.startswith("[dim]"):
                console.print(f"     [dim]Code:[/dim]")
                # Indent the code snippet
                for line in snippet.split('\n'):
                    console.print(f"     [dim]│[/dim] {line}")
            elif snippet.startswith("[dim]"):
                console.print(f"     {snippet}")
    
    console.print()
    
    # If there are matches, ask if user wants to query the first one
    if matches:
        first_match = matches[0]
        console.print(f"[dim]Tip: Query similar symbols with: --symbol {first_match['symbol']}[/dim]\n")


def interactive_mode(model, excluded_keywords: set = None, show_full: bool = False):
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
                display_results(model, query_input, top_k, show_full=show_full)
            else:
                # Try as keywords
                keywords_parsed = parse_keywords(query_input)
                search_and_display_results(model, keywords_parsed, top_k, excluded_keywords, show_full=show_full)

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
              default=['.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', '.tricoder'],
              help='Exclude these directories (can be specified multiple times).')
@click.option('--extensions', '--ext', default='py',
              help='Comma-separated list of file extensions to process (e.g., "py,js,ts"). Default: py')
@click.option('--exclude-keywords', '--exclude', multiple=True,
              help='Symbol names to exclude from extraction (can be specified multiple times). '
                   'These are appended to the default excluded keywords (Python builtins, etc.).')
@click.option('--output-nodes', '-n', default=None,
              help='Output file for nodes (default: .tricoder/nodes.jsonl)')
@click.option('--output-edges', '-d', default=None,
              help='Output file for edges (default: .tricoder/edges.jsonl)')
@click.option('--output-types', '-t', default=None,
              help='Output file for types (default: .tricoder/types.jsonl)')
@click.option('--no-gitignore', is_flag=True, default=False,
              help='Disable .gitignore filtering (enabled by default)')
def extract(input_dir, include_dirs, exclude_dirs, extensions, exclude_keywords, output_nodes, output_edges, output_types, no_gitignore):
    """Extract symbols and relationships from codebase."""
    from .extract import extract_from_directory
    from .model import DEFAULT_EXCLUDED_KEYWORDS

    # Use default .tricoder directory if paths not specified
    tricoder_dir = get_tricoder_dir()
    output_nodes_path = output_nodes if output_nodes else os.path.join(tricoder_dir, 'nodes.jsonl')
    output_edges_path = output_edges if output_edges else os.path.join(tricoder_dir, 'edges.jsonl')
    output_types_path = output_types if output_types else os.path.join(tricoder_dir, 'types.jsonl')

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
        output_nodes=output_nodes_path,
        output_edges=output_edges_path,
        output_types=output_types_path,
        use_gitignore=not no_gitignore
    )


@cli.command(name='optimize')
@click.option('--nodes', '-n', default=None, type=click.Path(),
              help='Path to nodes.jsonl file (default: .tricoder/nodes.jsonl)')
@click.option('--edges', '-e', default=None, type=click.Path(),
              help='Path to edges.jsonl file (default: .tricoder/edges.jsonl)')
@click.option('--types', '-t', default=None, type=click.Path(),
              help='Path to types.jsonl file (default: .tricoder/types.jsonl, optional)')
@click.option('--output-nodes', '-N', default=None, type=click.Path(),
              help='Output path for optimized nodes (default: overwrites input)')
@click.option('--output-edges', '-E', default=None, type=click.Path(),
              help='Output path for optimized edges (default: overwrites input)')
@click.option('--output-types', '-T', default=None, type=click.Path(),
              help='Output path for optimized types (default: overwrites input)')
@click.option('--min-edge-weight', default=0.3, type=float,
              help='Minimum edge weight to keep (default: 0.3)')
@click.option('--remove-isolated', is_flag=True, default=True,
              help='Remove nodes with no edges (default: True)')
@click.option('--keep-isolated', is_flag=True, default=False,
              help='Keep isolated nodes (overrides --remove-isolated)')
@click.option('--remove-generic', is_flag=True, default=True,
              help='Remove nodes with generic names (default: True)')
@click.option('--keep-generic', is_flag=True, default=False,
              help='Keep generic names (overrides --remove-generic)')
@click.option('--exclude-keywords', '--exclude', multiple=True,
              help='Additional keywords to exclude (can be specified multiple times)')
def optimize(nodes, edges, types, output_nodes, output_edges, output_types,
             min_edge_weight, remove_isolated, keep_isolated, remove_generic, keep_generic, exclude_keywords):
    """Optimize nodes and edges by filtering out low-value entries.
    
    This command removes:
    - Nodes with generic names (single letters, common names like 'temp', 'var', etc.)
    - Isolated nodes (nodes with no edges)
    - Low-weight edges (below minimum threshold)
    - Nodes matching excluded keywords
    
    This reduces the graph size while preserving meaningful relationships.
    """
    from .optimize import optimize_nodes_and_edges
    from .model import DEFAULT_EXCLUDED_KEYWORDS
    
    # Use default .tricoder directory if paths not specified
    tricoder_dir = get_tricoder_dir()
    nodes_path = nodes if nodes else os.path.join(tricoder_dir, 'nodes.jsonl')
    edges_path = edges if edges else os.path.join(tricoder_dir, 'edges.jsonl')
    types_path = types if types else os.path.join(tricoder_dir, 'types.jsonl')
    
    # Default output paths: overwrite input if not specified
    output_nodes_path = output_nodes if output_nodes else nodes_path
    output_edges_path = output_edges if output_edges else edges_path
    output_types_path = output_types if output_types else types_path
    
    # Build excluded keywords set
    excluded_keywords_set = DEFAULT_EXCLUDED_KEYWORDS
    if exclude_keywords:
        excluded_keywords_set = excluded_keywords_set | {kw.lower() for kw in exclude_keywords}
    
    # Check if input files exist
    if not os.path.exists(nodes_path):
        console.print(f"[bold red]Error: Nodes file not found: {nodes_path}[/bold red]")
        return
    if not os.path.exists(edges_path):
        console.print(f"[bold red]Error: Edges file not found: {edges_path}[/bold red]")
        return
    
    # Handle flags
    remove_isolated_nodes = remove_isolated and not keep_isolated
    remove_generic_names = remove_generic and not keep_generic
    
    console.print("[bold cyan]Optimizing nodes and edges...[/bold cyan]\n")
    console.print(f"[dim]Min edge weight: {min_edge_weight}[/dim]")
    console.print(f"[dim]Remove isolated nodes: {remove_isolated_nodes}[/dim]")
    console.print(f"[dim]Remove generic names: {remove_generic_names}[/dim]")
    console.print(f"[dim]Excluded keywords: {len(excluded_keywords_set)}[/dim]\n")
    
    try:
        nodes_removed, edges_removed, types_removed, stats = optimize_nodes_and_edges(
            nodes_path=nodes_path,
            edges_path=edges_path,
            types_path=types_path if types_path and os.path.exists(types_path) else None,
            output_nodes=output_nodes_path,
            output_edges=output_edges_path,
            output_types=output_types_path,
            min_edge_weight=min_edge_weight,
            remove_isolated=remove_isolated_nodes,
            remove_generic_names=remove_generic_names,
            excluded_keywords=excluded_keywords_set
        )
        
        console.print(f"\n[bold green]✓ Optimization complete![/bold green]\n")
        
        # Overall statistics
        from rich.table import Table
        stats_table = Table(title="Optimization Statistics", box=box.ROUNDED, show_header=True)
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Original", style="white", justify="right", width=12)
        stats_table.add_column("Final", style="green", justify="right", width=12)
        stats_table.add_column("Removed", style="yellow", justify="right", width=12)
        stats_table.add_column("Reduction", style="dim", justify="right", width=12)
        
        # Calculate percentages
        node_reduction = (nodes_removed / stats['original']['nodes'] * 100) if stats['original']['nodes'] > 0 else 0
        edge_reduction = (edges_removed / stats['original']['edges'] * 100) if stats['original']['edges'] > 0 else 0
        type_reduction = (types_removed / stats['original']['types'] * 100) if stats['original']['types'] > 0 else 0
        
        stats_table.add_row(
            "Nodes",
            f"{stats['original']['nodes']:,}",
            f"{stats['final']['nodes']:,}",
            f"{stats['removed']['nodes']:,}",
            f"{node_reduction:.1f}%"
        )
        stats_table.add_row(
            "Edges",
            f"{stats['original']['edges']:,}",
            f"{stats['final']['edges']:,}",
            f"{stats['removed']['edges']:,}",
            f"{edge_reduction:.1f}%"
        )
        if stats['original']['types'] > 0:
            stats_table.add_row(
                "Types",
                f"{stats['original']['types']:,}",
                f"{stats['final']['types']:,}",
                f"{stats['removed']['types']:,}",
                f"{type_reduction:.1f}%"
            )
        
        console.print(stats_table)
        
        # Removal reasons
        console.print(f"\n[bold cyan]Removal Breakdown:[/bold cyan]")
        reasons_table = Table(show_header=False, box=None)
        reasons_table.add_column("Reason", style="dim", width=30)
        reasons_table.add_column("Count", style="yellow", justify="right", width=15)
        
        if stats['removal_reasons']['excluded_keywords'] > 0:
            reasons_table.add_row("Excluded keywords", f"{stats['removal_reasons']['excluded_keywords']:,}")
        if stats['removal_reasons']['generic_names'] > 0:
            reasons_table.add_row("Generic names", f"{stats['removal_reasons']['generic_names']:,}")
        if stats['removal_reasons']['isolated'] > 0:
            reasons_table.add_row("Isolated nodes", f"{stats['removal_reasons']['isolated']:,}")
        if stats['removal_reasons']['orphaned_edges'] > 0:
            reasons_table.add_row("Orphaned edges (node removed)", f"{stats['removal_reasons']['orphaned_edges']:,}")
        if stats['removal_reasons']['low_weight_edges'] > 0:
            reasons_table.add_row(f"Low-weight edges (<{min_edge_weight})", f"{stats['removal_reasons']['low_weight_edges']:,}")
        
        console.print(reasons_table)
        
        # Statistics by kind
        console.print(f"\n[bold cyan]Statistics by Kind:[/bold cyan]")
        kind_table = Table(show_header=True, box=box.ROUNDED)
        kind_table.add_column("Kind", style="cyan", width=15)
        kind_table.add_column("Original", style="white", justify="right", width=12)
        kind_table.add_column("Removed", style="yellow", justify="right", width=12)
        kind_table.add_column("Final", style="green", justify="right", width=12)
        
        for kind in sorted(stats['by_kind'].keys()):
            kind_stats = stats['by_kind'][kind]
            if kind_stats['original'] > 0:
                kind_table.add_row(
                    kind,
                    f"{kind_stats['original']:,}",
                    f"{kind_stats['removed']:,}",
                    f"{kind_stats['final']:,}"
                )
        
        console.print(kind_table)
        
        # Show output paths
        console.print(f"\n[dim]Optimized files written to:[/dim]")
        console.print(f"  [dim]Nodes: {output_nodes_path}[/dim]")
        console.print(f"  [dim]Edges: {output_edges_path}[/dim]")
        if output_types_path and os.path.exists(output_types_path):
            console.print(f"  [dim]Types: {output_types_path}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error during optimization: {e}[/bold red]")
        raise


@cli.command(name='retrain')
@click.option('--model-dir', '-m', required=True, type=click.Path(exists=True),
              help='Path to existing model directory')
@click.option('--codebase-dir', '-c', default='.', type=click.Path(exists=True, file_okay=False),
              help='Path to codebase root directory (default: current directory)')
@click.option('--output-nodes', '-n', default=None,
              help='Temporary output file for nodes (default: .tricoder/nodes_retrain.jsonl)')
@click.option('--output-edges', '-d', default=None,
              help='Temporary output file for edges (default: .tricoder/edges_retrain.jsonl)')
@click.option('--output-types', '-t', default=None,
              help='Temporary output file for types (default: .tricoder/types_retrain.jsonl)')
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

    # Use default .tricoder directory if paths not specified
    if model_dir is None:
        model_dir = get_model_dir()
    tricoder_dir = get_tricoder_dir()
    output_nodes_path = output_nodes if output_nodes else os.path.join(tricoder_dir, 'nodes_retrain.jsonl')
    output_edges_path = output_edges if output_edges else os.path.join(tricoder_dir, 'edges_retrain.jsonl')
    output_types_path = output_types if output_types else os.path.join(tricoder_dir, 'types_retrain.jsonl')

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
        output_nodes=output_nodes_path,
        output_edges=output_edges_path,
        output_types=output_types_path,
        use_gitignore=True
    )

    # Filter extracted data to only include changed files
    console.print("[cyan]Filtering extracted data to changed files...[/cyan]")
    filtered_nodes = output_nodes_path + '.filtered'
    filtered_edges = output_edges_path + '.filtered'
    filtered_types = output_types_path + '.filtered'

    # Filter nodes
    node_count = 0
    with open(filtered_nodes, 'w') as out_f:
        with open(output_nodes_path, 'r') as in_f:
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
        with open(output_edges_path, 'r') as in_f:
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
    if os.path.exists(output_types_path):
        with open(filtered_types, 'w') as out_f:
            with open(output_types_path, 'r') as in_f:
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
        for f in [output_nodes_path, output_edges_path, output_types_path, filtered_nodes, filtered_edges, filtered_types]:
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
        random_state=random_state,
        use_gpu=False  # Retrain doesn't support GPU yet
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
    for f in [output_nodes_path, output_edges_path, output_types_path, filtered_nodes, filtered_edges, filtered_types]:
        if os.path.exists(f):
            os.remove(f)

    console.print("[bold green]✓ Incremental retraining complete![/bold green]")


if __name__ == '__main__':
    cli()
