#!/usr/bin/env python3
"""Query interface for TriVector Code Intelligence model."""
import click
from rich.console import Console
from rich.table import Table
from rich import box
from tvci import SymbolModel

console = Console()


@click.command()
@click.option('--model-dir', '-m', required=True, help='Path to model directory')
@click.option('--symbol', '-s', help='Symbol ID to query')
@click.option('--top-k', '-k', default=10, help='Number of results to return')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def query(model_dir, symbol, top_k, interactive):
    """Query the TriVector Code Intelligence model for similar symbols."""
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


if __name__ == '__main__':
    query()

