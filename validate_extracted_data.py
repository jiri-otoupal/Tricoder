#!/usr/bin/env python3
"""Validate extracted data format and content."""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def validate_nodes(nodes_path: str) -> tuple:
    """Validate nodes.jsonl format and content."""
    nodes = []
    issues = []
    
    required_fields = {'id', 'kind', 'name', 'meta'}
    required_meta_fields = {'file', 'lineno'}
    
    with open(nodes_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                node = json.loads(line)
                
                # Check required fields
                missing_fields = required_fields - set(node.keys())
                if missing_fields:
                    issues.append(f"Line {i}: Missing fields: {missing_fields}")
                    continue
                
                # Check meta structure
                if 'meta' not in node or not isinstance(node['meta'], dict):
                    issues.append(f"Line {i}: 'meta' must be a dictionary")
                    continue
                
                missing_meta = required_meta_fields - set(node['meta'].keys())
                if missing_meta:
                    issues.append(f"Line {i}: Missing meta fields: {missing_meta}")
                    continue
                
                nodes.append(node)
                
            except json.JSONDecodeError as e:
                issues.append(f"Line {i}: Invalid JSON: {e}")
    
    return nodes, issues


def validate_edges(edges_path: str) -> tuple:
    """Validate edges.jsonl format and content."""
    edges = []
    issues = []
    
    required_fields = {'src', 'dst', 'rel'}
    
    with open(edges_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                edge = json.loads(line)
                
                # Check required fields
                missing_fields = required_fields - set(edge.keys())
                if missing_fields:
                    issues.append(f"Line {i}: Missing fields: {missing_fields}")
                    continue
                
                # Check weight (optional but should be numeric if present)
                if 'weight' in edge:
                    try:
                        float(edge['weight'])
                    except (ValueError, TypeError):
                        issues.append(f"Line {i}: 'weight' must be numeric")
                        continue
                
                edges.append(edge)
                
            except json.JSONDecodeError as e:
                issues.append(f"Line {i}: Invalid JSON: {e}")
    
    return edges, issues


def validate_types(types_path: str) -> tuple:
    """Validate types.jsonl format and content."""
    types = []
    issues = []
    
    required_fields = {'symbol', 'type_token', 'count'}
    
    with open(types_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                type_entry = json.loads(line)
                
                # Check required fields
                missing_fields = required_fields - set(type_entry.keys())
                if missing_fields:
                    issues.append(f"Line {i}: Missing fields: {missing_fields}")
                    continue
                
                # Check count is numeric
                try:
                    int(type_entry['count'])
                except (ValueError, TypeError):
                    issues.append(f"Line {i}: 'count' must be an integer")
                    continue
                
                types.append(type_entry)
                
            except json.JSONDecodeError as e:
                issues.append(f"Line {i}: Invalid JSON: {e}")
    
    return types, issues


def check_data_consistency(nodes, edges, types):
    """Check consistency between nodes, edges, and types."""
    issues = []
    
    # Build node ID set
    node_ids = {node['id'] for node in nodes}
    
    # Check edges reference valid nodes
    edge_src_ids = {edge['src'] for edge in edges}
    edge_dst_ids = {edge['dst'] for edge in edges}
    
    missing_src = edge_src_ids - node_ids
    missing_dst = edge_dst_ids - node_ids
    
    if missing_src:
        issues.append(f"Edges reference {len(missing_src)} non-existent source nodes")
    
    if missing_dst:
        issues.append(f"Edges reference {len(missing_dst)} non-existent destination nodes")
    
    # Check types reference valid symbols
    if types:
        type_symbol_ids = {t['symbol'] for t in types}
        missing_type_symbols = type_symbol_ids - node_ids
        
        if missing_type_symbols:
            issues.append(f"Types reference {len(missing_type_symbols)} non-existent symbols")
    
    return issues


def main():
    """Main validation function."""
    nodes_path = Path('nodes.jsonl')
    edges_path = Path('edges.jsonl')
    types_path = Path('types.jsonl')
    
    console.print("[bold cyan]Validating Extracted Data[/bold cyan]\n")
    
    # Check files exist
    if not nodes_path.exists():
        console.print(f"[bold red]Error: {nodes_path} not found[/bold red]")
        return 1
    
    if not edges_path.exists():
        console.print(f"[bold red]Error: {edges_path} not found[/bold red]")
        return 1
    
    # Validate nodes
    console.print("[cyan]Validating nodes.jsonl...[/cyan]")
    nodes, node_issues = validate_nodes(str(nodes_path))
    
    if node_issues:
        console.print(f"[yellow]Found {len(node_issues)} issues in nodes:[/yellow]")
        for issue in node_issues[:10]:  # Show first 10
            console.print(f"  {issue}")
        if len(node_issues) > 10:
            console.print(f"  ... and {len(node_issues) - 10} more")
    else:
        console.print(f"[green]✓ Nodes valid: {len(nodes)} nodes[/green]")
    
    # Validate edges
    console.print("\n[cyan]Validating edges.jsonl...[/cyan]")
    edges, edge_issues = validate_edges(str(edges_path))
    
    if edge_issues:
        console.print(f"[yellow]Found {len(edge_issues)} issues in edges:[/yellow]")
        for issue in edge_issues[:10]:
            console.print(f"  {issue}")
        if len(edge_issues) > 10:
            console.print(f"  ... and {len(edge_issues) - 10} more")
    else:
        console.print(f"[green]✓ Edges valid: {len(edges)} edges[/green]")
    
    # Validate types (if exists)
    types = []
    type_issues = []
    if types_path.exists():
        console.print("\n[cyan]Validating types.jsonl...[/cyan]")
        types, type_issues = validate_types(str(types_path))
        
        if type_issues:
            console.print(f"[yellow]Found {len(type_issues)} issues in types:[/yellow]")
            for issue in type_issues[:10]:
                console.print(f"  {issue}")
            if len(type_issues) > 10:
                console.print(f"  ... and {len(type_issues) - 10} more")
        else:
            console.print(f"[green]✓ Types valid: {len(types)} type entries[/green]")
    else:
        console.print("\n[yellow]types.jsonl not found (optional)[/yellow]")
    
    # Check consistency
    console.print("\n[cyan]Checking data consistency...[/cyan]")
    consistency_issues = check_data_consistency(nodes, edges, types)
    
    if consistency_issues:
        console.print("[yellow]Consistency issues:[/yellow]")
        for issue in consistency_issues:
            console.print(f"  {issue}")
    else:
        console.print("[green]✓ Data is consistent[/green]")
    
    # Statistics
    console.print("\n[bold cyan]Data Statistics[/bold cyan]\n")
    
    stats_table = Table(box=box.ROUNDED, title="Summary")
    stats_table.add_column("Metric", style="cyan", width=30)
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Nodes", str(len(nodes)))
    stats_table.add_row("Total Edges", str(len(edges)))
    if types:
        stats_table.add_row("Total Type Entries", str(len(types)))
    
    # Node kind distribution
    if nodes:
        kind_counts = Counter(node['kind'] for node in nodes)
        stats_table.add_row("", "")
        stats_table.add_row("[bold]Node Kinds:[/bold]", "")
        for kind, count in kind_counts.most_common():
            stats_table.add_row(f"  {kind}", str(count))
    
    # Edge relationship distribution
    if edges:
        rel_counts = Counter(edge['rel'] for edge in edges)
        stats_table.add_row("", "")
        stats_table.add_row("[bold]Edge Relationships:[/bold]", "")
        for rel, count in rel_counts.most_common():
            stats_table.add_row(f"  {rel}", str(count))
    
    # Unique files
    if nodes:
        files = {node['meta'].get('file', '') for node in nodes if 'meta' in node}
        stats_table.add_row("", "")
        stats_table.add_row("Unique Files", str(len([f for f in files if f])))
    
    console.print(stats_table)
    
    # Overall status
    total_issues = len(node_issues) + len(edge_issues) + len(type_issues) + len(consistency_issues)
    
    console.print("")
    if total_issues == 0:
        console.print("[bold green]✓ All data is valid and consistent![/bold green]")
        return 0
    else:
        console.print(f"[bold yellow]⚠ Found {total_issues} issues total[/bold yellow]")
        return 1


if __name__ == '__main__':
    sys.exit(main())

