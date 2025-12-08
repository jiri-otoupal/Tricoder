"""Symbol extraction utilities for TriCoder."""
# This module contains the symbol extraction logic
# Imported from extract_symbols.py for package distribution

import ast
import fnmatch
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


class SymbolExtractor(ast.NodeVisitor):
    """AST visitor to extract symbols and relationships."""

    def __init__(self, file_path: str, excluded_keywords: Set[str] = None):
        self.file_path = file_path
        self.symbols = []
        self.edges = []
        self.current_class = None
        self.current_function = None
        self.imports = {}
        self.symbol_counter = 0
        self.symbol_map = {}
        self.type_tokens = defaultdict(int)
        self.added_symbol_ids = set()  # Track which symbol IDs have been added
        self.edge_weights = {}  # Track edge weights for aggregation: (src, dst, rel) -> weight
        self.excluded_keywords = excluded_keywords or set()  # Symbol names to exclude

    def _sanitize_name(self, name: str) -> str:
        """Sanitize symbol name for use in ID (remove/replace invalid characters)."""
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to avoid very long IDs
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unnamed'
        return sanitized.lower()
    
    def _get_symbol_id(self, name: str, kind: str) -> str:
        """Generate unique symbol ID with descriptive name."""
        key = f"{self.file_path}:{kind}:{name}"
        if key not in self.symbol_map:
            self.symbol_counter += 1
            sanitized_name = self._sanitize_name(name)
            sanitized_kind = kind.lower()
            # Format: {kind}_{name}_{counter}
            self.symbol_map[key] = f"{sanitized_kind}_{sanitized_name}_{self.symbol_counter:04d}"
        return self.symbol_map[key]

    def _add_symbol(self, name: str, kind: str, lineno: int,
                    extra_meta: Optional[Dict] = None):
        """Add a symbol to the collection."""
        # Skip if symbol name is in excluded keywords
        if name.lower() in self.excluded_keywords:
            return None
        
        symbol_id = self._get_symbol_id(name, kind)
        
        # Skip if symbol already added (prevents duplicates)
        if symbol_id in self.added_symbol_ids:
            return symbol_id
        
        meta = {
            "file": self.file_path,
            "lineno": lineno,
            "typing": []
        }
        if extra_meta:
            meta.update(extra_meta)

        self.symbols.append({
            "id": symbol_id,
            "kind": kind,
            "name": name,
            "meta": meta
        })
        self.added_symbol_ids.add(symbol_id)
        return symbol_id

    def _add_edge(self, src: str, dst: str, rel: str, weight: float = 1.0):
        """Add a relationship edge, aggregating weights for duplicate edges."""
        edge_key = (src, dst, rel)
        if edge_key in self.edge_weights:
            # Aggregate weights: use maximum (consistent with training code)
            self.edge_weights[edge_key] = max(self.edge_weights[edge_key], weight)
        else:
            self.edge_weights[edge_key] = weight

    def _add_type_token(self, symbol_id: str, type_token: str, count: int = 1):
        """Add type token information."""
        self.type_tokens[(symbol_id, type_token)] += count

    def visit_Module(self, node):
        """Visit module node."""
        file_symbol = self._add_symbol(
            os.path.basename(self.file_path),
            "file",
            node.lineno if hasattr(node, 'lineno') else 1
        )
        self.generic_visit(node)
        return file_symbol

    def visit_ClassDef(self, node):
        """Visit class definition."""
        bases = []
        for base in node.bases:
            if hasattr(ast, 'unparse'):
                bases.append(ast.unparse(base))
            elif isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(str(base))

        class_id = self._add_symbol(
            node.name,
            "class",
            node.lineno,
            {"bases": bases}
        )

        # Link class to file
        file_symbol = self._get_symbol_id(os.path.basename(self.file_path), "file")
        self._add_edge(file_symbol, class_id, "defines_in_file", 1.0)

        # Handle inheritance
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                base_id = self._get_symbol_id(base_name, "class")
                self._add_edge(class_id, base_id, "inherits", 1.0)

        old_class = self.current_class
        self.current_class = class_id
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        func_id = self._add_symbol(
            node.name,
            "function",
            node.lineno,
            {"args": len(node.args.args)}
        )

        # Link function to containing class or file
        if self.current_class:
            self._add_edge(self.current_class, func_id, "defines_in_file", 1.0)
        else:
            file_symbol = self._get_symbol_id(os.path.basename(self.file_path), "file")
            self._add_edge(file_symbol, func_id, "defines_in_file", 1.0)

        # Extract return type annotation
        if node.returns:
            return_type = self._extract_type_annotation(node.returns)
            if return_type:
                self._add_type_token(func_id, return_type, 1)

        # Extract parameter type annotations
        for arg in node.args.args:
            if arg.annotation:
                param_type = self._extract_type_annotation(arg.annotation)
                if param_type:
                    self._add_type_token(func_id, param_type, 1)

        old_function = self.current_function
        self.current_function = func_id
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def visit_Import(self, node):
        """Visit import statement."""
        for alias in node.names:
            module_name = alias.name
            import_name = alias.asname if alias.asname else alias.name.split('.')[0]
            import_id = self._get_symbol_id(import_name, "import")

            # Store import mapping
            self.imports[import_name] = import_id

            # Link import to file
            file_symbol = self._get_symbol_id(os.path.basename(self.file_path), "file")
            self._add_edge(file_symbol, import_id, "imports", 1.0)

    def visit_ImportFrom(self, node):
        """Visit from ... import statement."""
        module_name = node.module if node.module else ""
        for alias in node.names:
            import_name = alias.asname if alias.asname else alias.name
            import_id = self._get_symbol_id(import_name, "import")

            # Store import mapping
            self.imports[import_name] = import_id

            # Link import to file
            file_symbol = self._get_symbol_id(os.path.basename(self.file_path), "file")
            self._add_edge(file_symbol, import_id, "imports", 1.0)

    def visit_Call(self, node):
        """Visit function call."""
        if self.current_function or self.current_class:
            caller_id = self.current_function if self.current_function else self.current_class

            # Extract called function/class name
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
                callee_id = self._get_symbol_id(callee_name, "function")
                self._add_edge(caller_id, callee_id, "calls", 1.0)
            elif isinstance(node.func, ast.Attribute):
                # Handle method calls like obj.method()
                if isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id
                    method_name = node.func.attr
                    method_id = self._get_symbol_id(f"{obj_name}.{method_name}", "function")
                    self._add_edge(caller_id, method_id, "calls", 1.0)

            # Check for co-occurrence with arguments
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    var_id = self._get_symbol_id(arg.id, "var")
                    self._add_edge(caller_id, var_id, "cooccurs", 0.5)

        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visit variable assignment."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_id = self._add_symbol(
                    target.id,
                    "var",
                    node.lineno
                )

                # Link variable to containing function/class/file
                if self.current_function:
                    self._add_edge(self.current_function, var_id, "defines_in_file", 1.0)
                elif self.current_class:
                    self._add_edge(self.current_class, var_id, "defines_in_file", 1.0)
                else:
                    file_symbol = self._get_symbol_id(os.path.basename(self.file_path), "file")
                    self._add_edge(file_symbol, var_id, "defines_in_file", 1.0)

        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        """Visit annotated assignment (e.g., x: int = 5)."""
        if isinstance(node.target, ast.Name):
            var_id = self._add_symbol(
                node.target.id,
                "var",
                node.lineno
            )
            
            # Link variable to containing function/class/file
            if self.current_function:
                self._add_edge(self.current_function, var_id, "defines_in_file", 1.0)
            elif self.current_class:
                self._add_edge(self.current_class, var_id, "defines_in_file", 1.0)
            else:
                file_symbol = self._get_symbol_id(os.path.basename(self.file_path), "file")
                self._add_edge(file_symbol, var_id, "defines_in_file", 1.0)
            
            # Extract type annotation
            if node.annotation:
                var_type = self._extract_type_annotation(node.annotation)
                if var_type:
                    self._add_type_token(var_id, var_type, 1)
        
        self.generic_visit(node)

    def visit_Name(self, node):
        """Visit name node (variable reference)."""
        if isinstance(node.ctx, ast.Load):
            if self.current_function or self.current_class:
                referencer_id = self.current_function if self.current_function else self.current_class
                var_id = self._get_symbol_id(node.id, "var")
                self._add_edge(referencer_id, var_id, "cooccurs", 0.3)
        self.generic_visit(node)

    def _extract_type_annotation(self, node) -> Optional[str]:
        """Extract type annotation string from AST node."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Fallback for older Python versions
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name):
                        base = node.value.id
                        if isinstance(node.slice, ast.Name):
                            return f"{base}[{node.slice.id}]"
                        elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Name):
                            return f"{base}[{node.slice.value.id}]"
                        return base
                elif isinstance(node, ast.Attribute):
                    return f"{node.value.id}.{node.attr}" if isinstance(node.value, ast.Name) else None
        except:
            pass
        return None


class GitIgnoreMatcher:
    """Simple gitignore pattern matcher."""

    def __init__(self, gitignore_path: Path, root_path: Path):
        self.root_path = root_path.resolve()
        self.patterns = []
        self.load_patterns(gitignore_path)

    def load_patterns(self, gitignore_path: Path):
        """Load patterns from .gitignore file."""
        if not gitignore_path.exists():
            return

        with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Handle negation (patterns starting with !)
                negated = line.startswith('!')
                if negated:
                    pattern = line[1:]
                else:
                    pattern = line

                # Normalize pattern
                pattern = pattern.rstrip('/')

                self.patterns.append((pattern, negated))

    def matches(self, file_path: Path) -> bool:
        """Check if file/directory matches any gitignore pattern."""
        file_path = file_path.resolve()

        # Get relative path from root
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            # File is outside root, don't ignore
            return False

        rel_str = str(rel_path).replace('\\', '/')
        parts = rel_str.split('/')

        # Check against all patterns
        matched = False
        for pattern, negated in self.patterns:
            if self._match_pattern(pattern, rel_str, parts):
                if negated:
                    # Negation pattern - explicitly include
                    matched = False
                else:
                    matched = True

        return matched

    def _match_pattern(self, pattern: str, rel_str: str, parts: List[str]) -> bool:
        """Match a single gitignore pattern."""
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            # Match if any directory component matches
            return any(self._match_glob(pattern, part) for part in parts[:-1])

        # Handle absolute patterns (starting with /)
        if pattern.startswith('/'):
            pattern = pattern[1:]
            return self._match_glob(pattern, parts[0] if parts else '')

        # Handle patterns with ** (matches any number of directories)
        if '**' in pattern:
            # Convert ** to regex
            regex_pattern = pattern.replace('**', '.*').replace('*', '[^/]*')
            regex_pattern = regex_pattern.replace('?', '.')
            try:
                return bool(re.search(regex_pattern, rel_str))
            except:
                return fnmatch.fnmatch(rel_str, pattern)

        # Simple glob pattern - check if any part matches
        return any(self._match_glob(pattern, part) for part in parts)

    def _match_glob(self, pattern: str, text: str) -> bool:
        """Match glob pattern against text."""
        return fnmatch.fnmatch(text, pattern)


def find_gitignore(root_path: Path) -> Optional[Path]:
    """Find .gitignore file in directory hierarchy."""
    current = root_path.resolve()
    while current != current.parent:
        gitignore = current / '.gitignore'
        if gitignore.exists():
            return gitignore
        current = current.parent
    return None


def should_process_file(file_path: str, include_dirs: List[str],
                        exclude_dirs: List[str],
                        gitignore_matcher: Optional[GitIgnoreMatcher] = None) -> bool:
    """Check if file should be processed based on directory filters and gitignore."""
    path = Path(file_path)

    # Check gitignore first (if enabled)
    if gitignore_matcher and gitignore_matcher.matches(path):
        return False

    # Check exclude patterns
    for exclude in exclude_dirs:
        if exclude in str(path):
            return False

    # If include_dirs is empty, process all (except excluded)
    if not include_dirs:
        return True

    # Check if file is in any included directory
    for include in include_dirs:
        if include in str(path):
            return True

    return False


def extract_from_file(file_path: str, excluded_keywords: Set[str] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Extract symbols, edges, and types from a source file.
    
    Note: Currently only supports Python files (uses AST parsing).
    For other languages, additional parsers would need to be implemented.
    
    Args:
        file_path: Path to the source file
        excluded_keywords: Set of symbol names to exclude from extraction
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        tree = ast.parse(content, filename=file_path)
        extractor = SymbolExtractor(file_path, excluded_keywords=excluded_keywords)
        extractor.visit(tree)

        # Convert aggregated edges from dictionary to list format
        edges = []
        for (src, dst, rel), weight in extractor.edge_weights.items():
            edges.append({
                "src": src,
                "dst": dst,
                "rel": rel,
                "weight": weight
            })

        # Convert type_tokens to list format
        types = []
        for (symbol_id, type_token), count in extractor.type_tokens.items():
            types.append({
                "symbol": symbol_id,
                "type_token": type_token,
                "count": count
            })

        return extractor.symbols, edges, types
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], [], []


def extract_from_directory(root_dir: str, include_dirs: List[str] = None,
                           exclude_dirs: List[str] = None,
                           extensions: List[str] = None,
                           excluded_keywords: Set[str] = None,
                           output_nodes: str = "nodes.jsonl",
                           output_edges: str = "edges.jsonl",
                           output_types: str = "types.jsonl",
                           use_gitignore: bool = True):
    """Extract symbols from directory recursively.
    
    Args:
        root_dir: Root directory to scan
        include_dirs: List of directories to include (empty = all)
        exclude_dirs: List of directories to exclude
        extensions: List of file extensions to process
        excluded_keywords: Set of symbol names to exclude from extraction
        output_nodes: Output file for nodes
        output_edges: Output file for edges
        output_types: Output file for types
        use_gitignore: Whether to use .gitignore filtering
    """
    import click

    if include_dirs is None:
        include_dirs = []
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']
    if extensions is None:
        extensions = ['py']  # Default to Python files
    if excluded_keywords is None:
        excluded_keywords = set()

    root_path = Path(root_dir).resolve()
    all_symbols = []
    all_edges = []
    all_types = []
    seen_symbols = {}

    # Load gitignore if enabled
    gitignore_matcher = None
    if use_gitignore:
        gitignore_path = find_gitignore(root_path)
        if gitignore_path:
            gitignore_matcher = GitIgnoreMatcher(gitignore_path, root_path)
            click.echo(f"Using .gitignore from: {gitignore_path}")
        else:
            click.echo("No .gitignore found, skipping gitignore filtering")

    click.echo(f"Scanning directory: {root_dir}")
    click.echo(f"Include dirs: {include_dirs if include_dirs else 'all'}")
    click.echo(f"Exclude dirs: {exclude_dirs}")
    click.echo(f"Extensions: {', '.join(extensions)}")
    if excluded_keywords:
        click.echo(f"Excluded keywords: {len(excluded_keywords)} symbol names")
    click.echo(f"Gitignore: {'enabled' if use_gitignore else 'disabled'}")

    # Normalize extensions: ensure they start with dot
    normalized_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    source_files = []
    skipped_gitignore = 0

    for root, dirs, files in os.walk(root_path):
        # Filter out excluded directories
        dirs_to_remove = []
        for d in dirs:
            dir_path = Path(root) / d
            # Check gitignore
            if gitignore_matcher and gitignore_matcher.matches(dir_path):
                dirs_to_remove.append(d)
                skipped_gitignore += 1
                continue
            # Check exclude patterns
            if any(exclude in str(dir_path) for exclude in exclude_dirs):
                dirs_to_remove.append(d)

        for d in dirs_to_remove:
            dirs.remove(d)

        for file in files:
            # Check if file has one of the allowed extensions
            file_path_obj = Path(file)
            file_ext = file_path_obj.suffix.lower()
            if file_ext in normalized_extensions:
                file_path = Path(root) / file
                file_path_str = str(file_path)

                # Check gitignore
                if gitignore_matcher and gitignore_matcher.matches(file_path):
                    skipped_gitignore += 1
                    continue

                if should_process_file(file_path_str, include_dirs, exclude_dirs, gitignore_matcher):
                    source_files.append(file_path_str)

    ext_display = ', '.join(extensions)
    click.echo(f"Found {len(source_files)} file(s) with extensions [{ext_display}] to process")
    if skipped_gitignore > 0:
        click.echo(f"Skipped {skipped_gitignore} files/directories due to .gitignore")

    with click.progressbar(source_files, label='Processing files') as bar:
        for file_path in bar:
            symbols, edges, types = extract_from_file(file_path, excluded_keywords=excluded_keywords)

            # Deduplicate symbols by ID
            for symbol in symbols:
                symbol_id = symbol['id']
                if symbol_id not in seen_symbols:
                    seen_symbols[symbol_id] = symbol
                    all_symbols.append(symbol)

            all_edges.extend(edges)
            all_types.extend(types)

    excluded_msg = f" (excluded symbol names: {len(excluded_keywords)} keywords)" if excluded_keywords else ""
    click.echo(
        f"\nExtracted {len(all_symbols)} symbols, {len(all_edges)} edges, {len(all_types)} type tokens{excluded_msg}")

    # Write output files
    click.echo(f"Writing {output_nodes}...")
    with open(output_nodes, 'w') as f:
        for symbol in all_symbols:
            f.write(json.dumps(symbol) + '\n')

    click.echo(f"Writing {output_edges}...")
    with open(output_edges, 'w') as f:
        for edge in all_edges:
            f.write(json.dumps(edge) + '\n')

    click.echo(f"Writing {output_types}...")
    with open(output_types, 'w') as f:
        for type_entry in all_types:
            f.write(json.dumps(type_entry) + '\n')

    click.echo(click.style("✓ Extraction complete!", fg='green'))
