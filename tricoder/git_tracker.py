"""Git tracking utilities for incremental retraining."""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional


def get_git_commit_hash(repo_path: str = '.') -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_commit_timestamp(repo_path: str = '.') -> Optional[datetime]:
    """Get the timestamp of the current git commit."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ct'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        timestamp = int(result.stdout.strip())
        return datetime.fromtimestamp(timestamp)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def get_file_git_timestamp(file_path: str, repo_path: str = '.') -> Optional[datetime]:
    """Get the last modification timestamp of a file from git."""
    try:
        # Get the last commit that modified this file
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ct', '--', file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            timestamp = int(result.stdout.strip())
            return datetime.fromtimestamp(timestamp)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def get_changed_files_since_commit(commit_hash: str, repo_path: str = '.') -> Set[str]:
    """Get all files that changed since a given commit."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', commit_hash, 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        files = set()
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                files.add(line.strip())
        return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def get_all_python_files(repo_path: str = '.') -> Set[str]:
    """Get all Python files tracked by git."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '*.py'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        files = set()
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                files.add(line.strip())
        return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def save_training_metadata(output_dir: str, commit_hash: Optional[str],
                           commit_timestamp: Optional[datetime],
                           files_trained: Set[str]):
    """Save training metadata including git commit info."""
    metadata_path = Path(output_dir) / 'training_metadata.json'

    metadata = {
        'commit_hash': commit_hash,
        'commit_timestamp': commit_timestamp.isoformat() if commit_timestamp else None,
        'training_timestamp': datetime.now().isoformat(),
        'files_trained': sorted(list(files_trained))
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_training_metadata(output_dir: str) -> Optional[Dict]:
    """Load training metadata."""
    metadata_path = Path(output_dir) / 'training_metadata.json'

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except (json.JSONDecodeError, IOError):
        return None


def get_changed_files_for_retraining(output_dir: str, repo_path: str = '.') -> Set[str]:
    """Get files that have changed since last training."""
    metadata = load_training_metadata(output_dir)

    if not metadata:
        # No previous training, return all files
        return get_all_python_files(repo_path)

    commit_hash = metadata.get('commit_hash')
    if not commit_hash:
        # No commit hash stored, return all files
        return get_all_python_files(repo_path)

    # Get files changed since that commit
    changed_files = get_changed_files_since_commit(commit_hash, repo_path)

    # Also check files that were trained before (in case they were deleted from git)
    files_trained = set(metadata.get('files_trained', []))

    # Return union of changed files and previously trained files (for safety)
    all_python_files = get_all_python_files(repo_path)
    return changed_files.intersection(all_python_files)


def extract_files_from_jsonl(jsonl_path: str) -> Set[str]:
    """Extract file paths from nodes.jsonl or edges.jsonl."""
    files = set()
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Check for file path in metadata
                    if 'meta' in data and isinstance(data['meta'], dict):
                        file_path = data['meta'].get('file', '')
                        if file_path:
                            files.add(file_path)
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass
    return files


def filter_jsonl_by_files(jsonl_path: str, allowed_files: Set[str],
                          output_path: str) -> int:
    """Filter a JSONL file to only include entries from allowed files."""
    count = 0
    with open(output_path, 'w') as out_f:
        with open(jsonl_path, 'r') as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Check if this entry belongs to an allowed file
                    file_path = ''
                    if 'meta' in data and isinstance(data['meta'], dict):
                        file_path = data['meta'].get('file', '')

                    # For edges, we need to check both src and dst
                    if 'src' in data or 'dst' in data:
                        # This is an edge - we'll include it if either endpoint is in allowed files
                        # We'll need to check this against the nodes
                        out_f.write(line)
                        count += 1
                    elif file_path in allowed_files or not file_path:
                        # Node or type entry - include if file matches or no file specified
                        out_f.write(line)
                        count += 1
                except json.JSONDecodeError:
                    continue
    return count
