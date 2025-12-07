"""Subtoken extraction and normalization utilities."""
import re
import string
from typing import List, Tuple


class SimpleStemmer:
    """Lightweight Porter-like stemmer for code subtokens."""

    def __init__(self):
        # Common suffixes to remove
        self.suffixes = [
            ('ing', ''),
            ('ed', ''),
            ('er', ''),
            ('est', ''),
            ('ly', ''),
            ('tion', ''),
            ('sion', ''),
            ('ness', ''),
            ('ment', ''),
            ('able', ''),
            ('ible', ''),
            ('ful', ''),
            ('less', ''),
            ('ize', ''),
            ('ise', ''),
        ]

    def stem(self, word: str) -> str:
        """Apply stemming to a word."""
        if len(word) <= 3:
            return word

        word_lower = word.lower()

        # Try to remove suffixes
        for suffix, replacement in self.suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return word_lower[:-len(suffix)] + replacement

        return word_lower


def split_camel_case(name: str) -> List[str]:
    """Split camelCase or PascalCase identifier into tokens."""
    # Insert space before uppercase letters (but not if already preceded by space)
    # Handle sequences of uppercase letters (e.g., "XMLParser" -> "XML", "Parser")
    tokens = []
    current_token = ""

    for i, char in enumerate(name):
        if char.isupper():
            if current_token and not current_token[-1].isupper():
                # End of lowercase token, start new uppercase token
                tokens.append(current_token)
                current_token = char
            else:
                # Continue uppercase sequence
                current_token += char
        elif char.islower() or char.isdigit():
            if current_token and current_token[-1].isupper() and len(current_token) > 1:
                # End of uppercase sequence (except last char), start new token
                tokens.append(current_token[:-1])
                current_token = current_token[-1] + char
            else:
                current_token += char
        else:
            # Non-alphanumeric character
            if current_token:
                tokens.append(current_token)
                current_token = ""

    if current_token:
        tokens.append(current_token)

    return [t for t in tokens if t]


def split_snake_case(name: str) -> List[str]:
    """Split snake_case identifier into tokens."""
    return [t for t in name.split('_') if t]


def split_kebab_case(name: str) -> List[str]:
    """Split kebab-case identifier into tokens."""
    return [t for t in name.split('-') if t]


def split_numeric(name: str) -> List[str]:
    """Split identifier at numeric boundaries."""
    # Split on transitions between digits and non-digits
    parts = re.split(r'(\d+)', name)
    tokens = []
    for part in parts:
        if part:
            # Further split non-numeric parts by case changes
            if part.isdigit():
                tokens.append(part)
            else:
                # Split camelCase within numeric-separated parts
                camel_tokens = split_camel_case(part)
                tokens.extend(camel_tokens)
    return [t for t in tokens if t]


def extract_subtokens(symbol_name: str, normalize: bool = True) -> Tuple[List[str], List[str]]:
    """
    Extract subtokens from a symbol name.
    
    Args:
        symbol_name: name of the symbol
        normalize: whether to normalize subtokens (lowercase + stem)
    
    Returns:
        Tuple of (raw_subtokens, normalized_subtokens)
    """
    if not symbol_name:
        return [], []

    # Try different splitting strategies
    tokens = set()

    # Split by common delimiters first
    parts = re.split(r'[_\-\s\.]+', symbol_name)
    for part in parts:
        if not part:
            continue

        # Try camelCase splitting
        camel_tokens = split_camel_case(part)
        tokens.update(camel_tokens)

        # Try numeric splitting
        numeric_tokens = split_numeric(part)
        tokens.update(numeric_tokens)

    # Also try direct snake_case and kebab_case
    snake_tokens = split_snake_case(symbol_name)
    tokens.update(snake_tokens)

    kebab_tokens = split_kebab_case(symbol_name)
    tokens.update(kebab_tokens)

    # Filter out empty tokens and very short ones
    raw_tokens = [t for t in tokens if len(t) > 0]

    # Normalize if requested
    normalized_tokens = []
    if normalize:
        stemmer = SimpleStemmer()
        for token in raw_tokens:
            # Remove punctuation
            cleaned = token.strip(string.punctuation)
            if cleaned:
                # Lowercase and stem
                normalized = stemmer.stem(cleaned.lower())
                if normalized:
                    normalized_tokens.append(normalized)
    else:
        normalized_tokens = raw_tokens

    return raw_tokens, normalized_tokens


def get_file_hierarchy(file_path: str) -> Tuple[str, str, str]:
    """
    Extract file hierarchy information.
    
    Args:
        file_path: path to file
    
    Returns:
        Tuple of (file_name, directory_path, top_level_package)
    """
    if not file_path:
        return "", "", ""

    # Normalize path separators
    normalized = file_path.replace('\\', '/')

    # Extract components
    parts = normalized.split('/')
    file_name = parts[-1] if parts else ""

    # Directory is everything except filename
    directory_path = '/'.join(parts[:-1]) if len(parts) > 1 else ""

    # Top-level package is first non-empty component
    top_level = ""
    for part in parts:
        if part and part not in ['.', '..']:
            top_level = part
            break

    return file_name, directory_path, top_level
