# TriCoder Code Intelligence

[![image](https://img.shields.io/pypi/v/tricoder.svg)](https://pypi.org/project/tricoder/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tricoder)](https://pypi.org/project/tricoder/)

[![Build Status](https://travis-ci.com/jiri-otoupal/pycrosskit.svg?branch=master)](https://travis-ci.com/github/jiri-otoupal/tricoder)
[![Downloads](https://pepy.tech/badge/tricoder)](https://pepy.tech/project/tricoder)

## TriCoder learns high-quality symbol-level embeddings from codebases using three complementary views:

1. **Graph View**: Structural relationships via PPMI and SVD
2. **Context View**: Semantic context via Node2Vec random walks and Word2Vec
3. **Typed View**: Type information via type-token co-occurrence (optional)

## Features

- **Subtoken Semantic Graph**: Captures fine-grained semantic relationships through subtoken analysis
- **File & Module Hierarchy**: Leverages file/directory structure for better clustering
- **Static Call-Graph Expansion**: Propagates call relationships to depth 2-3
- **Type Semantic Expansion**: Expands composite types into constructors and primitives
- **Context Window Co-occurrence**: Captures lexical context within ±5 lines
- **Improved Negative Sampling**: Biased sampling for better temperature calibration
- **Hybrid Similarity Scoring**: Length-penalized cosine similarity
- **Iterative Embedding Smoothing**: Diffusion-based smoothing for better clustering
- **Query-Time Semantic Expansion**: Expands queries with subtokens and types

## Installation

### Using Poetry (Recommended)

```bash
poetry install
```

### Using pip

```bash
pip install .
```

## Usage

### 1. Extract Symbols from Codebase

```bash
tricoder-extract --input-dir /path/to/codebase --output-nodes nodes.jsonl --output-edges edges.jsonl --output-types types.jsonl
```

### 2. Train Model

```bash
tricoder-train --nodes nodes.jsonl --edges edges.jsonl --types types.jsonl --out model_output
```

### 3. Query Model

```bash
# Single query
tricoder-query --model-dir model_output --symbol sym_0001 --top-k 10

# Interactive mode
tricoder-query --model-dir model_output --interactive
```

## Advanced Options

### Training Options

- `--graph-dim`: Graph view dimensionality (default: auto)
- `--context-dim`: Context view dimensionality (default: auto)
- `--typed-dim`: Typed view dimensionality (default: auto)
- `--final-dim`: Final fused embedding dimensionality (default: auto)
- `--num-walks`: Number of random walks per node (default: 10)
- `--walk-length`: Length of each random walk (default: 80)
- `--train-ratio`: Fraction of edges for training (default: 0.8)
- `--random-state`: Random seed for reproducibility (default: 42)

### Extraction Options

- `--include-dirs`: Include only specific subdirectories
- `--exclude-dirs`: Exclude specific directories
- `--no-gitignore`: Disable .gitignore filtering

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- gensim >= 4.0.0
- annoy >= 1.17.0
- click >= 8.0.0
- rich >= 13.0.0

## License

TriCoder is available under a **Non-Commercial License**.

- ✅ **Free for non-commercial use**: Personal projects, education, research, open-source
- ❌ **Commercial license required**: Paid products, SaaS, commercial consulting, enterprise use

For commercial licensing inquiries, please contact: **j.f.otoupal@gmail.com**

See [LICENSE](LICENSE) for full terms and [LICENSE_COMMERCIAL.md](LICENSE_COMMERCIAL.md) for commercial license information.

<hr>
Did I made your life less painfull ? 
<br>
<br>
Support my coffee addiction ;)
<br>
<a href="https://www.buymeacoffee.com/jiriotoupal" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy me a Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
