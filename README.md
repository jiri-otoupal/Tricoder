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
- **GPU Acceleration**: Supports CUDA (NVIDIA) and MPS (Mac) for faster training
- **Keyword Search**: Search symbols by keywords and type tokens
- **Graph Optimization**: Filter out low-value nodes and edges to improve training efficiency

## Installation

### Using Poetry (Recommended)

```bash
poetry install
```

### Using pip

```bash
pip install .
```

### GPU Support (Optional)

For NVIDIA GPUs (CUDA):
```bash
pip install cupy-cuda12x
```

For Mac GPUs (MPS):
```bash
pip install torch
```

## Usage

### 1. Extract Symbols from Codebase

```bash
# Basic extraction (Python files only)
tricoder extract --input-dir /path/to/codebase

# Extract specific file types
tricoder extract --input-dir /path/to/codebase --extensions "py,js,ts"

# Exclude specific keywords from extraction
tricoder extract --input-dir /path/to/codebase --exclude-keywords debug --exclude-keywords temp

# Custom output files
tricoder extract --input-dir /path/to/codebase --output-nodes my_nodes.jsonl --output-edges my_edges.jsonl
```

**Extraction Options:**
- `--input-dir`, `--root`, `-r`: Input directory to scan (default: current directory)
- `--extensions`, `--ext`: Comma-separated file extensions (default: `py`)
- `--include-dirs`, `-i`: Include only specific subdirectories (can specify multiple)
- `--exclude-dirs`, `-e`: Exclude directories (default: `.venv`, `__pycache__`, `.git`, `node_modules`, `.pytest_cache`)
- `--exclude-keywords`, `--exclude`: Exclude symbol names (appended to default excluded keywords)
- `--output-nodes`, `-n`: Output file for nodes (default: `nodes.jsonl`)
- `--output-edges`, `-d`: Output file for edges (default: `edges.jsonl`)
- `--output-types`, `-t`: Output file for types (default: `types.jsonl`)
- `--no-gitignore`: Disable `.gitignore` filtering (enabled by default)

### 2. Optimize Graph (Optional)

Reduce graph size by filtering low-value nodes and edges:

```bash
# Basic optimization (overwrites input files)
tricoder optimize

# Custom output files
tricoder optimize --output-nodes nodes_opt.jsonl --output-edges edges_opt.jsonl

# Customize thresholds
tricoder optimize --min-edge-weight 0.5 --remove-isolated --remove-generic

# Keep isolated nodes
tricoder optimize --keep-isolated
```

**Optimization Options:**
- `--nodes`, `-n`: Input nodes file (default: `nodes.jsonl`)
- `--edges`, `-e`: Input edges file (default: `edges.jsonl`)
- `--types`, `-t`: Input types file (default: `types.jsonl`, optional)
- `--output-nodes`, `-N`: Output nodes file (default: overwrites input)
- `--output-edges`, `-E`: Output edges file (default: overwrites input)
- `--output-types`, `-T`: Output types file (default: overwrites input)
- `--min-edge-weight`: Minimum edge weight to keep (default: `0.3`)
- `--remove-isolated`: Remove nodes with no edges (default: `True`)
- `--keep-isolated`: Keep isolated nodes (overrides `--remove-isolated`)
- `--remove-generic`: Remove generic names (default: `True`)
- `--keep-generic`: Keep generic names (overrides `--remove-generic`)
- `--exclude-keywords`, `--exclude`: Additional keywords to exclude (can specify multiple)

### 3. Train Model

```bash
# Basic training
tricoder train --out model_output

# With GPU acceleration
tricoder train --out model_output --use-gpu

# Fast mode (faster training, slightly lower quality)
tricoder train --out model_output --fast

# Custom dimensions
tricoder train --out model_output --graph-dim 128 --context-dim 128 --final-dim 256

# Custom training parameters
tricoder train --out model_output --num-walks 20 --walk-length 100 --train-ratio 0.9
```

**Training Options:**
- `--nodes`, `-n`: Path to nodes.jsonl (default: `nodes.jsonl`)
- `--edges`, `-e`: Path to edges.jsonl (default: `edges.jsonl`)
- `--types`, `-t`: Path to types.jsonl (default: `types.jsonl`, optional)
- `--out`, `-o`: Output directory (required)
- `--graph-dim`: Graph view dimensionality (default: auto-calculated)
- `--context-dim`: Context view dimensionality (default: auto-calculated)
- `--typed-dim`: Typed view dimensionality (default: auto-calculated)
- `--final-dim`: Final fused embedding dimensionality (default: auto-calculated)
- `--num-walks`: Number of random walks per node (default: `10`)
- `--walk-length`: Length of each random walk (default: `80`)
- `--train-ratio`: Fraction of edges for training (default: `0.8`)
- `--random-state`: Random seed for reproducibility (default: `42`)
- `--fast`: Enable fast mode (reduces parameters for faster training)
- `--use-gpu`: Enable GPU acceleration (CUDA or MPS, falls back to CPU if unavailable)

### 4. Query Model

```bash
# Query by symbol ID
tricoder query --model-dir model_output --symbol function_my_function_0001 --top-k 10

# Search by keywords
tricoder query --model-dir model_output --keywords "database connection" --top-k 10

# Multi-word phrases (use quotes)
tricoder query --model-dir model_output --keywords '"user authentication" login'

# Exclude specific keywords from results
tricoder query --model-dir model_output --keywords handler --exclude-keywords debug --exclude-keywords temp

# Interactive mode
tricoder query --model-dir model_output --interactive
```

**Query Options:**
- `--model-dir`, `-m`: Path to model directory (required)
- `--symbol`, `-s`: Symbol ID to query
- `--keywords`, `-w`: Keywords to search for (use quotes for multi-word: `"my function"`)
- `--top-k`, `-k`: Number of results to return (default: `10`)
- `--exclude-keywords`, `--exclude`: Additional keywords to exclude (appended to default excluded keywords)
- `--interactive`, `-i`: Interactive mode

### 5. Incremental Retraining

Retrain only on changed files since last training:

```bash
# Basic retraining (detects changed files automatically)
tricoder retrain --model-dir model_output --codebase-dir /path/to/codebase

# Force full retraining
tricoder retrain --model-dir model_output --codebase-dir /path/to/codebase --force

# Custom training parameters
tricoder retrain --model-dir model_output --codebase-dir /path/to/codebase --num-walks 20
```

**Retrain Options:**
- `--model-dir`, `-m`: Path to existing model directory (required)
- `--codebase-dir`, `-c`: Path to codebase root (default: current directory)
- `--output-nodes`, `-n`: Temporary nodes file (default: `nodes_retrain.jsonl`)
- `--output-edges`, `-d`: Temporary edges file (default: `edges_retrain.jsonl`)
- `--output-types`, `-t`: Temporary types file (default: `types_retrain.jsonl`)
- `--graph-dim`, `--context-dim`, `--typed-dim`, `--final-dim`: Override model dimensions
- `--num-walks`, `--walk-length`, `--train-ratio`, `--random-state`: Training parameters
- `--force`: Force full retraining even if no files changed

## Examples

### Complete Workflow

```bash
# 1. Extract symbols from codebase
tricoder extract --input-dir ./my_project --extensions "py,js"

# 2. (Optional) Optimize the graph
tricoder optimize --min-edge-weight 0.4

# 3. Train model with GPU acceleration
tricoder train --out ./models/my_project --use-gpu

# 4. Query for similar symbols
tricoder query --model-dir ./models/my_project --keywords "database" --top-k 5

# 5. After code changes, retrain incrementally
tricoder retrain --model-dir ./models/my_project --codebase-dir ./my_project
```

### Keyword Search Examples

```bash
# Search for authentication-related code
tricoder query --model-dir model_output --keywords "auth login password"

# Search for specific function name
tricoder query --model-dir model_output --keywords '"process_payment"'

# Search excluding common keywords
tricoder query --model-dir model_output --keywords handler --exclude-keywords temp --exclude-keywords debug
```

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- gensim >= 4.0.0
- annoy >= 1.17.0
- click >= 8.0.0
- rich >= 13.0.0

**Optional (for GPU acceleration):**
- cupy-cuda12x >= 12.0.0 (for NVIDIA GPUs)
- torch >= 2.0.0 (for Mac GPUs or CUDA fallback)

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
