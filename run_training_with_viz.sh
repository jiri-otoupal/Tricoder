#!/bin/bash
# Run training and visualize statistics

set -e

# Configuration
NODES_FILE="${1:-nodes.jsonl}"
EDGES_FILE="${2:-edges.jsonl}"
TYPES_FILE="${3:-types.jsonl}"
OUTPUT_DIR="${4:-model_dir}"

echo "=========================================="
echo "TriVector Code Intelligence Training"
echo "=========================================="
echo ""
echo "Input files:"
echo "  Nodes: $NODES_FILE"
echo "  Edges: $EDGES_FILE"
echo "  Types: $TYPES_FILE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run training
echo "Starting training..."
python train_tvci.py \
    --nodes "$NODES_FILE" \
    --edges "$EDGES_FILE" \
    --types "$TYPES_FILE" \
    --out "$OUTPUT_DIR"

echo ""
echo "Training completed!"
echo ""

# Launch visualization
echo "Launching statistics visualization..."
python visualize_stats.py \
    --nodes "$NODES_FILE" \
    --edges "$EDGES_FILE" \
    --types "$TYPES_FILE"

