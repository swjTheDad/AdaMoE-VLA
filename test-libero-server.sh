#!/bin/bash
PROJECT_DIR="your_project_root_directory"
VENV_PATH="$PROJECT_DIR/.venv"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/your_config_name/your_experiments_name"
STEP=${1:-90000}
PORT=${2:-8001}
GPU_ID=${3:-0}

cd "$PROJECT_DIR" || exit 1
source "$VENV_PATH/bin/activate"

# Environment Variables
export OPENPI_DATA_HOME="$PROJECT_DIR/checkpoints"
export XDG_CACHE_HOME="$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Start Inference
python scripts/serve_policy.py \
  --port "$PORT" \
  --env "LIBERO policy:checkpoint" \
  --policy.config="your_config_name" \
  --policy.dir="$CHECKPOINT_DIR/$STEP"
