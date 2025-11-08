#!/bin/bash
PROJECT_DIR="your_project_root_directory"
VENV_PATH="$PROJECT_DIR/examples/libero/.venv"
RESULTS_DIR="$PROJECT_DIR/data/your_experiment_name"

SEED=${1:-7}
TASK=${2:-libero_10}
PORT=${3:-8001}
TRIALS=${5:-50}
STEP=${6:-90000}

cd "$PROJECT_DIR" || exit 1
source "$VENV_PATH/bin/activate"

export VIRTUAL_ENV="$VENV_PATH"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/third_party/libero"
export LIBERO_CONFIG_PATH="$PROJECT_DIR/third_party/libero/libero"

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/${TASK}_results_${STEP}_seed${SEED}.txt"

python examples/libero/main.py \
  --args.seed "$SEED" \
  --args.task_suite_name "$TASK" \
  --args.port "$PORT" \
  --args.results_file "$OUT" \
  --args.num_trials_per_task "$TRIALS"
