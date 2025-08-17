#!/bin/bash

# Forest Energy Balance - Training and Evaluation Script
# This script trains a MORL agent and then evaluates it across different preference weights
#
# Usage: ./train_and_eval.sh [agent] [run_dir_name] [save_interval]
#   agent: Agent type (default: eupg)
#   run_dir_name: Optional run directory name
#   save_interval: Save model every N episodes (default: 100)
#
# Environment variables:
#   AGENT: Agent type (default: eupg)
#   RUN_DIR: Run directory name
#   SAVE_INTERVAL: Save model every N episodes (default: 100)

echo "=== Forest Energy Balance MORL Training and Evaluation ==="
echo "Starting at: $(date)"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Error: Virtual environment not activated. Please ensure .venv exists."
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"
echo ""

# Determine agent (default: eupg). Accept from first CLI arg or AGENT env var.
AGENT_ARG=${1:-${AGENT:-eupg}}
# Optional run directory name from env RUN_DIR or 2nd arg
RUN_DIR_NAME=${2:-${RUN_DIR:-}} 
# Optional save interval from env SAVE_INTERVAL or 3rd arg (default: 100)
SAVE_INTERVAL=${3:-${SAVE_INTERVAL:-100}}
python main.py --train_then_eval --timesteps 300000 --agent ${AGENT_ARG} ${RUN_DIR_NAME:+--run_dir_name ${RUN_DIR_NAME}}

# Check if training was successfu
if [ $? -ne 0 ]; then
    echo "Error: Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"
echo "Model saved as: ${MODEL_FILE} (if supported by the selected agent)"
echo ""

echo "=== COMPLETED ==="
echo "Training and evaluation completed successfully!"
echo "Results saved as: plots/morl_pareto_front.png"
echo "Finished at: $(date)"