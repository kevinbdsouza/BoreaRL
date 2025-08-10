#!/bin/bash

# Forest Energy Balance - Training and Evaluation Script
# This script trains a MORL agent and then evaluates it across different preference weights

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
MODEL_FILE="models/${AGENT_ARG}_forest_manager.pth"

# Training phase
echo "=== PHASE 1: TRAINING ==="
echo "Training MORL agent (${AGENT_ARG}) with 10,000 timesteps..."
python main.py --train --timesteps 10000 --site_specific --agent ${AGENT_ARG}

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"
echo "Model saved as: ${MODEL_FILE} (if supported by the selected agent)"
echo ""

# Evaluation phase
echo "=== PHASE 2: EVALUATION ==="
echo "Evaluating model across different preference weights using agent ${AGENT_ARG}..."
python main.py --evaluate --model_path ${MODEL_FILE} --eval_episodes 100 --site_specific --config logs/config.yaml --agent ${AGENT_ARG}

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed."
    exit 1
fi

echo ""
echo "=== COMPLETED ==="
echo "Training and evaluation completed successfully!"
echo "Results saved as: plots/morl_pareto_front.png"
echo "Finished at: $(date)"