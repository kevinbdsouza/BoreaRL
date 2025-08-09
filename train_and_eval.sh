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

# Training phase
echo "=== PHASE 1: TRAINING ==="
echo "Training MORL agent with 500,000 timesteps..."
python main.py --train --timesteps 500000 --site_specific

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed. Exiting."
    exit 1
fi

echo "Training completed successfully!"
echo "Model saved as: models/eupg_forest_manager.pth"
echo ""

# Evaluation phase
echo "=== PHASE 2: EVALUATION ==="
echo "Evaluating model across different preference weights..."
python main.py --evaluate --model_path models/eupg_forest_manager.pth --eval_episodes 100

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