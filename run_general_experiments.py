#!/usr/bin/env python3
"""
Script to run experiments with different EUPG_DEFAULT_WEIGHTS in general mode.

This script runs training with USE_FIXED_PREFERENCE_DEFAULT = True and different
weight combinations, creating separate folders for each experiment.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Weight combinations to test
WEIGHT_COMBINATIONS = [
    (1.0, 0.0), 
    (0.75, 0.25), 
    (0.5, 0.5), 
    (0.25, 0.75),  
    (0.0, 1.0),    
]

def modify_constants_file(weights):
    """Temporarily modify constants.py to set the desired weights."""
    constants_file = "borearl/constants.py"
    
    # Read the current constants file
    with open(constants_file, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_file = constants_file + ".backup"
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Replace the EUPG_DEFAULT_WEIGHTS line
    old_line = f"EUPG_DEFAULT_WEIGHTS = (1.0, 0.0)"
    new_line = f"EUPG_DEFAULT_WEIGHTS = {weights}"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the modified content
        with open(constants_file, 'w') as f:
            f.write(content)
        
        return backup_file
    else:
        print(f"Warning: Could not find EUPG_DEFAULT_WEIGHTS line in {constants_file}")
        return None

def restore_constants_file(backup_file):
    """Restore the original constants.py file."""
    if backup_file and os.path.exists(backup_file):
        constants_file = "borearl/constants.py"
        with open(backup_file, 'r') as f:
            content = f.read()
        
        with open(constants_file, 'w') as f:
            f.write(content)
        
        # Remove backup file
        os.remove(backup_file)
        print(f"Restored {constants_file} from backup")

def run_experiment(weights, run_dir_name):
    """Run a single experiment with the given weights."""
    print(f"\n{'='*60}")
    print(f"Running experiment with weights {weights}")
    print(f"Run directory: {run_dir_name}")
    print(f"{'='*60}")
    
    # Modify constants file
    backup_file = modify_constants_file(weights)
    
    # Run the training and evaluation
    cmd = [
        sys.executable, "main.py",
        "--train_then_eval",
        "--timesteps", "100000",  # Adjust as needed
        "--agent", "eupg",
        "--run_dir_name", run_dir_name
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ Experiment completed successfully for {run_dir_name}")
    else:
        print(f"❌ Experiment failed for {run_dir_name}")
        return False
    
    # Always restore the constants file
    restore_constants_file(backup_file)
    
    return True

def main():
    """Main function to run all experiments."""
    print("🚀 Starting General Mode EUPG Weight Experiments")
    print(f"Total experiments to run: {len(WEIGHT_COMBINATIONS)}")
    print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    successful_runs = 0
    failed_runs = 0
    
    for i, weights in enumerate(WEIGHT_COMBINATIONS, 1):
        # Create folder name in format gen_x_y
        w1, w2 = weights
        run_dir_name = f"gen_{w1}_{w2}"
        
        print(f"\n📊 Experiment {i}/{len(WEIGHT_COMBINATIONS)}")
        print(f"Weights: {weights}")
        print(f"Folder: {run_dir_name}")
        
        # Run the experiment
        success = run_experiment(weights, run_dir_name)
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        
        # Small delay between experiments
        if i < len(WEIGHT_COMBINATIONS):
            print("⏳ Waiting 5 seconds before next experiment...")
            time.sleep(5)
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(WEIGHT_COMBINATIONS)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Success rate: {successful_runs/len(WEIGHT_COMBINATIONS)*100:.1f}%")
    print(f"Ending time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_runs == len(WEIGHT_COMBINATIONS):
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  Some experiments failed. Check the logs for details.")
    
    print(f"\n📁 Results saved in logs/ directory:")
    for weights in WEIGHT_COMBINATIONS:
        w1, w2 = weights
        run_dir_name = f"gen_{w1}_{w2}"
        print(f"  - {run_dir_name}/")

if __name__ == "__main__":
    main()
