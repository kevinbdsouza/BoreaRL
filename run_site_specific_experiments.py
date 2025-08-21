#!/usr/bin/env python3
"""
Script to run site-specific experiments with different EUPG_DEFAULT_WEIGHTS.

This script runs training with USE_FIXED_PREFERENCE_DEFAULT = True and different
weight combinations, creating separate folders for each experiment with ss_ prefix.
Runs 5 different weights with 5 seeds per weight.
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

# Seeds to test for each weight combination
SEEDS = [42, 123, 456, 789, 999]



def run_experiment(weights, seed, run_dir_name):
    """Run a single experiment with the given weights and seed."""
    print(f"\n{'='*60}")
    print(f"Running experiment with weights {weights}, seed {seed}")
    print(f"Run directory: {run_dir_name}")
    print(f"{'='*60}")
    
    # Create a temporary Python script that sets the constants and runs the experiment
    temp_script_content = f'''
import sys
import os

# Set the constants before importing any modules
import borearl.constants as const
const.EUPG_DEFAULT_WEIGHTS = {weights}
const.SITE_WEATHER_SEED_DEFAULT = {seed}

# Generate site-specific overrides for this seed
site_overrides = const.generate_site_overrides_from_seed({seed})

# Now run the main script
sys.argv = [
    "main.py",
    "--train_then_eval",
    "--timesteps", "50000",
    "--agent", "eupg", 
    "--run_dir_name", "{run_dir_name}",
    "--site_specific"
]

# Set environment variable for site_overrides
os.environ["BOREARL_SITE_OVERRIDES"] = str(site_overrides)

# Import and run main
from main import main
main()
'''
    
    # Write temporary script
    temp_script_path = f"temp_run_{seed}.py"
    with open(temp_script_path, 'w') as f:
        f.write(temp_script_content)
    
    # Run the temporary script
    cmd = [sys.executable, temp_script_path]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ Experiment completed successfully for {run_dir_name}")
    else:
        print(f"❌ Experiment failed for {run_dir_name}")
        return False
    
    return True

def main():
    """Main function to run all experiments."""
    total_experiments = len(WEIGHT_COMBINATIONS) * len(SEEDS)
    print("🚀 Starting Site-Specific EUPG Weight Experiments")
    print(f"Total experiments to run: {total_experiments}")
    print(f"Weights: {len(WEIGHT_COMBINATIONS)} combinations")
    print(f"Seeds per weight: {len(SEEDS)}")
    print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    successful_runs = 0
    failed_runs = 0
    experiment_count = 0
    
    for weights in WEIGHT_COMBINATIONS:
        w1, w2 = weights
        weight_dir_name = f"ss_{w1}_{w2}"
        
        print(f"\n📊 Weight combination: {weights}")
        print(f"Weight directory: {weight_dir_name}")
        
        for seed in SEEDS:
            experiment_count += 1
            run_dir_name = f"{weight_dir_name}_seed{seed}"
            
            print(f"\n  🔬 Experiment {experiment_count}/{total_experiments}")
            print(f"  Weights: {weights}, Seed: {seed}")
            print(f"  Folder: {run_dir_name}")
            
            # Run the experiment
            success = run_experiment(weights, seed, run_dir_name)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
            
            # Small delay between experiments
            if experiment_count < total_experiments:
                print("  ⏳ Waiting 3 seconds before next experiment...")
                time.sleep(3)
        
        # Longer delay between weight combinations
        if weights != WEIGHT_COMBINATIONS[-1]:
            print(f"\n⏳ Waiting 10 seconds before next weight combination...")
            time.sleep(10)
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Weight combinations: {len(WEIGHT_COMBINATIONS)}")
    print(f"Seeds per weight: {len(SEEDS)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Success rate: {successful_runs/total_experiments*100:.1f}%")
    print(f"Ending time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_runs == total_experiments:
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  Some experiments failed. Check the logs for details.")
    
    print(f"\n📁 Results saved in logs/ directory:")
    for weights in WEIGHT_COMBINATIONS:
        w1, w2 = weights
        weight_dir_name = f"ss_{w1}_{w2}"
        print(f"  - {weight_dir_name}_seed*/")

if __name__ == "__main__":
    main()
