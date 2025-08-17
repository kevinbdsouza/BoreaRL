#!/usr/bin/env python3
"""
Test script to demonstrate smart checkpointing functionality.

This script shows how the system now automatically saves model checkpoints
only when the scalarized_episodic_return improves.
"""

import os
import json
import subprocess
import sys

def test_smart_checkpointing():
    """Test the smart checkpointing functionality"""
    
    print("Testing smart checkpointing functionality...")
    print("=" * 60)
    
    # Test 1: Check if wandb summary file exists and can be read
    print("1. Checking wandb summary file access...")
    wandb_dir = os.path.join(os.getcwd(), 'wandb')
    if os.path.exists(wandb_dir):
        print(f"   ✓ Wandb directory found: {wandb_dir}")
        
        # Look for latest run
        import glob
        run_dirs = glob.glob(os.path.join(wandb_dir, 'run-*'))
        if not run_dirs:
            latest_run_dir = os.path.join(wandb_dir, 'latest-run')
            if os.path.exists(latest_run_dir):
                run_dirs = [latest_run_dir]
        
        if run_dirs:
            latest_run_dir = max(run_dirs, key=os.path.getmtime)
            print(f"   ✓ Latest run directory: {os.path.basename(latest_run_dir)}")
            
            summary_file = os.path.join(latest_run_dir, 'files', 'wandb-summary.json')
            if os.path.exists(summary_file):
                print(f"   ✓ Summary file found: {summary_file}")
                
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    scalarized_return = summary_data.get('metrics/scalarized_episodic_return')
                    if scalarized_return is not None:
                        print(f"   ✓ Scalarized episodic return: {scalarized_return}")
                    else:
                        print("   ⚠ Scalarized episodic return not found in summary")
                        
                except Exception as e:
                    print(f"   ✗ Error reading summary file: {e}")
            else:
                print(f"   ✗ Summary file not found: {summary_file}")
        else:
            print("   ⚠ No wandb run directories found")
    else:
        print(f"   ⚠ Wandb directory not found: {wandb_dir}")
    
    print("\n2. Usage examples:")
    print("   # Train with smart checkpointing (default behavior)")
    print("   python main.py --train --agent eupg --timesteps 10000")
    print("   ")
    print("   # Train with custom save interval")
    print("   python main.py --train --agent eupg --timesteps 10000 --save_interval 50")
    print("   ")
    print("   # Train with wandb disabled (will fall back to regular checkpointing)")
    print("   python main.py --train --agent eupg --timesteps 10000 --no_wandb")
    
    print("\n3. How it works:")
    print("   - The system automatically reads the latest scalarized_episodic_return from wandb/latest-run/files/wandb-summary.json")
    print("   - Only saves checkpoints when the metric improves compared to the previous best")
    print("   - If wandb is disabled or the metric can't be read, falls back to regular checkpointing")
    print("   - Provides informative console output about metric improvements")
    
    print("\n4. Benefits:")
    print("   - Saves disk space by only keeping the best performing models")
    print("   - Ensures you always have the best checkpoint available")
    print("   - Works offline (reads from local wandb files)")
    print("   - Graceful fallback if metric reading fails")
    print("   - No additional configuration needed - it's the default behavior!")

if __name__ == "__main__":
    test_smart_checkpointing()

