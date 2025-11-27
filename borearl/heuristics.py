import os
import sys
import numpy as np
import csv
from borearl.agents.common import make_env, set_env_preference

def run_heuristic_evaluation(
    heuristic_type: str = "target_density",
    target_density: int = 1000,
    tolerance: int = 100,
    num_episodes: int = 5,
    output_file: str = "heuristic_results.csv",
    csv_output_dir: str = "logs_heuristic"
):
    """
    Evaluates a heuristic policy.
    
    Args:
        heuristic_type: "target_density" or "conifer_restoration"
        target_density: Target stem density for "target_density" heuristic
        tolerance: Tolerance for "target_density" heuristic
        num_episodes: Number of episodes to evaluate
        output_file: Path to save results CSV
        csv_output_dir: Directory for environment logs
    """
    print(f"Evaluating Heuristic: {heuristic_type}")
    if heuristic_type == "target_density":
        print(f"  Target Density = {target_density} +/- {tolerance}")
    
    env_config = {'csv_output_dir': csv_output_dir}
    env = make_env(env_config)
    
    # Set to generalist mode evaluation settings
    current_env = env
    while hasattr(current_env, 'env'):
        setattr(current_env, 'in_evaluation', True)
        current_env = current_env.env
    setattr(current_env, 'in_evaluation', True)

    results = []
    
    # Evaluate across a range of preference weights to see how it performs on the "scalarized" metric
    # though the policy itself is static (doesn't change behavior based on preference)
    preferences = [0.0, 0.5, 1.0]
    
    for pref in preferences:
        print(f"  Testing with preference weight w_c = {pref}...")
        set_env_preference(env, pref)
        
        episode_rewards = []
        
        for i in range(num_episodes):
            seed = 1000 + i # Use same evaluation seeds as main experiments
            obs, info = env.reset(seed=seed)
            done = False
            total_carbon = 0.0
            total_thaw = 0.0
            
            while not done:
                # Extract density from observation or info
                # Obs index 2 is normalized density: density / 1500
                current_density = obs[2] * 1500.0
                
                if heuristic_type == "target_density":
                    # Target Density Heuristic
                    # - If density < target - tolerance: Plant (+50 or +100)
                    # - If density > target + tolerance: Thin (-50 or -100)
                    # - Else: Do nothing
                    
                    action_density_idx = 2 # Default: 0 change (index 2 in 0..4 scale)
                    
                    if current_density < target_density - tolerance:
                        # Plant
                        if current_density < target_density - tolerance - 50:
                            action_density_idx = 4 # +100
                        else:
                            action_density_idx = 3 # +50
                    elif current_density > target_density + tolerance:
                        # Thin
                        if current_density > target_density + tolerance + 50:
                            action_density_idx = 0 # -100
                        else:
                            action_density_idx = 1 # -50
                    
                    # Always maintain mixed species (e.g. 0.5 conifer) as a "reasonable" middle ground
                    action_mix_idx = 2 # 0.5 conifer fraction
                    
                elif heuristic_type == "conifer_restoration":
                    # Conifer Restoration Heuristic
                    # - Always try to maximize conifer fraction (Action Mix Index 4 = 1.0 conifer)
                    # - Manage density to stay within target range (same as target density but maybe looser?)
                    
                    action_density_idx = 2 # Default: 0 change
                    
                    if current_density < target_density - tolerance:
                        # Plant
                        if current_density < target_density - tolerance - 50:
                            action_density_idx = 4 # +100
                        else:
                            action_density_idx = 3 # +50
                    elif current_density > target_density + tolerance:
                        # Thin
                        if current_density > target_density + tolerance + 50:
                            action_density_idx = 0 # -100
                        else:
                            action_density_idx = 1 # -50
                            
                    action_mix_idx = 4 # 1.0 conifer fraction (Pure Conifer)
                
                else:
                    raise ValueError(f"Unknown heuristic type: {heuristic_type}")
                
                # Encode action
                # Action = density_idx * 5 + mix_idx
                action = action_density_idx * 5 + action_mix_idx
                
                obs, reward, done, truncated, step_info = env.step(action)
                
                total_carbon += reward[0]
                total_thaw += reward[1]
                
                if truncated:
                    done = True
            
            scalarized = pref * total_carbon + (1.0 - pref) * total_thaw
            episode_rewards.append({
                'seed': seed,
                'carbon': total_carbon,
                'thaw': total_thaw,
                'scalarized': scalarized
            })
            
        # Calculate stats
        avg_c = np.mean([r['carbon'] for r in episode_rewards])
        avg_t = np.mean([r['thaw'] for r in episode_rewards])
        avg_s = np.mean([r['scalarized'] for r in episode_rewards])
        
        results.append({
            'preference': pref,
            'avg_carbon': avg_c,
            'avg_thaw': avg_t,
            'avg_scalarized': avg_s
        })
        print(f"    Avg Carbon: {avg_c:.3f}, Avg Thaw: {avg_t:.3f}, Avg Scalarized: {avg_s:.3f}")

    # Write results
    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['preference', 'avg_carbon', 'avg_thaw', 'avg_scalarized'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage:
    # 1. Target Density (1000 stems/ha)
    run_heuristic_evaluation(
        heuristic_type="target_density", 
        target_density=1000, 
        output_file="heuristic_target_density_1000.csv"
    )
    
    # 2. Conifer Restoration (100% Conifer)
    run_heuristic_evaluation(
        heuristic_type="conifer_restoration",
        target_density=1000, 
        output_file="heuristic_conifer_restoration.csv"
    )
