#!/usr/bin/env python3
from __future__ import annotations

# Early disable wandb if requested
import sys, os
if "--no_wandb" in sys.argv:
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from borearl.agents.runner import train as train_morl, evaluate as evaluate_morl_policy
from borearl.agents import AGENTS as _AVAILABLE_AGENTS
from borearl.agents.common import load_simple_yaml
from borearl import constants as const
from borearl.utils.plotting import plot_profiling_statistics


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a MORL agent for forest management.")
    parser.add_argument("--train", action="store_true", help="Train a new model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a trained model.")
    parser.add_argument("--model_path", type=str, default="models/eupg_forest_manager.pth", help="Path to the MORL model file.")
    # Dynamically expose only available agents (based on optional deps present)
    _agent_choices = sorted(list(_AVAILABLE_AGENTS.keys())) or ["eupg"]
    parser.add_argument(
        "--agent",
        type=str,
        choices=_agent_choices,
        default=_agent_choices[0],
        help=f"Which MORL agent to use. Available: {', '.join(_agent_choices)}",
    )
    parser.add_argument("--timesteps", type=int, default=500000, help="Number of timesteps for training.")
    parser.add_argument("--site_specific", action="store_true", help="Enable site-specific mode (fixed weather seed, deterministic temp noise, no age jitter; uses defaults in constants unless overridden).")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes per weight for evaluation.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config to override eval settings (defaults to logs/config.yaml if present).")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging for both training and evaluation.")
    parser.add_argument("--plot_profile", type=str, default=None, help="Path to saved profiling JSON to plot. If omitted, plots current profiler data.")
    parser.add_argument("--baseline", action="store_true", help="Run baselines and counterfactual analysis and exit.")
    args = parser.parse_args()

    if args.no_wandb:
        print("Wandb logging disabled via environment variables")

    if args.baseline:
        from borearl.agents.baseline import run_baselines
        run_baselines(output_dir='logs', fixed_preference=const.COUNTERFACTUAL_PREF_DEFAULT)
    elif args.train:
        train_morl(
            total_timesteps=args.timesteps,
            use_wandb=not args.no_wandb,
            site_specific=args.site_specific,
            algorithm=args.agent,
        )
    elif args.evaluate:
        # If user kept default model path but chose a different agent, adjust to algorithm-specific default
        default_eupg_path = "models/eupg_forest_manager.pth"
        if args.model_path == default_eupg_path and args.agent != "eupg":
            alt_default = os.path.join("models", f"{args.agent}_forest_manager.pth")
            if os.path.exists(alt_default):
                args.model_path = alt_default
        # Load config overrides from YAML file if provided or if default exists
        config_overrides = None
        try:
            if args.config:
                if os.path.exists(args.config):
                    config_overrides = load_simple_yaml(args.config)
            else:
                default_cfg = os.path.join("logs", "config.yaml")
                if os.path.exists(default_cfg):
                    config_overrides = load_simple_yaml(default_cfg)
        except Exception as e:
            print(f"Warning: failed to load config from YAML: {e}")

        results = evaluate_morl_policy(
            model_path=args.model_path,
            n_eval_episodes=args.eval_episodes,
            use_wandb=not args.no_wandb,
            site_specific=args.site_specific,
            config_overrides=config_overrides,
            algorithm=args.agent,
        )
        # Optional: quick summary
        if results:
            print("\nWeight\t\tCarbon\t\tThaw\t\tScalarized")
            print("-" * 50)
            for w, c, t, s in zip(results['weights'], results['carbon_objectives'], results['thaw_objectives'], results['scalarized_rewards']):
                print(f"[{w[0]:.1f}, {w[1]:.1f}]\t{c:.3f}\t\t{t:.3f}\t\t{s:.3f}")
            # Pareto front plot
            plots_dir = os.path.join("plots")
            os.makedirs(plots_dir, exist_ok=True)
            plt.figure(figsize=(10, 6))
            plt.scatter(results['carbon_objectives'], results['thaw_objectives'], 
                        c=np.linspace(0, 1, len(results['weights'])), cmap='viridis', edgecolor='k', s=80)
            for (w, x, y) in zip(results['weights'], results['carbon_objectives'], results['thaw_objectives']):
                plt.annotate(f"({w[0]:.1f},{w[1]:.1f})", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8)
            plt.xlabel('Carbon Objective (kg C/m²)')
            plt.ylabel('Thaw Objective (-TDD)')
            plt.title('MORL Pareto Front')
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(plots_dir, 'morl_pareto_front.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"Pareto front plot saved as '{out_path}'")
    else:
        parser.print_help()

    if args.plot_profile is not None:
        plot_profiling_statistics(args.plot_profile)


if __name__ == "__main__":
    main()


