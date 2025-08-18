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
    # Dynamically expose only available agents (based on optional deps present)
    _agent_choices = sorted(list(_AVAILABLE_AGENTS.keys())) or ["eupg"]
    parser.add_argument(
        "--agent",
        type=str,
        choices=_agent_choices,
        default=_agent_choices[0],
        help=f"Which MORL agent to use. Available: {', '.join(_agent_choices)}",
    )
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps for training.")
    parser.add_argument("--site_specific", action="store_true", help="Enable site-specific mode (fixed weather seed, deterministic temp noise, no age jitter; uses defaults in constants unless overridden).")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes per weight for evaluation.")
    parser.add_argument("--run_dir_name", type=str, default=None, help="Name for the central run directory under logs/. If omitted, uses the run_id timestamp.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging for both training and evaluation.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save model every N episodes during training (default: 100).")
    parser.add_argument("--eval_interval", type=int, default=5000, help="Evaluate model every N steps during training (default: 1000).")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Number of episodes per weight for periodic evaluation (default: 10).")
    parser.add_argument("--plot_profile", type=str, default=None, help="Path to saved profiling JSON to plot. If omitted, plots current profiler data.")
    parser.add_argument("--baseline", action="store_true", help="Run baselines and counterfactual analysis and exit.")
    parser.add_argument("--train_then_eval", action="store_true", help="Train and then immediately evaluate in the same run.")
    args = parser.parse_args()

    #args.train_then_eval = True
    #args.timesteps = 400
    #args.eval_episodes = 3
    #args.save_interval = 2
    #args.eval_interval = 50
    #args.run_dir_name = "test"
    #args.no_wandb = True
    #args.site_specific = True

    if args.no_wandb:
        print("Wandb logging disabled via environment variables")

    results = None
    if args.baseline:
        from borearl.agents.baseline import run_baselines
        # When running baselines standalone, allow saving into a named run dir if provided
        out_dir = os.path.join('logs', args.run_dir_name) if args.run_dir_name else 'logs'
        os.makedirs(out_dir, exist_ok=True)
        run_baselines(output_dir=out_dir, fixed_preference=const.COUNTERFACTUAL_PREF_DEFAULT)
    elif args.train_then_eval:
        # Do combined flow
        train_info = train_morl(
            total_timesteps=args.timesteps,
            use_wandb=not args.no_wandb,
            site_specific=args.site_specific,
            algorithm=args.agent,
            run_dir_name=args.run_dir_name,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            n_eval_episodes=args.n_eval_episodes,
        )
        # Resolve run directory
        run_dir = os.path.join("logs", args.run_dir_name) if args.run_dir_name else os.environ.get("BOREARL_RUN_DIR")
        # Determine model path (prefer the model saved during training; fallback to default agent filename in run dir)
        model_path = None
        try:
            if isinstance(train_info, dict) and train_info.get("model_path") and os.path.exists(train_info["model_path"]):
                model_path = train_info["model_path"]
        except Exception:
            pass
        if (not model_path) and run_dir:
            agent_mod = _AVAILABLE_AGENTS.get(args.agent)
            if agent_mod and hasattr(agent_mod, 'default_model_filename'):
                candidate = os.path.join(run_dir, agent_mod.default_model_filename())
                if os.path.exists(candidate):
                    model_path = candidate
        # Load config overrides from the run directory if present
        config_overrides = None
        if run_dir:
            default_cfg = os.path.join(run_dir, "config.yaml")
            if os.path.exists(default_cfg):
                config_overrides = load_simple_yaml(default_cfg)

        results = evaluate_morl_policy(
            model_path=model_path,
            n_eval_episodes=args.eval_episodes,
            use_wandb=False,  # Disable wandb for evaluation
            site_specific=args.site_specific,
            config_overrides=config_overrides,
            algorithm=args.agent,
            run_dir_name=args.run_dir_name,
        )
    elif args.train:
        train_morl(
            total_timesteps=args.timesteps,
            use_wandb=not args.no_wandb,
            site_specific=args.site_specific,
            algorithm=args.agent,
            run_dir_name=args.run_dir_name,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            n_eval_episodes=args.n_eval_episodes,
        )
    elif args.evaluate:
        # Resolve run directory
        run_dir = os.path.join("logs", args.run_dir_name) if args.run_dir_name else os.environ.get("BOREARL_RUN_DIR")
        # Determine model path from run directory and agent default filename
        model_path = None
        if run_dir and os.path.isdir(run_dir):
            agent_mod = _AVAILABLE_AGENTS.get(args.agent)
            if agent_mod and hasattr(agent_mod, 'default_model_filename'):
                candidate = os.path.join(run_dir, agent_mod.default_model_filename())
                if os.path.exists(candidate):
                    model_path = candidate
        # Load config overrides from the run directory if present
        config_overrides = None
        if run_dir:
            default_cfg = os.path.join(run_dir, "config.yaml")
            if os.path.exists(default_cfg):
                config_overrides = load_simple_yaml(default_cfg)

        results = evaluate_morl_policy(
            model_path=model_path,
            n_eval_episodes=args.eval_episodes,
            use_wandb=False,  # Disable wandb for evaluation
            site_specific=args.site_specific,
            config_overrides=config_overrides,
            algorithm=args.agent,
            run_dir_name=args.run_dir_name,
        )
    else:
        parser.print_help()

    # Optional: quick summary
    if results:
        # Pareto front plot
        # Determine the run directory for saving plots
        run_dir = os.path.join("logs", args.run_dir_name) if args.run_dir_name else os.environ.get("BOREARL_RUN_DIR")
        if not run_dir:
            # Fallback to global plots directory if no run directory is available
            plots_dir = os.path.join("plots")
            os.makedirs(plots_dir, exist_ok=True)
        else:
            plots_dir = run_dir
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

    if args.plot_profile is not None:
        # Only show interactively when explicitly requested
        # Determine the run directory for saving plots
        run_dir = os.path.join("logs", args.run_dir_name) if args.run_dir_name else os.environ.get("BOREARL_RUN_DIR")
        plot_profiling_statistics(args.plot_profile, show=True, output_dir=run_dir)


if __name__ == "__main__":
    main()


