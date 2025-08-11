import json
import os
from datetime import datetime
import glob
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .profiling import profiler


def plot_profiling_statistics(profiling_data_file: Optional[str] = None, show: bool = False):
    if profiling_data_file:
        with open(profiling_data_file, 'r') as f:
            data = json.load(f)
        timers = data['timers']
        episode_timers = data['episode_timers']
    else:
        timers = profiler.get_all_statistics()
        episode_timers = profiler.get_episode_statistics()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RL Forest Simulation Performance Profiling', fontsize=16, fontweight='bold')

    # 1. Step-level timing breakdown
    ax1 = axes[0, 0]
    step_timers = {k: v for k, v in timers.items() if k in [
        'action_processing', 'physics_simulation', 'state_updates',
        'reward_calculation', 'episode_tracking', 'termination_checks', 'csv_logging'
    ]}
    if step_timers:
        labels = list(step_timers.keys())
        sizes = [step_timers[k]['mean'] for k in labels]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Step-Level Time Breakdown (Average)', fontweight='bold')

    # 2. Physics simulation detailed breakdown
    ax2 = axes[0, 1]
    physics_timers = {k: v for k, v in timers.items() if k in [
        'daily_physics_loop', 'timestep_physics', 'flux_calculation'
    ]}
    if physics_timers:
        labels = list(physics_timers.keys())
        means = [physics_timers[k]['mean'] * 1000 for k in labels]
        stds = [physics_timers[k]['std'] * 1000 for k in labels]
        x_pos = np.arange(len(labels))
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5,
                       color=plt.cm.viridis(np.linspace(0, 1, len(labels))))
        ax2.set_xlabel('Physics Components')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Physics Simulation Breakdown', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{mean:.2f}ms', ha='center', va='bottom', fontweight='bold')

    # 3. Episode-level timing trends
    ax3 = axes[1, 0]
    if 'total_episode_time' in episode_timers:
        episode_data = episode_timers['total_episode_time']
        if episode_data['count'] > 0:
            episodes = list(range(1, episode_data['count'] + 1))
            if profiling_data_file and 'raw_episode_history' in data:
                raw_times = data['raw_episode_history']['total_episode_time']
                ax3.plot(episodes, raw_times, 'b-', alpha=0.6, linewidth=1)
                ax3.scatter(episodes, raw_times, c='blue', s=20, alpha=0.7)
            mean_time = episode_data['mean']
            ax3.axhline(y=mean_time, color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {mean_time:.3f}s')
            ax3.set_xlabel('Episode Number')
            ax3.set_ylabel('Episode Time (seconds)')
            ax3.set_title('Episode Timing Trends', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

    # 4. Step timing distribution
    ax4 = axes[1, 1]
    if 'step_time' in episode_timers:
        step_data = episode_timers['step_time']
        if step_data['count'] > 0:
            if profiling_data_file and 'raw_episode_history' in data:
                raw_step_times = data['raw_episode_history']['step_time']
                ax4.hist(raw_step_times, bins=30, alpha=0.7, color='green', edgecolor='black')
                ax4.axvline(step_data['mean'], color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {step_data["mean"]*1000:.2f}ms')
                ax4.axvline(step_data['mean'] + step_data['std'], color='orange', linestyle=':', linewidth=2,
                            label=f'+1σ: {(step_data["mean"] + step_data["std"])*1000:.2f}ms')
                ax4.axvline(step_data['mean'] - step_data['std'], color='orange', linestyle=':', linewidth=2,
                            label=f'-1σ: {(step_data["mean"] - step_data["std"])*1000:.2f}ms')
            else:
                ax4.text(0.5, 0.5, f'Mean: {step_data["mean"]*1000:.2f}ms\n'
                         f'Std: {step_data["std"]*1000:.2f}ms\n'
                         f'Min: {step_data["min"]*1000:.2f}ms\n'
                         f'Max: {step_data["max"]*1000:.2f}ms\n'
                         f'Count: {step_data["count"]}',
                         transform=ax4.transAxes, ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                         fontsize=12, fontweight='bold')
            ax4.set_xlabel('Step Time (seconds)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Step Time Distribution', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure plots directory exists
    plots_dir = os.path.join("plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"profiling_plots_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Profiling plots saved to: {plot_filename}")
    if show:
        plt.show()
    else:
        plt.close(fig)



def _find_latest_step_metrics_csv(csv_logs_dir: str = "logs") -> Optional[str]:
    """Return the latest step_metrics_*.csv path from the given directory, if any."""
    pattern = os.path.join(csv_logs_dir, "step_metrics_*.csv")
    matches: List[str] = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def plot_step_reward_histograms(
    csv_path: Optional[str] = None,
    csv_logs_dir: str = "logs",
    bins: int = 50,
    figsize: tuple = (16, 10),
    save: bool = True,
    show: bool = True,
) -> Optional[str]:
    """
    Plot histograms for all reward-related columns in a step-metrics CSV.

    - If csv_path is None, the latest file matching csv_logs_dir/step_metrics_*.csv is used.
    - Reward columns are auto-detected as those whose column names contain the substring "reward".

    Returns the saved plot path if save=True, else None.
    """
    # Resolve CSV path
    if csv_path is None:
        csv_path = _find_latest_step_metrics_csv(csv_logs_dir)
        if csv_path is None:
            raise FileNotFoundError(f"No step_metrics_*.csv found under '{csv_logs_dir}'.")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    # Detect reward columns and include normalized asymmetric thaw
    reward_cols = [c for c in df.columns if "reward" in c.lower()]
    if "normalized_asymmetric_thaw" in df.columns and "normalized_asymmetric_thaw" not in reward_cols:
        reward_cols.append("normalized_asymmetric_thaw")

    # Fallback: if none found, try including common components often treated as rewards
    if not reward_cols:
        candidates = [
            "stock_bonus",
            "base_stock_bonus",
            "biomass_penalty",
            "soil_penalty",
            "total_carbon_limit_penalty",
            "limit_penalty",
            "max_density_penalty",
        ]
        reward_cols = [c for c in candidates if c in df.columns]

    if not reward_cols:
        raise ValueError("No reward-related columns found in the CSV.")

    # Create subplots grid
    num_plots = len(reward_cols)
    ncols = 3
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)
    fig.suptitle("Step Reward Histograms", fontsize=16, fontweight="bold")

    # Plot each histogram
    for idx, col in enumerate(reward_cols):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        series = df[col].dropna().astype(float)
        ax.hist(series, bins=bins, alpha=0.75, color="steelblue", edgecolor="black")
        mean_val = series.mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")
        ax.set_title(col, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

    # Hide any unused axes
    for j in range(num_plots, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    saved_path = None
    if save:
        plots_dir = os.path.join("plots")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join(plots_dir, f"step_reward_histograms_{timestamp}.png")
        plt.savefig(saved_path, dpi=300, bbox_inches="tight")
        print(f"Reward histograms saved to: {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path

