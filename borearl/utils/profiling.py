import time
import statistics
import json
from collections import defaultdict, deque
from typing import Dict


class TimeProfiler:
    """
    Comprehensive time profiler for tracking performance of different components.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.reset()

    def reset(self):
        self.timers = {}
        self.history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.episode_timers = {}
        self.episode_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.current_episode_start = None
        self.current_step_start = None

    def start_timer(self, name: str):
        self.timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        if name not in self.timers:
            return 0.0
        elapsed = time.time() - self.timers[name]
        self.history[name].append(elapsed)
        del self.timers[name]
        return elapsed

    def start_episode(self):
        self.current_episode_start = time.time()
        self.episode_timers = {}

    def end_episode(self) -> float:
        if self.current_episode_start is None:
            return 0.0
        total_time = time.time() - self.current_episode_start
        self.episode_history['total_episode_time'].append(total_time)
        self.current_episode_start = None
        return total_time

    def start_step(self):
        self.current_step_start = time.time()

    def end_step(self) -> float:
        if self.current_step_start is None:
            return 0.0
        step_time = time.time() - self.current_step_start
        self.episode_history['step_time'].append(step_time)
        self.current_step_start = None
        return step_time

    def get_statistics(self, name: str) -> Dict[str, float]:
        if name not in self.history or not self.history[name]:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        values = list(self.history[name])
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'total': sum(values)
        }

    def get_episode_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for name in self.episode_history:
            if self.episode_history[name]:
                values = list(self.episode_history[name])
                stats[name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        return stats

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        return {name: self.get_statistics(name) for name in self.history}

    def save_profiling_data(self, filename: str):
        data = {
            'timers': self.get_all_statistics(),
            'episode_timers': self.get_episode_statistics(),
            'raw_history': {name: list(values) for name, values in self.history.items()},
            'raw_episode_history': {name: list(values) for name, values in self.episode_history.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        print("\n=== TIME PROFILING SUMMARY ===")
        print("\n--- Step-Level Timers ---")
        for name, stats in self.get_all_statistics().items():
            if stats['count'] > 0:
                print(f"{name:30s}: {stats['mean']*1000:8.2f}ms ± {stats['std']*1000:6.2f}ms "
                      f"(min: {stats['min']*1000:6.2f}ms, max: {stats['max']*1000:6.2f}ms, "
                      f"total: {stats['total']:8.3f}s, count: {stats['count']:4d})")


# Global profiler instance
profiler = TimeProfiler()


