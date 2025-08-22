# BoreaRL
BoreaRL is a physically-grounded multi-objective reinforcement learning benchmark for boreal forest management, simulating coupled energy, carbon, and water fluxes to train agents that balance carbon sequestration with permafrost preservation.

### Why BoreaRL?
- **Physically-based simulator**: A class-based forest energy, water, and carbon model (`ForestSimulator`) with canopy, trunk space, soil (surface/deep), atmosphere, and snowpack, driven by latitude-aware climate and stochastic weather.
- **Multi-objective RL**: Learn to maximize carbon benefits while minimizing permafrost thaw using a 2D reward vector and preference-based scalarization (EUPG from `morl_baselines`).
- **Realistic management levers**: Annual actions jointly control stem density (plant/thin) and the conifer fraction of stems added/removed, with age-structured demography, thinning restricted to old trees, and harvested wood product (HWP) accounting.
- **Robustness via stochasticity**: Episodes use Monte Carlo sampling of climate, weather, and ecological parameters to stress-test policies.

### Repository layout
- `borearl/env/forest_env.py`: Multi-objective Gym environment (`ForestEnv`) and Gym registration (`ForestEnv-v0`).
- `borearl/constants.py`: Shared RL-level constants used by the environment and training.
- `borearl/agents/runner.py`: Unified training/evaluation entrypoints.
- `borearl/agents/eupg_agent.py`, `pcn_agent.py`, `ppo_gated.py`: Agent-specific factories.
- `borearl/agents/common.py`: Shared helpers for env and config.
- `borearl/agents/baseline.py`: Baselines and counterfactual sensitivity.
- `borearl/utils/`: Profiling and plotting utilities.
- `main.py`: Thin CLI wrapper for training and evaluation (routes via `borearl.agents.runner`).
- `borearl/physics/`: Modular physics package.
  - `energy_balance.py`: Simulator implementation (`ForestSimulator`).
  - `weather.py`: Weather/climate utilities.
  - `config.py`: Physics configuration and parameter ranges.
  - `constants.py`: Shared physics constants.
  - `demography.py`: Natural demography helpers.
- `logs/`, `plots/`: Outputs for metrics and figures.

## Environment overview
### Actions (annual management)
- Single discrete action encoding a pair: Δdensity and conifer-fraction choice.
  - **Δdensity (stems/ha)**: one of [-50, -20, 0, 20, 50] (`DENSITY_ACTIONS`). Negative = thinning, positive = planting.
  - **Conifer fraction**: one of {0.0, 0.25, 0.5, 0.75, 1.0} (`CONIFER_FRACTIONS`). Applied to stems added/removed this year.
- Thinning is restricted to old trees (101+ years). Planting adds seedlings.
- HWP accounting: when thinning, 95% of removed carbon is stored as HWP, 5% is lost.

### Observations (continuous vector)
Comprehensive state including, among others:
- Current year progress, stem density, conifer fraction, total carbon stock.
- Climate signals (winter/summer temps, latitude-driven features), disturbance history (fire, insects, drought).
- Carbon cycle details (biomass/soil changes, NPP/GPP, litterfall, HWP carbon).
- Management history (last-year actions/changes).
- Age distribution by species across classes (seedling, sapling, young, mature, old).
- Carbon limit flags and penalties, and the current preference weight for scalarization.

### Objectives and reward
- Vector: [carbon_reward, thaw_penalty]
  - Carbon uses HWP-adjusted net carbon change with stock bonuses and limit penalties.
  - Thaw uses an asymmetric thaw reward derived from conductive heat to deep soil (permafrost proxy), penalizing warming more than rewarding cooling.
- Scalarization: EUPG conditions on a preference weight and optimizes the weighted sum internally.

### Episode structure
- 50 decision steps per episode (≈ 50 years). Each step runs a full 365-day physical simulation with sub-daily timesteps and stochastic weather.

## Physics simulator highlights (`borearl.physics`)
- Implicit canopy energy balance (stable numerics) with radiative, sensible, latent, conductive, and melt terms.
- Photosynthesis via light-use efficiency; consistent energetic sink per gram of C fixed.
- Carbon cycle: GPP, autotrophic and soil respiration (Q10), NPP, litterfall to soil.
- Permafrost metric: thaw/cooling degree-days from deep boundary heat flux with asymmetric aggregation.
- Disturbances: climate/density-driven fire and insect outbreak mortality with carbon routing to atmosphere/soil.
- Age-structured demography: annual recruitment, density-dependent and natural mortality, aging transitions.
- Latitude-driven climate and phenology; rain suppresses diurnal temperature amplitude; temperature–precip relationships by season.
- Parameter ranges sampled per episode for robustness (soil conductivity, SWC capacity, wind, radiation factors, etc.).

## Installation
Use a local virtual environment and install dependencies from `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tip: keep all project commands inside the `.venv` virtual environment.

## Quickstart
### Train a MORL agent (EUPG)
```bash
python main.py --train --timesteps 500000
```
Flags:
- `--timesteps`: training timesteps (default 500000)
- `--no_wandb`: disable Weights & Biases logging

Alternatively, import the environment directly:
```python
import gymnasium as gym
import borearl.env.forest_env  # registers ForestEnv-v0
env = gym.make("ForestEnv-v0", disable_env_checker=True)
```

### Evaluate a trained agent across preferences
```bash
python main.py --evaluate --model_path models/eupg_forest_manager.pth --eval_episodes 100
```
Outputs a Pareto front plot `plots/morl_pareto_front.png` and summary stats.

### Standalone physics sanity check
```python
from borearl.physics import ForestSimulator
sim = ForestSimulator(coniferous_fraction=0.5, stem_density=800, weather_seed=123)
res = sim.run_annual_cycle(current_stem_density=800, current_biomass_carbon_kg_m2=10.0, current_soil_carbon_kg_m2=5.0)
print(res['final_biomass_carbon_kg_m2'], res['thaw_degree_days'])
```

## Logging, outputs, and artifacts
- **CSV metrics**: Per-step and per-episode logs in `logs/` with timestamps. Includes actions, rewards, state summaries, carbon changes, disturbances, and age-structure snapshots.
- **Episode plots**: Saved to `plots/episode_<N>_statistics.png` every 1000 episodes.
- **Profiling**: Comprehensive timing via `TimeProfiler`; after training, profiling plots are saved as `plots/profiling_plots_<timestamp>.png` (or repository root if run standalone). You can also run:

```bash
python -c "from borearl.utils.plotting import plot_profiling_statistics; plot_profiling_statistics()"
```

- **W&B**: Enable by default unless `--no_wandb` is passed.

## Configuration notes
- Gym ID is `ForestEnv-v0`; episode length is 50 years.
- Environment constants are centralized in `borearl/constants.py` (actions, weights, normalizers).
- Action grid:
  - Δdensity: −50, −20, 0, +20, +50 stems/ha
  - Conifer fraction: {0.0, 0.25, 0.5, 0.75, 1.0}
- Carbon limits are enforced with penalties at the RL layer and clipping in diagnostics.
- Thinning is preferentially applied to old trees; planting adds seedlings with chosen species mix.

## Citation
If you use BoreaRL in your research, please cite this repository. A formal citation will be added once a preprint is available.

## License
This project is released under the MIT License. See `LICENSE` for details.
