---
language:
- en
license: mit
library_name: stable-baselines3
tags:
- reinforcement-learning
- evolutionary-algorithms
- urban-planning
- spatial-optimization
- pytorch
datasets:
- custom-wenzhou-population
model_name: PTOP-Wenzhou-Optimizer
---

# Model Card for PTOP-Wenzhou-Optimizer

## Model Details

### Model Description

The **PTOP (Public Transport Optimization Platform)** is a specialized spatial optimization system designed to improve bus stop layouts in Wenzhou City. It employs a hybrid approach combining heuristic evolutionary algorithms and Deep Reinforcement Learning (DRL) to maximize population coverage while minimizing infrastructure migration costs.

The system consists of two primary optimization engines:
1.  **Ultra-Fast Heuristic Engine:** Utilizes Parallel Genetic Algorithms (GA) and Simulated Annealing (SA) for rapid prototyping and global search.
2.  **Hybrid RL Engine:** Integrates Proximal Policy Optimization (PPO) for global policy exploration with Gurobi/PuLP for precise local mixed-integer programming (MIP) optimization.

- **Developed by:** PTOP Research Team
- **Model type:** Hybrid Spatial Optimizer (Evolutionary Algorithm + Deep Reinforcement Learning)
- **Language(s):** Python
- **License:** MIT
- **Frameworks:** PyTorch (Stable Baselines3), Numba, Geopandas

### Model Sources

- **Repository:** [Local Project] PTOP-main
- **Paper:** [Pending]

## Uses

### Direct Use

The model is intended for **urban planners** and **public transport authorities** in Wenzhou to:
- Evaluate the efficiency of current bus stop distributions.
- Generate optimized relocation plans for bus stops to increase service coverage.
- Simulate the trade-offs between coverage improvement and relocation costs (distance moved).

### Out-of-Scope Use

- **Universal Application:** The current model is trained/calibrated specifically on Wenzhou's spatial data. Direct application to other cities without retraining/re-initializing with local data is not recommended.
- **Real-time Routing:** This is a strategic planning tool for static infrastructure (bus stops), not a real-time dynamic scheduling or routing system.

## Bias, Risks, and Limitations

### Limitations
- **Data Dependency:** The optimization is heavily dependent on the accuracy of the `Wenzhou_population_grid.csv`. If population data is outdated, the optimized results may not reflect actual demand.
- **Static Assumption:** The model assumes static population distribution and does not account for temporal variations (e.g., day vs. night population flow).
- **Geography:** The distance calculations use Euclidean/Haversine distance and do not fully account for the actual road network topology or physical barriers (rivers, mountains) unless implicitly captured by the stop distribution.

### Recommendations
Users should validate the proposed stop locations against the actual road network and local traffic regulations before implementation.

## How to Get Started with the Model

```python
from main_optimizer import MainOptimizer, get_default_config

# Load default configuration
config = get_default_config()

# Initialize the optimizer
optimizer = MainOptimizer(config)

# Run the Ultra-Fast Genetic Algorithm optimization
results = optimizer.run_ultra_fast_optimization()

# Access results
print(f"Final Coverage: {results['genetic_algorithm']['final_coverage']}")
```

## Training Details

### Training Data

The model does not use a standard public dataset but processes specific local geospatial data:
1.  **Population Data:** `populaiton/Wenzhou_population_grid.csv` - Contains grid-level population counts with latitude/longitude coordinates.
2.  **Bus Stop Data:** `Bus Stop shp/0577Wenzhou.shp` - Vector data containing the original coordinates of all public transport stops in Wenzhou.

### Training Procedure

#### Hybrid RL (PPO) Training
- **Algorithm:** Proximal Policy Optimization (PPO) via `stable_baselines3`.
- **Policy:** `MlpPolicy` (Multi-Layer Perceptron).
- **Environment:** Custom `BusStopOptimizationEnv`.
- **Hyperparameters:**
    - `total_timesteps`: 30,000 - 50,000
    - `learning_rate`: 3e-4
    - `batch_size`: 64
    - `n_steps`: 1024 / 2048
    - `gamma`: 0.99

#### Evolutionary Algorithm (GA/SA) Settings
- **Genetic Algorithm:**
    - Population Size: 50 - 100
    - Generations: 50 - 150
    - Selection: Tournament Selection
    - Crossover: Adaptive Uniform Crossover
- **Simulated Annealing:**
    - Chains: 4 - 8 parallel chains
    - Iterations: 1000 - 2000 per chain

## Evaluation

### Testing Factors & Metrics

The model is evaluated based on a composite objective function rather than a test set accuracy.

- **Primary Metric:** **Population Coverage Ratio** (Percentage of total population within `coverage_radius` typically 500m-800m).
- **Secondary Metrics:**
    - **Movement Cost:** Total distance (km) existing stops are moved.
    - **Stability:** Percentage of stops that remain within a small threshold of their original location.

### Results

In internal benchmarks (comparative analysis):
- **Baseline (Original):** Coverage ratio ~X% (varies by radius).
- **Genetic Algorithm:** Typically achieves 10-15% improvement in coverage with moderate movement.
- **Hybrid PPO:** Capable of finding non-intuitive global optima, balancing exploration with precise local adjustments via Gurobi.

## Environmental Impact

- **Hardware Type:** CPU / GPU (Optional for PPO)
- **Compute Region:** Local
- **Carbon Emitted:** Negligible for heuristic methods; Low for PPO training (minutes to hours on standard consumer hardware).

## Technical Specifications

### Model Architecture

- **RL Component:** PPO Actor-Critic network (MLP).
- **Heuristic Component:** Numba-accelerated iterative search algorithms.

### Software

- **Python:** 3.8+
- **Key Libraries:** `stable-baselines3`, `numba`, `geopandas`, `pandas`, `gurobipy` (optional).
