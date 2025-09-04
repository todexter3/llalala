# RL Order Execution System

A reinforcement learning-based system for optimal order execution in financial markets using PPO (Proximal Policy Optimization).

## Overview

This project implements an intelligent trading agent that learns to execute large orders in financial markets 
while minimizing transaction costs and market impact. The system uses deep reinforcement learning with PPO algorithm 
to develop adaptive execution strategies that outperform traditional methods like TWAP. By processing real-time limit 
order book data and learning from market microstructure, the agent makes sophisticated decisions about order timing, 
sizing, and pricing to achieve optimal execution quality.

### Key Features

- **PPO Algorithm**: Stable on-policy RL algorithm for continuous and discrete action spaces
- **Realistic Market Simulation**: Includes limit order semantics, partial fills, and market impact
- **Mixed Action Space**: Handles continuous (size, price) and discrete (order type) decisions
- **Adaptive Execution**: Increases urgency as time deadline approaches
- **Comprehensive Benchmarking**: Compares against TWAP baseline strategy

## Project Structure

```
.
├── config.py          # Configuration parameters
├── envs.py           # Market environment simulation
├── models.py         # Neural network architectures
├── agents.py         # PPO agent implementation
├── utils.py          # Utility functions
├── visualize.py      # Visualization tools
├── train.py          # Main training script
└── README.md         # This file
```

## Requirements

```bash
pip install numpy pandas torch matplotlib
```

## Data Format

The system expects CSV files with the following columns:
- Time series data with bid/ask prices and volumes (L1-L5)
- Required columns: `bidPrice1-5`, `askPrice1-5`, `bidVolume1-5`, `askVolume1-5`, `volume`

Data should be organized as:
```
data/
├── AA_Comdty_cpu/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── AA_Comdty_gpu/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
```

## How to Start


```bash
python train.py
```

### Custom Configuration

```python
# Modify train.py to change parameters
results = train_enhanced_system(
    data_path="data/AA_Comdty_cpu",
    n_episodes=500,
    save_interval=50,
    side='buy'  # or 'sell'
)
```


## Configuration Options

Key parameters in `config.py`:
- `initial_inventory`: Starting position size (default: 1000)
- `time_horizon`: Number of time steps to complete execution (default: 500)
- `side`: Trading direction ('buy' or 'sell')
- `lr_actor`: Learning rate (default: 3e-4)
- `batch_size`: PPO batch size (default: 128)

## Output

The system generates:
- **Model checkpoints**: `best_enhanced_model.pth`, `checkpoint_ep*.pth`
- **Visualization**: `rl_results.png` with training curves and performance metrics
- **Console logs**: Training progress and final test results (Output on the console)

## Performance Metrics

- **Completion Rate**: Percentage of target volume executed
- **Average Cost**: Execution cost in basis points vs mid-price
- **VWAP Shortfall**: Performance vs volume-weighted average price
- **Portfolio Value**: Final P&L considering market movements


