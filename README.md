# 🚦 Multi-Agent Reinforcement Learning for Smart Traffic Signal Control

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art implementation of **QMIX** algorithm for optimizing traffic signal control in urban networks using Multi-Agent Reinforcement Learning.

## 🎯 Features

- ✅ **QMIX Algorithm**: Value decomposition with mixing networks
- ✅ **Multi-Intersection Environment**: 2×2 grid with 4 coordinated traffic lights
- ✅ **Advanced Neural Networks**: GRU-based Q-networks for partial observability
- ✅ **Realistic Traffic Simulation**: Dynamic traffic patterns with rush hour modeling
- ✅ **Comprehensive Metrics**: Waiting time, queue length, throughput tracking
- ✅ **Experience Replay**: Stable training with replay buffer
- ✅ **Model Checkpointing**: Save best and periodic models

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/marl-traffic-control.git
cd marl-traffic-control

# Install dependencies
pip install -e .
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print('✓ Ready to train!')"


Training
Bash

# Train QMIX (500 episodes, ~15-20 minutes)
python scripts/train_qmix.py

# Quick test (5 episodes, ~30 seconds)
# Edit configs/qmix_config.yaml: total_episodes: 5
python scripts/train_qmix.py


🎓 Algorithm: QMIX
QMIX (Monotonic Value Function Factorisation) enables centralized training with decentralized execution:

Key Components
Agent Networks: GRU-based Q-networks (one shared network for all agents)

Input: Local observation (queue lengths, waiting times, phase info)
Output: Q-values for 4 traffic light phases
Mixing Network: Combines individual Q-values into joint Q-value

Hypernetworks generate mixing weights from global state
Monotonicity constraint ensures alignment
Training:

Centralized: Uses global state and all agent observations
Execution: Each agent acts based on local observation only


Architecture
text

Individual Agents → [GRU Q-Networks] → Individual Q-values
                                              ↓
Global State → [Hypernetwork] → Mixing Weights
                                              ↓
                                    [Mixer] → Total Q-value



📈 Results
Training Metrics
After training, you'll find:

Bash

# Saved models
checkpoints/qmix/
├── qmix_best.pth      # Best performing model
├── qmix_final.pth     # Final model
└── qmix_ep*.pth       # Periodic checkpoints

# Training metrics (JSON)
results/qmix_metrics.json
Expected Performance
Average Reward: ~-150 to -160 (lower is better - less congestion)
Queue Length: ~2.3 vehicles per lane
Waiting Time: ~45 seconds average
Throughput: ~450 vehicles per episode
Sample Output
text

Training: 100%|██████| 500/500 [16:23<00:00]

📊 Final Statistics (last 100 episodes):
   Average Reward: -156.82
   Average Queue Length: 2.34
   Average Waiting Time: 45.67s
   Average Throughput: 452 vehicles


   🧪 Testing
Bash

# Quick test (5 episodes)
python -c "
import yaml
with open('configs/qmix_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['training']['total_episodes'] = 5
with open('configs/qmix_config.yaml', 'w') as f:
    yaml.dump(config, f)
"
python scripts/train_qmix.py


🎯 Key Features Explained
1. Partial Observability
Each intersection observes:

Own queue lengths (N, S, E, W)
Own waiting times
Current phase and phase duration
Neighbor queue information
2. Coordination
Agents coordinate through:

Shared experiences in replay buffer
Mixing network that combines Q-values
Information about neighboring intersections
3. Dynamic Traffic
Poisson arrival process
Rush hour simulation (7-9 AM, 5-7 PM)
Adaptive green time based on demand
🏗️ Requirements
Python 3.8+
PyTorch 2.0+
NumPy
PyYAML
tqdm
Other dependencies in requirements.txt
📚 References
QMIX Paper: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (Rashid et al., 2018)
Multi-Agent RL: An Introduction to Multi-Agent RL


⭐ If you find this project useful, please give it a star!!!

