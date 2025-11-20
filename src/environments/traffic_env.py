"""
Multi-Intersection Traffic Environment
Supports 2x2 grid of traffic lights with realistic simulation
"""
import numpy as np
from typing import Dict, Tuple, List
import yaml

class MultiIntersectionEnv:
    """Advanced traffic environment with multiple intersections"""
    
    def __init__(self, config_path: str = "configs/env_config.yaml"):
        """Initialize environment from config"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.env_config = config['environment']
        self.obs_config = config['observation']
        self.reward_config = config['reward']
        self.traffic_config = config['traffic']
        
        # Grid setup
        self.grid_size = self.env_config['grid_size']
        self.num_intersections = np.prod(self.grid_size)
        self.agents = [f"intersection_{i}" for i in range(self.num_intersections)]
        
        # Action and observation spaces
        self.n_actions = 4  # 4 traffic light phases
        self.obs_dim = 14  # Will be calculated properly
        
        # Episode parameters
        self.max_steps = self.env_config['episode_length']
        self.delta_time = self.env_config['delta_time']
        
        # Traffic state for all intersections
        self.queues = np.zeros((self.num_intersections, 4))  # N, S, E, W queues
        self.waiting_times = np.zeros((self.num_intersections, 4))
        self.current_phases = np.zeros(self.num_intersections, dtype=int)
        self.phase_times = np.zeros(self.num_intersections)
        
        # Metrics
        self.total_throughput = 0
        self.current_step = 0
        
        # Traffic parameters
        self.arrival_rate = self.traffic_config['arrival_rate']
        self.peak_multiplier = self.traffic_config['peak_multiplier']
        
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.queues = np.random.randint(0, 3, size=(self.num_intersections, 4)).astype(float)
        self.waiting_times = np.zeros((self.num_intersections, 4))
        self.current_phases = np.zeros(self.num_intersections, dtype=int)
        self.phase_times = np.zeros(self.num_intersections)
        self.total_throughput = 0
        
        observations = self._get_observations()
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step"""
        # Apply actions to change traffic light phases
        for i, agent_id in enumerate(self.agents):
            action = actions[agent_id]
            if action != self.current_phases[i]:
                self.current_phases[i] = action
                self.phase_times[i] = 0
        
        # Simulate traffic flow
        self._simulate_traffic_flow()
        
        # Update time
        self.current_step += 1
        for i in range(self.num_intersections):
            self.phase_times[i] += 1
        
        # Get results
        observations = self._get_observations()
        rewards = self._calculate_rewards()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = self._get_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _simulate_traffic_flow(self):
        """Simulate traffic dynamics for all intersections"""
        # Calculate time-dependent arrival rate (rush hour simulation)
        hour = (self.current_step * self.delta_time) // 3600
        is_peak = hour in [7, 8, 17, 18]  # Morning and evening rush
        current_rate = self.arrival_rate * (self.peak_multiplier if is_peak else 1.0)
        
        for i in range(self.num_intersections):
            # Vehicle arrivals (Poisson process)
            arrivals = np.random.poisson(current_rate, size=4)
            self.queues[i] = np.minimum(self.queues[i] + arrivals, 50)  # Max queue: 50
            
            # Update waiting times
            self.waiting_times[i] += self.queues[i] * self.delta_time
            
            # Process vehicles based on current phase
            phase = self.current_phases[i]
            
            if phase == 0:  # North-South green
                departures_n = min(self.queues[i, 0], 3)
                departures_s = min(self.queues[i, 1], 3)
                self.queues[i, 0] -= departures_n
                self.queues[i, 1] -= departures_s
                self.total_throughput += (departures_n + departures_s)
                
            elif phase == 1:  # North-South yellow (minimal flow)
                for lane in [0, 1]:
                    departures = min(self.queues[i, lane], 1)
                    self.queues[i, lane] -= departures
                    self.total_throughput += departures
                    
            elif phase == 2:  # East-West green
                departures_e = min(self.queues[i, 2], 3)
                departures_w = min(self.queues[i, 3], 3)
                self.queues[i, 2] -= departures_e
                self.queues[i, 3] -= departures_w
                self.total_throughput += (departures_e + departures_w)
                
            elif phase == 3:  # East-West yellow (minimal flow)
                for lane in [2, 3]:
                    departures = min(self.queues[i, lane], 1)
                    self.queues[i, lane] -= departures
                    self.total_throughput += departures
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}
        for i, agent_id in enumerate(self.agents):
            obs = self._get_single_observation(i)
            observations[agent_id] = obs
        return observations
    
    def _get_single_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a single agent"""
        obs = []
        
        # Own intersection state
        obs.extend(self.queues[agent_idx])  # 4 values (N, S, E, W queues)
        obs.extend(self.waiting_times[agent_idx] / 100.0)  # 4 values (normalized)
        obs.append(self.current_phases[agent_idx] / 3.0)  # 1 value (normalized)
        obs.append(min(self.phase_times[agent_idx] / 30.0, 1.0))  # 1 value (normalized)
        
        # Neighbor information (if enabled)
        if self.obs_config.get('neighbor_info', False):
            neighbors = self._get_neighbors(agent_idx)
            for neighbor_idx in neighbors[:4]:  # Max 4 neighbors (N, S, E, W)
                if neighbor_idx is not None:
                    obs.append(np.mean(self.queues[neighbor_idx]))
                else:
                    obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_neighbors(self, agent_idx: int) -> List:
        """Get indices of neighboring intersections"""
        row = agent_idx // self.grid_size[1]
        col = agent_idx % self.grid_size[1]
        
        neighbors = []
        # Check all 4 directions: North, South, East, West
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]:
                neighbor_idx = new_row * self.grid_size[1] + new_col
                neighbors.append(neighbor_idx)
            else:
                neighbors.append(None)
        
        return neighbors
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {}
        weights = self.reward_config['weights']
        
        for i, agent_id in enumerate(self.agents):
            # Penalty for waiting time
            waiting_penalty = np.sum(self.waiting_times[i]) / 100.0
            
            # Penalty for queue length
            queue_penalty = np.sum(self.queues[i])
            
            # Combined reward
            reward = (
                weights['waiting_time'] * waiting_penalty +
                weights['queue_length'] * queue_penalty
            )
            
            # Normalize if configured
            if self.reward_config['normalize']:
                reward /= 4  # Normalize by number of lanes
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _check_terminated(self) -> Dict[str, bool]:
        """Check if episode is terminated"""
        terminated = {agent_id: False for agent_id in self.agents}
        terminated['__all__'] = False
        return terminated
    
    def _check_truncated(self) -> Dict[str, bool]:
        """Check if episode is truncated (time limit)"""
        is_truncated = self.current_step >= self.max_steps
        truncated = {agent_id: is_truncated for agent_id in self.agents}
        truncated['__all__'] = is_truncated
        return truncated
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'step': self.current_step,
            'avg_queue_length': np.mean(self.queues),
            'avg_waiting_time': np.mean(self.waiting_times),
            'total_throughput': self.total_throughput,
            'max_queue': np.max(self.queues)
        }
    
    def close(self):
        """Cleanup resources"""
        pass
    
    def render(self):
        """Render environment state (optional)"""
        print(f"\nStep: {self.current_step}")
        for i in range(self.num_intersections):
            print(f"Intersection {i}: Phase={self.current_phases[i]}, "
                  f"Queues={self.queues[i]}, Avg_Wait={np.mean(self.waiting_times[i]):.1f}s")