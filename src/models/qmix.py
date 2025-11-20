"""
QMIX Algorithm Implementation
Multi-Agent Q-Learning with Value Decomposition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict
import yaml
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.networks import GRUNetwork

class QMixingNetwork(nn.Module):
    """
    QMIX Mixing Network
    Combines individual agent Q-values into a total Q-value using hypernetworks
    """
    
    def __init__(self, n_agents: int, state_dim: int, mixing_embed_dim: int):
        """
        Initialize mixing network
        
        Args:
            n_agents: Number of agents
            state_dim: Dimension of global state
            mixing_embed_dim: Dimension of mixing network hidden layer
        """
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = mixing_embed_dim
        
        # Hypernetworks to generate mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, n_agents * mixing_embed_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        
        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
    
    def forward(self, agent_qs, states):
        """
        Mix agent Q-values
        
        Args:
            agent_qs: (batch_size, n_agents) - Individual Q-values
            states: (batch_size, state_dim) - Global state
            
        Returns:
            q_total: (batch_size,) - Mixed total Q-value
        """
        batch_size = agent_qs.size(0)
        
        # Generate mixing weights (use abs for monotonicity)
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.embed_dim)
        
        # First mixing layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # Second mixing layer
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        # Final Q-value
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.view(batch_size)


class QMixAgent:
    """
    QMIX Multi-Agent Reinforcement Learning Algorithm
    """
    
    def __init__(self, config_path: str, n_agents: int, obs_dim: int, n_actions: int):
        """
        Initialize QMIX agent
        
        Args:
            config_path: Path to configuration file
            n_agents: Number of agents
            obs_dim: Observation dimension
            n_actions: Number of actions
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        # Training hyperparameters
        self.gamma = self.config['training']['gamma']
        self.lr = self.config['training']['learning_rate']
        self.tau = self.config['training']['tau']
        self.epsilon = self.config['training']['epsilon_start']
        self.epsilon_end = self.config['training']['epsilon_end']
        self.epsilon_decay = self.config['training']['epsilon_decay']
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network parameters
        hidden_dim = self.config['network']['rnn_hidden_dim']
        mixing_dim = self.config['network']['mixer_hidden_dim']
        
        # Create agent networks (shared among all agents)
        self.agent_network = GRUNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_agent_network = GRUNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        
        # Create mixing networks
        state_dim = obs_dim * n_agents  # Global state = concatenated observations
        self.mixer = QMixingNetwork(n_agents, state_dim, mixing_dim).to(self.device)
        self.target_mixer = QMixingNetwork(n_agents, state_dim, mixing_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Optimizer for all networks
        self.optimizer = optim.Adam(
            list(self.agent_network.parameters()) + list(self.mixer.parameters()),
            lr=self.lr
        )
        
        # Hidden states for RNN (reset each episode)
        self.hidden_states = None
        self.training_step = 0
    
    def select_actions(self, observations: Dict[str, np.ndarray], explore: bool = True) -> Dict[str, int]:
        """
        Select actions for all agents
        
        Args:
            observations: Dict mapping agent_id to observation
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Dict mapping agent_id to action
        """
        actions = {}
        agent_ids = list(observations.keys())
        
        with torch.no_grad():
            for agent_id in agent_ids:
                # Convert observation to tensor
                obs = torch.FloatTensor(observations[agent_id]).unsqueeze(0).to(self.device)
                
                # Get or initialize hidden state
                if self.hidden_states is None:
                    hidden = self.agent_network.init_hidden(1, self.device)
                else:
                    hidden = self.hidden_states
                
                # Forward pass
                q_values, new_hidden = self.agent_network(obs, hidden)
                self.hidden_states = new_hidden
                
                # Epsilon-greedy action selection
                if explore and np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = q_values.argmax(1).item()
                
                actions[agent_id] = action
        
        return actions
    
    def reset_hidden_states(self):
        """Reset hidden states at the start of each episode"""
        self.hidden_states = None
    
    def train(self, batch: Dict) -> float:
        """
        Train on a batch of experiences
        
        Args:
            batch: Dict containing batched experiences
            
        Returns:
            loss: Training loss value
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        agent_ids = list(states.keys())
        batch_size = len(states[agent_ids[0]])
        
        # Compute Q-values for each agent
        agent_q_values = []
        target_q_values = []
        
        for agent_id in agent_ids:
            # Convert to tensors
            obs = torch.FloatTensor(states[agent_id]).to(self.device)
            next_obs = torch.FloatTensor(next_states[agent_id]).to(self.device)
            act = torch.LongTensor(actions[agent_id]).to(self.device)
            
            # Current Q-values
            q, _ = self.agent_network(obs)
            q_taken = q.gather(1, act.unsqueeze(1))
            agent_q_values.append(q_taken)
            
            # Target Q-values (using target network)
            with torch.no_grad():
                target_q, _ = self.target_agent_network(next_obs)
                target_q_max = target_q.max(1)[0].unsqueeze(1)
                target_q_values.append(target_q_max)
        
        # Stack Q-values: (batch_size, n_agents)
        agent_q_values = torch.cat(agent_q_values, dim=1)
        target_q_values = torch.cat(target_q_values, dim=1)
        
        # Create global state (concatenate all observations)
        global_state = torch.cat([
            torch.FloatTensor(states[aid]) for aid in agent_ids
        ], dim=1).to(self.device)
        
        next_global_state = torch.cat([
            torch.FloatTensor(next_states[aid]) for aid in agent_ids
        ], dim=1).to(self.device)
        
        # Mix Q-values
        q_total = self.mixer(agent_q_values, global_state)
        
        with torch.no_grad():
            target_q_total = self.target_mixer(target_q_values, next_global_state)
        
        # Compute total rewards and dones
        rewards_total = torch.FloatTensor([
            sum(rewards[aid][i] for aid in agent_ids) 
            for i in range(batch_size)
        ]).to(self.device)
        
        dones_total = torch.FloatTensor([
            float(any(dones[aid][i] for aid in agent_ids))
            for i in range(batch_size)
        ]).to(self.device)
        
        # TD target: r + γ * Q_target(s', a') * (1 - done)
        targets = rewards_total + self.gamma * target_q_total * (1 - dones_total)
        
        # Compute loss
        loss = F.mse_loss(q_total, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_network.parameters()) + list(self.mixer.parameters()), 
            10
        )
        
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_step += 1
        
        return loss.item()
    
    def update_target_networks(self):
        """Soft update of target networks"""
        # Update agent network
        for target_param, param in zip(
            self.target_agent_network.parameters(),
            self.agent_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # Update mixer network
        for target_param, param in zip(
            self.target_mixer.parameters(),
            self.mixer.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'agent_network': self.agent_network.state_dict(),
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_network.load_state_dict(checkpoint['agent_network'])
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.05)
        self.training_step = checkpoint.get('training_step', 0)