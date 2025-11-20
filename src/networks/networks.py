"""
Neural Network Architectures for MARL
"""
import torch
import torch.nn as nn
from typing import List

class GRUNetwork(nn.Module):
    """
    GRU-based Q-Network for QMIX
    Uses recurrent network to handle partial observability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize GRU network
        
        Args:
            input_dim: Dimension of input (observation)
            hidden_dim: Dimension of hidden state
            output_dim: Dimension of output (number of actions)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            hidden: Hidden state (1, batch_size, hidden_dim)
            
        Returns:
            q_values: Q-values for each action (batch_size, output_dim)
            hidden: Updated hidden state
        """
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # GRU forward pass
        out, hidden = self.gru(x, hidden)
        
        # Get Q-values from last timestep
        q_values = self.fc(out[:, -1, :])
        
        return q_values, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state"""
        return torch.zeros(1, batch_size, self.hidden_dim).to(device)


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron for general use
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        """
        Initialize MLP
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)