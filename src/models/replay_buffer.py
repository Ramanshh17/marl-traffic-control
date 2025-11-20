"""
Experience Replay Buffer for Multi-Agent RL
"""
import numpy as np
from collections import deque
import random
from typing import Dict

class ReplayBuffer:
    """
    Experience replay buffer for MARL
    Stores transitions and samples batches for training
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, states, actions, rewards, next_states, dones):
        """
        Add an experience to the buffer
        
        Args:
            states: Dict of observations for each agent
            actions: Dict of actions for each agent
            rewards: Dict of rewards for each agent
            next_states: Dict of next observations
            dones: Dict of done flags
        """
        experience = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict:
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dict containing batched experiences
        """
        # Adjust batch size if buffer is smaller
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Random sample
        batch = random.sample(self.buffer, batch_size)
        
        # Get agent IDs from first experience
        agent_ids = list(batch[0]['states'].keys())
        
        # Initialize batch dictionary
        batch_dict = {
            'states': {aid: [] for aid in agent_ids},
            'actions': {aid: [] for aid in agent_ids},
            'rewards': {aid: [] for aid in agent_ids},
            'next_states': {aid: [] for aid in agent_ids},
            'dones': {aid: [] for aid in agent_ids}
        }
        
        # Collect all experiences
        for exp in batch:
            for aid in agent_ids:
                batch_dict['states'][aid].append(exp['states'][aid])
                batch_dict['actions'][aid].append(exp['actions'][aid])
                batch_dict['rewards'][aid].append(exp['rewards'][aid])
                batch_dict['next_states'][aid].append(exp['next_states'][aid])
                batch_dict['dones'][aid].append(exp['dones'][aid])
        
        # Convert lists to numpy arrays
        for aid in agent_ids:
            batch_dict['states'][aid] = np.array(batch_dict['states'][aid])
            batch_dict['actions'][aid] = np.array(batch_dict['actions'][aid])
            batch_dict['rewards'][aid] = np.array(batch_dict['rewards'][aid])
            batch_dict['next_states'][aid] = np.array(batch_dict['next_states'][aid])
            batch_dict['dones'][aid] = np.array(batch_dict['dones'][aid])
        
        return batch_dict
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)