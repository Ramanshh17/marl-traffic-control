#!/usr/bin/env python3
"""
QMIX Training Script
Multi-Agent Reinforcement Learning for Smart Traffic Signal Control
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.environments.traffic_env import MultiIntersectionEnv
from src.models.qmix import QMixAgent
from src.models.replay_buffer import ReplayBuffer

def train_qmix():
    """Main training function for QMIX"""
    
    print("=" * 70)
    print(" " * 25 + "QMIX TRAINING")
    print("=" * 70)
    
    # Load configuration
    config_path = "configs/qmix_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    print("\n🌐 Creating environment...")
    env = MultiIntersectionEnv()
    print(f"   Intersections: {env.num_intersections}")
    print(f"   Observation dim: {env.obs_dim}")
    print(f"   Action space: {env.n_actions}")
    
    # Create QMIX agent
    print("\n🤖 Creating QMIX agent...")
    agent = QMixAgent(config_path, env.num_intersections, env.obs_dim, env.n_actions)
    print(f"   Device: {agent.device}")
    print(f"   Learning rate: {agent.lr}")
    print(f"   Gamma: {agent.gamma}")
    
    # Create replay buffer
    buffer_size = config['training']['buffer_size']
    replay_buffer = ReplayBuffer(buffer_size)
    print(f"   Buffer size: {buffer_size}")
    
    # Training parameters
    total_episodes = config['training']['total_episodes']
    batch_size = config['training']['batch_size']
    target_update_interval = config['training']['target_update_interval']
    save_interval = config['checkpoint']['save_interval']
    save_path = Path(config['checkpoint']['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics tracking
    metrics = {
        'episode_rewards': [],
        'avg_queue': [],
        'avg_waiting': [],
        'throughput': [],
        'losses': [],
        'epsilon': []
    }
    
    # Training state
    total_steps = 0
    best_reward = -float('inf')
    
    print(f"\n🚀 Starting training for {total_episodes} episodes...")
    print(f"⏱️  Estimated time: ~{total_episodes * 2 // 60} minutes")
    print("=" * 70 + "\n")
    
    # Training loop with progress bar
    pbar = tqdm(range(total_episodes), desc="Training", ncols=100)
    
    for episode in pbar:
        # Reset environment and agent
        obs, _ = env.reset(seed=episode)
        agent.reset_hidden_states()
        
        # Episode variables
        episode_reward = 0
        episode_loss = []
        done = False
        step = 0
        
        # Episode loop
        while not done:
            # Select actions for all agents
            actions = agent.select_actions(obs, explore=True)
            
            # Execute actions in environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Store experience in replay buffer
            replay_buffer.push(obs, actions, rewards, next_obs, terminated)
            
            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train(batch)
                episode_loss.append(loss)
                
                # Update target networks periodically
                if total_steps % target_update_interval == 0:
                    agent.update_target_networks()
            
            # Update state
            obs = next_obs
            episode_reward += sum(rewards.values())
            total_steps += 1
            step += 1
            
            # Check if episode is done
            done = terminated['__all__'] or truncated['__all__']
        
        # Record metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['avg_queue'].append(info['avg_queue_length'])
        metrics['avg_waiting'].append(info['avg_waiting_time'])
        metrics['throughput'].append(info['total_throughput'])
        metrics['epsilon'].append(agent.epsilon)
        
        if episode_loss:
            metrics['losses'].append(np.mean(episode_loss))
        
        # Update progress bar with statistics
        avg_reward_100 = np.mean(metrics['episode_rewards'][-100:])
        pbar.set_postfix({
            'Ep_R': f'{episode_reward:.1f}',
            'Avg100': f'{avg_reward_100:.1f}',
            'Queue': f'{info["avg_queue_length"]:.2f}',
            'ε': f'{agent.epsilon:.3f}'
        })
        
        # Save best model
        if avg_reward_100 > best_reward and episode > 50:
            best_reward = avg_reward_100
            agent.save(str(save_path / 'qmix_best.pth'))
        
        # Save periodic checkpoint
        if episode > 0 and episode % save_interval == 0:
            agent.save(str(save_path / f'qmix_ep{episode}.pth'))
            
            # Save metrics
            Path('results').mkdir(exist_ok=True)
            with open('results/qmix_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
    
    # Save final model
    agent.save(str(save_path / 'qmix_final.pth'))
    
    # Save final metrics
    with open('results/qmix_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Statistics (last 100 episodes):")
    print(f"   Average Reward: {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    print(f"   Average Queue Length: {np.mean(metrics['avg_queue'][-100:]):.2f}")
    print(f"   Average Waiting Time: {np.mean(metrics['avg_waiting'][-100:]):.2f}s")
    print(f"   Average Throughput: {np.mean(metrics['throughput'][-100:]):.0f} vehicles")
    print(f"\n💾 Models saved to: {save_path}")
    print(f"📈 Metrics saved to: results/qmix_metrics.json")
    print(f"🏆 Best model saved at: {save_path / 'qmix_best.pth'}")
    print("=" * 70 + "\n")
    
    env.close()
    return metrics

if __name__ == "__main__":
    try:
        print("\n" + "🚦" * 20)
        print("Multi-Agent RL for Smart Traffic Signal Control")
        print("🚦" * 20 + "\n")
        
        metrics = train_qmix()
        
        print("✓ Training completed successfully!")
        print("✓ You can now analyze results in results/qmix_metrics.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()