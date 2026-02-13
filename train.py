"""
Main training script for CartPole with PPO
"""

import os
import numpy as np
import torch
from datetime import datetime
from cartpole_env import CartPole
from ppo_agent import PPOAgent
from config import PPOConfig


def evaluate_policy(env, agent, num_episodes=10):
    """Evaluate the policy over multiple episodes"""
    total_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action[0])
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'max_reward': np.max(total_rewards)
    }


def train():
    """Main training loop"""
    # Setup
    config = PPOConfig()
    env = CartPole()
    agent = PPOAgent(state_dim=4, action_dim=1, config=config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Training log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.txt'
    
    # Training metrics
    best_reward = -float('inf')
    episode_rewards = []
    
    print("=" * 60)
    print("Starting CartPole PPO Training")
    print("=" * 60)
    print(f"Device: {agent.device}")
    print("=" * 60)
    
    # Training loop
    for episode in range(config.num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        # Collect trajectories
        for step in range(config.steps_per_episode):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action[0])
            
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                state = env.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0
                steps = 0
        
        # Get final value estimate
        _, _, next_value = agent.select_action(state)
        
        # Update policy
        update_metrics = agent.update(next_value)
        
        # Logging
        if (episode + 1) % config.log_interval == 0:
            recent_rewards = episode_rewards[-config.log_interval * 10:] if len(episode_rewards) > 0 else [0]
            mean_reward = np.mean(recent_rewards)
            
            log_msg = (f"Episode {episode + 1}/{config.num_episodes} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                      f"Value Loss: {update_metrics['value_loss']:.4f} | "
                      f"Entropy: {update_metrics['entropy']:.4f}")
            print(log_msg)
            
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
        
        # Evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_policy(env, agent, config.eval_episodes)
            
            eval_msg = (f"\n{'='*60}\n"
                       f"Evaluation at Episode {episode + 1}\n"
                       f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}\n"
                       f"Mean Length: {eval_metrics['mean_length']:.1f}\n"
                       f"Max Reward: {eval_metrics['max_reward']:.2f}\n"
                       f"{'='*60}\n")
            print(eval_msg)
            
            with open(log_file, 'a') as f:
                f.write(eval_msg)
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                agent.save(config.best_model_path)
                print(f"✓ New best model saved! Reward: {best_reward:.2f}\n")
        
        # Regular save
        if (episode + 1) % config.save_interval == 0:
            agent.save(config.model_path)
            print(f"Model saved at episode {episode + 1}\n")
    
    # Final save
    agent.save(config.model_path)
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Models saved in: models/")
    print(f"Logs saved in: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    train()