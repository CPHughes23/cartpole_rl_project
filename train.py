"""
Main training script for CartPole with PPO
Includes recovery training and checkpoint rollback
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
    recovery_successes = 0
    
    for ep in range(num_episodes):
        # Test with both normal and challenging starts
        if ep < num_episodes // 2:
            state = env.reset(recovery_training=False)
        else:
            state = env.reset(recovery_training=True)
            
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action[0])
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if ep >= num_episodes // 2 and steps > 100:
            recovery_successes += 1
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'max_reward': np.max(total_rewards),
        'recovery_rate': recovery_successes / (num_episodes // 2) * 100
    }


def train():
    """Main training loop with curriculum and checkpoint rollback"""
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
    last_best_episode = 0
    episode_rewards = []
    
    # Rollback settings
    rollback_threshold = 100  # Check every N episodes
    performance_drop_threshold = 0.3  # If performance drops 30%, rollback
    
    print("=" * 60)
    print("Starting CartPole PPO Training - FULL RANGE")
    print("With Curriculum Learning and Checkpoint Rollback")
    print("=" * 60)
    print(f"Device: {agent.device}")
    print(f"Network: {config.hidden_sizes}")
    print(f"Episodes: {config.num_episodes}")
    print("=" * 60)
    
    # Training loop
    for episode in range(config.num_episodes):
        # CURRICULUM: Gradually increase difficulty
        progress = episode / config.num_episodes
        
        if progress < 0.33:
            max_angle = 15 * np.pi/180
            max_pos = 1.0
            phase = "Easy"
        elif progress < 0.66:
            max_angle = 60 * np.pi/180
            max_pos = 1.5
            phase = "Medium"
        else:
            max_angle = np.pi
            max_pos = 2.0
            phase = "Hard"
        
        # Custom reset with curriculum
        if config.recovery_training:
            x = np.random.uniform(-max_pos, max_pos)
            theta = np.random.uniform(-max_angle, max_angle)
            x_dot = np.random.uniform(-1.0, 1.0)
            theta_dot = np.random.uniform(-1.5, 1.5)
            env.state = np.array([x, x_dot, theta, theta_dot])
            env.steps = 0
            env.edge_timer = 0
            state = env.state
        else:
            state = env.reset()
        
        episode_reward = 0
        steps = 0
        
        # Collect trajectories
        for step in range(config.steps_per_episode):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action[0])
            
            # Recovery bonus
            if config.recovery_training:
                angle_improvement = abs(state[2]) - abs(next_state[2])
                if angle_improvement > 0:
                    reward += 0.3 * angle_improvement
            
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                # Reset with current curriculum
                if config.recovery_training:
                    x = np.random.uniform(-max_pos, max_pos)
                    theta = np.random.uniform(-max_angle, max_angle)
                    x_dot = np.random.uniform(-1.0, 1.0)
                    theta_dot = np.random.uniform(-1.5, 1.5)
                    env.state = np.array([x, x_dot, theta, theta_dot])
                    env.steps = 0
                    env.edge_timer = 0
                    state = env.state
                else:
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
            mean_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
            
            log_msg = (f"Episode {episode + 1}/{config.num_episodes} | Phase: {phase} | "
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
                       f"Recovery Success Rate: {eval_metrics['recovery_rate']:.1f}%\n"
                       f"{'='*60}\n")
            print(eval_msg)
            
            with open(log_file, 'a') as f:
                f.write(eval_msg)
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                last_best_episode = episode
                agent.save(config.best_model_path)
                print(f"✓ New best model saved! Reward: {best_reward:.2f}\n")
            
            # CHECKPOINT ROLLBACK: If performance dropped significantly, reload best
            elif eval_metrics['mean_reward'] < best_reward * (1 - performance_drop_threshold):
                if episode - last_best_episode >= rollback_threshold:
                    print(f"⚠ Performance dropped {performance_drop_threshold*100}%! Rolling back to best model...")
                    print(f"   Current: {eval_metrics['mean_reward']:.2f}, Best: {best_reward:.2f}")
                    try:
                        agent.load(config.best_model_path)
                        print(f"✓ Rolled back to episode {last_best_episode}\n")
                    except:
                        print("✗ Could not load best model, continuing...\n")
        
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