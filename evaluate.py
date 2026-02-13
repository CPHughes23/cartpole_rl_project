"""
Evaluate trained model
"""

import torch
import numpy as np
from cartpole_env import CartPole
from ppo_agent import PPOAgent
from config import PPOConfig


def evaluate_model(model_path, num_episodes=10, render=False):
    """Evaluate a trained model"""
    # Setup
    config = PPOConfig()
    env = CartPole()
    agent = PPOAgent(state_dim=4, action_dim=1, config=config)
    
    # Load model
    agent.load(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Evaluate
    total_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action[0])
            episode_reward += reward
            steps += 1
            
            if render and steps % 10 == 0:
                print(f"  Step {steps}: x={info['x']:.3f}, theta={info['theta']:.3f}")
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"  Reward: {episode_reward:.2f}, Steps: {steps}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min Reward:  {np.min(total_rewards):.2f}")
    print(f"Max Reward:  {np.max(total_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/ppo_cartpole_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Print state information')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.episodes, args.render)