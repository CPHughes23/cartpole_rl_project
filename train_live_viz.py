"""
PPO Training with Live Visualization
Shows real-time training progress with detailed console output
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from cartpole_env import CartPole
from ppo_agent import PPOAgent
from config import PPOConfig


class LiveTrainingVisualizer:
    """Real-time visualization of training progress"""
    
    def __init__(self):
        # Setup figure with subplots
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('Live PPO Training Visualization', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax_reward = self.fig.add_subplot(3, 3, 1)
        self.ax_loss = self.fig.add_subplot(3, 3, 2)
        self.ax_entropy = self.fig.add_subplot(3, 3, 3)
        self.ax_episode_length = self.fig.add_subplot(3, 3, 4)
        self.ax_recovery = self.fig.add_subplot(3, 3, 5)
        self.ax_phase = self.fig.add_subplot(3, 3, 6)
        self.ax_value_estimates = self.fig.add_subplot(3, 3, 7)
        self.ax_loss_components = self.fig.add_subplot(3, 3, 8)
        self.ax_stats = self.fig.add_subplot(3, 3, 9)
        
        # Data storage
        self.episodes = []
        self.rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.episode_lengths = []
        self.recovery_rates = []
        self.phases = []
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show(block=False)
        plt.pause(0.1)
    
    def update(self, episode, reward, policy_loss, value_loss, entropy, 
               episode_length, recovery_rate, phase):
        """Update all plots with new data"""
        
        # Store data
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.episode_lengths.append(episode_length)
        self.recovery_rates.append(recovery_rate)
        self.phases.append(phase)
        
        try:
            # Update plots (only redraw every 5 episodes for performance)
            if episode % 5 == 0 or episode < 10:
                self._update_reward_plot()
                self._update_loss_plot()
                self._update_entropy_plot()
                self._update_episode_length_plot()
                self._update_recovery_plot()
                self._update_phase_plot()
                self._update_loss_components_plot()
                self._update_stats_display()
                
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.001)
        except Exception as e:
            # Silently ignore matplotlib errors
            pass
    
    def _update_reward_plot(self):
        """Plot reward curve"""
        self.ax_reward.clear()
        if len(self.rewards) > 0:
            self.ax_reward.plot(self.episodes, self.rewards, alpha=0.4, color='blue', marker='o', markersize=3)
            
            # Smooth curve
            if len(self.rewards) > 5:
                window = min(10, len(self.rewards) // 3)
                smoothed = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
                self.ax_reward.plot(self.episodes[window-1:], smoothed, color='darkblue', linewidth=2.5, label='Smoothed')
            
            self.ax_reward.set_xlabel('Episode')
            self.ax_reward.set_ylabel('Mean Reward')
            self.ax_reward.set_title('Training Reward')
            self.ax_reward.grid(True, alpha=0.3)
            if len(self.rewards) > 5:
                self.ax_reward.legend()
    
    def _update_loss_plot(self):
        """Plot policy and value loss"""
        self.ax_loss.clear()
        if len(self.policy_losses) > 0:
            self.ax_loss.plot(self.episodes, self.policy_losses, label='Policy Loss', 
                            color='orange', alpha=0.7, linewidth=2)
            self.ax_loss.plot(self.episodes, self.value_losses, label='Value Loss', 
                            color='green', alpha=0.7, linewidth=2)
            self.ax_loss.set_xlabel('Episode')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.set_title('Training Losses')
            self.ax_loss.legend()
            self.ax_loss.grid(True, alpha=0.3)
    
    def _update_entropy_plot(self):
        """Plot entropy (exploration)"""
        self.ax_entropy.clear()
        if len(self.entropies) > 0:
            self.ax_entropy.plot(self.episodes, self.entropies, color='purple', 
                               alpha=0.7, linewidth=2)
            self.ax_entropy.set_xlabel('Episode')
            self.ax_entropy.set_ylabel('Entropy')
            self.ax_entropy.set_title('Policy Entropy (Exploration)')
            self.ax_entropy.grid(True, alpha=0.3)
    
    def _update_episode_length_plot(self):
        """Plot episode lengths"""
        self.ax_episode_length.clear()
        if len(self.episode_lengths) > 0:
            self.ax_episode_length.plot(self.episodes, self.episode_lengths, 
                                       alpha=0.5, color='teal', marker='o', markersize=3)
            
            if len(self.episode_lengths) > 5:
                window = min(10, len(self.episode_lengths) // 3)
                smoothed = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
                self.ax_episode_length.plot(self.episodes[window-1:], smoothed, 
                                          color='darkcyan', linewidth=2.5)
            
            self.ax_episode_length.set_xlabel('Episode')
            self.ax_episode_length.set_ylabel('Steps')
            self.ax_episode_length.set_title('Episode Length')
            self.ax_episode_length.grid(True, alpha=0.3)
    
    def _update_recovery_plot(self):
        """Plot recovery success rate"""
        self.ax_recovery.clear()
        if len(self.recovery_rates) > 0:
            self.ax_recovery.plot(self.episodes, self.recovery_rates, color='red', linewidth=2.5)
            self.ax_recovery.fill_between(self.episodes, 0, self.recovery_rates, alpha=0.3, color='red')
            self.ax_recovery.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
            self.ax_recovery.axhline(y=75, color='green', linestyle='--', alpha=0.5, label='75% target')
            self.ax_recovery.set_xlabel('Episode')
            self.ax_recovery.set_ylabel('Recovery Rate (%)')
            self.ax_recovery.set_title('Recovery Success Rate')
            self.ax_recovery.set_ylim([0, 100])
            self.ax_recovery.legend(fontsize=8)
            self.ax_recovery.grid(True, alpha=0.3)
    
    def _update_phase_plot(self):
        """Show current curriculum phase"""
        self.ax_phase.clear()
        if len(self.phases) > 0:
            phase_nums = [0 if p == 'Easy' else 1 if p == 'Medium' else 2 for p in self.phases]
            
            self.ax_phase.plot(self.episodes, phase_nums, linewidth=3, color='green', marker='s', markersize=5)
            self.ax_phase.fill_between(self.episodes, phase_nums, alpha=0.3, color='green')
            self.ax_phase.set_xlabel('Episode')
            self.ax_phase.set_ylabel('Difficulty')
            self.ax_phase.set_title('Curriculum Phase')
            self.ax_phase.set_yticks([0, 1, 2])
            self.ax_phase.set_yticklabels(['Easy', 'Medium', 'Hard'])
            self.ax_phase.set_ylim([-0.2, 2.2])
            self.ax_phase.grid(True, alpha=0.3)
    
    def _update_loss_components_plot(self):
        """Show loss ratio"""
        self.ax_loss_components.clear()
        if len(self.policy_losses) > 0 and len(self.value_losses) > 0:
            ratios = [p / (v + 1e-8) for p, v in zip(self.policy_losses, self.value_losses)]
            self.ax_loss_components.plot(self.episodes, ratios, color='brown', linewidth=2)
            self.ax_loss_components.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            self.ax_loss_components.set_xlabel('Episode')
            self.ax_loss_components.set_ylabel('Ratio')
            self.ax_loss_components.set_title('Policy/Value Loss Ratio')
            self.ax_loss_components.grid(True, alpha=0.3)
    
    def _update_stats_display(self):
        """Display current statistics"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if len(self.rewards) > 0:
            current_reward = self.rewards[-1]
            best_reward = max(self.rewards)
            current_recovery = self.recovery_rates[-1] if self.recovery_rates else 0
            current_phase = self.phases[-1] if self.phases else "N/A"
            
            stats_text = f"""
TRAINING STATISTICS

Episode: {self.episodes[-1]}
Phase: {current_phase}

Current Reward: {current_reward:.1f}
Best Reward: {best_reward:.1f}

Recovery Rate: {current_recovery:.0f}%

Policy Loss: {self.policy_losses[-1]:.4f}
Value Loss: {self.value_losses[-1]:.4f}
Entropy: {self.entropies[-1]:.4f}

Episode Length: {self.episode_lengths[-1]:.0f}
            """
            
            self.ax_stats.text(0.1, 0.5, stats_text, fontsize=10, 
                             verticalalignment='center', family='monospace',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def evaluate_policy(env, agent, num_episodes=10):
    """Evaluate policy"""
    total_rewards = []
    episode_lengths = []
    recovery_successes = 0
    
    for ep in range(num_episodes):
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


def train_with_visualization():
    """Main training loop with live visualization"""
    config = PPOConfig()
    env = CartPole()
    agent = PPOAgent(state_dim=4, action_dim=1, config=config)
    
    # Create visualizer
    viz = LiveTrainingVisualizer()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_viz_{timestamp}.txt'
    
    best_reward = -float('inf')
    last_best_episode = 0
    episode_rewards = []
    
    rollback_threshold = 100
    performance_drop_threshold = 0.3
    
    print("=" * 60)
    print("Starting CartPole PPO Training - LIVE VISUALIZATION")
    if config.recovery_training:
        print("RECOVERY MODE: Training with challenging initial states")
    print("=" * 60)
    print(f"Device: {agent.device}")
    print(f"Network: {config.hidden_sizes}")
    print(f"Episodes: {config.num_episodes}")
    print("=" * 60)
    
    try:
        for episode in range(config.num_episodes):
            # Curriculum
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
            
            # Reset environment
            if config.recovery_training:
                x = np.random.uniform(-max_pos, max_pos)
                theta = np.random.uniform(-max_angle, max_angle)
                x_dot = np.random.uniform(-1.0, 1.0)
                theta_dot = np.random.uniform(-1.5, 1.5)
                env.state = np.array([x, x_dot, theta, theta_dot])
                env.steps = 0
                env.edge_timer = 0
                env.prev_action = 0.0
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
                    if config.recovery_training:
                        x = np.random.uniform(-max_pos, max_pos)
                        theta = np.random.uniform(-max_angle, max_angle)
                        x_dot = np.random.uniform(-1.0, 1.0)
                        theta_dot = np.random.uniform(-1.5, 1.5)
                        env.state = np.array([x, x_dot, theta, theta_dot])
                        env.steps = 0
                        env.edge_timer = 0
                        env.prev_action = 0.0
                        state = env.state
                    else:
                        state = env.reset()
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    steps = 0
            
            # Get final value
            _, _, next_value = agent.select_action(state)
            
            # Update policy
            update_metrics = agent.update(next_value)
            
            # Calculate mean reward for logging
            recent_rewards = episode_rewards[-config.log_interval * 10:] if len(episode_rewards) > 0 else [0]
            mean_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
            
            # Update visualization every episode
            viz.update(episode + 1, mean_reward,
                      update_metrics['policy_loss'],
                      update_metrics['value_loss'],
                      update_metrics['entropy'],
                      np.mean(recent_rewards[-10:]) if len(recent_rewards) > 0 else 0,
                      0,  # Will update recovery rate during eval
                      phase)
            
            # Console logging (like train.py)
            if (episode + 1) % config.log_interval == 0:
                log_msg = (f"Episode {episode + 1}/{config.num_episodes} | Phase: {phase} | "
                          f"Mean Reward: {mean_reward:.2f} | "
                          f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                          f"Value Loss: {update_metrics['value_loss']:.4f} | "
                          f"Entropy: {update_metrics['entropy']:.4f}")
                print(log_msg)
                
                with open(log_file, 'a') as f:
                    f.write(log_msg + '\n')
            
            # Evaluation (like train.py)
            if (episode + 1) % config.eval_interval == 0:
                eval_metrics = evaluate_policy(env, agent, config.eval_episodes)
                
                # Update visualization with eval metrics
                viz.update(episode + 1, eval_metrics['mean_reward'],
                          update_metrics['policy_loss'],
                          update_metrics['value_loss'],
                          update_metrics['entropy'],
                          eval_metrics['mean_length'],
                          eval_metrics['recovery_rate'],
                          phase)
                
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
                
                # Save best
                if eval_metrics['mean_reward'] > best_reward:
                    best_reward = eval_metrics['mean_reward']
                    last_best_episode = episode
                    agent.save(config.best_model_path)
                    print(f"✓ New best model saved! Reward: {best_reward:.2f}\n")
                
                # Rollback check
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
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        # Final save
        agent.save(config.model_path)
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Models saved in: models/")
        print(f"Logs saved in: {log_file}")
        print("=" * 60)
        print("\nKeeping visualization window open...")
        print("Close the window to exit.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    train_with_visualization()