"""
Visualize training progress from log files
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """Parse training log file"""
    episodes = []
    rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse training logs
            match = re.search(r'Episode (\d+)/\d+ \| Mean Reward: ([\d.]+) \| Policy Loss: ([\d.]+) \| Value Loss: ([\d.]+) \| Entropy: ([\d.]+)', line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
                policy_losses.append(float(match.group(3)))
                value_losses.append(float(match.group(4)))
                entropies.append(float(match.group(5)))
    
    return {
        'episodes': episodes,
        'rewards': rewards,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropies': entropies
    }


def plot_training_curves(log_path, save_path='training_curves.png'):
    """Plot training curves"""
    data = parse_log_file(log_path)
    
    if not data['episodes']:
        print(f"No data found in {log_path}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Reward curve
    ax = axes[0, 0]
    ax.plot(data['episodes'], data['rewards'], linewidth=2, alpha=0.7)
    if len(data['rewards']) > 10:
        # Smooth curve
        window = min(20, len(data['rewards']) // 10)
        smoothed = np.convolve(data['rewards'], np.ones(window)/window, mode='valid')
        ax.plot(data['episodes'][window-1:], smoothed, linewidth=2, color='red', label='Smoothed')
        ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward Progress')
    ax.grid(True, alpha=0.3)
    
    # Policy loss
    ax = axes[0, 1]
    ax.plot(data['episodes'], data['policy_losses'], linewidth=2, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # Value loss
    ax = axes[1, 0]
    ax.plot(data['episodes'], data['value_losses'], linewidth=2, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)
    
    # Entropy
    ax = axes[1, 1]
    ax.plot(data['episodes'], data['entropies'], linewidth=2, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, help='Path to specific log file')
    parser.add_argument('--dir', type=str, default='logs', help='Directory with log files')
    parser.add_argument('--output', type=str, default='training_curves.png', 
                       help='Output plot filename')
    
    args = parser.parse_args()
    
    if args.log:
        plot_training_curves(args.log, args.output)
    else:
        # Find most recent log file
        if os.path.exists(args.dir):
            log_files = [f for f in os.listdir(args.dir) if f.endswith('.txt')]
            if log_files:
                latest_log = max([os.path.join(args.dir, f) for f in log_files], 
                               key=os.path.getctime)
                print(f"Using most recent log: {latest_log}")
                plot_training_curves(latest_log, args.output)
            else:
                print(f"No log files found in {args.dir}")
                print("Run training first: python train.py")
        else:
            print(f"Directory {args.dir} doesn't exist")
            print("Run training first: python train.py")
