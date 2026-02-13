"""
Configuration file for CartPole PPO training
Adjust these hyperparameters for optimal performance
"""

class PPOConfig:
    # Environment settings
    max_episode_steps = 500
    
    # Network architecture
    hidden_sizes = [128, 128]  # Two hidden layers with 128 units each
    activation = 'tanh'  # 'tanh' or 'relu'
    
    # PPO hyperparameters
    learning_rate = 3e-4
    gamma = 0.99  # Discount factor
    gae_lambda = 0.95  # GAE parameter
    clip_ratio = 0.2  # PPO clip parameter
    value_loss_coef = 0.5  # Value loss coefficient
    entropy_coef = 0.01  # Entropy bonus coefficient
    max_grad_norm = 0.5  # Gradient clipping
    
    # Training settings
    num_episodes = 100
    steps_per_episode = 2048  # Collect this many steps before update
    ppo_epochs = 10  # Number of epochs for PPO update
    minibatch_size = 64  # Minibatch size for updates
    
    # Logging and evaluation
    log_interval = 10  # Log every N episodes
    eval_interval = 50  # Evaluate every N episodes
    eval_episodes = 10  # Number of episodes for evaluation
    save_interval = 100  # Save model every N episodes
    
    # Model saving
    model_path = "models/ppo_cartpole.pth"
    best_model_path = "models/ppo_cartpole_best.pth"