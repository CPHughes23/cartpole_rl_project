from cartpole_env import CartPole
from ppo_train import train

env = CartPole()

policy = train(env, num_episodes=500, steps_per_update=2000)

import torch
torch.save(policy.state_dict(), "ppo_cartpole.pth")
