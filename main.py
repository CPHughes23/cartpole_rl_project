import gymnasium as gym
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1", render_mode="human")
obs = env.reset()
terminated = False

while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
env.close()
