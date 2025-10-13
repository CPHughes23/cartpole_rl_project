import numpy as np
from cartpole_env import CartPole

# for equations and more information: https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

class LinearPolicy:
    def __init__(self, state_dim, action_std=1.0):
        self.theta = np.random.randn(state_dim) * 0.01
        self.action_std = action_std

    def get_action(self, state):
        mean = np.dot(self.theta, state)
        action = np.random.normal(mean, self.action_std)
        return action, mean
    
    def log_prob(self, state, action):
        mean = np.dot(self.theta, state)
        return -0.5 * ((action - mean) ** 2) / (self.action_std ** 2)
    
    def run_episode(env, policy, gamma=0.99):
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []

        terminated = False
        while not terminated:
            action, mean = policy.get_action(state)
            next_state, reward, terminated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(policy.log_prob(state, action))

            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        return states, actions, log_probs, returns

    def update_policy(policy, states, actions, log_probs, returns, lr=1e-2):
        for state, action, logp, G in zip(states, actions, log_probs, returns):
            mean = np.dot(policy.theta, state)
            grad_logp = ((action - mean) / (policy.action_std ** 2)) * state
            policy.theta += lr * grad_logp * G

    env = CartPoleEnv()
    policy = LinearPolicy(state_dim=4, action_std=1.0)
    sim = CartPoleSim(env, window_width=600, window_height=400)

    num_episodes = 500

    for episode in range(num_episodes):
        states, actions, log_probs, returns = run_episode(env, policy)
        update_policy(policy, states, actions, log_probs, returns)

        total_reward = sum(returns)
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
