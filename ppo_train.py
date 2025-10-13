import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        # Policy mean output
        self.mu = nn.Linear(hidden_size, action_dim)
        # Log std (trainable scalar)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Value function
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, state):
        features = self.shared(state)
        mu = self.mu(features)
        std = torch.exp(self.log_std)
        value = self.value(features)
        return mu, std, value


def compute_returns_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    This uses Generalized Advantage Estimation (GAE)
    For more info see https://arxiv.org/abs/1506.02438
    """
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(policy, optimizer, states, actions, log_probs_old, returns, advantages, clip_ratio=0.2, epochs=10):
    for _ in range(epochs):
        mu, std, values = policy(states)
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1).mean()

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = ((returns - values.squeeze()) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(env, num_episodes=500, steps_per_update=2000, gamma=0.99, lam=0.95, lr=3e-4):
    state_dim = 4
    action_dim = 1
    policy = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        state = torch.tensor(env.reset(), dtype=torch.float32)

        for step in range(steps_per_update):
            mu, std, value = policy(state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action.detach())
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.detach())
            values.append(value.detach())

            state = torch.tensor(next_state, dtype=torch.float32)
            if done:
                state = torch.tensor(env.reset(), dtype=torch.float32)

        with torch.no_grad():
            _, _, next_value = policy(state)
        values.append(next_value)

        advantages, returns = compute_returns_advantages(rewards, [v.item() for v in values], dones, gamma, lam)
        ppo_update(policy, optimizer,
                   torch.stack(states),
                   torch.stack(actions),
                   torch.stack(log_probs),
                   returns,
                   advantages)

        print(f"Episode {episode+1}/{num_episodes}: mean reward = {np.mean(rewards):.2f}")

    torch.save(policy.state_dict(), "ppo_cartpole.pth")
    return policy
