# train_marl_ppo_plot.py
"""
Multi-Agent Train Induction RL
- DQN vs PPO comparison
- Auto-save plot of total episode rewards
- Fully debugged for VSCode/macOS headless
"""

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from collections import deque, namedtuple

# ---------- ENVIRONMENT ----------
class TrainInductionEnv:
    def __init__(self, n_agents=3, max_q=3, platform_time=3, spawn_prob=0.4, max_steps=200):
        self.n = n_agents
        self.max_q = max_q
        self.platform_time_total = platform_time
        self.spawn_prob = spawn_prob
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.queues = [0 for _ in range(self.n)]
        self.platform_busy = False
        self.platform_time_left = 0
        self.platform_agent = None
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.n):
            obs_i = np.array([
                self.queues[i] / float(self.max_q),
                1.0 if self.platform_busy else 0.0,
                (self.platform_time_left / float(self.platform_time_total)) if self.platform_busy else 0.0
            ], dtype=np.float32)
            obs.append(obs_i)
        return obs

    def step(self, actions):
        rewards = [0.0 for _ in range(self.n)]
        collision = False

        candidates = [i for i, a in enumerate(actions) if a == 1 and self.queues[i] > 0]
        if self.platform_busy:
            if candidates:
                collision = True
        else:
            if len(candidates) > 1:
                collision = True
            elif len(candidates) == 1:
                ag = candidates[0]
                self.platform_busy = True
                self.platform_agent = ag
                self.platform_time_left = self.platform_time_total
                self.queues[ag] -= 1
                rewards[ag] += 1.0

        if collision:
            for i in range(self.n):
                rewards[i] -= 5.0

        if self.platform_busy:
            self.platform_time_left -= 1
            if self.platform_time_left <= 0:
                self.platform_busy = False
                self.platform_agent = None
                self.platform_time_left = 0

        for i in range(self.n):
            if random.random() < self.spawn_prob and self.queues[i] < self.max_q:
                self.queues[i] += 1
                rewards[i] -= 0.05

        for i in range(self.n):
            rewards[i] -= 0.02 * self.queues[i]

        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_obs(), rewards, done, {'collision': collision}


# ---------- DQN ----------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, hidden=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=800):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(obs_dim).to(self.device)
        self.target_net = DQN(obs_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(5000)
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.batch_size = 64
        self.update_target_every = 200

    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(2)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(s)
        return int(q.argmax(dim=1).item())

    def store(self, *args):
        self.replay.push(*args)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        trans = self.replay.sample(self.batch_size)
        state = torch.tensor(np.vstack(trans.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(trans.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(trans.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.vstack(trans.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target = reward + (1 - done) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()


# ---------- PPO ----------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, output_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.net(x)
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, obs_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.policy(s)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def compute_returns(self, rewards, dones, next_value):
        R = next_value
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - d)
            returns.insert(0, R)
        return returns

    def update(self, trajectories):
        if len(trajectories) == 0:
            return 0.0
        states = torch.tensor(np.vstack([t[0] for t in trajectories]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in trajectories], dtype=torch.int64, device=self.device)
        old_log_probs = torch.stack([t[2] for t in trajectories]).detach()
        rewards = [t[3] for t in trajectories]
        dones = [t[5] for t in trajectories]

        _, state_values = self.policy(states)
        state_values = state_values.squeeze()
        returns = self.compute_returns(rewards, dones, 0)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns - state_values

        logits, _ = self.policy(states)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.functional.mse_loss(state_values, returns)
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ---------- TRAINING LOOP ----------
def train(env, agents, episodes=200, max_steps=200, method="DQN"):
    episode_returns = []
    print(f"=== Starting {method} training for {episodes} episodes ===", flush=True)
    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_rewards = [0.0 for _ in range(env.n)]
        done = False
        trajectories = [[] for _ in range(env.n)]
        while not done:
            actions = []
            log_probs = []
            for i, ag in enumerate(agents):
                if method == "DQN":
                    a = ag.select_action(obs[i])
                    actions.append(a)
                elif method == "PPO":
                    a, logp = ag.select_action(obs[i])
                    actions.append(a)
                    log_probs.append(logp)

            next_obs, rewards, done, info = env.step(actions)

            for i, ag in enumerate(agents):
                ep_rewards[i] += rewards[i]
                if method == "DQN":
                    ag.store(obs[i], actions[i], rewards[i], next_obs[i], float(done))
                    ag.train_step()
                elif method == "PPO":
                    trajectories[i].append((obs[i], actions[i], log_probs[i], rewards[i], next_obs[i], float(done)))

            obs = next_obs

        if method == "PPO":
            for i, ag in enumerate(agents):
                ag.update(trajectories[i])

        total = sum(ep_rewards)
        episode_returns.append(total)
        if ep % 5 == 0 or ep == 1:
            avg_recent = np.mean(episode_returns[-5:])
            print(f"[{method}] Episode {ep:4d} | Avg return (last 5): {avg_recent:.3f}", flush=True)

    print(f"=== Finished {method} training ===\n", flush=True)
    return agents, episode_returns


# ---------- MAIN ----------
if __name__ == "__main__":
    n_agents = 3
    episodes = 200
    env = TrainInductionEnv(n_agents=n_agents)

    # DQN
    dqn_agents = [DQNAgent(obs_dim=3) for _ in range(n_agents)]
    dqn_agents, dqn_returns = train(env, dqn_agents, episodes=episodes, method="DQN")

    # PPO
    ppo_agents = [PPOAgent(obs_dim=3) for _ in range(n_agents)]
    ppo_agents, ppo_returns = train(env, ppo_agents, episodes=episodes, method="PPO")

    # Save models
    os.makedirs("saved_models", exist_ok=True)
    for i, a in enumerate(dqn_agents):
        torch.save(a.policy_net.state_dict(), f"saved_models/dqn_agent_{i}_policy.pth")
    for i, a in enumerate(ppo_agents):
        torch.save(a.policy.state_dict(), f"saved_models/ppo_agent_{i}_policy.pth")

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12,5))
    plt.plot(dqn_returns, label="DQN")
    plt.plot(ppo_returns, label="PPO")
    plt.title("Episode Return Over Time (Sum of all agents)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/dqn_vs_ppo.png")
    print("Plot saved to plots/dqn_vs_ppo.png", flush=True)
    # Optional: comment this out if running headless
    # plt.show()
