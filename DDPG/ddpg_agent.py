import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))  # Output in [0, 1]
        return x


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.relu(self.l1(torch.cat([x, u], 1)))
        x = torch.relu(self.l2(x))
        return self.l3(x)


# Ornstein-Uhlenbeck Noise with Decay
class OUActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None, decay=0.9995):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.decay = decay
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        self.sigma *= self.decay  # Decay the noise
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)


# DDPG Agent with Target Network Update Frequency
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005,
                 lr_actor=1e-4, lr_critic=1e-3, target_update_frequency=2):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.noise = OUActionNoise(mu=np.zeros(action_dim), sigma=0.1, decay=0.995)

        self.gamma = gamma
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.train_step_counter = 0

    def select_action(self, state, explore=True):
        state = torch.FloatTensor(state).reshape(1, -1)
        action = self.actor(state).detach().numpy()[0]
        if explore:
            action += self.noise()
        action = np.clip(action, 0.0, 1.0)  # Clip between 0 and 1
        return action

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        self.train_step_counter += 1

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).reshape(-1, 1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).reshape(-1, 1)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Periodic target network updates
        if self.train_step_counter % self.target_update_frequency == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
