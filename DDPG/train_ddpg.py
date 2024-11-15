import numpy as np
from pettingzoo.mpe import simple_tag_v3
from ddpg_agent import DDPGAgent
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/DDPG_simple_tag")

# Best hyperparameters (as per tune_ddpg.py)
best_params = {
    "gamma": 0.9245719521461825,
    "lr_actor": 0.00019700746629363635,
    "lr_critic": 0.0007123313579529518,
    "tau": 0.01598388776262126,
    "batch_size": 48
}

# Initialize environment
env = simple_tag_v3.parallel_env(continuous_actions=True)
observations, infos = env.reset()

# Determine the maximum observation dimension
obs_spaces = [env.observation_space(agent).shape[0] for agent in env.agents]
state_dim = max(obs_spaces)

# Get action dimension and max_action
action_dim = env.action_space(env.agents[0]).shape[0]
max_action = float(env.action_space(env.agents[0]).high[0])

# Initialize DDPG agent
agent = DDPGAgent(
    state_dim, action_dim, max_action,
    gamma=best_params["gamma"],
    tau=best_params["tau"],
    lr_actor=best_params["lr_actor"],
    lr_critic=best_params["lr_critic"]
)


# Pad observation to the required state dimension
def pad_observation(observation, target_dim=state_dim):
    if len(observation) < target_dim:
        # Pad observation with zeros to reach target dimension
        padded_observation = np.zeros(target_dim)
        padded_observation[:len(observation)] = observation
        return padded_observation
    return observation


# Training loop
num_episodes = 10000
max_steps = 100
batch_size = best_params["batch_size"]

for episode in range(num_episodes):
    observations, infos = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        actions = {}
        for agent_name in env.agents:
            observation = observations[agent_name]
            observation = pad_observation(observation)
            action = agent.select_action(observation)
            actions[agent_name] = action
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent_name in env.agents:
            observation = observations[agent_name]
            observation = pad_observation(observation)
            action = actions[agent_name]
            reward = rewards[agent_name]
            next_observation = pad_observation(next_observations[agent_name])
            done = terminations[agent_name] or truncations[agent_name]
            agent.replay_buffer.add(observation, action, reward, next_observation, done)
            episode_reward += reward
        agent.train(batch_size)
        observations = next_observations
        if all(terminations.values()) or all(truncations.values()):
            break
    # Log rewards to TensorBoard
    writer.add_scalar("Rewards/Episode", episode_reward, episode)
    print(f"Episode {episode + 1}, Reward: {episode_reward}")

writer.close()
