import optuna
import numpy as np
from pettingzoo.mpe import simple_tag_v3
from ddpg_agent import DDPGAgent


# Pad observation to the required state dimension
def pad_observation(observation, target_dim):
    if len(observation) < target_dim:
        padded_observation = np.zeros(target_dim)
        padded_observation[:len(observation)] = observation
        return padded_observation
    return observation


def train_agent(params, num_episodes=100, max_steps=100):
    env = simple_tag_v3.parallel_env(continuous_actions=True)
    agent_name = env.agents[0]

    # Determine the maximum observation dimension
    obs_spaces = [env.observation_space(agent).shape[0] for agent in env.agents]
    state_dim = max(obs_spaces)

    # Define agent parameters
    action_dim = env.action_space(agent_name).shape[0]
    max_action = float(env.action_space(agent_name).high[0])

    # Initialize DDPG agent
    agent = DDPGAgent(
        state_dim, action_dim, max_action,
        gamma=params["gamma"],
        tau=params["tau"],
        lr_actor=params["lr_actor"],
        lr_critic=params["lr_critic"]
    )

    cumulative_reward = 0
    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            actions = {}
            for agent_name in env.agents:
                observation = pad_observation(observations[agent_name], state_dim)
                action = agent.select_action(observation)
                actions[agent_name] = action
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent_name in env.agents:
                observation = pad_observation(observations[agent_name], state_dim)
                action = actions[agent_name]
                reward = rewards[agent_name]
                next_observation = pad_observation(next_observations[agent_name], state_dim)
                done = terminations[agent_name] or truncations[agent_name]

                agent.replay_buffer.add(observation, action, reward, next_observation, done)
                episode_reward += reward
            agent.train(params["batch_size"])
            observations = next_observations

            if all(terminations.values()) or all(truncations.values()):
                break
        cumulative_reward += episode_reward
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    env.close()
    return cumulative_reward / num_episodes


# Define Optuna objective function
def objective(trial):
    params = {
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "lr_actor": trial.suggest_float("lr_actor", 1e-5, 1e-3),
        "lr_critic": trial.suggest_float("lr_critic", 1e-5, 1e-3),
        "tau": trial.suggest_float("tau", 0.005, 0.05),
        "batch_size": trial.suggest_int("batch_size", 32, 256),
    }
    average_reward = train_agent(params, num_episodes=100, max_steps=100)
    return -average_reward


# Set up and run
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
