import os
import copy
import torch
from torch import multiprocessing
from matplotlib import pyplot as plt
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.envs import (
    check_env_specs,
    RewardSum,
    TransformedEnv,
    VmasEnv,
)
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from tqdm import tqdm
from tensordict.nn import TensorDictSequential
from models import create_policies, create_critics
from torch.utils.tensorboard import SummaryWriter

experiment_name = "MADDPG_simple_tag"

# Create directories for results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
plots_dir = os.path.join(results_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
logs_dir = os.path.join(results_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

experiment_log_dir = os.path.join(logs_dir, experiment_name)
os.makedirs(experiment_log_dir, exist_ok=True)

existing_runs = [
    d for d in os.listdir(experiment_log_dir)
    if os.path.isdir(os.path.join(experiment_log_dir, d)) and d.startswith('run')
]
run_numbers = [int(d[3:]) for d in existing_runs if d[3:].isdigit()]
if run_numbers:
    next_run_number = max(run_numbers) + 1
else:
    next_run_number = 1
run_name = f"run{next_run_number}"

log_dir = os.path.join(experiment_log_dir, run_name)
os.makedirs(log_dir, exist_ok=True)
print(f"TensorBoard logs will be saved to {log_dir}")

plot_filename = os.path.join(
    plots_dir, f"{experiment_name}_training_plot_{run_name}.png"
)

# Hyperparameters
seed = 0
torch.manual_seed(seed)

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
)

# Sampling
frames_per_batch = 1000
n_iters = 100
total_frames = frames_per_batch * n_iters

# Training control
iteration_when_stop_training_evaders = n_iters // 2

# Replay buffer
memory_size = 1000000

# Training
n_optimiser_steps = 100
train_batch_size = 128
lr = 3e-4
max_grad_norm = 1.0

# DDPG
gamma = 0.99
polyak_tau = 0.005

# Environment setup
max_steps = 100
n_chasers = 2
n_evaders = 1
n_obstacles = 2

num_vmas_envs = frames_per_batch // max_steps
base_env = VmasEnv(
    scenario="simple_tag",
    num_envs=num_vmas_envs,
    continuous_actions=True,
    max_steps=max_steps,
    device=device,
    seed=seed,
    num_good_agents=n_evaders,
    num_adversaries=n_chasers,
    num_landmarks=n_obstacles,
)

env = TransformedEnv(
    base_env,
    RewardSum(
        in_keys=base_env.reward_keys,
        reset_keys=["_reset"] * len(base_env.group_map.keys()),
    ),
)

check_env_specs(env)

# Create policies and critics
policies, exploration_policies = create_policies(env, device, total_frames)
critics = create_critics(env, device)

# Test policies and critics
reset_td = env.reset()
for group in env.group_map.keys():
    print(
        f"Running value and policy for group '{group}':",
        critics[group](policies[group](reset_td)),
    )

# Data collector
agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

collector = SyncDataCollector(
    env,
    agents_exploration_policy,
    device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

# Replay buffers
replay_buffers = {}
for group in env.group_map.keys():
    replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(memory_size),
        sampler=RandomSampler(),
        batch_size=train_batch_size,
    )
    if device.type != "cpu":
        replay_buffer.append_transform(lambda x: x.to(device))
    replay_buffers[group] = replay_buffer

# Loss functions and optimizers
losses = {}
for group in env.group_map.keys():
    loss_module = DDPGLoss(
        actor_network=policies[group],
        value_network=critics[group],
        delay_value=True,
        loss_function="l2",
    )
    loss_module.set_keys(
        state_action_value=(group, "state_action_value"),
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    losses[group] = loss_module

target_updaters = {
    group: SoftUpdate(loss_module, tau=polyak_tau)
    for group, loss_module in losses.items()
}

optimizers = {
    group: {
        "loss_actor": torch.optim.Adam(
            loss_module.actor_network_params.flatten_keys().values(), lr=lr
        ),
        "loss_value": torch.optim.Adam(
            loss_module.value_network_params.flatten_keys().values(), lr=lr
        ),
    }
    for group, loss_module in losses.items()
}

# TensorBoard logging
writer = SummaryWriter(log_dir=log_dir)


# Training utility function
def process_batch(batch: TensorDictBase) -> TensorDictBase:
    for group in env.group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch


# Training loop
pbar = tqdm(
    total=n_iters,
    desc=", ".join(
        [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
    ),
)
episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
train_group_map = copy.deepcopy(env.group_map)

for iteration, batch in enumerate(collector):
    current_frames = batch.numel()
    batch = process_batch(batch)
    # Loop over groups
    for group in train_group_map.keys():
        group_batch = batch.exclude(
            *[
                key
                for _group in env.group_map.keys()
                if _group != group
                for key in [_group, ("next", _group)]
            ]
        )
        group_batch = group_batch.reshape(-1)
        replay_buffers[group].extend(group_batch)

        for _ in range(n_optimiser_steps):
            subdata = replay_buffers[group].sample()
            loss_vals = losses[group](subdata)

            for loss_name in ["loss_actor", "loss_value"]:
                loss = loss_vals[loss_name]
                optimizer = optimizers[group][loss_name]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]["params"], max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()

            target_updaters[group].step()

        exploration_policies[group][-1].step(current_frames)

    if iteration == iteration_when_stop_training_evaders:
        del train_group_map["agent"]

    # Logging
    for group in env.group_map.keys():
        episode_reward_mean = (
            batch.get(("next", group, "episode_reward"))[
                batch.get(("next", group, "done"))
            ]
            .mean()
            .item()
        )
        episode_reward_mean_map[group].append(episode_reward_mean)

        writer.add_scalar(
            f"{group}/episode_reward_mean",
            episode_reward_mean,
            iteration,
        )

    pbar.set_description(
        ", ".join(
            [
                f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]:.2f}"
                for group in env.group_map.keys()
            ]
        ),
        refresh=False,
    )
    pbar.update()

writer.close()

# Plot results
fig, axs = plt.subplots(len(env.group_map.keys()), 1, figsize=(8, 6))
for i, group in enumerate(env.group_map.keys()):
    axs[i].plot(episode_reward_mean_map[group], label=f"Episode reward mean {group}")
    axs[i].set_ylabel("Reward")
    axs[i].axvline(
        x=iteration_when_stop_training_evaders,
        label="Agent (evader) stop training",
        color="orange",
    )
    axs[i].legend()
axs[-1].set_xlabel("Training iterations")
plt.tight_layout()

# Save the plot
plt.savefig(plot_filename)
print(f"Training plot saved to {plot_filename}")
plt.close()
