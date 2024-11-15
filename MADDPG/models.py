import torch
from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
    AdditiveGaussianModule,
)
from tensordict.nn import TensorDictModule, TensorDictSequential


def create_policies(env, device, total_frames):
    policy_modules = {}
    for group, agents in env.group_map.items():
        share_parameters_policy = True

        policy_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1],
            n_agent_outputs=env.full_action_spec[group, "action"].shape[-1],
            n_agents=len(agents),
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        )

        policy_module = TensorDictModule(
            policy_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "param")],
        )
        policy_modules[group] = policy_module

    policies = {}
    for group in env.group_map.keys():
        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            distribution_class=TanhDelta,
            distribution_kwargs={
                "low": env.full_action_spec[group, "action"].space.low,
                "high": env.full_action_spec[group, "action"].space.high,
            },
            return_log_prob=False,
        )
        policies[group] = policy

    exploration_policies = {}
    for group in env.group_map.keys():
        exploration_policy = TensorDictSequential(
            policies[group],
            AdditiveGaussianModule(
                spec=policies[group].spec,
                annealing_num_steps=total_frames // 2,
                action_key=(group, "action"),
                sigma_init=0.9,
                sigma_end=0.1,
            ),
        )
        exploration_policies[group] = exploration_policy

    return policies, exploration_policies


def create_critics(env, device):
    critics = {}
    for group, agents in env.group_map.items():
        share_parameters_critic = True
        MADDPG = True

        cat_module = TensorDictModule(
            lambda obs, action: torch.cat([obs, action], dim=-1),
            in_keys=[(group, "observation"), (group, "action")],
            out_keys=[(group, "obs_action")],
        )

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=env.observation_spec[group, "observation"].shape[-1]
                + env.full_action_spec[group, "action"].shape[-1],
                n_agent_outputs=1,
                n_agents=len(agents),
                centralised=MADDPG,
                share_params=share_parameters_critic,
                device=device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[(group, "obs_action")],
            out_keys=[(group, "state_action_value")],
        )

        critics[group] = TensorDictSequential(cat_module, critic_module)

    return critics
