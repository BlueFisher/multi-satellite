from argparse import Namespace
from typing import List, Dict, Any

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

from algorithms.Manager import Manager


class Evaluator:
    def __init__(
        self,
        env: gym.Env,
        name_agents: List[str],
        alg: Manager,  # 只接受一个算法
        args: Namespace,
    ) -> None:
        self.env = env
        self.name_agents = name_agents
        self.alg = alg  # 保存单个算法
        self.args = args

    def eval_episode(
        self,
        max_episode_steps: int,
        render: bool = False,
        scaler=None,
        recorder=None,
        episode_index: int = 0,
    ) -> (Dict[str, List[float]], Dict[str, Any]):

        tag_generate = (
            self.args.just_eval
            and hasattr(self.args, "generate_image")
            and self.args.generate_image
        )

        # 直接调用单个算法的钩子
        self.alg.before_episode(-1)

        observations, info = self.env.reset()

        if recorder is not None:
            recorder.begin_episode(
                episode_index, self.env.unwrapped, observations
            )

        episode_rewards = {agent: [] for agent in self.name_agents}
        obs_list = [
            np.asarray(observations[agent]).reshape(1, -1) for agent in self.name_agents
        ]

        if scaler is not None:
            obs_list = scaler.transform(obs_list)

        if tag_generate:
            trail = {}

        reached_step_limit = True
        for step in range(max_episode_steps):
            # 批处理：一次性传入所有智能体的观测
            choose_ret = self.alg.choose_action(
                agent_list=self.name_agents, obs=obs_list, eval=True
            )

            # 解析批处理返回的动作（假设与 name_agents 顺序对应）
            actions = {}
            for idx, agent in enumerate(self.name_agents):
                action = choose_ret["action"][idx]
                if hasattr(action, "reshape"):
                    action = action.reshape(-1)  # 确保动作是一维数组
                actions[agent] = action

            next_observations, rewards, terminations, truncations, info = self.env.step(
                actions
            )

            if recorder is not None:
                recorder.record_transition(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    termination=terminations,
                    truncation=truncations,
                    info=info,
                    env=self.env.unwrapped,
                    communication_diagnostics=choose_ret.get(
                        "communication_diagnostics"
                    ),
                )

            if tag_generate:
                state = self.env.unwrapped.get_all_satellite_state()
                for key, value in state.items():
                    if key not in trail:
                        trail[key] = []
                    trail[key].append(value)

            if render:
                self.env.render()

            next_obs_list = [
                next_observations[agent].reshape(1, -1) for agent in self.name_agents
            ]

            if scaler is not None:
                next_obs_list = scaler.transform(next_obs_list)

            obs_list = next_obs_list
            observations = next_observations
            for idx, agent in enumerate(self.name_agents):
                episode_rewards[agent].append(rewards[idx])

            if terminations or truncations:
                reached_step_limit = False
                break

        if tag_generate:
            info["trail"] = trail

        if recorder is not None:
            recorder.finish_episode(
                info, reached_step_limit=reached_step_limit
            )

        self.alg.after_episode(-1)

        return episode_rewards, info
