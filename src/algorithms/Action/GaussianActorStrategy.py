from argparse import Namespace
from typing import List, Dict, Union

import numpy as np
import torch
from torch.distributions import Normal

from .IActorStrategy import IActorStrategy


class GaussianActorStrategy(IActorStrategy):
    """
    基于高斯策略网络的动作选择策略，用于连续动作。

    该类支持多智能体一次性计算动作。

    属性:
        args (Namespace): 包含配置参数的命名空间对象。
        actors (List[torch.nn.Module]): 智能体的策略网络列表。
    """

    def __init__(self, actors: List[torch.nn.Module], args: Namespace) -> None:
        """
        初始化 GaussianActorStrategy 实例。

        参数:
            actors (List[torch.nn.Module]): 智能体的策略网络列表。
            args (Namespace): 包含配置参数的命名空间对象。
        """
        self.args = args
        self.actors = actors

    def select_action(
        self,
        agent_list,  # 智能体列表
        observation: np.ndarray = None,  # 形状: (num_agents, observation_dim)
        global_state: np.ndarray = None,
        evaluation: bool = False,
    ) -> Dict[str, Union[list, np.ndarray]]:
        """
        为所有智能体选择动作。

        参数:
            agent_list: 智能体的 ID 列表。
            observation (np.ndarray): 智能体的观测值数组，形状为 (num_agents, observation_dim)。
            global_state (np.ndarray): 全局状态数组（可选）。
            evaluation (bool): 是否为评估模式，默认为 False。

        返回:
            Dict[str, Union[list, np.ndarray]]: 包含所有智能体动作的字典。

        示例:
            >>> strategy = GaussianActorStrategy(actors, args)
            >>> actions = strategy.select_action(agent_list, observation, evaluation=True)
            >>> print(actions)
        """
        all_actions = []

        # 转换观测为张量
        obs_tensor = torch.FloatTensor(observation).to(self.args.device)

        # 为每个智能体计算动作
        for agent_idx, agent_id in enumerate(agent_list):
            # 获取当前智能体的索引和观测
            actor_index = self.args.name_agents.index(agent_id)
            agent_obs = obs_tensor[agent_idx]

            # 计算高斯分布参数
            with torch.no_grad():
                mean, log_std = self.actors[actor_index](agent_obs)
                std = torch.exp(log_std)
                action_dist = Normal(mean, std)
                action = action_dist.sample()
                exec_action = torch.tanh(action).cpu().numpy()

            # 收集结果
            all_actions.append(exec_action)

        return {"action": all_actions}
