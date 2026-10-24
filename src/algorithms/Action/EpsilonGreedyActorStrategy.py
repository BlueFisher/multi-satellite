from argparse import Namespace
from typing import List, Dict, Optional

import numpy as np
import torch

from IActorStrategy import IActorStrategy


class EpsilonGreedyActorStrategy(IActorStrategy):
    """
    ε-贪心动作选择策略，支持多智能体和离散动作。

    该类统一格式返回所有智能体的动作。

    属性:
        config (Namespace): 包含配置参数的命名空间对象。
        actors (List[torch.nn.Module]): 智能体的策略网络列表。
        epsilon (float): ε-贪心策略中的探索概率。
        epsilon_min (float): ε的最小值。
        epsilon_decay (float): ε的衰减因子。
    """

    def __init__(
        self,
        actors: List[torch.nn.Module],
        epsilon: float,
        config: Namespace,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        """
        初始化 EpsilonGreedyActorStrategy 实例。

        参数:
            actors (List[torch.nn.Module]): 智能体的策略网络列表。
            epsilon (float): ε-贪心策略中的探索概率。
            config (Namespace): 包含配置参数的命名空间对象。
            epsilon_min (float): ε的最小值，默认为 0.01。
            epsilon_decay (float): ε的衰减因子，默认为 0.995。
        """
        self.config = config
        self.actors = actors
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(
        self,
        agent_list,
        observation: np.ndarray,
        global_state: Optional[np.ndarray] = None,
        evaluation: bool = False,
    ) -> Dict[str, list]:
        """
        为所有智能体选择动作。

        参数:
            agent_list: 智能体的 ID 列表。
            observation (np.ndarray): 智能体的观测值数组，形状为 (num_agents, observation_dim)。
            global_state (Optional[np.ndarray]): 全局状态数组（可选）。
            evaluation (bool): 是否为评估模式，默认为 False。

        返回:
            Dict[str, list]: 包含所有智能体动作的字典。

        示例:
            >>> strategy = EpsilonGreedyActorStrategy(actors, epsilon=0.1, config=config)
            >>> actions = strategy.select_action(agent_list, observation, evaluation=True)
            >>> print(actions)
        """
        all_actions = []

        # 转换观测为张量
        obs_tensor = torch.FloatTensor(observation).to(self.config.device)

        # 为每个智能体计算动作
        for agent_idx, agent_id in enumerate(agent_list):
            # 获取当前智能体的观测
            agent_obs = obs_tensor[agent_idx].unsqueeze(0)  # 增加批次维度
            # 计算Q值
            with torch.no_grad():
                q_values = self.actors[agent_idx](agent_obs).squeeze(0)

            # ε-贪心选择动作
            if evaluation or np.random.random() > self.epsilon:
                action = torch.argmax(q_values).item()
            else:
                action = np.random.choice(np.arange(len(q_values)))

            all_actions.append(action)

        return {"action": all_actions}

    def after_action_selection(self) -> None:
        """
        在动作选择后更新 ε 值。

        如果当前 ε 值大于最小值，则根据衰减因子进行更新。

        示例:
            >>> strategy = EpsilonGreedyActorStrategy(actors, epsilon=0.1, config=config)
            >>> strategy.after_action_selection()
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
