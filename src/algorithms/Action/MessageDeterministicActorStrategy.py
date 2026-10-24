from argparse import Namespace
from typing import List, Union, Dict

import numpy as np
import torch

from .IActorStrategy import IActorStrategy
from ..utils import Noise
from ..utils import OUNoise


class MessageDeterministicActorStrategy(IActorStrategy):
    """
    基于确定性策略网络的动作选择策略，支持离散和连续动作。

    该类统一格式返回所有智能体的动作。

    属性:
        args (Namespace): 包含配置参数的命名空间对象。
        actors (List[torch.nn.Module]): 智能体的策略网络列表。
        action_continuous (bool): 是否为连续动作。
        noise (Noise): 用于连续动作的噪声生成器。
    """

    def __init__(
        self,
        actors: List[torch.nn.Module],
        args: Namespace,
        action_continuous: bool = False,
    ) -> None:
        """
        初始化 DeterministicActorStrategy 实例。

        参数:
            actors (List[torch.nn.Module]): 智能体的策略网络列表。
            args (Namespace): 包含配置参数的命名空间对象。
            action_continuous (bool): 是否为连续动作，默认为 False。
        """
        self.args = args
        self.actors = actors
        self.action_continuous = action_continuous
        self.noise = Noise()

    def select_action(
        self,
        agent_list,
        observation: np.ndarray = None,  # 形状: (num_agents, observation_dim)
        global_state: np.ndarray = None,
        evaluation: bool = False,
    ) -> Dict[str, Union[np.ndarray, list]]:
        """
        为所有智能体选择动作。

        参数:
            agent_list: 智能体的 ID 列表。
            observation (np.ndarray): 智能体的观测值数组，形状为 (num_agents, observation_dim)。
            global_state (np.ndarray): 全局状态数组（可选）。
            evaluation (bool): 是否为评估模式，默认为 False。

        返回:
            Dict[str, Union[np.ndarray, list]]: 包含动作和独热编码（仅离散动作）的字典。

        示例:
            >>> strategy = DeterministicActorStrategy(actors, args, action_continuous=True)
            >>> actions = strategy.select_action(agent_list, observation, evaluation=True)
            >>> print(actions)
        """
        all_actions = []
        all_one_hot = [] if not self.action_continuous else None

        if isinstance(observation, list):
            observation = np.array(observation)

        obs_tensor = torch.FloatTensor(observation).to(self.args.device)

        for i, agent_id in enumerate(agent_list):
            agent_index = self.args.name_agents.index(agent_id)
            agent_obs = obs_tensor[i]  # 保持批次维度
            # 提取其他智能体的观测作为消息
            other_obs = torch.cat([obs_tensor[:i], obs_tensor[i + 1 :]], dim=0)
            with torch.no_grad():
                output = self.actors[agent_index](agent_obs, messages=other_obs)

            if self.action_continuous:
                # 处理连续动作
                if not evaluation:
                    output = self.noise(
                        output, self.args.action_min, self.args.action_max
                    )
                action = output.cpu().numpy()
                all_actions.append(action)
            else:
                # 处理离散动作
                action_probs = torch.exp(output)
                if evaluation:
                    action = torch.argmax(action_probs).item()
                else:
                    action = (
                        torch.distributions.Categorical(action_probs).sample().item()
                    )
                all_actions.append(action)
                all_one_hot.append(np.eye(len(action_probs))[action])

        # 构建返回字典
        result = {"action": all_actions}
        # 仅离散动作添加one_hot键
        if not self.action_continuous:
            result["one_hot"] = all_one_hot

        return result
