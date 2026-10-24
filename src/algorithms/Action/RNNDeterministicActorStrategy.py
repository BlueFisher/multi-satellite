from argparse import Namespace
from typing import List, Union, Dict

import numpy as np
import torch

from .IActorStrategy import IActorStrategy
from ..utils import Noise


class RNNDeterministicActorStrategy(IActorStrategy):
    """
    基于共享确定性策略网络的多智能体动作选择策略（批处理版本）。
    使用单个Actor网络，输入所有智能体的观测，输出所有智能体的动作。
    支持连续和离散动作空间，所有操作均向量化。
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        args: Namespace,
        action_continuous: bool = False,
        use_agent_id: bool = False,
    ) -> None:
        self.args = args
        self.actor = actor
        self.action_continuous = action_continuous
        self.use_agent_id = use_agent_id
        self.noise = Noise()

        if not self.action_continuous:
            raise NotImplementedError("The discrete strategy is not implemented.")

        self.num_layers = 1

        n_agents = len(args.name_agents)
        batch_size = args.num_envs

        # 训练时的隐藏状态: (num_layers, batch_size, n_agents, hidden_size)
        self.hidden_state = torch.zeros(
            self.num_layers,
            batch_size,
            n_agents,
            args.hidden_size,
            device=args.device,
        )

        # 评估时的隐藏状态: (num_layers, 1, n_agents, hidden_size)
        self.eval_hidden_state = torch.zeros(
            self.num_layers,
            1,
            n_agents,
            args.hidden_size,
            device=args.device,
        )

    def select_action(
        self,
        agent_list,
        observation: np.ndarray = None,
        global_state: np.ndarray = None,
        evaluation: bool = False,
    ) -> Dict[str, Union[np.ndarray, list]]:

        if observation is None:
            raise ValueError("observation cannot be None")
        if isinstance(observation, list):
            observation = np.array(observation)

        # observation 形状: (batch_size, n_agents, obs_dim)
        obs_tensor = (
            torch.FloatTensor(observation).permute(1, 0, 2).to(self.args.device)
        )
        batch_size, n_agents, obs_dim = obs_tensor.shape

        # 重塑为 (batch_size * n_agents, seq_len=1, obs_dim)
        obs_tensor = obs_tensor.reshape(batch_size * n_agents, 1, obs_dim)

        if self.use_agent_id:
            # 生成 one-hot: (n_agents, n_agents)
            agent_ids = torch.arange(n_agents, device=self.args.device)
            one_hot = torch.nn.functional.one_hot(
                agent_ids, num_classes=n_agents
            ).float()
            # 扩展为 (batch_size, n_agents, n_agents)
            one_hot = one_hot.unsqueeze(0).expand(batch_size, -1, -1)
            # 重塑为 (batch_size * n_agents, n_agents) -> (batch_size * n_agents, 1, n_agents)
            one_hot = one_hot.reshape(batch_size * n_agents, n_agents).unsqueeze(1)
            # 拼接观测和 one-hot
            obs_tensor = torch.cat([obs_tensor, one_hot], dim=-1)

        # 选择当前使用的隐藏状态
        if evaluation:
            hidden = self.eval_hidden_state  # (num_layers, 1, n_agents, hidden_size)
        else:
            hidden = (
                self.hidden_state
            )  # (num_layers, batch_size, n_agents, hidden_size)

        # 将隐藏状态重塑为 actor 需要的形状: (num_layers, batch_size * n_agents, hidden_size)
        hidden = hidden.view(
            self.num_layers, batch_size * n_agents, self.args.hidden_size
        )

        with torch.no_grad():
            actions_tensor, next_hidden = self.actor(obs_tensor, hidden)
            # actions_tensor: (batch_size * n_agents, 1, act_dim)
            # next_hidden: (num_layers, batch_size * n_agents, hidden_size)

        # 将新的隐藏状态重塑回 (num_layers, batch_size, n_agents, hidden_size)
        next_hidden = next_hidden.view(
            self.num_layers, batch_size, n_agents, self.args.hidden_size
        )

        if evaluation:
            self.eval_hidden_state = next_hidden
        else:
            self.hidden_state = next_hidden

        if not evaluation:
            # 添加高斯探索噪声
            actions_tensor = self.noise(
                actions_tensor, self.args.action_min, self.args.action_max
            )

        # 转换为动作列表: 形状 (n_agents, batch_size, act_dim)
        actions = (
            actions_tensor.cpu()
            .squeeze(1)
            .view(batch_size, n_agents, -1)
            .permute(1, 0, 2)
        ).numpy()
        action_list = [actions[i] for i in range(actions.shape[0])]
        return {"action": action_list}

    def reset_hidden_state(self, env_idx: int = -1) -> None:
        """
        重置隐藏状态。
        - env_idx == -1: 重置所有环境的隐藏状态（仅评估模式常用）
        - env_idx >= 0: 重置指定环境索引的隐藏状态
        """
        if env_idx == -1:
            # 重置所有评估环境的隐藏状态
            self.eval_hidden_state = torch.zeros_like(self.eval_hidden_state)
        else:
            # 重置指定环境索引的训练隐藏状态（注意 batch 维度）
            if env_idx < self.hidden_state.size(1):
                self.hidden_state[:, env_idx, :, :] = 0.0
            else:
                raise IndexError(
                    f"env_idx {env_idx} out of range (batch_size={self.hidden_state.size(1)})"
                )
