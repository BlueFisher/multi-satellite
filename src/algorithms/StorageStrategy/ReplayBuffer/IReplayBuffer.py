from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class IReplayBuffer(ABC):
    @property
    @abstractmethod
    def capacity(self) -> int:
        """最大存储容量"""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """当前存储量"""
        pass

    @abstractmethod
    def add(
        self,
        obs_list: List,
        actions_list: List,
        rewards: List[float],
        next_obs_list: List,
        dones: List[bool],
        **kwargs
    ) -> None:
        """添加经验数据

        Args:
            obs_list: 各智能体当前观测 [agent1_obs, agent2_obs,...]
            actions_list: 各智能体动作 [agent1_action, agent2_action,...]
            rewards: 奖励数组 (num_agents,) 或 (1, num_agents)
            next_obs_list: 各智能体下一观测 [agent1_next_obs,...]
            dones: 终止标记 (num_agents,) 或 (1, num_agents)
            kwargs: 算法特定数据（如MAPPO需要log_probs）
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[
        List[torch.Tensor],  # obs
        List[torch.Tensor],  # actions
        torch.Tensor,  # rewards
        List[torch.Tensor],  # next_obs
        torch.Tensor,  # dones
        dict,  # 额外数据
    ]:
        """采样批次数据

        Returns:
            tuple: (obs_tensors, action_tensors, rewards, next_obs_tensors, dones, extra_data)
            extra_data: 包含kwargs中存储的额外字段
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空缓冲区"""
        pass

    def __len__(self) -> int:
        return self.size
