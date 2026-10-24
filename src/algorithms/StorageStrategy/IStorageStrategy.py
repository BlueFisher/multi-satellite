from abc import ABC, abstractmethod
from typing import List


class IStorageStrategy(ABC):
    """
    经验存储策略的基类，定义了如何存储和管理智能体的经验数据
    """

    @abstractmethod
    def store_experience(
        self,
        observations: List,
        actions: List,
        rewards: List[float],
        next_observations: List,
        dones: List[bool],
        **kwargs
    ) -> None:
        """
        存储经验数据

        参数:
            observations: 当前观测
            actions: 执行的动作
            rewards: 获得的奖励
            next_observations: 下一个观测
            dones: 是否为终止状态
        """
        pass

    def before_storage(self) -> None:
        """在存储经验前执行的操作"""
        pass

    def after_storage(self) -> None:
        """在存储经验后执行的操作"""
        pass

    @abstractmethod
    def can_update(self) -> bool:
        """判断是否可以进行模型更新"""
        pass
