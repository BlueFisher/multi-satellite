from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict

from ..StorageStrategy.ReplayBuffer.IReplayBuffer import IReplayBuffer


class ITrainer(ABC):
    """
    训练策略的基类，定义了如何从经验中学习和更新模型
    """

    def __init__(self, replay_buffer: IReplayBuffer, args: Namespace) -> None:
        self.replay_buffer = replay_buffer
        self.args = args

    def before_training(self) -> None:
        """在训练前执行的操作"""
        pass

    def after_training(self) -> None:
        """在训练后执行的操作"""
        pass

    @abstractmethod
    def save_models(self, directory: str) -> None:
        """保存模型到指定目录"""
        pass

    @abstractmethod
    def load_models(self, directory: str) -> None:
        """从指定目录加载模型"""
        pass

    @abstractmethod
    def train(self) -> Dict[str, float]:
        """
        执行训练步骤

        返回:
            包含训练指标的字典，如 {"loss": ..., "entropy": ...}
        """
        pass
