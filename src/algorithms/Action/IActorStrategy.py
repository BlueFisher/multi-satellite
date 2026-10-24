from abc import ABC, abstractmethod
from typing import Dict, Union, List

import numpy as np


class IActorStrategy(ABC):
    """
    动作选择策略的基类，定义了智能体选择动作的方式
    """

    @abstractmethod
    def select_action(
        self,
        agent_list: List[str],
        observation: np.ndarray,
        global_state: np.ndarray = None,
        evaluation: bool = False,
    ) -> Dict[str, Union[np.ndarray, int, float]]:
        """
        根据当前观测选择动作

        参数:
            agent_id: 智能体标识符
            observation: 当前观测
            evaluation: 是否为评估模式

        返回:
            包含动作和其他必要信息的字典，如 {"action": ..., "log_prob": ..., "value": ...}
        """
        pass

    def before_action_selection(self) -> None:
        """在动作选择前执行的操作"""
        pass

    def after_action_selection(self) -> None:
        """在动作选择后执行的操作"""
        pass
