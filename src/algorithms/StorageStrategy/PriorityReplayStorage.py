from argparse import Namespace
from typing import Tuple, List

from .IStorageStrategy import IStorageStrategy
from .ReplayBuffer import PriorityReplayBuffer


class PriorityReplayStorage(IStorageStrategy):
    """
    使用标准经验回放缓冲区的存储策略。

    该类实现了一个存储策略，使用 `PriorityReplayBuffer` 来存储和管理经验数据。

    Attributes:
        args (Namespace): 包含超参数和配置的命名空间对象。
        replay_buffer (PriorityReplayBuffer): 用于存储经验的优先级回放缓冲区。
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shapes: List[Tuple[int]],
        action_shapes: List[Tuple[int]],
        num_agents: int,
        args: Namespace,
    ) -> None:
        """
        初始化优先级经验存储策略。

        Args:
            buffer_size (int): 缓冲区的最大容量。
            obs_shapes (List[Tuple[int]]): 每个智能体的观测数据形状。
            action_shapes (List[Tuple[int]]): 每个智能体的动作数据形状。
            num_agents (int): 智能体的数量。
            args (Namespace): 包含超参数和配置的命名空间对象。

        Example:
            ```python
            from argparse import Namespace

            args = Namespace(batch_size=32, priority_alpha=0.6)
            storage = PriorityReplayStorage(
                buffer_size=10000,
                obs_shapes=[(4,), (4,)],
                action_shapes=[(2,), (2,)],
                num_agents=2,
                args=args
            )
            ```
        """
        self.args = args
        self.replay_buffer = PriorityReplayBuffer(
            capacity=buffer_size,
            alpha=self.args.priority_alpha,
            num_agents=num_agents,
            state_dim=obs_shapes[0][0],
            action_dim=action_shapes[0][0],
        )

    def store_experience(
        self,
        obs: List,
        actions: List,
        rewards: List[float],
        next_obs: List,
        dones: List[bool],
        **kwargs
    ) -> None:
        """
        存储经验数据到优先级回放缓冲区。

        Args:
            obs (List): 当前观测数据。
            actions (List): 当前动作数据。
            rewards (List[float]): 当前奖励数据。
            next_obs (List): 下一观测数据。
            dones (List[bool]): 终止标志。
            **kwargs: 额外需要存储的数据。

        Example:
            ```python
            obs = [np.random.rand(4), np.random.rand(4)]
            actions = [np.random.rand(2), np.random.rand(2)]
            rewards = [1.0, 0.5]
            next_obs = [np.random.rand(4), np.random.rand(4)]
            dones = [False, True]

            storage.store_experience(obs, actions, rewards, next_obs, dones)
            ```
        """
        self.replay_buffer.add(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

    def can_update(self) -> bool:
        """
        检查是否可以进行更新。

        Returns:
            bool: 如果缓冲区中的经验数量大于等于批量大小，则返回 True，否则返回 False。

        Example:
            ```python
            if storage.can_update():
                print("可以进行更新")
            else:
                print("数据不足，无法更新")
            ```
        """
        return len(self.replay_buffer) >= self.args.batch_size
