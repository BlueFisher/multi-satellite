from argparse import Namespace
from typing import Tuple, List

from .IStorageStrategy import IStorageStrategy
from .ReplayBuffer.EpisodeBuffer import EpisodeReplayBuffer


class EpisodeReplayStorage(IStorageStrategy):
    """
    使用回合经验回放缓冲区的存储策略。

    该类实现了一个存储策略，使用 `EpisodeReplayBuffer` 来存储和管理经验数据。

    Attributes:
        args (Namespace): 包含超参数和配置的命名空间对象。
        replay_buffer (EpisodeReplayBuffer): 用于存储经验的回放缓冲区。
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_shapes: List[tuple[int]],
        action_shapes: List[tuple[int]],
        args: Namespace,
        action_continuous: bool = False,
        use_states: bool = False,
        state_shapes: Tuple[int] = (0,),
        use_mask: bool = False,
        max_episode_length: int = 200,
        device: str = "cpu",
    ) -> None:
        """
        初始化回合经验存储策略。

        Args:
            buffer_size (int): 缓冲区的最大容量。
            num_agents (int): 智能体的数量。
            obs_shapes (List[tuple[int]]): 每个智能体的观测数据形状。
            action_shapes (List[tuple[int]]): 每个智能体的动作数据形状。
            args (Namespace): 包含超参数和配置的命名空间对象。
            action_continuous (bool, optional): 是否为连续动作。默认为 False。
            use_states (bool, optional): 是否使用全局状态。默认为 False。
            state_shapes (Tuple[int], optional): 全局状态的形状。默认为 (0,).
            use_mask (bool, optional): 是否使用掩码。默认为 False。
            max_episode_length (int, optional): 每回合的最大长度。默认为 200。
            device (str, optional): 存储设备（如 "cpu" 或 "cuda"）。默认为 "cpu"。

        Example:
            ```python
            from argparse import Namespace

            args = Namespace(batch_size=32)
            storage = EpisodeReplayStorage(
                buffer_size=10000,
                num_agents=2,
                obs_shapes=[(4,), (4,)],
                action_shapes=[(2,), (2,)],
                args=args,
                action_continuous=True,
                max_episode_length=100,
                device="cuda"
            )
            ```
        """
        self.args = args
        self.replay_buffer = EpisodeReplayBuffer(
            capacity=buffer_size,
            num_agents=num_agents,
            obs_shapes=obs_shapes,
            action_shapes=action_shapes,
            action_continuous=action_continuous,
            use_states=use_states,
            state_shapes=state_shapes,
            use_mask=use_mask,
            max_episode_length=max_episode_length,
            num_envs=args.num_envs,
            device=device,
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
        存储经验数据到回放缓冲区。

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
            **kwargs
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
        return self.replay_buffer.size >= self.args.batch_size
