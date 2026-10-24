import numpy as np
import torch

from .IReplayBuffer import IReplayBuffer


class StandardReplayBuffer(IReplayBuffer):
    """
    标准经验回放缓冲区。

    该类实现了一个标准的经验回放缓冲区，支持多智能体的经验存储和采样，
    并支持额外数据的存储。

    Attributes:
        capacity (int): 缓冲区的最大容量。
        obs_shapes (List[Tuple]): 每个智能体的观测数据形状。
        action_shapes (List[Tuple]): 每个智能体的动作数据形状。
        num_agents (int): 智能体的数量。
        action_continuous (bool): 是否为连续动作。
        obs_buffers (List[np.ndarray]): 存储观测数据的缓冲区。
        action_buffers (List[np.ndarray]): 存储动作数据的缓冲区。
        next_obs_buffers (List[np.ndarray]): 存储下一观测数据的缓冲区。
        rewards_buffer (np.ndarray): 存储奖励数据的缓冲区。
        dones_buffer (np.ndarray): 存储终止标志的缓冲区。
        extra_buffers (Dict[str, np.ndarray]): 存储额外数据的缓冲区。
        position (int): 当前存储位置的索引。
        size (int): 当前缓冲区中的经验数量。
    """

    def __init__(
        self,
        capacity,
        obs_shapes,
        action_shapes,
        num_agents,
        action_continuous=True,
    ):
        self._capacity = capacity
        self.obs_shapes = obs_shapes
        self.action_shapes = action_shapes
        self.num_agents = num_agents
        self.action_continuous = action_continuous

        # 初始化存储
        self.obs_buffers = self._init_buffers(obs_shapes)
        self.action_buffers = self._init_buffers(action_shapes)
        self.next_obs_buffers = self._init_buffers(obs_shapes)
        self.rewards_buffer = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones_buffer = np.zeros((capacity, num_agents), dtype=np.float32)
        self.position = 0
        self._size = 0

        self.extra_buffers = {}

    def _init_buffers(self, shapes):
        return [
            np.zeros((self._capacity, *shape), dtype=np.float32) for shape in shapes
        ]

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        **kwargs,
    ):
        """
        支持并行环境的数据添加，输入数据包含num_envs维度时会自动展平处理

        Args:
            obs: 各智能体观察列表 [array1, array2,...]，每个array形状为 (num_envs, *obs_shape)
            actions: 各智能体动作列表 [array1, array2,...]，每个array形状为 (num_envs, *action_shape)
            rewards: 所有智能体奖励数组 (num_envs, num_agents)
            next_obs: 各智能体下一观察列表 [array1, array2,...]，每个array形状为 (num_envs, *obs_shape)
            dones: 是否终止状态 (num_envs, num_agents)
            **kwargs: 额外需要保存的数据，每个数据应包含num_envs维度
        """

        # 确定并行环境数量 (num_envs)，只检查第一个元素
        num_envs = obs[0].shape[0]

        # 快速验证并行数据，只检查必要元素
        if self._validate_parallel_data_fast(
            obs, actions, next_obs, rewards, dones, kwargs, num_envs
        ):
            # 计算需要添加的数据总量
            total_to_add = num_envs
            new_position = (self.position + total_to_add) % self._capacity

            # 处理循环存储的两种情况
            if self.position + total_to_add <= self._capacity:
                # 无需环绕，直接添加块
                self._add_batch(
                    obs,
                    actions,
                    rewards,
                    next_obs,
                    dones,
                    kwargs,
                    self.position,
                    self.position + total_to_add,
                )
            else:
                # 需要环绕，分两部分添加
                part1_size = self._capacity - self.position
                self._add_batch(
                    obs,
                    actions,
                    rewards,
                    next_obs,
                    dones,
                    kwargs,
                    self.position,
                    self._capacity,
                    end_idx=part1_size,
                )
                self._add_batch(
                    obs,
                    actions,
                    rewards,
                    next_obs,
                    dones,
                    kwargs,
                    0,
                    new_position,
                    start_idx=part1_size,
                )

            self.position = new_position
            self._size = min(self._size + total_to_add, self._capacity)

    def _validate_parallel_data_fast(
        self, obs, actions, next_obs, rewards, dones, kwargs, num_envs
    ):
        """快速验证并行环境数据的维度一致性"""
        # 检查obs的第一个元素即可（假设所有元素结构一致）
        if len(obs) > 0 and obs[0].shape[0] != num_envs:
            raise ValueError(f"obs[0]环境数量不符: {obs[0].shape[0]} != {num_envs}")

        # 检查actions的第一个元素
        if len(actions) > 0 and actions[0].shape[0] != num_envs:
            raise ValueError(
                f"actions[0]环境数量不符: {actions[0].shape[0]} != {num_envs}"
            )

        # 检查next_obs的第一个元素
        if len(next_obs) > 0 and next_obs[0].shape[0] != num_envs:
            raise ValueError(
                f"next_obs[0]环境数量不符: {next_obs[0].shape[0]} != {num_envs}"
            )

        # 检查rewards和dones
        if rewards.shape[0] != num_envs:
            raise ValueError(f"rewards环境数量不符: {rewards.shape[0]} != {num_envs}")

        if dones.shape[0] != num_envs:
            raise ValueError(f"dones环境数量不符: {dones.shape[0]} != {num_envs}")

        # 检查额外数据的第一个元素
        if kwargs:
            first_key = next(iter(kwargs.keys()))
            if kwargs[first_key].shape[0] != num_envs:
                raise ValueError(
                    f"额外数据'{first_key}'环境数量不符: {kwargs[first_key].shape[0]} != {num_envs}"
                )

        return True

    def _add_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        kwargs,
        buffer_start,
        buffer_end,
        start_idx=0,
        end_idx=None,
    ):
        """批量添加数据到缓冲区，提高效率"""
        end_idx = end_idx or len(rewards)
        batch_size = end_idx - start_idx

        # 批量处理观测数据
        for i in range(self.num_agents):
            self.obs_buffers[i][buffer_start:buffer_end] = obs[i][start_idx:end_idx]

            if not self.action_continuous:
                # 批量处理离散动作的独热编码
                actions_np = actions[i][start_idx:end_idx]
                one_hot = np.zeros(
                    (batch_size, self.action_shapes[i][0]), dtype=np.float32
                )
                one_hot[np.arange(batch_size), actions_np.astype(int)] = 1.0
                self.action_buffers[i][buffer_start:buffer_end] = one_hot
            else:
                self.action_buffers[i][buffer_start:buffer_end] = actions[i][
                    start_idx:end_idx
                ]

            self.next_obs_buffers[i][buffer_start:buffer_end] = next_obs[i][
                start_idx:end_idx
            ]

        # 批量处理奖励和终止状态
        self.rewards_buffer[buffer_start:buffer_end] = rewards[start_idx:end_idx]
        self.dones_buffer[buffer_start:buffer_end] = dones[start_idx:end_idx]

        # 批量处理额外数据
        for key, data in kwargs.items():
            if key not in self.extra_buffers:
                data_shape = data.shape[1:]  # 排除num_envs维度
                self.extra_buffers[key] = np.zeros(
                    (self.capacity, *data_shape), dtype=data.dtype
                )
            self.extra_buffers[key][buffer_start:buffer_end] = data[start_idx:end_idx]

    def sample(self, batch_size, device="cpu"):
        """返回包含额外数据的字典"""
        if batch_size > self.size:
            raise ValueError(
                f"Batch size {batch_size} is larger than the current buffer size {self.size}."
            )

        indices = np.random.choice(self.size, batch_size)

        # 采样各智能体数据
        obs_samples = self._sample_buffers(self.obs_buffers, indices, device)
        action_samples = self._sample_buffers(self.action_buffers, indices, device)
        next_obs_samples = self._sample_buffers(self.next_obs_buffers, indices, device)

        rewards = torch.as_tensor(self.rewards_buffer[indices], dtype=torch.float32).to(
            device
        )
        dones = torch.as_tensor(self.dones_buffer[indices], dtype=torch.float32).to(
            device
        )

        # 处理额外数据
        extra_samples = {}
        for key, buf in self.extra_buffers.items():
            data = torch.as_tensor(buf[indices]).to(device)
            extra_samples[key] = data

        return (
            obs_samples,
            action_samples,
            rewards,
            next_obs_samples,
            dones,
            extra_samples,
        )

    def _sample_buffers(self, buffers, indices, device):
        return [
            torch.as_tensor(buf[indices], dtype=torch.float32).to(device)
            for buf in buffers
        ]

    def clear(self):
        self.position = 0
        self._size = 0
        # 清空额外数据缓冲区
        self.extra_buffers.clear()
