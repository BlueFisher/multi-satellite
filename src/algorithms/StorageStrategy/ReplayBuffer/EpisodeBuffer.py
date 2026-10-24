from collections import deque
from typing import List, Tuple, Dict

import numpy as np
import torch

from .IReplayBuffer import IReplayBuffer


class EpisodeReplayBuffer(IReplayBuffer):
    def __init__(
        self,
        capacity: int,
        num_agents: int,
        num_envs: int,  # 并行环境数量（核心参数）
        obs_shapes: list,  # 每个智能体观测维度（不含num_envs），如[(4,), (4,)]
        action_shapes: list,  # 每个智能体动作维度（不含num_envs），如[(2,), (2,)]
        action_continuous=False,
        use_states: bool = False,
        state_shapes: tuple = None,  # 全局状态维度（不含num_envs）
        use_mask: bool = False,
        max_episode_length: int = 200,
        device: str = "cpu",
    ):
        """
        支持并行环境的回合制经验回放缓冲区（reward/done维度：num_envs × num_agents）

        :param capacity: 最大缓存的 Episode 数量（所有环境共享）
        :param num_agents: 智能体数量
        :param num_envs: 并行环境数量
        :param obs_shapes: 每个智能体的观测维度列表，例如[(4,), (4,)]
        :param action_shapes: 每个智能体的动作维度列表（离散填类别数，连续填动作维度）
        :param action_continuous: 是否为连续动作空间
        :param use_states: 是否使用全局状态（区别于单个智能体观测）
        :param state_shapes: 全局状态维度，如(10,)
        :param max_episode_length: 单个episode的最大步数（防止内存溢出）
        :param device: 数据最终转移的设备（CPU/GPU）
        """
        self._capacity = capacity
        self.num_agents = num_agents
        self.num_envs = num_envs  # 并行环境数量
        self.obs_shapes = obs_shapes
        self.state_shape = state_shapes
        self.action_shapes = action_shapes
        self.action_continuous = action_continuous
        self.max_episode_length = max_episode_length
        self.device = device
        self.use_mask = use_mask
        self.use_states = use_states

        # 存储所有环境产生的完整episode
        self.buffer = deque(maxlen=capacity)
        # 为每个环境维护独立的当前episode缓存
        self.current_episodes: List[Dict] = [
            self._init_single_episode() for _ in range(num_envs)
        ]
        # 每个环境的当前episode步数（解决并行环境进度不同步）
        self.episode_steps = np.zeros(num_envs, dtype=int)  # 形状: (num_envs,)
        self.episode_count = 0  # 已缓存的完整episode总数

    def _init_single_episode(self) -> Dict:
        """初始化单个环境的episode缓存结构（预分配内存）"""
        single_ep = {
            # 每个智能体的观测：(max_episode_length, *obs_shape)
            "obs": [
                np.zeros((self.max_episode_length, *shape), dtype=np.float64)
                for shape in self.obs_shapes
            ],
            # 每个智能体的动作：(max_episode_length, *action_shape)
            "actions": [
                np.zeros((self.max_episode_length, *shape), dtype=np.float64)
                for shape in self.action_shapes
            ],
            # 每个智能体的下一观测：(max_episode_length, *obs_shape)
            "next_obs": [
                np.zeros((self.max_episode_length, *shape), dtype=np.float64)
                for shape in self.obs_shapes
            ],
            # 奖励：(max_episode_length, num_agents) （单环境单步的所有智能体奖励）
            "rewards": np.zeros(
                (self.max_episode_length, self.num_agents), dtype=np.float64
            ),
            # 终止标志：(max_episode_length, num_agents) （单环境单步的所有智能体终止状态）
            "dones": np.zeros(
                (self.max_episode_length, self.num_agents), dtype=np.bool_
            ),
            # 有效数据标志：(max_episode_length, num_agents) （区分真实数据与预分配0）
            "filled": np.zeros(
                (self.max_episode_length, self.num_agents), dtype=np.bool_
            ),
        }

        # 若启用全局状态，补充状态字段
        if self.use_states:
            single_ep["states"] = np.zeros(
                (self.max_episode_length, *self.state_shape), dtype=np.float64
            )
            single_ep["next_states"] = np.zeros(
                (self.max_episode_length, *self.state_shape), dtype=np.float64
            )

        return single_ep

    def add(
        self,
        obs: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: np.ndarray,
        next_obs: List[np.ndarray],
        dones: np.ndarray,
        **kwargs,
    ) -> None:
        """
        向缓冲区添加并行环境的经验数据（核心方法）

        输入数据维度规范（以2个智能体、3个并行环境为例）：
        - obs: List[np.ndarray]，长度=num_agents，每个元素维度=(num_envs, *obs_shape) → 如[(3,4), (3,4)]
        - actions: List[np.ndarray]，长度=num_agents，每个元素维度=(num_envs, *action_shape) → 离散[(3,), (3,)] / 连续[(3,2), (3,2)]
        - rewards: np.ndarray，维度=(num_envs, num_agents) → 如(3,2)（3个环境，每个环境2个智能体的奖励）
        - next_obs: List[np.ndarray]，同obs → 如[(3,4), (3,4)]
        - dones: np.ndarray，维度=(num_envs, num_agents) → 如(3,2)（3个环境，每个环境2个智能体的终止标志）
        - **kwargs: 额外数据（如全局状态），维度需为(num_envs, *data_shape)
        """
        num_envs = self.num_envs
        num_agents = self.num_agents

        # -------------------------- 1. 输入维度合法性检查 --------------------------
        # 检查智能体数量匹配（obs/actions/next_obs）
        if (
            len(obs) != num_agents
            or len(actions) != num_agents
            or len(next_obs) != num_agents
        ):
            raise ValueError(
                f"obs/actions/next_obs长度需等于智能体数量{num_agents}，"
                f"实际分别为{len(obs)}/{len(actions)}/{len(next_obs)}"
            )

        # 检查并行环境数量匹配（obs/actions/next_obs的第0维）
        for agent_idx in range(num_agents):
            if obs[agent_idx].shape[0] != num_envs:
                raise ValueError(
                    f"Agent{agent_idx}的obs维度第0维需为环境数量{num_envs}，"
                    f"实际为{obs[agent_idx].shape[0]}"
                )
            if actions[agent_idx].shape[0] != num_envs:
                raise ValueError(
                    f"Agent{agent_idx}的actions维度第0维需为环境数量{num_envs}，"
                    f"实际为{actions[agent_idx].shape[0]}"
                )
            if next_obs[agent_idx].shape[0] != num_envs:
                raise ValueError(
                    f"Agent{agent_idx}的next_obs维度第0维需为环境数量{num_envs}，"
                    f"实际为{next_obs[agent_idx].shape[0]}"
                )

        # 检查rewards/dones维度（核心调整：必须为(num_envs, num_agents)）
        if rewards.shape != (num_envs, num_agents):
            raise ValueError(
                f"rewards维度需为(num_envs, num_agents)=({num_envs}, {num_agents})，"
                f"实际为{rewards.shape}"
            )
        if dones.shape != (num_envs, num_agents):
            raise ValueError(
                f"dones维度需为(num_envs, num_agents)=({num_envs}, {num_agents})，"
                f"实际为{dones.shape}"
            )

        # -------------------------- 2. 逐个环境处理数据 --------------------------
        for env_idx in range(num_envs):
            current_step = self.episode_steps[env_idx]
            current_ep = self.current_episodes[env_idx]

            # 检查当前环境是否已达episode最大长度（超过则强制终止）
            if current_step >= self.max_episode_length:
                self._end_single_episode(env_idx)  # 终止该环境的episode
                current_step = self.episode_steps[env_idx]  # 重置后为0
                current_ep = self.current_episodes[env_idx]

            # 2.1 存储智能体观测、动作、下一观测
            for agent_idx in range(num_agents):
                # 观测：(num_envs, *obs_shape) → 取当前环境的观测
                current_ep["obs"][agent_idx][current_step] = obs[agent_idx][env_idx]

                # 动作：离散转独热编码，连续直接存储
                raw_action = actions[agent_idx][env_idx]
                if not self.action_continuous:
                    # 离散动作：输入为标量索引（如0/1），转独热编码
                    current_ep["actions"][agent_idx][current_step] = np.eye(
                        self.action_shapes[agent_idx][0], dtype=np.float64
                    )[raw_action]
                else:
                    # 连续动作：直接存储向量
                    current_ep["actions"][agent_idx][current_step] = raw_action

                # 下一观测：同观测逻辑
                current_ep["next_obs"][agent_idx][current_step] = next_obs[agent_idx][
                    env_idx
                ]

            # 2.2 存储奖励与终止标志（核心调整：直接从二维数组索引）
            current_ep["rewards"][current_step] = rewards[
                env_idx
            ]  # (num_agents,) → 对应单步所有智能体奖励
            current_ep["dones"][current_step] = dones[
                env_idx
            ]  # (num_agents,) → 对应单步所有智能体终止标志
            current_ep["filled"][
                current_step
            ] = True  # 标记该步数据有效（所有智能体共享同一填充标志）

            # 2.3 存储全局状态（若启用）
            if self.use_states and "states" in kwargs and "next_states" in kwargs:
                states = kwargs["states"]  # 维度：(num_envs, *state_shape)
                next_states = kwargs["next_states"]
                if states.shape[0] != num_envs or next_states.shape[0] != num_envs:
                    raise ValueError(
                        f"states/next_states维度第0维需为环境数量{num_envs}"
                    )
                current_ep["states"][current_step] = states[env_idx]
                current_ep["next_states"][current_step] = next_states[env_idx]

            # 2.4 存储额外自定义数据（修复：正确处理各种形式的额外数据）
            for key, value in kwargs.items():
                if key in ["states", "next_states"]:  # 已单独处理的字段跳过
                    continue

                # 检查数据维度
                if isinstance(value, (list, tuple)):
                    # 列表形式：每个元素对应一个智能体，维度应为(num_envs, *agent_data_shape)
                    if len(value) != self.num_agents:
                        raise ValueError(
                            f"额外数据'{key}'为列表时，长度应等于智能体数量 {self.num_agents}，实际为 {len(value)}"
                        )

                    # 首次出现该字段时初始化缓存空间
                    if key not in current_ep:
                        # 为每个智能体创建存储空间
                        current_ep[key] = []
                        for agent_idx in range(self.num_agents):
                            data_shape = value[agent_idx].shape[
                                1:
                            ]  # 排除num_envs后的维度
                            current_ep[key].append(
                                np.zeros(
                                    (self.max_episode_length, *data_shape),
                                    dtype=value[agent_idx].dtype,
                                )
                            )

                    # 存储每个智能体的数据
                    for agent_idx in range(self.num_agents):
                        if value[agent_idx].shape[0] != num_envs:
                            raise ValueError(
                                f"额外数据'{key}'（智能体 {agent_idx}）环境数量不符: {value[agent_idx].shape[0]} != {num_envs}"
                            )
                        current_ep[key][agent_idx][current_step] = value[agent_idx][
                            env_idx
                        ]

                else:
                    # 单个数组形式：维度应为(num_envs, *data_shape)
                    if value.shape[0] != num_envs:
                        raise ValueError(
                            f"额外数据'{key}'环境数量不符: {value.shape[0]} != {num_envs}"
                        )

                    # 首次出现该字段时初始化缓存空间
                    if key not in current_ep:
                        data_shape = value.shape[1:]  # 排除num_envs后的维度
                        current_ep[key] = np.zeros(
                            (self.max_episode_length, *data_shape), dtype=value.dtype
                        )

                    # 存储当前环境的数据
                    current_ep[key][current_step] = value[env_idx]

            self.episode_steps[env_idx] += 1
            # 2.5 判断当前环境的episode是否终止（并行环境独立终止）
            env_done = np.all(dones[env_idx])
            if env_done or (current_step >= self.max_episode_length):
                self._end_single_episode(env_idx)

    def _end_single_episode(self, env_idx: int) -> None:
        """终止指定环境的当前episode，将其加入缓冲区并重置"""
        current_step = self.episode_steps[env_idx]
        if current_step == 0:  # 空episode跳过
            return

        # 提取该环境的有效episode数据（截断预分配的0填充部分）
        valid_episode = self._extract_valid_episode(env_idx, current_step)
        # 加入缓冲区（超容时自动删除最早的episode）
        self.buffer.append(valid_episode)
        self.episode_count = min(self.episode_count + 1, self._capacity)

        # 重置该环境的episode状态（准备下一轮）
        self._reset_single_episode(env_idx)
        self.episode_steps[env_idx] = 0

    def _extract_valid_episode(self, env_idx: int, current_step: int) -> Dict:
        """提取指定环境的有效episode数据（截断到实际步数）"""
        raw_ep = self.current_episodes[env_idx]
        valid_ep = {
            # 智能体相关数据：每个agent的数组截断到current_step步
            "obs": [agent_obs[:current_step].copy() for agent_obs in raw_ep["obs"]],
            "actions": [
                agent_act[:current_step].copy() for agent_act in raw_ep["actions"]
            ],
            "next_obs": [
                agent_next[:current_step].copy() for agent_next in raw_ep["next_obs"]
            ],
            # 全局数据：截断到current_step步（rewards/dones已为(num_agents,) per step）
            "rewards": raw_ep["rewards"][
                :current_step
            ].copy(),  # (current_step, num_agents)
            "dones": raw_ep["dones"][
                :current_step
            ].copy(),  # (current_step, num_agents)
            "filled": raw_ep["filled"][
                :current_step
            ].copy(),  # (current_step, num_agents)
        }

        # 补充全局状态（若启用）
        if self.use_states:
            valid_ep["states"] = raw_ep["states"][:current_step].copy()
            valid_ep["next_states"] = raw_ep["next_states"][:current_step].copy()

        # 补充额外自定义数据（修复：正确处理各种形式的额外数据）
        for key in raw_ep:
            if key not in valid_ep and key not in ["states", "next_states"]:
                if isinstance(raw_ep[key], list):
                    # 列表形式：每个智能体的数据分别截断
                    valid_ep[key] = [arr[:current_step].copy() for arr in raw_ep[key]]
                else:
                    # 单个数组形式：直接截断
                    valid_ep[key] = raw_ep[key][:current_step].copy()

        return valid_ep

    def _reset_single_episode(self, env_idx: int) -> None:
        """重置指定环境的episode缓存（填充0）"""
        current_ep = self.current_episodes[env_idx]
        for key in current_ep:
            if isinstance(current_ep[key], list):
                # 智能体相关字段（如obs/actions/额外列表数据）：列表内每个数组填充0
                for arr in current_ep[key]:
                    arr.fill(0)
            else:
                # 全局字段（如rewards/dones/额外数组数据）：直接填充0
                current_ep[key].fill(0)

    def sample(self, batch_size: int, device: str = None) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        List[torch.Tensor],
        torch.Tensor,
        dict,
    ]:
        """
        采样批次经验数据（返回格式不变，rewards/dones为(batch_size, max_length, num_agents)）

        返回值说明：
        - obs_tensors: List[Tensor]，长度=num_agents，每个维度=(batch_size, max_length, *obs_shape)
        - action_tensors: List[Tensor]，长度=num_agents，每个维度=(batch_size, max_length, *action_shape)
        - rewards_tensor: Tensor，维度=(batch_size, max_length, num_agents)
        - next_obs_tensors: List[Tensor]，同obs_tensors
        - dones_tensor: Tensor，维度=(batch_size, max_length, num_agents)
        - extra_data: Dict，包含填充标志、episode长度、全局状态等
        """
        device = device or self.device
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"缓冲区episode数量不足（当前{len(self.buffer)}），无法满足批次大小{batch_size}"
            )

        # 1. 随机采样batch_size个episode（无关来源环境）
        sample_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled_episodes = [self.buffer[idx] for idx in sample_indices]

        # 2. 确定批次内最长episode长度（用于统一Padding）
        max_ep_length = max(len(ep["rewards"]) for ep in sampled_episodes)

        # 3. 初始化Padding后的批次数据结构
        # 3.1 智能体相关数据（obs/actions/next_obs）
        batch_obs = self._init_padded_batch(self.obs_shapes, batch_size, max_ep_length)
        batch_actions = self._init_padded_batch(
            self.action_shapes, batch_size, max_ep_length
        )
        batch_next_obs = self._init_padded_batch(
            self.obs_shapes, batch_size, max_ep_length
        )

        # 3.2 全局数据（rewards/dones/filled）：维度已适配(num_agents)
        batch_rewards = np.zeros(
            (batch_size, max_ep_length, self.num_agents), dtype=np.float64
        )
        batch_dones = np.zeros(
            (batch_size, max_ep_length, self.num_agents), dtype=np.bool_
        )
        batch_filled = np.zeros(
            (batch_size, max_ep_length, self.num_agents), dtype=np.bool_
        )

        # 3.3 全局状态（若启用）
        if self.use_states:
            batch_states = np.zeros(
                (batch_size, max_ep_length, *self.state_shape), dtype=np.float64
            )
            batch_next_states = np.zeros(
                (batch_size, max_ep_length, *self.state_shape), dtype=np.float64
            )

        # 3.4 额外自定义数据（修复：正确处理各种形式的额外数据）
        batch_extras = {}
        if sampled_episodes:
            # 获取所有额外数据的键（排除基础字段）
            base_keys = {
                "obs",
                "actions",
                "next_obs",
                "rewards",
                "dones",
                "filled",
                "states",
                "next_states",
            }
            extra_keys = set(sampled_episodes[0].keys()) - base_keys

            for key in extra_keys:
                first_data = sampled_episodes[0][key]
                if isinstance(first_data, list):
                    # 列表形式：为每个智能体创建存储空间
                    batch_extras[key] = []
                    for agent_idx in range(len(first_data)):
                        data_shape = first_data[agent_idx].shape[
                            1:
                        ]  # 排除时间步后的维度
                        batch_extras[key].append(
                            np.zeros(
                                (batch_size, max_ep_length, *data_shape),
                                dtype=first_data[agent_idx].dtype,
                            )
                        )
                else:
                    # 单个数组形式
                    data_shape = first_data.shape[1:]  # 排除时间步后的维度
                    batch_extras[key] = np.zeros(
                        (batch_size, max_ep_length, *data_shape), dtype=first_data.dtype
                    )

        # 4. 填充数据（按episode逐个Padding到max_ep_length）
        for batch_idx, ep in enumerate(sampled_episodes):
            ep_length = len(ep["rewards"])  # 当前episode的实际长度

            # 4.1 填充智能体相关数据
            for agent_idx in range(self.num_agents):
                batch_obs[agent_idx][batch_idx, :ep_length] = ep["obs"][agent_idx]
                batch_actions[agent_idx][batch_idx, :ep_length] = ep["actions"][
                    agent_idx
                ]
                batch_next_obs[agent_idx][batch_idx, :ep_length] = ep["next_obs"][
                    agent_idx
                ]

            # 4.2 填充奖励与终止标志（直接对应维度）
            batch_rewards[batch_idx, :ep_length] = ep[
                "rewards"
            ]  # (ep_length, num_agents) → 直接赋值
            batch_dones[batch_idx, :ep_length] = ep["dones"]  # 同上
            batch_filled[batch_idx, :ep_length] = ep["filled"]  # 同上

            # 4.3 填充全局状态
            if self.use_states:
                batch_states[batch_idx, :ep_length] = ep["states"]
                batch_next_states[batch_idx, :ep_length] = ep["next_states"]

            # 4.4 填充额外数据（修复：正确处理各种形式的额外数据）
            for key in batch_extras:
                if isinstance(batch_extras[key], list):
                    # 列表形式：分别填充每个智能体的数据
                    for agent_idx in range(len(batch_extras[key])):
                        batch_extras[key][agent_idx][batch_idx, :ep_length] = ep[key][
                            agent_idx
                        ]
                else:
                    # 单个数组形式
                    batch_extras[key][batch_idx, :ep_length] = ep[key]

        # 5. 转换为PyTorch Tensor并转移到目标设备
        obs_tensors = [torch.from_numpy(arr).float().to(device) for arr in batch_obs]
        action_tensors = [
            torch.from_numpy(arr).float().to(device) for arr in batch_actions
        ]
        next_obs_tensors = [
            torch.from_numpy(arr).float().to(device) for arr in batch_next_obs
        ]
        rewards_tensor = torch.from_numpy(batch_rewards).float().to(device)
        dones_tensor = (
            torch.from_numpy(batch_dones).float().to(device)
        )  # bool转float便于计算
        filled_tensor = (
            torch.from_numpy(batch_filled).float().to(device)
        )  # 有效数据掩码

        # 6. 构建额外数据字典（修复：正确处理各种形式的额外数据）
        extra_data = {
            "filled": filled_tensor,  # 标记有效数据位置（非Padding）
            "episode_lengths": torch.tensor(  # 每个episode的实际长度
                [len(ep["rewards"]) for ep in sampled_episodes], device=device
            ),
        }

        # 处理额外数据
        for key, value in batch_extras.items():
            if isinstance(value, list):
                # 列表形式：转换为Tensor列表
                extra_data[key] = [torch.from_numpy(arr).to(device) for arr in value]
            else:
                # 单个数组形式：转换为Tensor
                extra_data[key] = torch.from_numpy(value).to(device)

        # 补充全局状态（若启用）
        if self.use_states:
            extra_data["states"] = torch.from_numpy(batch_states).float().to(device)
            extra_data["next_states"] = (
                torch.from_numpy(batch_next_states).float().to(device)
            )

        return (
            obs_tensors,
            action_tensors,
            rewards_tensor,
            next_obs_tensors,
            dones_tensor,
            extra_data,
        )

    def _init_padded_batch(
        self, shapes: list, batch_size: int, max_length: int
    ) -> List[np.ndarray]:
        """初始化Padding后的批次数据结构（用于智能体相关字段）"""
        return [
            np.zeros((batch_size, max_length, *shape), dtype=np.float64)
            for shape in shapes
        ]

    def clear(self) -> None:
        """清空缓冲区及所有环境的当前episode状态"""
        self.buffer.clear()
        self.episode_count = 0
        self.episode_steps = np.zeros(self.num_envs, dtype=int)
        for env_idx in range(self.num_envs):
            self._reset_single_episode(env_idx)

    # -------------------------- 基础属性访问器 --------------------------
    @property
    def size(self) -> int:
        """当前缓存的完整episode数量"""
        return len(self.buffer)

    @property
    def capacity(self) -> int:
        """缓冲区最大容量（episode数量）"""
        return self._capacity

    def __len__(self) -> int:
        """支持len(buffer)语法，返回当前缓存的episode数量"""
        return self.size
