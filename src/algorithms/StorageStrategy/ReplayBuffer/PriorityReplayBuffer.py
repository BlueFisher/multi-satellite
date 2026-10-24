import random
from typing import Dict

import numpy as np
import torch


class PriorityReplayBuffer:
    """
    高效多智能体优先级经验回放缓冲区。

    该类实现了一个优先级经验回放缓冲区，支持多智能体的经验存储和采样，
    并使用优先级树来高效地进行基于优先级的采样。

    Args:
        capacity (int): 缓冲区的最大容量。
        alpha (float): 优先级采样的权重因子，控制采样偏向程度。
        num_agents (int): 智能体的数量。
        states (torch.Tensor): 存储状态的张量，形状为 (capacity, num_agents, state_dim)。
        actions (torch.Tensor): 存储动作的张量，形状为 (capacity, num_agents, action_dim)。
        rewards (torch.Tensor): 存储奖励的张量，形状为 (capacity, num_agents)。
        next_states (torch.Tensor): 存储下一状态的张量，形状为 (capacity, num_agents, state_dim)。
        dones (torch.Tensor): 存储终止标志的张量，形状为 (capacity, num_agents)。
        priority_tree (torch.Tensor): 优先级树，用于高效采样。
        max_priority (float): 当前缓冲区中的最大优先级。
        next_idx (int): 下一个存储位置的索引。
        size (int): 当前缓冲区中的经验数量。
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        num_agents: int,
        state_dim: int,
        action_dim: int,
    ):
        """
        初始化优先级经验回放缓冲区。

        Args:
            capacity (int): 缓冲区的最大容量。
            alpha (float): 优先级采样的权重因子。
            num_agents (int): 智能体的数量。
            state_dim (int): 状态的维度。
            action_dim (int): 动作的维度。

        Example:
            ```python
            buffer = PriorityReplayBuffer(
                capacity=10000, alpha=0.6, num_agents=2, state_dim=4, action_dim=2
            )
            ```
        """
        self.capacity = capacity
        self.alpha = alpha
        self.num_agents = num_agents

        # 使用PyTorch张量存储数据，提高效率
        self.states = torch.zeros(
            (capacity, num_agents, state_dim), dtype=torch.float32
        )
        self.actions = torch.zeros(
            (capacity, num_agents, action_dim), dtype=torch.float32
        )
        self.rewards = torch.zeros((capacity, num_agents), dtype=torch.float32)
        self.next_states = torch.zeros(
            (capacity, num_agents, state_dim), dtype=torch.float32
        )
        self.dones = torch.zeros((capacity, num_agents), dtype=torch.bool)

        # 优先级树使用PyTorch张量实现
        self.priority_tree = torch.zeros(2 * capacity, dtype=torch.float32)
        self.max_priority = 1.0
        self.tree_capacity = capacity

        # 缓冲区的当前状态
        self.next_idx = 0
        self.size = 0

    def __len__(self) -> int:
        """
        获取缓冲区中当前存储的经验数量。

        Returns:
            int: 当前缓冲区中的经验数量。

        Example:
            ```python
            buffer = PriorityReplayBuffer(10000, 0.6, 2, 4, 2)
            print(len(buffer))
            ```
        """
        return self.size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        添加联合经验到缓冲区。

        Args:
            obs (np.ndarray): 当前状态，形状为 (num_agents, state_dim)。
            actions (np.ndarray): 当前动作，形状为 (num_agents, action_dim)。
            rewards (np.ndarray): 当前奖励，形状为 (num_agents, )。
            next_obs (np.ndarray): 下一状态，形状为 (num_agents, state_dim)。
            dones (np.ndarray): 终止标志，形状为 (num_agents, )。

        Example:
            ```python
            buffer = PriorityReplayBuffer(10000, 0.6, 2, 4, 2)
            obs = np.random.rand(2, 4)
            actions = np.random.rand(2, 2)
            rewards = np.random.rand(2)
            next_obs = np.random.rand(2, 4)
            dones = np.array([False, True])

            buffer.add(obs, actions, rewards, next_obs, dones)
            ```
        """
        idx = self.next_idx

        # 存储经验
        self.states[idx] = torch.from_numpy(np.array(obs)).float()
        self.actions[idx] = torch.from_numpy(np.array(actions)).float()
        self.rewards[idx] = torch.from_numpy(np.array(rewards)).float()
        self.next_states[idx] = torch.from_numpy(np.array(next_obs)).float()
        self.dones[idx] = torch.from_numpy(np.array(dones)).bool()

        # 更新优先级树
        priority = self.max_priority**self.alpha
        self._update_priority_tree(idx, priority)

        # 更新缓冲区状态
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def _update_priority_tree(self, idx: int, priority: float) -> None:
        """更新优先级树中的单个节点"""
        tree_idx = idx + self.tree_capacity
        self.priority_tree[tree_idx] = priority

        # 向上更新父节点
        while tree_idx > 1:
            tree_idx //= 2
            left_child = 2 * tree_idx
            right_child = 2 * tree_idx + 1

            # 更新父节点为子节点的和
            self.priority_tree[tree_idx] = (
                self.priority_tree[left_child] + self.priority_tree[right_child]
            )

    def _update_priority_tree_batch(
        self, indexes: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        """批量更新优先级树"""
        # 转换为树索引
        tree_indexes = indexes + self.tree_capacity
        self.priority_tree[tree_indexes] = priorities

        # 批量更新父节点
        unique_parents = set()
        for tree_idx in tree_indexes.tolist():
            parent_idx = tree_idx
            while parent_idx > 1:
                parent_idx //= 2
                unique_parents.add(parent_idx)

        # 更新所有受影响的父节点
        for parent_idx in sorted(unique_parents, reverse=True):
            left_child = 2 * parent_idx
            right_child = 2 * parent_idx + 1
            self.priority_tree[parent_idx] = (
                self.priority_tree[left_child] + self.priority_tree[right_child]
            )

    def _sample_proportional(self, batch_size: int) -> torch.Tensor:
        """基于优先级采样"""
        indexes = []
        total_priority = self.priority_tree[1].item()

        for _ in range(batch_size):
            mass = random.random() * total_priority
            idx = 1  # 从根节点开始

            # 向下遍历树
            while idx < self.tree_capacity:
                left_child = 2 * idx
                right_child = left_child + 1

                if mass <= self.priority_tree[left_child]:
                    idx = left_child
                else:
                    mass -= self.priority_tree[left_child]
                    idx = right_child

            indexes.append(idx - self.tree_capacity)

        return torch.tensor(indexes, dtype=torch.long)

    def sample(
        self, batch_size: int, beta: float, device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        从缓冲区中采样一批经验。

        Args:
            batch_size (int): 采样的批量大小。
            beta (float): 重要性采样的修正因子。
            device (str): 返回数据的设备（如 "cpu" 或 "cuda"）。

        Returns:
            Dict[str, torch.Tensor]: 包含采样数据的字典，包括：
                - states (list[torch.Tensor]): 当前状态。
                - actions (list[torch.Tensor]): 当前动作。
                - rewards (torch.Tensor): 当前奖励。
                - next_states (list[torch.Tensor]): 下一状态。
                - dones (torch.Tensor): 终止标志。
                - weights (torch.Tensor): 重要性采样权重。
                - indexes (torch.Tensor): 采样的索引。

        Example:
            ```python
            buffer = PriorityReplayBuffer(10000, 0.6, 2, 4, 2)
            batch = buffer.sample(batch_size=32, beta=0.4, device="cpu")
            print(batch["states"])
            ```
        """
        # 采样索引
        indexes = self._sample_proportional(batch_size)

        # 计算重要性采样权重
        priorities = self.priority_tree[indexes + self.tree_capacity]
        min_priority = (
            self.priority_tree[self.tree_capacity : self.tree_capacity + self.size]
            .min()
            .item()
        )

        if min_priority <= 0:
            min_priority = 1e-6  # 避免除以零

        max_weight = (min_priority * self.size) ** (-beta)
        weights = (priorities * self.size) ** (-beta) / max_weight

        # 准备返回数据
        result = {
            "states": [],
            "actions": [],
            "rewards": self.rewards[indexes].to(device),
            "next_states": [],
            "dones": self.dones[indexes].float().to(device),  # 转换为float
            "weights": weights.to(device),
            "indexes": indexes.to(device),
        }

        # 按智能体拆分状态和动作
        for i in range(self.num_agents):
            result["states"].append(self.states[indexes, i].to(device))
            result["actions"].append(self.actions[indexes, i].to(device))
            result["next_states"].append(self.next_states[indexes, i].to(device))

        return result

    def update_priorities(self, indexes: torch.Tensor, priorities: np.ndarray) -> None:
        """
        更新缓冲区中指定索引的优先级。

        Args:
            indexes (torch.Tensor): 要更新的索引。
            priorities (np.ndarray): 对应的优先级值。

        Example:
            ```python
            buffer = PriorityReplayBuffer(10000, 0.6, 2, 4, 2)
            indexes = torch.tensor([0, 1, 2])
            priorities = np.array([0.9, 0.8, 0.7])

            buffer.update_priorities(indexes, priorities)
            ```
        """
        # 更新最大优先级
        self.max_priority = max(self.max_priority, np.max(priorities))

        # 计算alpha调整后的优先级
        priorities_alpha = torch.tensor(priorities**self.alpha, dtype=torch.float32)

        # 批量更新优先级树
        self._update_priority_tree_batch(indexes.cpu(), priorities_alpha)

    def is_full(self) -> bool:
        """
        检查缓冲区是否已满。

        Returns:
            bool: 如果缓冲区已满，则返回 True，否则返回 False。

        Example:
            ```python
            buffer = PriorityReplayBuffer(10000, 0.6, 2, 4, 2)
            print(buffer.is_full())
            ```
        """
        return self.size == self.capacity


# class MADDPG_PER_Agent:
#     def __init__(
#         self,
#         num_agents,
#         state_dim,
#         action_dim,
#         actor_hidden=64,
#         critic_hidden=64,
#         buffer_capacity=1000000,
#         alpha=0.6,
#         beta=0.4,
#     ):
#         self.num_agents = num_agents
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#
#         # 创建Actor和Critic网络
#         self.actors = [
#             self._build_actor(state_dim, action_dim, actor_hidden)
#             for _ in range(num_agents)
#         ]
#         self.critics = [
#             self._build_critic(state_dim, action_dim * num_agents, critic_hidden)
#             for _ in range(num_agents)
#         ]
#
#         # 目标网络
#         self.target_actors = [
#             self._build_actor(state_dim, action_dim, actor_hidden)
#             for _ in range(num_agents)
#         ]
#         self.target_critics = [
#             self._build_critic(state_dim, action_dim * num_agents, critic_hidden)
#             for _ in range(num_agents)
#         ]
#
#         # 初始化目标网络权重
#         for i in range(num_agents):
#             self._update_target(self.target_actors[i], self.actors[i], 1.0)
#             self._update_target(self.target_critics[i], self.critics[i], 1.0)
#
#         # 优化器
#         self.actor_optimizers = [
#             optim.Adam(self.actors[i].parameters(), lr=0.001) for i in range(num_agents)
#         ]
#         self.critic_optimizers = [
#             optim.Adam(self.critics[i].parameters(), lr=0.001)
#             for i in range(num_agents)
#         ]
#
#         # 多智能体PER缓冲区
#         self.buffer = PriorityReplayBuffer(
#             buffer_capacity, alpha, num_agents, state_dim
#         )
#         self.beta = beta
#         self.beta_increment = (1.0 - beta) / 50000  # 5万步内线性增加到1.0
#
#     def _build_actor(self, state_dim, action_dim, hidden_dim):
#         """构建Actor网络"""
#         return nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#             nn.Tanh(),
#         )
#
#     def _build_critic(self, state_dim, total_action_dim, hidden_dim):
#         """构建Critic网络（输入全局状态和所有动作）"""
#         return nn.Sequential(
#             nn.Linear(state_dim + total_action_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#         )
#
#     def act(self, state, agent_id, noise_scale=0.1):
#         """为指定智能体选择动作（带探索噪声）"""
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         with torch.no_grad():
#             action = self.actors[agent_id](state_tensor).numpy().squeeze(0)
#         # 添加探索噪声
#         noise = noise_scale * np.random.randn(self.action_dim)
#         return np.clip(action + noise, -1, 1)
#
#     def update(self, batch_size):
#         """更新所有智能体"""
#         # 更新beta值
#         self.beta = min(1.0, self.beta + self.beta_increment)
#
#         # 从PER采样批次
#         batch = self.buffer.sample(batch_size, self.beta)
#         states = torch.FloatTensor(batch["states"])
#         actions = torch.FloatTensor(batch["actions"])
#         rewards = torch.FloatTensor(batch["rewards"])
#         next_states = torch.FloatTensor(batch["next_states"])
#         dones = torch.FloatTensor(batch["dones"])
#         weights = torch.FloatTensor(batch["weights"])
#         indices = batch["indexes"]
#
#         # 计算每个智能体的TD误差
#         all_agents_td_errors = []
#
#         for agent_idx in range(self.num_agents):
#             # 更新Critic
#             critic_loss = self._update_critic(
#                 agent_idx,
#                 states,
#                 actions,
#                 rewards[:, agent_idx],
#                 next_states,
#                 dones,
#                 weights,
#             )
#
#             # 更新Actor
#             actor_loss = self._update_actor(agent_idx, states, actions)
#
#             # 收集TD误差用于优先级更新
#             all_agents_td_errors.append(critic_loss.detach().abs().numpy())
#
#         # 更新优先级（使用最大TD误差）
#         max_td_errors = np.max(all_agents_td_errors, axis=0)
#         self.buffer.update_priorities(indices, max_td_errors + 1e-6)
#
#         # 软更新目标网络
#         for agent_idx in range(self.num_agents):
#             self._update_target(
#                 self.target_actors[agent_idx], self.actors[agent_idx], 0.01
#             )
#             self._update_target(
#                 self.target_critics[agent_idx], self.critics[agent_idx], 0.01
#             )
#
#     def _update_critic(
#         self, agent_idx, states, all_actions, rewards, next_states, dones, weights
#     ):
#         """更新指定智能体的Critic网络"""
#         # 获取目标动作
#         with torch.no_grad():
#             target_actions = []
#             for i in range(self.num_agents):
#                 target_actions.append(self.target_actors[i](next_states))
#             target_actions = torch.cat(target_actions, dim=1)
#
#             # 计算目标Q值
#             target_q = self.target_critics[agent_idx](
#                 torch.cat([next_states, target_actions], dim=1)
#             )
#             target_q = rewards + 0.99 * (1 - dones) * target_q.squeeze()
#
#         # 计算当前Q值
#         current_q = self.critics[agent_idx](
#             torch.cat([states, all_actions.view(all_actions.size(0), -1)], dim=1)
#         ).squeeze()
#
#         # 计算TD误差和加权损失
#         td_errors = current_q - target_q
#         critic_loss = (weights * td_errors.pow(2)).mean()
#
#         # 优化Critic
#         self.critic_optimizers[agent_idx].zero_grad()
#         critic_loss.backward()
#         self.critic_optimizers[agent_idx].step()
#
#         return td_errors
#
#     def _update_actor(self, agent_idx, states, all_actions):
#         """更新指定智能体的Actor网络"""
#         # 冻结其他智能体的动作
#         detached_actions = []
#         for i in range(self.num_agents):
#             if i == agent_idx:
#                 # 当前智能体的动作由策略生成
#                 current_action = self.actors[agent_idx](states)
#                 detached_actions.append(current_action)
#             else:
#                 # 其他智能体的动作固定（从缓冲区取）
#                 detached_actions.append(all_actions[:, i].detach())
#
#         # 组合所有动作
#         joint_actions = torch.cat(detached_actions, dim=1)
#
#         # 计算策略梯度
#         actor_loss = -self.critics[agent_idx](
#             torch.cat([states, joint_actions], dim=1)
#         ).mean()
#
#         # 优化Actor
#         self.actor_optimizers[agent_idx].zero_grad()
#         actor_loss.backward()
#         self.actor_optimizers[agent_idx].step()
#
#         return actor_loss
#
#     def _update_target(self, target, source, tau):
#         """软更新目标网络"""
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
#
#     def save_experience(self, state, actions, rewards, next_state, done):
#         """保存联合经验到缓冲区"""
#         self.buffer.add(state, actions, rewards, next_state, done)
#
#     def train(self, env, episodes=1000, batch_size=1024):
#         """训练循环"""
#         for episode in range(episodes):
#             state = env.reset()
#             episode_rewards = np.zeros(self.num_agents)
#             done = False
#
#             while not done:
#                 # 所有智能体选择动作
#                 actions = []
#                 for agent_id in range(self.num_agents):
#                     action = self.act(state, agent_id)
#                     actions.append(action)
#
#                 # 执行动作
#                 next_state, rewards, done, _ = env.step(actions)
#
#                 # 存储经验
#                 self.save_experience(state, actions, rewards, next_state, done)
#
#                 # 更新网络
#                 if self.buffer.size >= batch_size:
#                     self.update(batch_size)
#
#                 state = next_state
#                 episode_rewards += rewards
#
#             print(
#                 f"Episode {episode+1}/{episodes} | Avg Reward: {episode_rewards.mean():.2f}"
#             )
