import os
from copy import deepcopy
from typing import Dict

import torch
from torch.functional import F
from torch.optim import Adam

from .Action.DeterministicActorStrategy import DeterministicActorStrategy
from .Manager import Manager
from .Network.Actor.DeterministicActor import DeterministicActor
from .Network.Critic.QValueCritic import QValueCritic
from .StorageStrategy.StandardReplayStorage import StandardReplayStorage
from .Trainer.ITrainer import ITrainer
from .utils import soft_update


class MADDPG_STrainer(ITrainer):
    """
    MADDPG算法训练器类，使用共享的单个Critic网络。

    这是一个实现了多智能体深度确定性策略梯度(MADDPG)算法的训练器，
    所有智能体共享同一个Critic网络来评估联合动作的价值。每个智能体有
    自己独立的Actor网络，但在训练时使用相同的Critic进行价值评估。

    Args:
        _actor (List[torch.nn.Module]): 所有智能体的Actor网络列表
        _critic (torch.nn.Module): 共享的Critic网络
        _target_actor (List[torch.nn.Module]): 所有智能体的目标Actor网络列表
        _target_critic (torch.nn.Module): 共享的目标Critic网络
        _actor_optimizer (List[torch.optim.Optimizer]): 所有智能体的Actor优化器列表
        _critic_optimizer (torch.optim.Optimizer): 共享的Critic优化器
        _update_counter (int): 训练更新计数器
        _num_agents (int): 智能体数量
        _agent_ids (torch.Tensor): 智能体ID的one-hot编码
    """

    def __init__(
        self,
        buffer,
        args,
        actors,
        critic,  # 现在是共享的单个Critic
        optimizer,
        target_actor,
        target_critic,  # 现在是共享的单个目标Critic
    ):
        """
        初始化MADDPG训练器。

        Args:
            buffer: 经验回放缓冲区
            args: 训练参数配置对象
            actors: 所有智能体的Actor网络列表
            critic: 共享的Critic网络
            optimizer: 优化器字典，包含"actor"和"critic"键
            target_actor: 所有智能体的目标Actor网络列表
            target_critic: 共享的目标Critic网络

        Example:
            >>> trainer = MADDPG_STrainer(
            >>>     buffer=replay_buffer,
            >>>     args=config_args,
            >>>     actors=[actor1, actor2, actor3],
            >>>     critic=shared_critic,
            >>>     optimizer={
            >>>         "actor": [optim.Adam(actor1.parameters()),
            >>>                   optim.Adam(actor2.parameters()),
            >>>                   optim.Adam(actor3.parameters())],
            >>>         "critic": [optim.Adam(shared_critic.parameters())]
            >>>     },
            >>>     target_actor=[target_actor1, target_actor2, target_actor3],
            >>>     target_critic=target_shared_critic
            >>> )
        """
        super().__init__(buffer, args)
        self._actor = actors
        self._critic = critic
        self._target_actor = target_actor
        self._target_critic = target_critic
        self._actor_optimizer = optimizer["actor"]
        # 注意：这里需要确保 optimizer["critic"] 是一个列表，且包含一个优化器实例
        self._critic_optimizer = optimizer["critic"][0]  # 单个Critic优化器，直接取出
        self._update_counter = 0
        self._num_agents = args.num_agents

        # 创建agent_id的one-hot编码，如果不需要可以移除
        self._agent_ids = torch.eye(self._num_agents).to(args.device)

    def train(self) -> Dict[str, float]:
        """
        执行一次训练步骤，包括Critic和Actor的更新。

        该方法执行标准的MADDPG训练流程：
        1. 从经验回放缓冲区采样一批经验数据
        2. 计算目标Q值和当前Q值
        3. 更新共享Critic网络
        4. 更新每个智能体的Actor网络
        5. 软更新目标网络

        Returns:
            Dict[str, float]: 包含各项损失的字典，键为"critic_loss{index}"和"actor_loss{index}"，
                              值为对应的损失值。
        Example:
            >>> trainer = MADDPG_STrainer(...)
            >>> for episode in range(num_episodes):
            >>>     # 收集经验数据到缓冲区...
            >>>     losses = trainer.train()
            >>>     print(f"Critic损失: {losses['critic_loss0']}, Actor损失: {losses['actor_loss0']}")
        """
        # 从buffer采样
        obs, action, reward, next_obs, done, extra = self.replay_buffer.sample(
            self.args.batch_size, device=self.args.device
        )

        next_union_actions = []
        for index in range(self._num_agents):
            next_union_actions.append(self._target_actor[index](next_obs[index]))
        next_union_actions = torch.cat(next_union_actions, dim=1)

        # 拼接所有智能体的当前观测
        union_obs = torch.cat([obs[idx] for idx in range(self._num_agents)], dim=1)
        # 拼接所有智能体的下一观测
        next_union_obs = torch.cat(
            [next_obs[idx] for idx in range(self._num_agents)], dim=1
        )
        # 拼接所有智能体在经验回放中实际采取的动作
        # 这是标准的MADDPG做法，评论家评估的是实际发生的动作
        union_actions_sampled = torch.cat(
            [action[idx] for idx in range(self._num_agents)], dim=1
        )

        loss = {}

        # 首先更新Critic（共享的）
        self._critic_optimizer.zero_grad()

        critic_losses = []
        for index in range(self._num_agents):
            # 添加agent_id到输入 (如果不需要可以移除此部分)
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )

            # 使用共享Critic评估当前Q值，输入包含所有智能体的联合观测、**采样动作**和agent_id
            # 注意这里使用了 union_actions_sampled，这是关键修改
            critic_input = torch.cat((union_obs, union_actions_sampled, agent_id), -1)
            q_values = self._critic(critic_input)

            # 计算目标Q值，同样包含agent_id
            next_critic_input = torch.cat(
                (next_union_obs, next_union_actions, agent_id), -1
            )
            next_q_values = self._target_critic(next_critic_input).detach()

            t_done = done[:, index].view(self.args.batch_size, 1)
            t_reward = reward[:, index].view(self.args.batch_size, 1)
            td_target = t_reward + self.args.gamma * (1 - t_done) * next_q_values

            critic_loss = F.mse_loss(q_values, td_target)
            critic_losses.append(critic_loss)

            loss.update({"critic_loss" + str(index): critic_loss.item()})

        # 累积所有智能体的Critic损失并反向传播
        total_critic_loss = sum(critic_losses)
        total_critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self._critic.parameters(), max_norm=self.args.norm_grad
        )

        self._critic_optimizer.step()

        # 然后更新每个智能体的Actor
        for index in range(self._num_agents):
            # 构建当前智能体的联合动作，其中当前智能体使用其当前策略，其他智能体使用其当前策略的 .detach() 版本
            # 这样梯度只会流向当前正在优化的actor
            union_actions_for_actor_loss = []
            for idx in range(self._num_agents):
                if idx == index:
                    # 正在优化的智能体的动作
                    actor_output = self._actor[idx](obs[idx])
                else:
                    # 其他智能体的动作，需要 .detach() 以阻断梯度
                    actor_output = self._actor[idx](obs[idx]).detach()
                union_actions_for_actor_loss.append(actor_output)

            union_actions_for_actor_loss = torch.cat(
                union_actions_for_actor_loss, dim=-1
            )

            # 添加agent_id到输入 (如果不需要可以移除此部分)
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )

            # 使用共享Critic评估，计算actor loss
            critic_input_for_actor_loss = torch.cat(
                (union_obs, union_actions_for_actor_loss, agent_id), -1
            )

            self._actor_optimizer[index].zero_grad()
            actor_loss = -self._critic(
                critic_input_for_actor_loss
            ).mean()  # 使用共享Critic评估
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._actor[index].parameters(), max_norm=self.args.norm_grad
            )

            self._actor_optimizer[index].step()

            loss.update({"actor_loss" + str(index): actor_loss.item()})

        # 软更新目标网络
        soft_update(self._target_critic, self._critic, self.args.tau)
        for index in range(self._num_agents):
            soft_update(self._target_actor[index], self._actor[index], self.args.tau)

        return loss

    def save_models(self, directory: str):
        """
        保存所有Actor网络和共享Critic网络的状态字典。

        将训练好的模型参数保存到指定目录，文件名为"actor_critic.pth"。
        保存的字典包含所有智能体的Actor参数和共享Critic的参数。

        Args:
            directory (str): 保存模型的目录路径

        Example:
            >>> trainer = MADDPG_STrainer(...)
            >>> trainer.save_models("./saved_models/checkpoint_1000")
            >>> # 会在"./saved_models/checkpoint_1000"目录下创建actor_critic.pth文件
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        actor_dict = {
            f"actor{idx}": self._actor[idx].state_dict()
            for idx in range(self._num_agents)
        }

        # 保存共享Critic
        critic_dict = {"critic": self._critic.state_dict()}

        final_dict = {}
        final_dict.update(actor_dict)
        final_dict.update(critic_dict)
        torch.save(final_dict, os.path.join(directory, "actor_critic.pth"))

    def load_models(self, directory: str):
        """
        从指定目录加载所有Actor网络和共享Critic网络的状态字典。

        加载保存的模型参数，并更新目标网络为加载的网络参数的深拷贝。

        Args:
            directory (str): 加载模型的目录路径

        Example:
            >>> trainer = MADDPG_STrainer(...)
            >>> trainer.load_models("./saved_models/checkpoint_1000")
            >>> # 从"./saved_models/checkpoint_1000/actor_critic.pth"加载模型参数
        """

        final_dict = torch.load(
            os.path.join(directory, "actor_critic.pth"),
            weights_only=True,
            map_location=self.args.device,
        )

        [
            self._actor[idx].load_state_dict(final_dict[f"actor{idx}"])
            for idx in range(self._num_agents)
        ]

        # 加载共享Critic
        self._critic.load_state_dict(final_dict["critic"])

        self._target_actor = deepcopy(self._actor)
        self._target_critic = deepcopy(self._critic)


class MADDPG_S(Manager):
    """
    多智能体环境中的 MADDPG 管理器类。

    该类负责构建 MADDPG 框架中的关键组件，包括：
    - 多智能体的 Actor 网络
    - 共享 Critic 网络
    - 对应的目标网络
    - 回放缓冲区
    - 优化器配置
    - 策略执行模块 ActorStrategy
    - 训练器 Trainer

    MADDPG（Multi-Agent Deep Deterministic Policy Gradient）是用于多主体协作 /
    对抗任务的经典强化学习算法，本类主要负责完成模型构建、初始化与资源管理。

    Args:
        name_agents (List[str]): 所有智能体名称列表。
        obs_shapes (List[int]): 每个智能体的观测空间维度。
        global_state_shape (int): 全局状态的维度（可能用于某些扩展算法）。
        action_shapes (List[int]): 每个智能体的动作维度。
        args (Any): 包含超参数、训练参数、设备信息的配置对象。
        action_continuous (bool): 是否为连续动作空间，默认为 False。

    Example:
        >>> maddpg = MADDPG_S(
        ...     name_agents=["agent_1", "agent_2"],
        ...     obs_shapes=[24, 24],
        ...     global_state_shape=48,
        ...     action_shapes=[4, 4],
        ...     args=args,
        ...     action_continuous=True,
        ... )
        >>> actions = maddpg.actor_strategy.get_actions(obs)
    """

    def __init__(
        self,
        name_agents: list[str],
        obs_shapes: list[int],
        global_state_shape: int,
        action_shapes: list[int],
        args,
        action_continuous: bool = False,
    ):
        """
        初始化 MADDPG_S 管理器，创建 Actor、Critic、回放缓冲区、优化器与训练器。

        此方法会执行以下步骤：
        1. 读取智能体名称与数量
        2. 创建多智能体共享的经验回放缓冲区 StandardReplayStorage
        3. 为每个智能体构建独立的确定性 Actor 网络
        4. 构建共享的 Critic 网络（输入由所有智能体的状态、动作和 one-hot agent_id 构成）
        5. 构建对应的目标网络（深拷贝）
        6. 配置 Actor 与 Critic 的优化器
        7. 创建 MADDPG_STrainer 训练器
        8. 初始化继承自 Manager 的组件

        Args:
            name_agents (List[str]): 智能体名称列表。
            obs_shapes (List[int]): 每个智能体的观测空间维度。
            global_state_shape (int): 全局状态维度。
            action_shapes (List[int]): 每个智能体的动作维度。
            args (Any): 包含训练超参数（学习率、设备、网络结构等）。
            action_continuous (bool): 是否为连续动作空间。

        Example:
            >>> maddpg = MADDPG_S(
            ...     name_agents=["robot1", "robot2"],
            ...     obs_shapes=[10, 10],
            ...     global_state_shape=20,
            ...     action_shapes=[2, 2],
            ...     args=args,
            ...     action_continuous=True,
            ... )
            >>> print(maddpg.num_agents)
            2
        """
        self.args = args
        self.name_agents = name_agents
        self.num_agents = len(self.name_agents)

        self.storage = StandardReplayStorage(
            buffer_size=args.buffer_size,
            obs_shapes=[(obs_shapes[i],) for i in range(self.num_agents)],
            action_shapes=[(action_shapes[i],) for i in range(self.num_agents)],
            num_agents=self.num_agents,
            action_continuous=action_continuous,
            args=args,
        )

        # 初始化所有网络
        actors = []

        obs_dim_sum = sum(args.n_obs)
        actions_dim_sum = sum(args.n_actions)
        agent_id_dim = self.num_agents  # one-hot编码的维度

        for index in range(args.num_agents):
            actor_input_size = args.n_obs[index]
            actor_output_size = args.n_actions[index]
            actors.append(
                DeterministicActor(
                    input_dim=actor_input_size,
                    output_dim=actor_output_size,
                    hidden_dim=args.hidden_size,
                ).to(args.device)
            )

        # 创建共享Critic（所有智能体共享），输入维度增加agent_id的维度
        # 如果不需要agent_id，这里需要调整 critic_input_size
        critic_input_size = obs_dim_sum + actions_dim_sum + agent_id_dim
        critic = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )

        # 目标网络
        target_actors = deepcopy(actors)
        target_critic = deepcopy(critic)

        self.actor_strategy = DeterministicActorStrategy(
            actors=actors, args=args, action_continuous=action_continuous
        )

        # 优化器
        optimizers = {
            "actor": [
                Adam(actors[index].parameters(), lr=args.lr_actor)  # type: ignore
                for index in range(args.num_agents)
            ],
            "critic": [
                Adam(
                    critic.parameters(), lr=args.lr_critic
                )  # 单个Critic优化器，包装成列表
            ],
        }

        self.trainer = MADDPG_STrainer(
            buffer=self.storage.replay_buffer,
            args=args,
            actors=actors,
            critic=critic,
            optimizer=optimizers,
            target_actor=target_actors,
            target_critic=target_critic,
        )
        super().__init__(
            actor_strategy=self.actor_strategy,
            storage_strategy=self.storage,
            trainer=self.trainer,
            args=self.args,
        )
