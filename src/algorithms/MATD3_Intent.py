import os
from copy import deepcopy
from typing import Dict

import torch
from torch.functional import F
from torch.optim import Adam

from .Action.DeterministicActorStrategy import DeterministicActorStrategy
from .Manager import Manager
from .Network.Actor.IntentPredictActor import IntentDeterministicActor
from .Network.Critic.QValueCritic import QValueCritic
from .StorageStrategy.PriorityReplayStorage import PriorityReplayStorage
from .StorageStrategy.StandardReplayStorage import StandardReplayStorage
from .Trainer.ITrainer import ITrainer
from .utils import soft_update


class MATD3Trainer(ITrainer):
    """
    多智能体 Twin Delayed Deep Deterministic Policy Gradient (MATD3) 训练器。

    该类负责执行多智能体 TD3 的全部训练逻辑，包括：
    - 从经验回放中采样（支持优先级经验回放 PER）
    - 计算双 Critic（Q1/Q2）目标值（target networks）
    - 策略平滑（policy smoothing）+ 噪声裁剪
    - 更新双 Critic 网络（带 IS 权重）
    - 延迟策略更新（delayed policy updates）
    - 多智能体独立 Actor 更新（仅自身动作保留梯度）
    - 软更新目标网络（Polyak averaging）
    - 学习率调度（Actor 与 Critic）
    - 更新 PRIORITY（若开启 PER）
    - 模型保存与加载

    MATD3 用于解决多智能体连续控制问题，在 TD3 基础上扩展为每个 agent
    均有独立 Actor，而 Critic 接受拼接后的全局信息（centralized critic）。

    Attributes:
        _actor (List[nn.Module]):
            多智能体 Actor 网络，每个智能体一个。
        _critic1, _critic2 (nn.Module):
            两个 Q 网络（Twin Critic），缓解 Q-value 过估计。
        _target_actor (List[nn.Module]):
            Actor 目标网络，定期进行软更新。
        _target_critic1, _target_critic2 (nn.Module):
            Critic 目标网络。
        _actor_optimizer (List[Optimizer]):
            每个 Actor 的优化器。
        _critic_optimizers (List[Optimizer]):
            两个 Critic 的优化器。
        actor_lr_scheduler (List[lr_scheduler]):
            每个 Actor 的学习率调度器。
        critic1_lr_scheduler, critic2_lr_scheduler:
            Critic 的学习率调度器。
        _agent_ids (Tensor):
            多智能体 one-hot 编码，用于 Critic 输入中区分 agent。
        global_step (int):
            全局步数计数，用于 PER β 计算。

    Example:
        >>> trainer = MATD3Trainer(
        ...     buffer=replay_buffer,
        ...     args=args,
        ...     actors=actors,
        ...     critics=(critic1, critic2),
        ...     optimizer=optimizers,
        ...     target_actor=target_actors,
        ...     target_critics=(target_c1, target_c2),
        ... )
        >>> metrics = trainer.train()
        >>> print(metrics["critic1_loss0"])
    """

    def __init__(
        self,
        buffer,
        args,
        actors,
        critics,  # 改为两个Critic网络
        optimizer,
        target_actor,
        target_critics,  # 改为两个目标Critic网络
    ):
        """
        初始化多智能体 MATD3 训练器。

        包含以下初始化步骤：
        1. 保存 replay buffer 和算法参数
        2. 构建 Actor / Critic / Target 网络
        3. 记录优化器与学习率调度器
        4. 配置多智能体 one-hot agent-ID 作为 Critic 额外输入
        5. 用于策略延迟更新的 update_counter 初始化

        Args:
            buffer:
                回放缓冲区对象（普通或优先级）。
            args:
                参数对象，包含：
                - gamma, tau, policy_freq
                - batch_size
                - num_agents
                - action_min, action_max
                - policy_noise, noise_clip
                - lr_actor, lr_critic
                - device 等
            actors (List[nn.Module]):
                多智能体 Actor 网络（每个 agent 一个）。
            critics (Tuple[nn.Module, nn.Module]):
                双 Critic（critic1, critic2）。
            optimizer (dict):
                字典格式优化器：
                {
                    "actor": [actor优化器列表],
                    "critic": [critic1_optimizer, critic2_optimizer]
                }
            target_actor (List[nn.Module]):
                多智能体目标 Actor。
            target_critics (Tuple[nn.Module, nn.Module]):
                critic1_target, critic2_target。

        Example:
            >>> trainer = MATD3Trainer(
            ...     buffer, args, actors, (critic1, critic2),
            ...     optimizer, target_actors, (tcritic1, tcritic2)
            ... )
        """
        super().__init__(buffer, args)
        self._actor = actors
        self._critic1, self._critic2 = critics  # 双Critic
        self._target_actor = target_actor
        self._target_critic1, self._target_critic2 = target_critics  # 双目标Critic
        self._actor_optimizer = optimizer["actor"]
        self._critic_optimizers = optimizer["critic"]  # 两个Critic优化器
        self._update_counter = 0  # 用于延迟策略更新
        self._num_agents = args.num_agents

        self.actor_lr_scheduler = []
        for optimizer in self._actor_optimizer:
            self.actor_lr_scheduler.append(
                torch.optim.lr_scheduler.StepLR(optimizer, 1000 * args.num_envs, 0.99)
            )
        self.critic1_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._critic_optimizers[0], 1000 * args.num_envs, 0.99
        )
        self.critic2_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._critic_optimizers[1], 1000 * args.num_envs, 0.99
        )

        # 智能体ID的one-hot编码
        self._agent_ids = torch.eye(self._num_agents).to(args.device)
        self.global_step = 0

    def train(self) -> Dict[str, float]:
        """
        执行一次 MATD3 多智能体训练过程。

        该方法从经验回放中采样数据，计算目标值并更新 Critic；
        在满足延迟策略更新条件时更新各 Agent 的 Actor；
        最终返回包含损失数值、学习率等信息的字典，用于日志记录和可视化。

        Returns:
            dict: 记录一次训练步骤中关键统计指标的字典。
        """
        # ----------------------------
        # 1. 采样逻辑（根据是否启用优先级采样适配）
        # ----------------------------
        if self.args.use_priority:
            self.global_step += 1
            # 计算当前beta（线性增长到1.0）
            beta = min(
                self.args.priority_beta_end,
                self.args.priority_beta_start
                + (self.args.priority_beta_end - self.args.priority_beta_start)
                * self.global_step
                / self.args.priority_beta_frames,
            )

            # 优先级采样：返回包含权重和索引的样本
            samples = self.replay_buffer.sample(
                self.args.batch_size, beta=beta, device=self.args.device
            )
            obs = samples["states"]
            action = samples["actions"]
            reward = samples["rewards"]
            next_obs = samples["next_states"]
            done = samples["dones"]
            weights = samples["weights"]  # 重要性采样权重 (batch_size,)
            indexes = samples["indexes"]  # 样本索引，用于更新优先级
        else:
            # 普通采样：权重默认为1.0
            obs, action, reward, next_obs, done, _ = self.replay_buffer.sample(
                self.args.batch_size, device=self.args.device
            )
            weights = torch.ones(
                self.args.batch_size, device=self.args.device
            )  # 无权重时为1
            indexes = None

        # ----------------------------
        # 2. 目标动作计算（加入策略噪声平滑）
        # ----------------------------
        next_union_actions = []
        for index in range(self._num_agents):
            # 目标Actor输出基础动作
            base_action = self._target_actor[index](next_obs[index])
            # 加入噪声并裁剪（目标策略平滑）
            noise = torch.clamp(
                torch.randn_like(base_action) * self.args.policy_noise,
                -self.args.noise_clip,
                self.args.noise_clip,
            )
            smoothed_action = torch.clamp(
                base_action + noise, self.args.action_min, self.args.action_max
            )
            next_union_actions.append(smoothed_action)
        next_union_actions = torch.cat(next_union_actions, dim=1)

        # 拼接观测和动作
        union_obs = torch.cat([obs[idx] for idx in range(self._num_agents)], dim=1)
        next_union_obs = torch.cat(
            [next_obs[idx] for idx in range(self._num_agents)], dim=1
        )
        union_actions_sampled = torch.cat(
            [action[idx] for idx in range(self._num_agents)], dim=1
        )

        loss = {}

        # ----------------------------
        # 3. 更新双Critic网络（带优先级权重修正）
        # ----------------------------
        # 清零梯度
        self._critic_optimizers[0].zero_grad()  # critic1优化器
        self._critic_optimizers[1].zero_grad()  # critic2优化器

        # 计算目标Q值（双Critic取最小值）
        target_q_values = []
        for index in range(self._num_agents):
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )
            # 目标Critic输入
            next_critic_input = torch.cat(
                (next_union_obs, next_union_actions, agent_id), -1
            )
            # 两个目标Critic计算Q值，取最小值（减少过高估计）
            q1_target = self._target_critic1(next_critic_input)
            q2_target = self._target_critic2(next_critic_input.clone())
            target_q = torch.min(q1_target, q2_target).detach()

            # TD目标计算
            t_done = done[:, index].view(self.args.batch_size, 1)
            t_reward = reward[:, index].view(self.args.batch_size, 1)
            td_target = t_reward + self.args.gamma * (1 - t_done) * target_q
            target_q_values.append(td_target)

        # 计算两个Critic的损失（带重要性采样权重）
        total_critic1_loss = 0.0
        total_critic2_loss = 0.0
        td_errors = []  # 存储每个样本的TD误差（用于更新优先级）
        for index in range(self._num_agents):
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )
            critic_input = torch.cat((union_obs, union_actions_sampled, agent_id), -1)

            # 两个Critic当前Q值
            q1 = self._critic1(critic_input)
            q2 = self._critic2(critic_input)

            # 计算单个样本的损失（不直接取mean，保留batch维度用于加权）
            critic1_loss_per_sample = F.mse_loss(
                q1, target_q_values[index], reduction="none"
            )  # (batch, 1)
            critic2_loss_per_sample = F.mse_loss(
                q2, target_q_values[index], reduction="none"
            )  # (batch, 1)

            # 用重要性采样权重加权（权重扩维匹配损失维度）
            weighted_critic1_loss = (
                weights.view(-1, 1) * critic1_loss_per_sample
            ).mean()
            weighted_critic2_loss = (
                weights.view(-1, 1) * critic2_loss_per_sample
            ).mean()

            total_critic1_loss += weighted_critic1_loss
            total_critic2_loss += weighted_critic2_loss

            # 记录单个智能体的TD误差（取绝对值，用于优先级更新）
            td_error = (
                torch.abs(q1 - target_q_values[index]).detach().squeeze(1)
            )  # (batch,)
            td_errors.append(td_error)

            # 记录损失
            loss.update(
                {
                    f"critic1_loss{index}": weighted_critic1_loss.item(),
                    f"critic2_loss{index}": weighted_critic2_loss.item(),
                }
            )

        # 计算所有智能体的平均TD误差（作为样本优先级）
        if self.args.use_priority:
            # 对多智能体的TD误差取平均（单个样本的综合优先级）
            avg_td_errors = torch.mean(torch.stack(td_errors, dim=1), dim=1)  # (batch,)

        # 反向传播更新
        total_critic1_loss.backward()
        total_critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._critic1.parameters(), max_norm=self.args.norm_grad
        )
        torch.nn.utils.clip_grad_norm_(
            self._critic2.parameters(), max_norm=self.args.norm_grad
        )
        self._critic_optimizers[0].step()
        self._critic_optimizers[1].step()

        # # ----------------------------
        # # 4. 延迟更新Actor网络（每policy_freq次更新一次）
        # # ----------------------------
        # self._update_counter += 1
        # if self._update_counter % self.args.policy_freq != 0:
        #     # 未达到更新频率，仅返回Critic损失（若使用优先级，此处已完成采样但不更新Actor）
        #     return loss

        # 重置计数器
        self._update_counter = 0

        # 更新每个智能体的Actor
        for index in range(self._num_agents):
            # 构建联合动作（当前智能体动作保留梯度，其他智能体动作阻断梯度）
            other_actor_output = []
            for idx in range(self._num_agents):
                if idx == index:
                    actor_output = self._actor[idx](obs[idx])  # 保留梯度
                else:
                    actor_output = self._actor[idx](obs[idx]).detach()  # 阻断梯度
                other_actor_output.append(actor_output)
            union_actions_for_actor = torch.cat(other_actor_output, dim=-1)

            # 计算Actor损失（基于critic1的Q值）
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )
            critic_input = torch.cat((union_obs, union_actions_for_actor, agent_id), -1)

            loss_actor = -self._critic1(critic_input).mean()

            actor_mu = self._actor[index].mu
            log_sigma = self._actor[index].log_sigma
            predict = self._actor[index].predict  # n_angent, batch_size, n_obs

            predict = torch.stack(predict, dim=0)  # n_agents, batch_size, n_actions

            predict_label = torch.stack(other_actor_output, dim=0).detach()
            predict_loss = F.mse_loss(predict, predict_label)

            actor_mu = torch.stack(actor_mu, dim=0)
            log_sigma = torch.stack(log_sigma, dim=0)
            kl_loss = (
                torch.distributions.kl.kl_divergence(
                    torch.distributions.Normal(actor_mu, torch.exp(log_sigma)),
                    torch.distributions.Normal(
                        torch.zeros_like(actor_mu), torch.ones_like(log_sigma)
                    ),
                )
                .sum(-1)
                .mean()
            )

            self._actor_optimizer[index].zero_grad()
            # Actor目标是最大化Q值（因此损失为负的Q值均值）
            actor_loss = loss_actor + predict_loss + self.args.kl_coef * kl_loss
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._actor[index].parameters(), max_norm=self.args.norm_grad
            )
            self._actor_optimizer[index].step()

            loss.update(
                {
                    f"actor_loss{index}": actor_loss.item(),
                    f"predict_loss{index}": predict_loss.item(),
                    f"kl_loss{index}": kl_loss.item(),
                }
            )

        # ----------------------------
        # 5. 软更新目标网络
        # ----------------------------
        soft_update(self._target_critic1, self._critic1, self.args.tau)
        soft_update(self._target_critic2, self._critic2, self.args.tau)
        for index in range(self._num_agents):
            soft_update(self._target_actor[index], self._actor[index], self.args.tau)

        for lr_scheduler in self.actor_lr_scheduler:
            lr_scheduler.step()
        self.critic1_lr_scheduler.step()
        self.critic2_lr_scheduler.step()
        loss["actor_lr"] = self.actor_lr_scheduler[0].get_lr()[0]
        loss["ciritc_lr"] = self.critic1_lr_scheduler.get_lr()[0]
        # ----------------------------
        # 6. 更新优先级回放缓冲区的优先级（若启用）
        # ----------------------------
        if self.args.use_priority:
            self.replay_buffer.update_priorities(
                indexes=indexes,
                priorities=avg_td_errors.cpu().numpy(),  # 将TD误差作为新优先级
            )

        # 记录beta值（可选）
        if self.args.use_priority:
            loss["priority_beta"] = beta

        return loss

    def save_models(self, directory: str):
        """
        保存当前多智能体 MATD3 的全部模型参数。

        包括所有 Actor、Critic 以及对应的目标网络；
        用于中断训练后的继续训练或模型部署。

        Args:
            path (str): 模型保存的文件目录路径。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存Actor和双Critic
        actor_dict = {
            f"actor{idx}": self._actor[idx].state_dict()
            for idx in range(self._num_agents)
        }
        critic_dict = {
            "critic1": self._critic1.state_dict(),
            "critic2": self._critic2.state_dict(),
        }

        actor_optimizer_dict = {
            f"actor_optim{idx}": self._actor_optimizer[idx].state_dict()
            for idx in range(self._num_agents)
        }

        critic_optimizer_dict = {
            "critic_optim0": self._critic_optimizers[0].state_dict(),
            "critic_optim1": self._critic_optimizers[1].state_dict(),
        }

        final_dict = {
            **actor_dict,
            **critic_dict,
            **actor_optimizer_dict,
            **critic_optimizer_dict,
        }
        torch.save(final_dict, os.path.join(directory, "actor_critic.pth"))

    def load_models(self, directory: str):
        """
        从指定路径加载多智能体 MATD3 的全部模型参数。

        包括所有 Actor、Critic 以及对应的目标网络；
        用于恢复训练进度或执行推理。

        Args:
            path (str): 模型文件所在的目录路径。
        """
        final_dict = torch.load(
            os.path.join(directory, "actor_critic.pth"),
            weights_only=False,
            map_location=self.args.device,
        )

        # 加载Actor和双Critic
        for idx in range(self._num_agents):
            self._actor[idx].load_state_dict(final_dict[f"actor{idx}"])
            if final_dict.get(f"actor_optim{idx}"):
                self._actor_optimizer[idx].load_state_dict(
                    final_dict[f"actor_optim{idx}"]
                )

        self._critic1.load_state_dict(final_dict["critic1"])
        self._critic2.load_state_dict(final_dict["critic2"])

        if final_dict.get("critic_optim0"):
            self._critic_optimizers[0].load_state_dict(final_dict["critic_optim0"])
            self._critic_optimizers[1].load_state_dict(final_dict["critic_optim1"])

        # 更新目标网络
        self._target_actor = deepcopy(self._actor)
        self._target_critic1 = deepcopy(self._critic1)
        self._target_critic2 = deepcopy(self._critic2)


class MATD3_Intent(Manager):
    """
    多智能体版本的 Twin Delayed DDPG（MATD3）算法管理器。

    该类负责初始化多智能体系统所需的 Actor、Critic、目标网络、
    优化器以及经验存储模块，并统一封装策略执行、训练与模型管理流程。

    主要特点包括：
    - 支持多智能体之间共享或独立 Actor / Critic；
    - 支持标准经验回放与优先经验回放；
    - 提供与 Manager 基类统一的策略接口。

    Attributes:
        name_agents (list[str]): 多个智能体的名称列表。
        num_agents (int): 智能体数量。
        storage (ReplayStorage): 用于存储交互数据的经验回放模块。
        actor_strategy (ActorStrategy): 用于执行动作策略的策略对象。
        trainer (MATD3Trainer): 用于优化策略与价值网络的训练器。

    Example:
        >>> matd3 = MATD3(
        ...     name_agents=["agent_1", "agent_2"],
        ...     obs_shapes=[24, 24],
        ...     global_state_shape=48,
        ...     action_shapes=[2, 2],
        ...     args=args,
        ... )
        >>> actions = matd3.actor_strategy.get_actions(observations)
    """

    def __init__(
        self,
        name_agents: list[str],
        obs_shapes: list[int],
        global_state_shape: int,
        action_shapes: list[int],
        args,
        action_continuous: bool = True,
    ):
        """
        初始化 MATD3 算法的全部核心组件。

        包括经验回放、Actor、Critic、目标网络、优化器、训练器以及策略执行器等。
        该方法只负责搭建结构，不执行训练逻辑。

        Args:
            name_agents (list[str]): 智能体名称列表。
            obs_shapes (list[int]): 每个智能体对应的观测向量维度。
            global_state_shape (int): 全局状态维度。
            action_shapes (list[int]): 每个智能体动作空间维度。
            args: 超参数配置对象。
            action_continuous (bool): 动作空间是否为连续型。

        Example:
            >>> matd3 = MATD3(
            ...     ["A1", "A2"],
            ...     [24, 24],
            ...     48,
            ...     [2, 2],
            ...     args,
            ...     action_continuous=True,
            ... )
            >>> print(matd3.num_agents)
        """
        self.args = args
        self.name_agents = name_agents
        self.num_agents = len(self.name_agents)

        if args.just_eval:
            self.storage = None
        else:
            if self.args.use_priority:
                self.storage = PriorityReplayStorage(
                    buffer_size=args.buffer_size,
                    obs_shapes=[(obs_shapes[i],) for i in range(self.num_agents)],
                    action_shapes=[(action_shapes[i],) for i in range(self.num_agents)],
                    num_agents=self.num_agents,
                    args=args,
                )
            else:
                # 初始化经验回放
                self.storage = StandardReplayStorage(
                    buffer_size=args.buffer_size,
                    obs_shapes=[(obs_shapes[i],) for i in range(self.num_agents)],
                    action_shapes=[(action_shapes[i],) for i in range(self.num_agents)],
                    num_agents=self.num_agents,
                    action_continuous=action_continuous,
                    args=args,
                )

        # 计算输入维度
        obs_dim_sum = sum(args.n_obs)
        actions_dim_sum = sum(args.n_actions)
        agent_id_dim = self.num_agents  # one-hot维度
        critic_input_size = obs_dim_sum + actions_dim_sum + agent_id_dim

        # 初始化Actor和Intent网络
        actors = []
        for index in range(args.num_agents):
            actor = IntentDeterministicActor(
                input_dim=args.n_obs[index],
                output_dim=args.n_actions[index],
                hidden_dim=args.hidden_size,
                intent_predictor_count=self.num_agents,
                intent_predictor_output_dim=args.n_actions[index],
            ).to(args.device)
            actors.append(actor)

        # 初始化双Critic网络（共享参数结构，独立参数）
        critic1 = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )
        critic2 = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )

        # 目标网络
        target_actors = deepcopy(actors)
        target_critic1 = deepcopy(critic1)
        target_critic2 = deepcopy(critic2)

        # 动作策略
        self.actor_strategy = DeterministicActorStrategy(
            actors=actors, args=args, action_continuous=action_continuous
        )

        # 优化器
        optimizers = {
            "actor": [Adam(actor.parameters(), lr=args.lr_actor) for actor in actors],
            "critic": [
                Adam(critic1.parameters(), lr=args.lr_critic),  # critic1优化器
                Adam(critic2.parameters(), lr=args.lr_critic),  # critic2优化器
            ],
        }

        # 初始化Trainer
        self.trainer = MATD3Trainer(
            buffer=self.storage.replay_buffer if not self.args.just_eval else None,
            args=args,
            actors=actors,
            critics=(critic1, critic2),  # 传入双Critic
            optimizer=optimizers,
            target_actor=target_actors,
            target_critics=(target_critic1, target_critic2),  # 传入双目标Critic
        )

        super().__init__(
            actor_strategy=self.actor_strategy,
            storage_strategy=self.storage,
            trainer=self.trainer,
            args=self.args,
        )
