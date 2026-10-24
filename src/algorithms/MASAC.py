import os
from copy import deepcopy
from typing import Dict

import torch
from torch.functional import F
from torch.optim import Adam

from .Action.GaussianActorStrategy import GaussianActorStrategy
from .Manager import Manager
from .Network.Actor.ContinuousActor import ContinuousActor
from .Network.Critic.QValueCritic import QValueCritic
from .StorageStrategy.PriorityReplayStorage import PriorityReplayStorage
from .StorageStrategy.StandardReplayStorage import StandardReplayStorage
from .Trainer.ITrainer import ITrainer
from .utils import soft_update


class MASACTrainer(ITrainer):
    """
    多智能体 Soft Actor-Critic（MASAC）训练器。

    本类实现了多智能体版 SAC 的完整训练逻辑，支持以下核心机制：
    - 多智能体独立 Actor 网络
    - 双 Critic 网络（Twin Q）
    - 双目标 Critic 网络（Target Twin Q）
    - 温度参数 α 的自动调整（自动熵调节）
    - 优先经验回放（PER）
    - 多智能体联合动作构建
    - 多智能体单独 Actor 更新 + 多智能体共享 Critic 输入结构
    - 重参数化技巧（reparameterization trick）
    - Soft Update 软更新目标网络
    - 学习率调度器（LR Scheduler）

    Args:
        buffer: 经验回放缓冲区。
        args: 超参数配置对象，包含 SAC/MASAC 的全部超参数。
        actors (List[nn.Module]): 多智能体 Actor 网络列表。
        critics (Tuple[nn.Module, nn.Module]): 双 Critic 网络。
        target_critics (Tuple[nn.Module, nn.Module]): 双目标 Critic 网络。
        alpha (List[torch.Tensor]): 多智能体可学习的温度参数。
        optimizers (dict): 包含 actor/critic/alpha 三类优化器的字典。
        target_entropy (List[float]): 每个智能体的目标熵。

    Example:
        >>> trainer = MASACTrainer(
        ...     buffer=replay_buffer,
        ...     args=args,
        ...     actors=actor_list,
        ...     critics=(critic1, critic2),
        ...     target_critics=(target_c1, target_c2),
        ...     alpha=[torch.tensor(0.1), torch.tensor(0.1)],
        ...     optimizers=optims,
        ...     target_entropy=[-3.0, -3.0],
        ... )
        >>> loss_dict = trainer.train()
    """

    def __init__(
        self,
        buffer,
        args,
        actors,
        critics,  # 双Critic网络
        target_critics,  # 双目标Critic网络
        alpha,  # 温度参数
        optimizers,  # 所有优化器（从外部传入）
        target_entropy,  # 目标熵
    ):

        super().__init__(buffer, args)
        self._actor = actors
        self._critic1, self._critic2 = critics
        self._target_critic1, self._target_critic2 = target_critics

        # 从外部传入的优化器
        self._actor_optimizers = optimizers["actor"]
        self._critic_optimizers = optimizers["critic"]
        self._alpha_optimizers = optimizers["alpha"]

        # 温度参数相关
        self._alpha = alpha
        self._target_entropy = target_entropy

        self._num_agents = args.num_agents
        self.global_step = 0

        # 学习率调度器
        self.actor_lr_scheduler = []
        for optimizer in self._actor_optimizers:
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

    def sample_action(self, mean, log_std, reparameterize=True):
        """
        依据高斯分布采样动作，并进行 tanh 映射，同时计算动作的对数概率。

        该函数实现：
        - SAC 中经典的重参数化技巧：a = tanh(μ + σ·ε)
        - tanh 后概率密度修正（包含雅可比项）
        - 返回动作与修正后的 log_prob

        Args:
            mean (Tensor): Actor 输出的动作均值。
            log_std (Tensor): Actor 输出的动作对数标准差。
            reparameterize (bool): 是否启用重参数化技巧。

        Returns:
            action (Tensor): [-1, 1] 区间的动作采样结果。
            log_prob (Tensor): 采样动作对应的对数概率。

        Example:
            >>> mean, log_std = actor(obs)
            >>> action, log_prob = trainer.sample_action(mean, log_std)
        """
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)

        if reparameterize:
            x_t = normal.rsample()  # 重参数化采样
        else:
            x_t = normal.sample()  # 普通采样

        # 动作映射到[-1, 1]区间
        action = torch.tanh(x_t)

        # 计算对数概率（考虑tanh变换的修正）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def train(self) -> Dict[str, float]:
        """
        执行一次多智能体 MASAC 的完整训练流程。

        完整训练步骤包括：

        1. **从经验回放中采样**（支持优先采样 PER）
        2. **为下一状态采样下一个动作**（由当前 Actor 产生）
        3. **更新双 Critic（TD 目标 + 熵项）**
        4. **更新 Actor（最大熵目标：max Q - α log π）**
        5. **更新温度参数 α（自动熵调节）**
        6. **软更新目标 Critic 网络**
        7. **更新学习率调度器**
        8. **在使用 PER 时更新优先级**

        Returns:
            Dict[str, float]: 包含 actor/critic/alpha 的损失与诊断指标。

        Example:
            >>> for step in range(10000):
            ...     loss_info = trainer.train()
            ...     print(loss_info["actor_loss0"])
        """

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

            # 优先级采样
            samples = self.replay_buffer.sample(
                self.args.batch_size, beta=beta, device=self.args.device
            )
            obs = samples["states"]
            action = samples["actions"]
            reward = samples["rewards"]
            next_obs = samples["next_states"]
            done = samples["dones"]
            weights = samples["weights"]
            indexes = samples["indexes"]
        else:
            # 普通采样
            obs, action, reward, next_obs, done, _ = self.replay_buffer.sample(
                self.args.batch_size, device=self.args.device
            )
            weights = torch.ones(self.args.batch_size, device=self.args.device)
            indexes = None

        # ----------------------------
        # 2. 计算目标动作和对数概率
        # ----------------------------
        next_union_actions = []
        next_log_probs = []
        for index in range(self._num_agents):
            # 从当前Actor采样下一个动作（SAC没有目标Actor）
            mean, log_std = self._actor[index](next_obs[index])
            next_action, next_log_prob = self.sample_action(
                mean, log_std, reparameterize=False
            )
            next_union_actions.append(next_action)
            next_log_probs.append(next_log_prob)

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
        # 3. 更新双Critic网络
        # ----------------------------
        self._critic_optimizers[0].zero_grad()
        self._critic_optimizers[1].zero_grad()

        # 计算目标Q值（包含熵项）
        target_q_values = []
        for index in range(self._num_agents):
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )
            next_critic_input = torch.cat(
                (next_union_obs, next_union_actions, agent_id), -1
            )

            with torch.no_grad():
                # 双目标Critic取最小值
                q1_target = self._target_critic1(next_critic_input)
                q2_target = self._target_critic2(next_critic_input)
                target_q = torch.min(q1_target, q2_target).detach()

                # 加入熵项：Q - α·logπ(a|s)
                target_q = (
                    target_q - self._alpha[index].detach() * next_log_probs[index]
                )

                # TD目标计算
                t_done = done[:, index].view(self.args.batch_size, 1)
                t_reward = reward[:, index].view(self.args.batch_size, 1)
                td_target = t_reward + self.args.gamma * (1 - t_done) * target_q
                target_q_values.append(td_target)

        # 计算Critic损失
        total_critic1_loss = 0.0
        total_critic2_loss = 0.0
        td_errors = []
        for index in range(self._num_agents):
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )
            critic_input = torch.cat((union_obs, union_actions_sampled, agent_id), -1)

            # 当前Q值
            q1 = self._critic1(critic_input)
            q2 = self._critic2(critic_input.clone())

            # 单个样本损失
            critic1_loss_per_sample = F.mse_loss(
                q1, target_q_values[index], reduction="none"
            )
            critic2_loss_per_sample = F.mse_loss(
                q2, target_q_values[index], reduction="none"
            )

            # 重要性采样加权
            weighted_critic1_loss = (
                weights.view(-1, 1) * critic1_loss_per_sample
            ).mean()
            weighted_critic2_loss = (
                weights.view(-1, 1) * critic2_loss_per_sample
            ).mean()

            total_critic1_loss += weighted_critic1_loss
            total_critic2_loss += weighted_critic2_loss

            # 记录TD误差
            td_error = torch.abs(q1 - target_q_values[index]).detach().squeeze(1)
            td_errors.append(td_error)

            # 记录损失
            loss.update(
                {
                    f"critic1_loss{index}": weighted_critic1_loss.item(),
                    f"critic2_loss{index}": weighted_critic2_loss.item(),
                }
            )

        # 反向传播
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

        # ----------------------------
        # 4. 更新Actor网络
        # ----------------------------
        for index in range(self._num_agents):
            self._actor_optimizers[index].zero_grad()

            # 重参数化采样动作
            mean, log_std = self._actor[index](obs[index])
            action, log_prob = self.sample_action(mean, log_std, reparameterize=True)

            # 构建联合动作（其他智能体动作阻断梯度）
            union_actions_for_actor = []
            for idx in range(self._num_agents):
                if idx == index:
                    union_actions_for_actor.append(action)
                else:
                    mean_other, log_std_other = self._actor[idx](obs[idx])
                    action_other, _ = self.sample_action(mean_other, log_std_other)
                    union_actions_for_actor.append(action_other.detach())

            union_actions_for_actor = torch.cat(union_actions_for_actor, dim=-1)

            # 计算Actor损失
            agent_id = (
                self._agent_ids[index].unsqueeze(0).expand(self.args.batch_size, -1)
            )
            critic_input = torch.cat((union_obs, union_actions_for_actor, agent_id), -1)

            q1 = self._critic1(critic_input)
            # 最大熵目标：max E[Q(s,a) - α·logπ(a|s)]
            actor_loss = -(q1 - self._alpha[index] * log_prob).mean()

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._actor[index].parameters(), max_norm=self.args.norm_grad
            )
            self._actor_optimizers[index].step()

            loss.update({f"actor_loss{index}": actor_loss.item()})

        # ----------------------------
        # 5. 更新温度参数α
        # ----------------------------
        for index in range(self._num_agents):
            self._alpha_optimizers[index].zero_grad()

            # 重新计算当前策略的对数概率
            mean, log_std = self._actor[index](obs[index])
            _, log_prob = self.sample_action(mean, log_std, reparameterize=False)

            # 温度损失：α·(-logπ - H_target)
            alpha_loss = (
                self._alpha[index] * (-log_prob - self._target_entropy[index])
            ).mean()
            alpha_loss.backward()
            self._alpha_optimizers[index].step()

            loss.update(
                {
                    f"alpha_loss{index}": alpha_loss.item(),
                    f"alpha{index}": self._alpha[index].item(),
                    f"entropy{index}": -log_prob.mean().item(),
                }
            )

        # ----------------------------
        # 6. 软更新目标网络
        # ----------------------------
        soft_update(self._target_critic1, self._critic1, self.args.tau)
        soft_update(self._target_critic2, self._critic2, self.args.tau)

        # 更新学习率
        for lr_scheduler in self.actor_lr_scheduler:
            lr_scheduler.step()
        self.critic1_lr_scheduler.step()
        self.critic2_lr_scheduler.step()
        loss["actor_lr"] = self.actor_lr_scheduler[0].get_lr()[0]
        loss["critic_lr"] = self.critic1_lr_scheduler.get_lr()[0]

        # ----------------------------
        # 7. 更新优先级
        # ----------------------------
        if self.args.use_priority:
            avg_td_errors = torch.mean(torch.stack(td_errors, dim=1), dim=1)
            self.replay_buffer.update_priorities(
                indexes=indexes, priorities=avg_td_errors.cpu().numpy()
            )
            loss["priority_beta"] = beta

        return loss

    def save_models(self, directory: str):
        """
        保存 MASAC 的全部网络参数与优化器状态。

        保存内容包括：
        - 每个智能体 Actor 网络参数
        - critic1、critic2 参数
        - 目标 critic1、critic2 参数
        - 所有 Actor/Critic/Alpha 优化器状态
        - 温度参数 alpha

        Args:
            directory (str): 模型保存路径，若不存在会自动创建。

        Example:
            >>> trainer.save_models("checkpoints/masac/")
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 保存网络参数
        actor_dict = {
            f"actor{idx}": self._actor[idx].state_dict()
            for idx in range(self._num_agents)
        }
        critic_dict = {
            "critic1": self._critic1.state_dict(),
            "critic2": self._critic2.state_dict(),
            "target_critic1": self._target_critic1.state_dict(),
            "target_critic2": self._target_critic2.state_dict(),
        }

        # 保存优化器参数
        actor_optim_dict = {
            f"actor_optim{idx}": self._actor_optimizers[idx].state_dict()
            for idx in range(self._num_agents)
        }
        critic_optim_dict = {
            "critic_optim0": self._critic_optimizers[0].state_dict(),
            "critic_optim1": self._critic_optimizers[1].state_dict(),
        }

        # 保存温度参数及优化器
        alpha_dict = {
            f"alpha{idx}": self._alpha[idx].item() for idx in range(self._num_agents)
        }
        alpha_optim_dict = {
            f"alpha_optim{idx}": self._alpha_optimizers[idx].state_dict()
            for idx in range(self._num_agents)
        }

        final_dict = {
            **actor_dict,
            **critic_dict,
            **actor_optim_dict,
            **critic_optim_dict,
            **alpha_dict,
            **alpha_optim_dict,
        }
        torch.save(final_dict, os.path.join(directory, "actor_critic.pth"))

    def load_models(self, directory: str):
        """
        加载模型文件并恢复：
        - Actor 参数与优化器
        - Critic 参数与优化器
        - 目标网络参数
        - 温度参数 alpha 与其优化器

        Args:
            directory (str): 模型目录路径，应包含 actor_critic.pth。

        Example:
            >>> trainer.load_models("checkpoints/masac/")
        """
        final_dict = torch.load(
            os.path.join(directory, "actor_critic.pth"),
            weights_only=False,
            map_location=self.args.device,
        )

        # 加载网络参数
        for idx in range(self._num_agents):
            self._actor[idx].load_state_dict(final_dict[f"actor{idx}"])
            if final_dict.get(f"actor_optim{idx}"):
                self._actor_optimizers[idx].load_state_dict(
                    final_dict[f"actor_optim{idx}"]
                )

        self._critic1.load_state_dict(final_dict["critic1"])
        self._critic2.load_state_dict(final_dict["critic2"])
        self._target_critic1.load_state_dict(final_dict["target_critic1"])
        self._target_critic2.load_state_dict(final_dict["target_critic2"])

        # 加载优化器
        if final_dict.get("critic_optim0"):
            self._critic_optimizers[0].load_state_dict(final_dict["critic_optim0"])
            self._critic_optimizers[1].load_state_dict(final_dict["critic_optim1"])

        # 加载温度参数及优化器
        for idx in range(self._num_agents):
            self._alpha[idx].data = torch.tensor(
                final_dict[f"alpha{idx}"], device=self.args.device
            )
            if final_dict.get(f"alpha_optim{idx}"):
                self._alpha_optimizers[idx].load_state_dict(
                    final_dict[f"alpha_optim{idx}"]
                )


class MASAC(Manager):
    """
    多智能体 Soft Actor-Critic（MASAC）算法主控制器。

    本类负责：
    - 多智能体 Actor 网络初始化（高斯策略）
    - 多智能体 Critic 网络与目标 Critic 网络构建
    - 温度参数 α 与目标熵的初始化
    - 优先或普通经验回放缓冲区初始化
    - 多智能体动作策略封装（GaussianActorStrategy）
    - 构建 MASACTrainer 用于执行梯度训练
    - 继承 Manager，实现与环境交互 + 采样 + 训练 三者逻辑整合

    MASAC 支持：
    - 多智能体独立 Actor
    - 双 Critic + 双目标 Critic（Twin Q）
    - Gaussian 动作分布（均值 + log_std）
    - 自动熵调节 α（每个智能体独立学习）
    - 标准经验回放 & 优先经验回放
    - 软更新
    - 与 MATD3 结构兼容（方便切换）

    Attributes:
        name_agents (List[str]): 智能体名称列表。
        num_agents (int): 智能体数量。
        storage: 回放缓冲区对象（PriorityReplayStorage 或 StandardReplayStorage）。
        actor_strategy: 多智能体动作策略（采样/选择动作）。
        trainer: MASACTrainer，包含所有训练步骤。

    Example:
        >>> masac = MASAC(
        ...     name_agents=["agent1", "agent2"],
        ...     obs_shapes=[24, 24],
        ...     global_state_shape=48,
        ...     action_shapes=[4, 4],
        ...     args=args,
        ... )
        >>> obs = env.reset()
        >>> actions = masac.act(obs)
        >>> masac.trainer.train()
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
        初始化多智能体 MASAC 算法的全部组件。

        主要完成以下流程：
        1. 创建经验回放缓冲区（支持 PER 和普通 Buffer）
        2. 构建多智能体 Actor（Gaussian 连续策略）
        3. 构建双 Critic 网络与对应的目标网络
        4. 初始化温度参数 α 与目标熵（每个 agent 各一份）
        5. 构建 Actor/Critic/Alpha 的优化器
        6. 构建多智能体动作策略（GaussianActorStrategy）
        7. 构建训练器 MASACTrainer
        8. 调用 Manager 完成最终组合

        Args:
            name_agents (List[str]):
                智能体名称列表，长度即为多智能体数量。
            obs_shapes (List[int]):
                每个智能体的观测维度。
            global_state_shape (int):
                全局状态维度（用于中心化 Critic）。
            action_shapes (List[int]):
                每个智能体的动作维度。
            args:
                参数集合，包含：
                - lr_actor, lr_critic, lr_alpha
                - buffer_size
                - hidden_size
                - n_obs, n_actions
                - use_priority
                - device 等
            action_continuous (bool):
                是否为连续动作空间，MASAC 默认使用连续动作。

        Example:
            >>> masac = MASAC(
            ...     name_agents=["uav1", "uav2"],
            ...     obs_shapes=[30, 30],
            ...     global_state_shape=60,
            ...     action_shapes=[2, 2],
            ...     args=args,
            ... )
            >>> print(masac.num_agents)
            2
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
                self.storage = StandardReplayStorage(
                    buffer_size=args.buffer_size,
                    obs_shapes=[(obs_shapes[i],) for i in range(self.num_agents)],
                    action_shapes=[(action_shapes[i],) for i in range(self.num_agents)],
                    num_agents=self.num_agents,
                    action_continuous=action_continuous,
                    args=args,
                )

        # 计算Critic输入维度
        obs_dim_sum = sum(args.n_obs)
        actions_dim_sum = sum(args.n_actions)
        agent_id_dim = self.num_agents  # one-hot维度
        critic_input_size = obs_dim_sum + actions_dim_sum + agent_id_dim

        # 初始化Actor网络（随机策略，输出均值和方差）
        actors = []
        for index in range(args.num_agents):
            actor = ContinuousActor(
                input_dim=args.n_obs[index],
                output_dim=args.n_actions[index],
                hidden_dim=args.hidden_size,
            ).to(args.device)
            actors.append(actor)

        # 初始化双Critic网络
        critic1 = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )
        critic2 = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )

        # 目标Critic网络
        target_critic1 = deepcopy(critic1)
        target_critic2 = deepcopy(critic2)

        # 初始化温度参数和目标熵
        target_entropy = [
            -action_shapes[i] for i in range(self.num_agents)
        ]  # 目标熵通常设为-动作维度
        alpha = [
            torch.tensor(1.0, device=args.device, requires_grad=True)
            for _ in range(self.num_agents)
        ]

        # 优化器统一在MASAC类中初始化（与MATD3保持一致）
        optimizers = {
            "actor": [Adam(actor.parameters(), lr=args.lr_actor) for actor in actors],
            "critic": [
                Adam(critic1.parameters(), lr=args.lr_critic),
                Adam(critic2.parameters(), lr=args.lr_critic),
            ],
            "alpha": [
                Adam([alpha[i]], lr=args.lr_alpha) for i in range(self.num_agents)
            ],
        }

        # 动作策略
        self.actor_strategy = GaussianActorStrategy(actors=actors, args=args)

        # 初始化Trainer（优化器通过参数传入）
        self.trainer = MASACTrainer(
            buffer=self.storage.replay_buffer if not self.args.just_eval else None,
            args=args,
            actors=actors,
            critics=(critic1, critic2),
            target_critics=(target_critic1, target_critic2),
            alpha=alpha,
            optimizers=optimizers,  # 传入优化器字典
            target_entropy=target_entropy,
        )

        super().__init__(
            actor_strategy=self.actor_strategy,
            storage_strategy=self.storage,
            trainer=self.trainer,
            args=self.args,
        )
