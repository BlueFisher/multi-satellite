import os
from copy import deepcopy
from typing import Dict

import torch
from torch.optim import Adam

from .Action.SharedMessageDeterministicActorStrategy import (
    SharedMessageDeterministicActorStrategy,
)
from .Manager import Manager
from .Network.Actor.SharedMessageDeterministicActor import (
    SharedMessageDeterministicActor,
)
from .Network.Critic.QValueCritic import QValueCritic
from .StorageStrategy.PriorityReplayStorage import PriorityReplayStorage
from .StorageStrategy.StandardReplayStorage import StandardReplayStorage
from .Trainer.ITrainer import ITrainer
from .utils import soft_update


# ==================== 修改后的 MATD3 训练器 ====================
class SharedMessageMATD3Trainer(ITrainer):
    def __init__(
        self,
        buffer,
        args,
        actor,  # 单个共享 Actor
        critics,  # (critic1, critic2)
        optimizer,  # {"actor": opt, "critic": [opt1, opt2]}
        target_actor,  # 单个目标 Actor
        target_critics,  # (target_critic1, target_critic2)
    ):
        super().__init__(buffer, args)
        self._actor = actor
        self._critic1, self._critic2 = critics
        self._target_actor = target_actor
        self._target_critic1, self._target_critic2 = target_critics
        self._actor_optimizer = optimizer["actor"]
        self._critic_optimizers = optimizer["critic"]
        self._update_counter = 0
        self._num_agents = args.num_agents

        # 学习率调度器（单个）
        self.actor_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._actor_optimizer, 1000 * args.num_envs, 0.99
        )
        self.critic1_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._critic_optimizers[0], 1000 * args.num_envs, 0.99
        )
        self.critic2_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._critic_optimizers[1], 1000 * args.num_envs, 0.99
        )

        self._agent_ids = torch.eye(self._num_agents).to(args.device)
        self.global_step = 0

    def train(self) -> Dict[str, float]:
        # 1. 采样
        if self.args.use_priority:
            self.global_step += 1
            beta = min(
                self.args.priority_beta_end,
                self.args.priority_beta_start
                + (self.args.priority_beta_end - self.args.priority_beta_start)
                * self.global_step
                / self.args.priority_beta_frames,
            )
            samples = self.replay_buffer.sample(
                self.args.batch_size, beta=beta, device=self.args.device
            )
            obs = samples["states"]  # list of [batch, obs_dim]
            action = samples["actions"]  # list of [batch, act_dim]
            reward = samples["rewards"]  # [batch, num_agents]
            next_obs = samples["next_states"]  # list of [batch, obs_dim]
            done = samples["dones"]  # [batch, num_agents]
            weights = samples["weights"]  # [batch]
            indexes = samples["indexes"]
        else:
            obs, action, reward, next_obs, done, _ = self.replay_buffer.sample(
                self.args.batch_size, device=self.args.device
            )
            weights = torch.ones(self.args.batch_size, device=self.args.device)
            indexes = None

        batch_size = self.args.batch_size
        num_agents = self._num_agents
        obs_dim = obs[0].shape[-1]
        act_dim = action[0].shape[-1]

        # 堆叠观测和动作
        obs_stacked = torch.stack(obs, dim=1)  # [B, num, obs_dim]
        actions_stacked = torch.stack(action, dim=1)  # [B, num, act_dim]
        next_obs_stacked = torch.stack(next_obs, dim=1)  # [B, num, obs_dim]

        # 联合观测/动作（拼接所有智能体）
        union_obs = obs_stacked.reshape(batch_size, -1)  # [B, num*obs_dim]
        union_actions_sampled = actions_stacked.reshape(
            batch_size, -1
        )  # [B, num*act_dim]
        next_union_obs = next_obs_stacked.reshape(batch_size, -1)
        actor_input_agent_ids = self._agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        agent_ids = actor_input_agent_ids.reshape(batch_size * num_agents, -1)

        loss = {}

        # ========== 2. 目标动作计算（加噪声平滑） ==========
        with torch.no_grad():
            target_actor_input = (
                torch.cat([next_obs_stacked, actor_input_agent_ids], dim=-1)
                if self.args.use_agent_id
                else next_obs_stacked
            )
            next_all_actions, _ = self._target_actor(
                target_actor_input,
            )  # [B, num, act_dim]
            noise = torch.clamp(
                torch.randn_like(next_all_actions) * self.args.policy_noise,
                -self.args.noise_clip,
                self.args.noise_clip,
            )
            next_all_actions = torch.clamp(
                next_all_actions + noise, self.args.action_min, self.args.action_max
            )
            next_union_actions = next_all_actions.reshape(
                batch_size, -1
            )  # [B, num*act_dim]

        # ========== 3. 计算目标 Q 值（双 Critic 取 min） ==========
        # 扩展联合观测/动作至 [B*num, ...]
        union_obs_exp = (
            union_obs.unsqueeze(1)
            .expand(batch_size, num_agents, -1)
            .reshape(batch_size * num_agents, -1)
        )
        next_union_obs_exp = (
            next_union_obs.unsqueeze(1)
            .expand(batch_size, num_agents, -1)
            .reshape(batch_size * num_agents, -1)
        )
        next_union_act_exp = (
            next_union_actions.unsqueeze(1)
            .expand(batch_size, num_agents, -1)
            .reshape(batch_size * num_agents, -1)
        )

        critic_input_target = torch.cat(
            [next_union_obs_exp, next_union_act_exp, agent_ids], dim=-1
        )
        q1_target_all = self._target_critic1(critic_input_target).view(
            batch_size, num_agents
        )
        q2_target_all = self._target_critic2(critic_input_target).view(
            batch_size, num_agents
        )
        target_q_all = torch.min(q1_target_all, q2_target_all).detach()  # [B, num]

        td_target = reward + self.args.gamma * (1 - done) * target_q_all  # [B, num]

        # ========== 4. 更新 Critic ==========
        # 当前 Critic 输入（使用采样的联合动作）
        critic_input_cur = torch.cat(
            [
                union_obs_exp,
                union_actions_sampled.unsqueeze(1)
                .expand(batch_size, num_agents, -1)
                .reshape(batch_size * num_agents, -1),
                agent_ids,
            ],
            dim=-1,
        )
        q1_all = self._critic1(critic_input_cur).view(batch_size, num_agents)
        q2_all = self._critic2(critic_input_cur).view(batch_size, num_agents)

        # 损失计算（加权）
        weights_exp = weights.view(batch_size, 1)  # [B, 1]
        critic1_loss = (weights_exp * (q1_all - td_target.detach()) ** 2).mean()
        critic2_loss = (weights_exp * (q2_all - td_target.detach()) ** 2).mean()

        # 记录每个智能体的损失
        for i in range(num_agents):
            loss[f"critic1_loss{i}"] = (
                (weights_exp * (q1_all[:, i] - td_target[:, i].detach()) ** 2)
                .mean()
                .item()
            )
            loss[f"critic2_loss{i}"] = (
                (weights_exp * (q2_all[:, i] - td_target[:, i].detach()) ** 2)
                .mean()
                .item()
            )

        self._critic_optimizers[0].zero_grad()
        self._critic_optimizers[1].zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self.args.norm_grad)
        torch.nn.utils.clip_grad_norm_(self._critic2.parameters(), self.args.norm_grad)
        self._critic_optimizers[0].step()
        self._critic_optimizers[1].step()

        # 每个样本的平均 TD 误差（用于优先级）
        td_errors = torch.abs(q1_all - td_target).detach().mean(dim=1)  # [B]

        actor_input = (
            torch.cat([obs_stacked, actor_input_agent_ids], dim=-1)
            if self.args.use_agent_id
            else obs_stacked
        )

        # 当前 Actor 输出所有动作
        all_actions, extra = self._actor(actor_input)  # [B, num, act_dim]
        union_actions_actor = all_actions.reshape(batch_size, -1)

        # 构建 Critic 输入（使用 all_actions）
        union_obs_exp_actor = (
            union_obs.unsqueeze(1)
            .expand(batch_size, num_agents, -1)
            .reshape(batch_size * num_agents, -1)
        )

        union_act_exp_actor = (
            union_actions_actor.unsqueeze(1)
            .expand(batch_size, num_agents, -1)
            .reshape(batch_size * num_agents, -1)
        )
        critic_input_actor = torch.cat(
            [union_obs_exp_actor, union_act_exp_actor, agent_ids], dim=-1
        )

        q_actor = self._critic1(critic_input_actor).view(batch_size, num_agents)
        actor_loss = -q_actor.mean()

        if self.args.use_ndq:
            mu = extra["mu"]
            log_std = extra["log_std"]
            decode_action = extra["decode_action"]

            dist = torch.distributions.Normal(mu, torch.exp(log_std))

            kl_loss = (
                torch.distributions.kl.kl_divergence(
                    dist,
                    torch.distributions.Normal(
                        torch.zeros_like(mu), torch.ones_like(log_std)
                    ),
                )
                .sum(-1)
                .mean()
            )

            action_loss = torch.nn.functional.mse_loss(decode_action, all_actions)

            actor_loss = actor_loss + self.args.kl_coef * kl_loss + action_loss

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self.args.norm_grad)
        self._actor_optimizer.step()

        loss["actor_loss"] = actor_loss.item()

        # 软更新目标网络
        soft_update(self._target_critic1, self._critic1, self.args.tau)
        soft_update(self._target_critic2, self._critic2, self.args.tau)
        soft_update(self._target_actor, self._actor, self.args.tau)

        # 学习率步进
        self.actor_lr_scheduler.step()
        loss["actor_lr"] = self.actor_lr_scheduler.get_last_lr()[0]
        # 仅步进 Critic 学习率
        self.critic1_lr_scheduler.step()
        self.critic2_lr_scheduler.step()
        loss["critic1_lr"] = self.critic1_lr_scheduler.get_last_lr()[0]
        loss["critic2_lr"] = self.critic2_lr_scheduler.get_last_lr()[0]
        # ========== 6. 更新优先级 ==========
        if self.args.use_priority and indexes is not None:
            self.replay_buffer.update_priorities(
                indexes=indexes,
                priorities=td_errors.cpu().numpy(),
            )
            loss["priority_beta"] = beta

        return loss

    def save_models(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "critic1": self._critic1.state_dict(),
                "critic2": self._critic2.state_dict(),
                "actor_optim": self._actor_optimizer.state_dict(),
                "critic_optim0": self._critic_optimizers[0].state_dict(),
                "critic_optim1": self._critic_optimizers[1].state_dict(),
            },
            os.path.join(directory, "actor_critic.pth"),
        )

    def load_models(self, directory: str):
        data = torch.load(
            os.path.join(directory, "actor_critic.pth"), map_location=self.args.device
        )
        self._actor.load_state_dict(data["actor"])
        self._critic1.load_state_dict(data["critic1"])
        self._critic2.load_state_dict(data["critic2"])
        self._actor_optimizer.load_state_dict(data["actor_optim"])
        self._critic_optimizers[0].load_state_dict(data["critic_optim0"])
        self._critic_optimizers[1].load_state_dict(data["critic_optim1"])
        # 更新目标网络
        # self._target_actor = deepcopy(self._actor)
        # self._target_critic1 = deepcopy(self._critic1)
        # self._target_critic2 = deepcopy(self._critic2)


# ==================== 修改后的 MATD3 管理器 ====================
class SharedMessageMATD3(Manager):
    def __init__(
        self,
        name_agents: list[str],
        obs_shapes: list[int],
        global_state_shape: int,
        action_shapes: list[int],
        args,
        action_continuous: bool = True,
    ):

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

        # 假设所有智能体观测/动作维度相同（同质智能体）
        obs_dim = obs_shapes[0]
        action_dim = action_shapes[0]
        assert all(
            od == obs_dim for od in obs_shapes
        ), "All agents must have same obs dimension for shared actor"
        assert all(
            ad == action_dim for ad in action_shapes
        ), "All agents must have same action dimension for shared actor"

        actor_obs_dim = obs_dim if not args.use_agent_id else obs_dim + self.num_agents
        # 共享 Actor
        actor = SharedMessageDeterministicActor(
            obs_dim=actor_obs_dim,
            action_dim=action_dim,
            num_agents=self.num_agents,
            hidden_dim=args.hidden_size,
            args=args,
        ).to(args.device)

        # Critic 输入维度：所有观测 + 所有动作 + agent_id
        obs_dim_sum = sum(args.n_obs)
        actions_dim_sum = sum(args.n_actions)
        agent_id_dim = self.num_agents
        critic_input_size = obs_dim_sum + actions_dim_sum + agent_id_dim

        # 双 Critic
        critic1 = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )
        critic2 = QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
            args.device
        )

        # 目标网络
        target_actor = SharedMessageDeterministicActor(
            obs_dim=actor_obs_dim,
            action_dim=action_dim,
            num_agents=self.num_agents,
            hidden_dim=args.hidden_size,
            args=args,
        ).to(args.device)
        target_actor.load_state_dict(actor.state_dict())

        target_critic1 = deepcopy(critic1)
        target_critic2 = deepcopy(critic2)

        # 动作策略（需对应修改为接收单个共享 Actor）
        self.actor_strategy = SharedMessageDeterministicActorStrategy(
            actor=actor,
            args=args,
            action_continuous=action_continuous,
            use_agent_id=args.use_agent_id,
        )

        # 优化器
        optimizers = {
            "actor": Adam(actor.parameters(), lr=args.lr_actor),
            "critic": [
                Adam(critic1.parameters(), lr=args.lr_critic),
                Adam(critic2.parameters(), lr=args.lr_critic),
            ],
        }

        # 训练器
        self.trainer = SharedMessageMATD3Trainer(
            buffer=self.storage.replay_buffer if not self.args.just_eval else None,
            args=args,
            actor=actor,
            critics=(critic1, critic2),
            optimizer=optimizers,
            target_actor=target_actor,
            target_critics=(target_critic1, target_critic2),
        )

        super().__init__(
            actor_strategy=self.actor_strategy,
            storage_strategy=self.storage,
            trainer=self.trainer,
            args=self.args,
        )
