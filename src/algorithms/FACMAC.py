import os

import torch
from torch.optim import Adam
from typing import Dict
from copy import deepcopy

from .Action.RNNDeterministicActorStrategy import RNNDeterministicActorStrategy

from .Manager import Manager

from .Network.Actor.RNNDeterministicActor import RNNDeterministicActor
from .Network.Critic.QValueCritic import QValueCritic
from .StorageStrategy.EpisodeReplayStorage import EpisodeReplayStorage
from .Trainer.ITrainer import ITrainer
from .utils import soft_update


class FACMACTrainer(ITrainer):
    def __init__(
        self,
        buffer,
        args,
        actor,
        critics,
        mixers,  # (mixer1, mixer2)
        optimizer,  # {"actor": opt, "critic": [opt1, opt2]}
        target_actor,
        target_critics,
        target_mixers,  # (target_mixer1, target_mixer2)
    ):
        super().__init__(buffer, args)
        self._actor = actor
        self._critic1, self._critic2 = critics
        self._target_actor = target_actor
        self._target_critic1, self._target_critic2 = target_critics
        self._actor_optimizer = optimizer["actor"]
        self._critic_optimizers = optimizer["critic"]
        self._num_agents = args.num_agents
        self._use_agent_id = getattr(args, "use_agent_id", False)

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

        self.mixer1, self.mixer2 = mixers
        self._target_mixer1, self._target_mixer2 = target_mixers
        for p in self._target_mixer1.parameters():
            p.requires_grad = False
        for p in self._target_mixer2.parameters():
            p.requires_grad = False

    def train(self) -> Dict[str, float]:
        # ========== 1. 采样（保留时间维度） ==========
        samples = self.replay_buffer.sample(
            self.args.batch_size, device=self.args.device
        )
        obs_list, actions_list, rewards, next_obs_list, dones, extra_data = samples

        batch_size, max_T, num_agents = rewards.shape
        device = self.args.device

        filled = extra_data["filled"]
        valid_mask = filled[..., 0].bool()  # (B, T)

        use_states = self.replay_buffer.use_states
        if use_states:
            states = extra_data["states"]  # (B, T, state_dim)
            next_states = extra_data["next_states"]
        else:
            states = torch.cat(obs_list, dim=-1)
            next_states = torch.cat(next_obs_list, dim=-1)

        # 观测和动作堆叠: (B, T, N, obs_dim/act_dim)
        obs_stacked = torch.stack(obs_list, dim=2)  # (B, T, N, obs_dim)
        actions_stacked = torch.stack(actions_list, dim=2)  # (B, T, N, act_dim)
        next_obs_stacked = torch.stack(next_obs_list, dim=2)

        # ========== 2. 目标动作（正确 reshape 适配 Actor） ==========
        with torch.no_grad():
            # 准备目标 Actor 输入: (B, T, N, input_dim)
            if self._use_agent_id:
                # 扩展 agent_ids 到 (1, 1, N, N) 然后广播到 (B, T, N, N)
                agent_ids = self._agent_ids.view(1, 1, num_agents, -1)
                target_actor_input_4d = torch.cat(
                    [next_obs_stacked, agent_ids.expand(batch_size, max_T, -1, -1)],
                    dim=-1,
                )
            else:
                target_actor_input_4d = next_obs_stacked
            # 重塑为 (B*N, T, input_dim) 以满足 RNN 要求
            B, T, N = batch_size, max_T, num_agents
            input_dim = target_actor_input_4d.shape[-1]
            target_actor_input_flat = target_actor_input_4d.permute(0, 2, 1, 3).reshape(
                B * N, T, input_dim
            )
            # 目标 Actor 前向
            next_all_actions_flat, _ = self._target_actor(target_actor_input_flat)
            # 重塑回 (B, T, N, act_dim)
            act_dim = next_all_actions_flat.shape[-1]
            next_all_actions = next_all_actions_flat.reshape(B, N, T, act_dim).permute(
                0, 2, 1, 3
            )

            # 添加噪声
            noise = torch.clamp(
                torch.randn_like(next_all_actions) * self.args.policy_noise,
                -self.args.noise_clip,
                self.args.noise_clip,
            )
            next_all_actions = torch.clamp(
                next_all_actions + noise, self.args.action_min, self.args.action_max
            )

        # ========== 3. 目标 Q（使用双目标 Mixer） ==========
        with torch.no_grad():
            # 联合观测和动作: (B*T, N*obs_dim) 和 (B*T, N*act_dim)
            union_next_obs = next_obs_stacked.view(B * T, N, -1).reshape(B * T, -1)
            union_next_actions = next_all_actions.reshape(B * T, N, -1).reshape(
                B * T, -1
            )
            # 扩展为每个智能体一份: (B*T*N, ...)
            union_next_obs_exp = (
                union_next_obs.unsqueeze(1).expand(B * T, N, -1).reshape(B * T * N, -1)
            )
            union_next_act_exp = (
                union_next_actions.unsqueeze(1)
                .expand(B * T, N, -1)
                .reshape(B * T * N, -1)
            )
            agent_ids_exp = (
                self._agent_ids.view(1, 1, N, -1)
                .expand(B, T, N, -1)
                .reshape(B * T * N, -1)
            )

            critic_input_target = torch.cat(
                [union_next_obs_exp, union_next_act_exp, agent_ids_exp], dim=-1
            )
            q1_target_all = self._target_critic1(critic_input_target).view(
                B * T, N
            )  # (B*T, N)
            q1_target = q1_target_all.view(B, T, N)
            q2_target_all = self._target_critic2(critic_input_target).view(B * T, N)
            q2_target = q2_target_all.view(B, T, N)

            total_target_q1 = self._target_mixer1(q1_target, next_states).squeeze(
                -1
            )  # (B, T)
            total_target_q2 = self._target_mixer2(q2_target, next_states).squeeze(-1)
            total_target_q = torch.min(total_target_q1, total_target_q2)

            total_reward = rewards.sum(dim=-1)  # (B, T)
            global_done = dones.any(dim=-1).float()  # (B, T)
            target_total_q = (
                total_reward + self.args.gamma * (1 - global_done) * total_target_q
            )

        # ========== 4. 更新 Critic 和 Mixer ==========
        # 当前 Critic 输入（使用采样动作）
        union_obs = obs_stacked.view(B * T, N, -1).reshape(B * T, -1)
        union_actions = actions_stacked.view(B * T, N, -1).reshape(B * T, -1)
        union_obs_exp = (
            union_obs.unsqueeze(1).expand(B * T, N, -1).reshape(B * T * N, -1)
        )
        union_act_exp = (
            union_actions.unsqueeze(1).expand(B * T, N, -1).reshape(B * T * N, -1)
        )

        critic_input_cur = torch.cat(
            [union_obs_exp, union_act_exp, agent_ids_exp], dim=-1
        )
        q1_all = self._critic1(critic_input_cur).view(B * T, N)  # (B*T, N)
        q1 = q1_all.view(B, T, N)
        q2_all = self._critic2(critic_input_cur).view(B * T, N)
        q2 = q2_all.view(B, T, N)

        total_q1 = self.mixer1(q1, states).squeeze(-1)  # (B, T)
        total_q2 = self.mixer2(q2, states).squeeze(-1)

        mask = valid_mask.float()
        critic1_loss = (mask * (total_q1 - target_total_q) ** 2).sum() / (
            mask.sum() + 1e-8
        )
        critic2_loss = (mask * (total_q2 - target_total_q) ** 2).sum() / (
            mask.sum() + 1e-8
        )

        loss_dict = {}
        for i in range(N):
            loss_dict[f"critic1_loss_agent{i}"] = (
                mask * (q1[:, :, i] - q1_target[:, :, i]) ** 2
            ).sum() / (mask.sum() + 1e-8).item()
            loss_dict[f"critic2_loss_agent{i}"] = (
                mask * (q2[:, :, i] - q2_target[:, :, i]) ** 2
            ).sum() / (mask.sum() + 1e-8).item()

        self._critic_optimizers[0].zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self.args.norm_grad)
        torch.nn.utils.clip_grad_norm_(self.mixer1.parameters(), self.args.norm_grad)
        self._critic_optimizers[0].step()

        self._critic_optimizers[1].zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic2.parameters(), self.args.norm_grad)
        torch.nn.utils.clip_grad_norm_(self.mixer2.parameters(), self.args.norm_grad)
        self._critic_optimizers[1].step()

        # ========== 5. 更新 Actor ==========
        # 准备 Actor 输入
        if self._use_agent_id:
            agent_ids_4d = self._agent_ids.view(1, 1, N, -1).expand(B, T, N, -1)
            actor_input_4d = torch.cat([obs_stacked, agent_ids_4d], dim=-1)
        else:
            actor_input_4d = obs_stacked
        # 重塑为 (B*N, T, input_dim)
        input_dim_actor = actor_input_4d.shape[-1]
        actor_input_flat = actor_input_4d.permute(0, 2, 1, 3).reshape(
            B * N, T, input_dim_actor
        )
        # Actor 前向
        all_actions_flat, _ = self._actor(actor_input_flat)
        # 重塑回 (B, T, N, act_dim)
        all_actions = all_actions_flat.reshape(B, N, T, act_dim).permute(
            0, 2, 1, 3
        )  # (B, T, N, act_dim)

        # 计算当前 Q（使用第一个 Critic 和 Mixer）
        union_actions_actor = all_actions.reshape(B * T, N, -1).reshape(B * T, -1)
        union_act_actor_exp = (
            union_actions_actor.unsqueeze(1).expand(B * T, N, -1).reshape(B * T * N, -1)
        )
        critic_input_actor = torch.cat(
            [union_obs_exp, union_act_actor_exp, agent_ids_exp], dim=-1
        )
        q_actor_individual = self._critic1(critic_input_actor).view(B * T, N)
        q_actor = q_actor_individual.view(B, T, N)
        total_q_actor = self.mixer1(q_actor, states).squeeze(-1)  # (B, T)

        actor_loss = -(mask * total_q_actor).sum() / (mask.sum() + 1e-8)

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self.args.norm_grad)
        self._actor_optimizer.step()

        loss_dict["actor_loss"] = actor_loss.item()
        loss_dict["critic1_loss"] = critic1_loss.item()
        loss_dict["critic2_loss"] = critic2_loss.item()
        loss_dict["total_q1_mean"] = (mask * total_q1).sum() / (
            mask.sum() + 1e-8
        ).item()
        loss_dict["total_q2_mean"] = (mask * total_q2).sum() / (
            mask.sum() + 1e-8
        ).item()

        # ========== 6. 软更新 ==========
        soft_update(self._target_critic1, self._critic1, self.args.tau)
        soft_update(self._target_critic2, self._critic2, self.args.tau)
        soft_update(self._target_actor, self._actor, self.args.tau)
        soft_update(self._target_mixer1, self.mixer1, self.args.tau)
        soft_update(self._target_mixer2, self.mixer2, self.args.tau)

        self.actor_lr_scheduler.step()
        self.critic1_lr_scheduler.step()
        self.critic2_lr_scheduler.step()
        loss_dict["actor_lr"] = self.actor_lr_scheduler.get_last_lr()[0]
        loss_dict["critic1_lr"] = self.critic1_lr_scheduler.get_last_lr()[0]
        loss_dict["critic2_lr"] = self.critic2_lr_scheduler.get_last_lr()[0]

        return loss_dict

    def save_models(self, directory: str):
        import os

        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(
            {
                "actor": self._actor.state_dict(),
                "critic1": self._critic1.state_dict(),
                "critic2": self._critic2.state_dict(),
                "mixer1": self.mixer1.state_dict(),
                "mixer2": self.mixer2.state_dict(),
                "actor_optim": self._actor_optimizer.state_dict(),
                "critic_optim0": self._critic_optimizers[0].state_dict(),
                "critic_optim1": self._critic_optimizers[1].state_dict(),
            },
            os.path.join(directory, "actor_critic_mixer.pth"),
        )

    def load_models(self, directory: str):
        data = torch.load(
            os.path.join(directory, "actor_critic_mixer.pth"),
            map_location=self.args.device,
        )
        self._actor.load_state_dict(data["actor"])
        self._critic1.load_state_dict(data["critic1"])
        self._critic2.load_state_dict(data["critic2"])
        self.mixer1.load_state_dict(data["mixer1"])
        self.mixer2.load_state_dict(data["mixer2"])
        self._actor_optimizer.load_state_dict(data["actor_optim"])
        self._critic_optimizers[0].load_state_dict(data["critic_optim0"])
        self._critic_optimizers[1].load_state_dict(data["critic_optim1"])

        self._target_actor = deepcopy(self._actor)
        self._target_critic1 = deepcopy(self._critic1)
        self._target_critic2 = deepcopy(self._critic2)
        self._target_mixer1 = deepcopy(self.mixer1)
        self._target_mixer2 = deepcopy(self.mixer2)


class FACMAC(Manager):
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
            self.storage = EpisodeReplayStorage(
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
        actor = RNNDeterministicActor(
            input_size=actor_obs_dim,
            output_size=action_dim,
            hidden_size=args.hidden_size,
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
        target_actor = RNNDeterministicActor(
            input_size=actor_obs_dim,
            output_size=action_dim,
            hidden_size=args.hidden_size,
        ).to(args.device)
        target_actor.load_state_dict(actor.state_dict())

        target_critic1 = deepcopy(critic1)
        target_critic2 = deepcopy(critic2)

        mixer = args.mixer
        # 创建 Mixer 及目标 Mixer
        if mixer == "qmixer":
            from .Network.Mixer.QMixer import QMixer

            mixers = [
                QMixer(
                    n_agents=args.num_agents,
                    state_dim=obs_dim * args.num_agents,
                    hyper_dim=args.hyper_dim,
                    mixer_dim=args.mixer_dim,
                ).to(args.device)
                for _ in range(2)
            ]

        elif mixer == "vdn":
            from .Network.Mixer.VNDMixer import VDNMixer

            mixers = [
                VDNMixer(
                    n_agents=args.num_agents,
                    state_dim=obs_dim * args.num_agents,
                    hyper_dim=args.hyper_dim,
                ).to(args.device)
                for _ in range(2)
            ]
        else:
            raise NotImplementedError(f"Unknown mixer: {mixer}")

        mixer1 = mixers[0]
        mixer2 = mixers[1]

        target_mixer1 = deepcopy(mixer1)
        target_mixer2 = deepcopy(mixer2)

        # 动作策略（需对应修改为接收单个共享 Actor）
        self.actor_strategy: RNNDeterministicActorStrategy = (
            RNNDeterministicActorStrategy(
                actor=actor,
                args=args,
                action_continuous=action_continuous,
                use_agent_id=args.use_agent_id,
            )
        )

        # 优化器
        optimizers = {
            "actor": Adam(actor.parameters(), lr=args.lr_actor),
            "critic": [
                Adam(
                    list(critic1.parameters()) + list(mixer1.parameters()),
                    lr=args.lr_critic,
                ),
                Adam(
                    list(critic2.parameters()) + list(mixer2.parameters()),
                    lr=args.lr_critic,
                ),
            ],
        }

        # 训练器
        self.trainer = FACMACTrainer(
            buffer=self.storage.replay_buffer if not self.args.just_eval else None,
            args=args,
            actor=actor,
            critics=(critic1, critic2),
            optimizer=optimizers,
            target_actor=target_actor,
            target_critics=(target_critic1, target_critic2),
            mixers=(mixer1, mixer2),
            target_mixers=(target_mixer1, target_mixer2),
        )

        super().__init__(
            actor_strategy=self.actor_strategy,
            storage_strategy=self.storage,
            trainer=self.trainer,
            args=self.args,
        )

    def before_episode(self, env_idx: int):
        self.actor_strategy.reset_hidden_state(env_idx)
