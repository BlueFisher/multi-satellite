import os
from copy import deepcopy
from typing import Dict

import torch
from torch.functional import F

from .ITrainer import ITrainer
from ..utils import soft_update


class MADDPGTrainer(ITrainer):
    """
    多智能体深度确定性策略梯度（MADDPG）训练器。

    该类实现了MADDPG算法的训练逻辑，包括策略网络和价值网络的更新，
    以及模型的保存与加载。

    Attributes:
        replay_buffer: 用于存储经验的回放缓冲区。
        args: 包含超参数和配置的命名空间对象。
        _actor: 策略网络列表，每个智能体对应一个网络。
        _critic: 价值网络列表，每个智能体对应一个网络。
        _target_actor: 目标策略网络列表。
        _target_critic: 目标价值网络列表。
        _actor_optimizer: 策略网络的优化器。
        _critic_optimizer: 价值网络的优化器。
        _update_counter: 更新计数器。
    """

    def __init__(
        self,
        buffer,
        args,
        actors,
        critics,
        optimizers,
        target_actors,
        target_critics,
    ):
        """
        初始化MADDPG训练器。

        Args:
            buffer: 用于存储经验的回放缓冲区。
            args: 包含超参数和配置的命名空间对象。
            actors: 策略网络列表。
            critics: 价值网络列表。
            optimizers: 包含策略网络和价值网络优化器的字典。
            target_actors: 目标策略网络列表。
            target_critics: 目标价值网络列表。

        Example:
            ```python
            trainer = MADDPGTrainer(buffer, args, actors, critics, optimizers, target_actors, target_critics)
            ```
        """
        super().__init__(buffer, args)
        self._actor = actors
        self._critic = critics
        self._target_actor = target_actors
        self._target_critic = target_critics
        self._actor_optimizer = optimizers["actor"]
        self._critic_optimizer = optimizers["critic"]
        self._update_counter = 0

    def train(self) -> Dict[str, float]:
        """
        执行一次训练迭代。

        Returns:
            Dict[str, float]: 包含每个智能体的策略网络和价值网络的损失。

        Example:
            ```python
            loss = trainer.train()
            print(loss)
            ```
        """

        # 从buffer采样
        obs, action, reward, next_obs, done, extra = self.replay_buffer.sample(
            self.args.batch_size, device=self.args.device
        )

        next_union_actions = []
        for index in range(self.args.num_agents):
            next_union_actions.append(self._target_actor[index](next_obs[index]))
        next_union_actions = torch.cat(next_union_actions, dim=1)
        union_obs = torch.cat([obs[idx] for idx in range(self.args.num_agents)], dim=1)
        next_union_obs = torch.cat(
            [next_obs[idx] for idx in range(self.args.num_agents)], dim=1
        )

        loss = {}
        for index in range(self.args.num_agents):
            # 重新计算 union action 避免梯度混乱
            # union_actions = []
            # for idx in range(self.args.num_agents):
            #     actor_output = self._actor[idx](obs[idx])
            #     union_actions.append(
            #         actor_output if idx == index else actor_output.detach()
            #     )
            # union_actions = torch.cat(union_actions, dim=-1)
            union_actions = []
            for idx in range(self.args.num_agents):
                if idx == index:
                    actor_output = self._actor[idx](obs[idx])  # 训练 agent-i
                else:
                    actor_output = action[idx]
                union_actions.append(actor_output)

            union_actions = torch.cat(union_actions, dim=-1)

            critic_input = torch.cat((union_obs, union_actions), -1)

            self._actor_optimizer[index].zero_grad()
            actor_loss = -self._critic[index](critic_input).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(
                self._actor[index].parameters(), clip_value=self.args.norm_grad
            )

            self._actor_optimizer[index].step()

            # actor的梯度不能影响 critic
            critic_input = torch.cat((union_obs, union_actions.detach()), -1)
            # critic_input = torch.cat((obs[index], union_actions.detach()), -1)
            q_values = self._critic[index](critic_input)

            next_critic_input = torch.cat((next_union_obs, next_union_actions), -1)
            next_q_values = self._target_critic[index](next_critic_input).detach()

            t_done = done[:, index].view(self.args.batch_size, 1)
            t_reward = reward[:, index].view(self.args.batch_size, 1)
            td_target = t_reward + self.args.gamma * (1 - t_done) * next_q_values

            self._critic_optimizer[index].zero_grad()
            critic_loss = F.mse_loss(q_values, td_target)
            critic_loss.backward()

            torch.nn.utils.clip_grad_value_(
                self._critic[index].parameters(), clip_value=self.args.norm_grad
            )

            self._critic_optimizer[index].step()
            soft_update(self._target_actor[index], self._actor[index], self.args.tau)
            soft_update(self._target_critic[index], self._critic[index], self.args.tau)
            loss.update(
                {
                    "critic_loss" + str(index): critic_loss.item(),
                    "actor_loss" + str(index): actor_loss.item(),
                }
            )
        return loss

    def save_models(self, directory: str):
        """
        保存模型到指定目录。

        Args:
            directory (str): 保存模型的目录路径。

        Example:
            ```python
            trainer.save_models("./models")
            ```
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        actor_dict = {
            f"actor{idx}": self._actor[idx].state_dict()
            for idx in range(self.args.num_agents)
        }

        critic_dict = {
            f"critic{idx}": self._critic[idx].state_dict()
            for idx in range(self.args.num_agents)
        }

        final_dict = {}
        final_dict.update(actor_dict)
        final_dict.update(critic_dict)
        torch.save(final_dict, os.path.join(directory, "actor_critic.pth"))

    def load_models(self, directory: str):
        """
        从指定目录加载模型。

        Args:
            directory (str): 模型文件所在的目录路径。

        Example:
            ```python
            trainer.load_models("./models")
            ```
        """

        final_dict = torch.load(
            os.path.join(directory, "actor_critic.pth"),
            weights_only=True,
            map_location=self.args.device,
        )

        [
            self._actor[idx].load_state_dict(final_dict[f"actor{idx}"])
            for idx in range(self.args.num_agents)
        ]

        [
            self._critic[idx].load_state_dict(final_dict[f"critic{idx}"])
            for idx in range(self.args.num_agents)
        ]

        self._target_actor = deepcopy(self._actor)
        self._target_critic = deepcopy(self._critic)
