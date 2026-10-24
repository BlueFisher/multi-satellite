from argparse import Namespace
from typing import List, Union, Dict

import numpy as np
import torch

from .IActorStrategy import IActorStrategy
from ..utils import Noise


def build_communication_diagnostics(
    actions: torch.Tensor, extra: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """从单批次 actor 输出构造逐智能体通信诊断。"""

    required = (
        "communication_edges",
        "message",
        "mu",
        "log_std",
        "decode_action",
    )
    missing = [name for name in required if name not in extra]
    if missing:
        raise ValueError(f"communication diagnostics missing tensor(s): {missing}")
    if actions.ndim < 1 or actions.shape[0] != 1:
        raise ValueError(
            "communication diagnostics require actor batch size 1, "
            f"got {tuple(actions.shape)}"
        )

    message = extra["message"]
    mu = extra["mu"]
    log_std = extra["log_std"]
    reconstruction_error = (extra["decode_action"] - actions).pow(2).mean(dim=-1)
    kl = 0.5 * (
        mu.pow(2) + torch.exp(2 * log_std) - 1 - 2 * log_std
    ).sum(dim=-1)
    tensors = {
        "communication_edges": extra["communication_edges"][0].bool(),
        "message_kl": kl[0],
        "message_reconstruction_error": reconstruction_error[0],
        "message_mean": message[0].mean(dim=-1),
        "message_std": message[0].std(dim=-1, correction=0),
        "message_l2_norm": torch.linalg.vector_norm(message[0], dim=-1),
        "message_min": message[0].amin(dim=-1),
        "message_max": message[0].amax(dim=-1),
    }
    return {
        name: value.detach().cpu().numpy() for name, value in tensors.items()
    }


def _requires_communication_diagnostics(args: Namespace, evaluation: bool) -> bool:
    return (
        evaluation
        and getattr(args, "env_name", None) == "SatelliteEnv3v1Hvt"
        and getattr(args, "alg_name", None) == "SharedMessageMATD3"
        and getattr(args, "on_plane", None) is False
        and getattr(args, "limit_accel", None) is True
    )


class SharedMessageDeterministicActorStrategy(IActorStrategy):
    """
    基于共享确定性策略网络的多智能体动作选择策略（批处理版本）。
    使用单个Actor网络，输入所有智能体的观测，输出所有智能体的动作。
    支持连续和离散动作空间，所有操作均向量化。
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        args: Namespace,
        action_continuous: bool = False,
        use_agent_id: bool = False,
    ) -> None:
        self.args = args
        self.actor = actor
        self.action_continuous = action_continuous
        self.use_agent_id = use_agent_id
        self.noise = Noise()

    def select_action(
        self,
        agent_list,
        observation: np.ndarray = None,
        global_state: np.ndarray = None,
        evaluation: bool = False,
    ) -> Dict[str, Union[np.ndarray, list]]:
        if observation is None:
            raise ValueError("observation cannot be None")
        if isinstance(observation, list):
            observation = np.array(observation)

        # 转换为张量并添加批次维度 (1, num_agents, obs_dim)
        obs_tensor = (
            torch.FloatTensor(observation)
            .permute(1, 0, 2)  # batch_size, n_agents, obs_dim
            .to(self.args.device)
        )

        batch_size, n_agents, obs_dim = obs_tensor.shape

        if self.use_agent_id:
            agent_id = torch.arange(0, len(agent_list), 1)  # n_agents, 1
            one_hot = torch.nn.functional.one_hot(
                agent_id, len(agent_list)
            )  # n_agents, n_agents
            one_hot = (
                one_hot.unsqueeze(0)
                .expand(batch_size, n_agents, -1)
                .to(obs_tensor.device)
            )

            obs_tensor = torch.cat((obs_tensor, one_hot), dim=-1)

        with torch.no_grad():
            actor_output = self.actor(obs_tensor)  # (1, num_agents, act_dim)
            if isinstance(actor_output, tuple):
                actions_tensor, extra = actor_output
            else:
                actions_tensor, extra = actor_output, None

        if self.action_continuous:
            if not evaluation:
                # 添加高斯探索噪声（向量化）
                actions_tensor = self.noise(
                    actions_tensor, self.args.action_min, self.args.action_max
                )
            # 转换为列表（兼容原接口）
            actions = (
                actions_tensor.cpu().permute(1, 0, 2).numpy()
            )  # n_agents, batch_size, action_dim
            action_list = [actions[i] for i in range(actions.shape[0])]
            result = {"action": action_list}
            if _requires_communication_diagnostics(self.args, evaluation):
                if extra is None:
                    raise ValueError(
                        "target inference requires actor communication tensors"
                    )
                result["communication_diagnostics"] = (
                    build_communication_diagnostics(actions_tensor, extra)
                )
            return result
        else:
            # 离散动作
            action_probs = torch.softmax(
                actions_tensor, dim=-1
            )  # (num_agents, act_dim)
            if evaluation:
                actions = torch.argmax(action_probs, dim=-1)  # (num_agents,)
            else:
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()  # (num_agents,)
            actions = actions.cpu().numpy().tolist()
            # 批量生成 one-hot
            one_hot = torch.nn.functional.one_hot(
                torch.tensor(actions), num_classes=action_probs.shape[-1]
            ).numpy()
            one_hot_list = [one_hot[i] for i in range(one_hot.shape[0])]
            return {"action": actions, "one_hot": one_hot_list}
