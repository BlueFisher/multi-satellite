from copy import deepcopy

from torch.optim import Adam

from .Action.DeterministicActorStrategy import DeterministicActorStrategy
from .Manager import Manager
from .Network.Actor.DeterministicActor import DeterministicActor
from .Network.Critic.QValueCritic import QValueCritic
from .StorageStrategy.StandardReplayStorage import StandardReplayStorage
from .Trainer.MADDPGTrainer import MADDPGTrainer


class MADDPG(Manager):
    """
    多智能体独立 Critic 版本的 MADDPG 管理器类。

    该类实现了多智能体深度确定性策略梯度（MADDPG）算法的系统构建。
    与共享 Critic 的版本不同，本类为每个智能体分别创建独立的 Critic 网络，
    从而允许不同智能体拥有各自的价值评估模型，更适用于异构场景。

    本类负责：
    - 初始化多个 Actor（每个智能体一个）
    - 初始化多个 Critic（每个智能体一个）
    - 构建目标网络（target actors 与 target critics）
    - 构建经验回放缓冲区
    - 配置多 Actor、多 Critic 的优化器
    - 初始化 MADDPGTrainer 训练器
    - 继承 Manager 以统一管理策略执行、训练与数据存储模块

    Args:
        name_agents (List[str]): 智能体名称列表。
        obs_shapes (List[int]): 每个智能体观测空间维度。
        action_shapes (List[int]): 每个智能体动作空间维度。
        global_state_shape (int): 全局状态维度（部分扩展算法可能使用）。
        args (Any): 算法与网络的超参数配置对象。
        action_continuous (bool): 是否为连续动作空间，默认为 False。

    Example:
        >>> maddpg = MADDPG(
        ...     name_agents=["uav_1", "uav_2", "uav_3"],
        ...     obs_shapes=[30, 30, 30],
        ...     action_shapes=[4, 4, 4],
        ...     global_state_shape=90,
        ...     args=args,
        ...     action_continuous=True,
        ... )
        >>> actions = maddpg.actor_strategy.get_actions(observations)
    """
    def __init__(
        self,
        name_agents: list[str],
        obs_shapes: list[int],
        action_shapes: list[int],
        global_state_shape: int,
        args,
        action_continuous: bool = False,
    ):
        """
    初始化 MADDPG 管理器，构建多智能体 Actor / Critic、目标网络、优化器与训练器。

    初始化过程包含：
    1. 保存基本配置信息（agent 名称、数量等）
    2. 创建 StandardReplayStorage 回放缓冲区
    3. 为每个智能体创建独立 Actor 与独立 Critic
    4. 创建所有 Actor 与 Critic 对应的目标网络（target networks）
    5. 根据 agent 数量创建多组 Actor 与 Critic 优化器
    6. 实例化 MADDPGTrainer，用于执行训练
    7. 调用基类 Manager 的构造函数进行模块绑定

    Args:
        name_agents (List[str]): 智能体名称列表。
        obs_shapes (List[int]): 各智能体观测维度。
        action_shapes (List[int]): 各智能体动作维度。
        global_state_shape (int): 全局状态空间维度。
        args (Any): 超参数与设备配置对象。
        action_continuous (bool): 是否为连续动作空间。

    Example:
        >>> maddpg = MADDPG(
        ...     name_agents=["robot_A", "robot_B"],
        ...     obs_shapes=[12, 12],
        ...     action_shapes=[3, 3],
        ...     global_state_shape=24,
        ...     args=args,
        ...     action_continuous=False,
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
        critics = []

        obs_dim_sum = sum(args.n_obs)
        actions_dim_sum = sum(args.n_actions)

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

            critic_input_size = obs_dim_sum + actions_dim_sum
            critics.append(
                QValueCritic(critic_input_size, hidden_dim=args.hidden_size).to(
                    args.device
                )
            )

        # 目标网络
        target_actors = deepcopy(actors)
        target_critics = deepcopy(critics)

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
                Adam(critics[index].parameters(), lr=args.lr_critic)
                for index in range(args.num_agents)
            ],
        }

        self.trainer = MADDPGTrainer(
            buffer=self.storage.replay_buffer,
            args=args,
            actors=actors,
            critics=critics,
            optimizers=optimizers,
            target_actors=target_actors,
            target_critics=target_critics,
        )
        super().__init__(
            actor_strategy=self.actor_strategy,
            storage_strategy=self.storage,
            trainer=self.trainer,
            args=self.args,
        )
