import torch
from torch import nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """
    策略网络 (Actor)，将智能体的观测映射到动作分布的参数（均值和对数标准差）.

    属性:
        action_shape (int): 动作的维度。
        shared_net (nn.Sequential): 提取特征的共享网络。
        mean_layer (nn.Linear): 输出动作均值的全连接层。
        log_std_layer (nn.Parameter): 输出动作对数标准差的可学习参数。
    """

    def __init__(self, obs_shape, action_shape, hidden_size):
        """
        初始化 ActorNetwork 实例。

        参数:
            obs_shape (int): 观测的维度。
            action_shape (int): 动作的维度。
            hidden_size (int): 隐藏层的维度。
        """
        super(ActorNetwork, self).__init__()
        self.action_shape = action_shape

        # 共享特征提取器
        self.shared_net = nn.Sequential(
            nn.Linear(obs_shape, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # 均值输出层
        self.mean_layer = nn.Linear(hidden_size, action_shape)

        # 对数标准差输出层（独立参数）
        self.log_std_layer = nn.Parameter(torch.zeros(1, action_shape))

    def forward(self, obs):
        """
        前向传播，返回一个正态分布。

        参数:
            obs (torch.Tensor): 智能体的观测，形状为 (batch_size, obs_shape)。

        返回:
            torch.distributions.Normal: 表示动作分布的正态分布对象。

        示例:
            >>> actor = ActorNetwork(obs_shape=4, action_shape=2, hidden_size=64)
            >>> obs = torch.randn(3, 4)
            >>> action_dist = actor(obs)
            >>> print(action_dist.mean.shape)
        """
        features = self.shared_net(obs)
        mean = self.mean_layer(features)

        # 对数标准差是可学习的，但可以通过 clamp 限制范围以保证数值稳定性
        log_std = self.log_std_layer.expand_as(mean)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return Normal(mean, log_std.exp())


class CriticNetwork(nn.Module):
    """
    价值网络 (Critic)，将全局状态映射到状态值。

    属性:
        net (nn.Sequential): 用于计算状态值的神经网络。
    """

    def __init__(self, state_shape, hidden_size):
        """
        初始化 CriticNetwork 实例。

        参数:
            state_shape (int): 状态的维度。
            hidden_size (int): 隐藏层的维度。
        """
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        """
        前向传播，计算状态值。

        参数:
            state (torch.Tensor): 全局状态，形状为 (batch_size, state_shape)。

        返回:
            torch.Tensor: 状态值张量，形状为 (batch_size, 1)。

        示例:
            >>> critic = CriticNetwork(state_shape=4, hidden_size=64)
            >>> state = torch.randn(3, 4)
            >>> value = critic(state)
            >>> print(value.shape)
        """
        return self.net(state)
