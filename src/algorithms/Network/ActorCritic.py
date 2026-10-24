import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络，用于同时生成动作和状态值。

    属性:
        actor_fc1 (nn.Linear): Actor 的第一层全连接层。
        actor_fc2 (nn.Linear): Actor 的第二层全连接层。
        actor_fc_out (nn.Linear): Actor 的输出层，用于生成动作。
        critic_fc1 (nn.Linear): Critic 的第一层全连接层。
        critic_fc2 (nn.Linear): Critic 的第二层全连接层。
        critic_fc_out (nn.Linear): Critic 的输出层，用于生成状态值。
        relu (nn.ReLU): ReLU 激活函数。
        tanh (nn.Tanh): Tanh 激活函数，用于将动作限制在 [-1, 1] 范围内。
    """

    def __init__(self, input_dim, action_dim, hidden_dim=64):
        """
        初始化 ActorCritic 实例。

        参数:
            input_dim (int): 输入维度。
            action_dim (int): 动作维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(ActorCritic, self).__init__()

        # Actor 部分
        self.actor_fc1 = nn.Linear(input_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_fc_out = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Critic 部分
        self.critic_fc1 = nn.Linear(input_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        前向传播，生成动作和状态值。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - action (torch.Tensor): 动作张量，形状为 (batch_size, action_dim)，范围在 [-1, 1]。
                - value (torch.Tensor): 状态值张量，形状为 (batch_size, 1)。

        示例:
            >>> model = ActorCritic(input_dim=4, action_dim=2)
            >>> x = torch.randn(3, 4)
            >>> action, value = model(x)
            >>> print(action.shape, value.shape)
        """

        # Actor 前向传播
        actor_x = self.relu(self.actor_fc1(x))
        actor_x = self.relu(self.actor_fc2(actor_x))
        action = self.tanh(self.actor_fc_out(actor_x))

        # Critic 前向传播
        critic_x = self.relu(self.critic_fc1(x))
        critic_x = self.relu(self.critic_fc2(critic_x))
        value = self.critic_fc_out(critic_x)

        return action, value
