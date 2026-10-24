import torch.nn as nn


class DeterministicActor(nn.Module):
    """
    确定性动作策略网络，用于生成连续动作。

    属性:
        fc1 (nn.Linear): 第一层全连接层。
        fc2 (nn.Linear): 第二层全连接层。
        fc_out (nn.Linear): 输出动作的全连接层。
        relu (nn.ReLU): ReLU 激活函数。
        tanh (nn.Tanh): Tanh 激活函数，用于将动作限制在 [-1, 1] 范围内。
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        初始化 DeterministicActor 实例。

        参数:
            input_dim (int): 输入维度。
            output_dim (int): 输出维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(DeterministicActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # 假设动作空间是 [-1, 1]，使用 tanh 激活函数
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        前向传播，生成动作。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 动作张量，形状为 (batch_size, output_dim)，范围在 [-1, 1]。

        示例:
            >>> actor = DeterministicActor(input_dim=4, output_dim=2)
            >>> x = torch.randn(3, 4)
            >>> action = actor(x)
            >>> print(action.shape)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc_out(x))
        return action
