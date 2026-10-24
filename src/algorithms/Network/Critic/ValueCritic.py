import torch.nn as nn


class ValueCritic(nn.Module):
    """
    状态值评估网络，用于评估给定状态的值函数。

    属性:
        fc1 (nn.Linear): 第一层全连接层。
        fc2 (nn.Linear): 第二层全连接层。
        fc_value (nn.Linear): 输出状态值的全连接层。
        relu (nn.ReLU): ReLU 激活函数。
    """

    def __init__(self, input_dim, hidden_dim=64):
        """
        初始化 ValueCritic 实例。

        参数:
            input_dim (int): 输入维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(ValueCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播，计算状态值。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 状态值张量，形状为 (batch_size, 1)。

        示例:
            >>> critic = ValueCritic(input_dim=4)
            >>> x = torch.randn(3, 4)
            >>> value = critic(x)
            >>> print(value.shape)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc_value(x)
        return value
