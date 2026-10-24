import torch.nn as nn


class QValueCritic(nn.Module):
    """
    Q 值评估网络，用于评估给定状态动作对的 Q 值。

    属性:
        fc1 (nn.Linear): 第一层全连接层。
        fc2 (nn.Linear): 第二层全连接层。
        fc_q (nn.Linear): 输出 Q 值的全连接层。
        relu (nn.ReLU): ReLU 激活函数。
    """

    def __init__(self, input_dim, hidden_dim=64):
        """
        初始化 QValueCritic 实例。

        参数:
            input_dim (int): 输入维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(QValueCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播，计算 Q 值。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: Q 值张量，形状为 (batch_size, 1)。

        示例:
            >>> critic = QValueCritic(input_dim=4)
            >>> x = torch.randn(3, 4)
            >>> q_value = critic(x)
            >>> print(q_value.shape)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc_q(x)
        return q_value
