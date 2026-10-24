import torch
import torch.nn as nn


class ContinuousActor(nn.Module):
    """
    连续动作策略网络，用于生成动作的均值和对数标准差。

    属性:
        fc1 (nn.Linear): 第一层全连接层。
        fc2 (nn.Linear): 第二层全连接层。
        fc_mean (nn.Linear): 输出动作均值的全连接层。
        fc_log_std (nn.Linear): 输出动作对数标准差的全连接层。
        relu (nn.ReLU): ReLU 激活函数。
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        初始化 ContinuousActor 实例。

        参数:
            input_dim (int): 输入维度。
            output_dim (int): 输出维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(ContinuousActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_log_std = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播，生成动作的均值和对数标准差。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - mean (torch.Tensor): 动作的均值，形状为 (batch_size, output_dim)。
                - log_std (torch.Tensor): 动作的对数标准差，形状为 (batch_size, output_dim)。

        示例:
            >>> actor = ContinuousActor(input_dim=4, output_dim=2)
            >>> x = torch.randn(3, 4)
            >>> mean, log_std = actor(x)
            >>> print(mean.shape, log_std.shape)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制log_std范围，防止数值问题
        return mean, log_std
