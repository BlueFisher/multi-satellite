import torch
import torch.nn as nn


class DiscreteActor(nn.Module):
    """
    离散动作策略网络，用于生成动作的对数概率分布。

    属性:
        fc1 (nn.Linear): 第一层全连接层。
        fc2 (nn.Linear): 第二层全连接层。
        fc3 (nn.Linear): 输出动作 logits 的全连接层。
        relu (nn.ReLU): ReLU 激活函数。
        softmax (nn.Softmax): Softmax 激活函数，用于生成概率分布。
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        初始化 DiscreteActor 实例。

        参数:
            input_dim (int): 输入维度。
            output_dim (int): 输出维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(DiscreteActor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播，生成动作的对数概率分布。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 动作的对数概率分布，形状为 (batch_size, output_dim)。

        示例:
            >>> actor = DiscreteActor(input_dim=4, output_dim=3)
            >>> x = torch.randn(5, 4)
            >>> log_probs = actor(x)
            >>> print(log_probs.shape)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = self.softmax(logits)
        return torch.log(probs)
