import torch
import torch.nn as nn


class IntentPredictor(nn.Module):
    """
    对手建模版本的动作预测网络
    """

    def __init__(self, intput_size, hidden_size, output_size):
        super(IntentPredictor, self).__init__()

        self.fc1 = nn.Linear(intput_size, hidden_size)
        self.relu = nn.ReLU()
        self.mu_layer = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_layer = nn.Linear(hidden_size, hidden_size)

        self.predictor = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x)

        dist = torch.distributions.Normal(mu, torch.exp(log_sigma))
        return mu, log_sigma, self.tanh(self.predictor(dist.rsample()))


class IntentDeterministicActor(nn.Module):
    """
    确定性动作策略网络，用于生成连续动作。对手建模版本

    属性:
        fc1 (nn.Linear): 第一层全连接层。
        fc2 (nn.Linear): 第二层全连接层。
        fc_out (nn.Linear): 输出动作的全连接层。
        relu (nn.ReLU): ReLU 激活函数。
        tanh (nn.Tanh): Tanh 激活函数，用于将动作限制在 [-1, 1] 范围内。
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        intent_predictor_count=0,
        intent_predictor_output_dim=0,
    ):
        """
        初始化 DeterministicActor 实例。

        参数:
            input_dim (int): 输入维度。
            output_dim (int): 输出维度。
            hidden_dim (int): 隐藏层维度，默认为 64。
        """
        super(IntentDeterministicActor, self).__init__()
        self.intent_predictor_count = intent_predictor_count
        self.intent_predictor_output_dim = intent_predictor_output_dim
        assert intent_predictor_count >= 0
        if self.intent_predictor_count > 0:
            assert self.intent_predictor_output_dim >= 0

        fc1_input_dim = (
            self.intent_predictor_output_dim * self.intent_predictor_count + input_dim
        )

        self.fc1 = nn.Linear(fc1_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.intent_predictor = nn.ModuleList(
            [
                IntentPredictor(input_dim, hidden_dim, self.intent_predictor_output_dim)
                for _ in range(intent_predictor_count)
            ]
        )

        self.mu = None
        self.log_sigma = None
        self.predict = None

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
        unsqueeze = False
        if x.dim() == 1:
            unsqueeze = True
            x = x.unsqueeze(0)

        mu_output = []
        log_sigma_output = []
        predictor_output = []
        for predictor in self.intent_predictor:
            mu, log_sigma, pred = predictor(x)
            mu_output.append(mu)
            log_sigma_output.append(log_sigma)
            predictor_output.append(pred)

        self.mu = mu_output
        self.log_sigma = log_sigma_output
        self.predict = predictor_output

        intent_output = torch.cat(self.predict, dim=1)

        x_input = torch.cat([intent_output, x], dim=1)
        x = self.relu(self.fc1(x_input))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc_out(x))

        if unsqueeze:
            action = action.squeeze()

        return action
