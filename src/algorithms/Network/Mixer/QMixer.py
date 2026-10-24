import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hyper_dim: int,
        mixer_dim: int,
        linear: bool = True,
    ):
        """
        QMixer网络

        参数:
            n_agents: 智能体数量
            state_dim: 全局状态维度
            hyper_dim: 超网络隐藏层维度
            mixer_dim: 混合网络隐藏层维度
            linear: 是否保证单调性（True时确保权重非负，False时不做限制）
        """
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_dim = hyper_dim
        self.mixer_dim = mixer_dim
        self.linear = linear

        # 第一层超网络 - 生成权重和偏置
        self.hyper_w1 = nn.Linear(state_dim, hyper_dim)
        self.hyper_w1_fc = nn.Linear(hyper_dim, n_agents * mixer_dim)

        self.hyper_b1 = nn.Linear(state_dim, mixer_dim)

        # 第二层超网络 - 生成权重和偏置
        self.hyper_w2 = nn.Linear(state_dim, hyper_dim)
        self.hyper_w2_fc = nn.Linear(hyper_dim, mixer_dim)

        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_dim, 1),
        )

        # 用于保证权重非负的激活函数
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            q_values: 各智能体的Q值，形状为 (batch_size, seq_len, n_agents)
            state: 全局状态，形状为 (batch_size, seq_len, state_dim)

        返回:
            混合后的全局Q值，形状为 (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = q_values.shape

        # 将序列维度与批次维度合并
        state_flat = state.reshape(-1, self.state_dim)
        q_values_flat = q_values.reshape(-1, 1, self.n_agents)

        # 第一层超网络
        w1 = self.hyper_w1(state_flat)
        w1 = self.relu(w1)
        w1 = self.hyper_w1_fc(w1)
        w1 = w1.reshape(-1, self.n_agents, self.mixer_dim)

        b1 = self.hyper_b1(state_flat)
        b1 = b1.reshape(-1, 1, self.mixer_dim)

        # 保证单调性
        if self.linear:
            w1 = self.softplus(w1)
            b1 = self.softplus(b1)

        # 第一层混合网络
        hidden = F.elu(torch.bmm(q_values_flat, w1) + b1)

        # 第二层超网络
        w2 = self.hyper_w2(state_flat)
        w2 = self.relu(w2)
        w2 = self.hyper_w2_fc(w2)
        w2 = w2.reshape(-1, self.mixer_dim, 1)

        b2 = self.hyper_b2(state_flat)
        b2 = b2.reshape(-1, 1, 1)

        # 保证单调性
        if self.linear:
            w2 = self.softplus(w2)

        # 第二层混合网络
        total_q = torch.bmm(hidden, w2) + b2

        # 恢复原始形状
        total_q = total_q.reshape(batch_size, seq_len, 1)

        return total_q
