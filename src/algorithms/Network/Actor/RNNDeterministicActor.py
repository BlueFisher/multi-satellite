import torch.nn as nn
import torch
from torch.nn import functional as F
from typing import Optional, Tuple


class RNNDeterministicActor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(x.shape) == 3  # n_agents * batch_size, seq_len, input_dim
        x = F.relu(self.fc1(x))
        x, hidden = self.rnn(x, hidden)
        x = F.tanh(self.fc2(x))
        return x, hidden
