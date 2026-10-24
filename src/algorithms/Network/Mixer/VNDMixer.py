import torch
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self, n_agents, *args, **kwargs):
        super(VDNMixer, self).__init__()

        self.n_agents = n_agents

    def forward(self, q_values: torch.Tensor, state: torch.Tensor):
        """
        q_values: (batch_size, seq_length, n_agents)
        """

        if q_values.shape[-1] != self.n_agents:
            raise ValueError("The last dim of q_values should be equal to n_agents")

        return torch.sum(q_values, dim=-1, keepdim=True)
