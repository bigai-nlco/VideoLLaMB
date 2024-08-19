import torch
from torch import nn

class MLPProjector(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.mlp = nn.Sequential(*modules)
    def forward(self, hidden_states):
        return self.mlp(hidden_states)