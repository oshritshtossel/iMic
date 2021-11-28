import torch
from torch import nn

from PLSuperModel import SuperModel


class Naeive(SuperModel):
    # GDM 276
    # BGU 210
    def __init__(self, params, task="reg", mode=None, weighted=False):
        super().__init__(params, mode, task, weighted)
        in_dim = self.in_dim
        self.params = params
        self.mode = mode
        self.task = task

        self.lin = nn.Sequential(
            nn.Linear(in_dim, params["linear_dim_1"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["linear_dim_1"], params["linear_dim_2"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(params["linear_dim_2"], 1)
        )

    def forward(self, x, b=None):
        x = x.type(torch.float32)
        if b is None:
            x = self.lin(x).type(torch.float32)
        else:
            x = torch.cat([x, b], dim=1).type(torch.float32)
            x = self.lin(x).type(torch.float32)
        return x
