import torch

from torch import nn
from math import floor
from PLSuperModel import SuperModel
import math


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size, kernel_size)
    d = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    h = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[2] + (2 * pad) - (dilation * (kernel_size[2] - 1)) - 1) / stride) + 1)
    return d, h, w


class CNN_D3(SuperModel):
    def __init__(self, params, mode=None, task="reg", weighted=False):
        super().__init__(params, mode, task, weighted)

        in_dim = self.in_dim

        self.cnn = nn.Sequential(nn.Conv3d(1,
                                           params["channels"],
                                           # padding=params["padding"],
                                           kernel_size=(
                                               params["kernel_size_a"], params["kernel_size_b"],
                                               params["kernel_size_c"]), stride=params["stride"]),
                                 self.activation())

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(1, params["channels"], kernel_size=(params["kernel_size_a"], params["kernel_size_b"]),
        #               stride=params["stride"], padding=params["padding"]),
        #     self.activation(),
        #
        #     nn.Conv2d(params["channels"], params["channels_2"],
        #               kernel_size=(params["kernel_size_a_2"], params["kernel_size_b_2"]),
        #               stride=params["stride_2"], padding=params["padding_2"]),
        #     self.activation(),
        # )

        add = 0
        # if mode is not None:
        #     in_dim = (in_dim[0], in_dim[1] - mode.shape[1])
        #     add = mode.shape[1]

        cos = conv_output_shape(in_dim, (params["kernel_size_a"], params["kernel_size_b"], params["kernel_size_c"]),
                                stride=params["stride"])
        conv_out_dim = int(cos[0] * cos[1] * cos[2] * params["channels"]) + add

        if conv_out_dim > self.threshold:
            self.use_max_pool = True
            max_pool_factor = int(((conv_out_dim - add) // self.threshold) ** 0.5)
            #####################
            if max_pool_factor <= 1:
                max_pool_factor = 2
            #####################
            conv_out_dim = (math.prod([max(math.ceil(i / max_pool_factor), 1) for i in cos]) * params["channels"]) + add
            self.MP = nn.MaxPool3d(max_pool_factor, ceil_mode=True)
        else:
            self.use_max_pool = False

        self.lin = nn.Sequential(
            nn.Linear(conv_out_dim, conv_out_dim // params["linear_dim_divider_1"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_1"],
                      conv_out_dim // params["linear_dim_divider_2"]),
            self.activation(),
            nn.Dropout(params["dropout"]),
            nn.Linear(conv_out_dim // params["linear_dim_divider_2"], 1)
        )

    def forward(self, x, b=None):
        x = x.type(torch.float32)
        x = torch.unsqueeze(x, 1)
        if (self.in_dim[0] - x.shape[2]) != 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.in_dim[0] - x.shape[2]))
        x = self.cnn(x)
        if self.use_max_pool:
            x = self.MP(x)
        x = torch.flatten(x, 1)
        if b is not None:
            x = torch.cat([x, b], dim=1).type(torch.float32)
        return self.lin(x).type(torch.float32)
