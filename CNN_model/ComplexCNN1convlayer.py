import nni
import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from nni.nas.pytorch import mutables
from complexPyTorch import complexLayers

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class CNN_1l(pl.LightningModule):
    def __init__(self, params, b, in_dim=(8, 210)):
        super().__init__()
        self.params = params
        self.mode = b

        if params["activation"] == "relu":
            activation = complexLayers.ComplexReLU
        elif params["activation"] == "elu":
            activation = complexLayers.ComplexELU
        elif params["activation"] == "tanh":
            activation = complexLayers.ComplexTanh

        self.cnn = nn.Sequential(
            complexLayers.ComplexConv2d(1, params["channels"],
                                        kernel_size=(params["kernel_size_a"], params["kernel_size_b"]),
                                        stride=params["stride"]),
            activation()
        )

        cos = conv_output_shape(in_dim, (params["kernel_size_a"], params["kernel_size_b"]), stride=params["stride"])
        if self.mode is None:
            conv_out_dim = int(cos[0] * cos[1] * params["channels"])
        else:
            conv_out_dim = int(cos[0] * cos[1] * params["channels"]) + 37

        self.lin = nn.Sequential(
            complexLayers.ComplexLinear(conv_out_dim, conv_out_dim // params["linear_dim_divider_1"]),
            activation(),
            complexLayers.ComplexDropout(params["dropout"]),
            complexLayers.ComplexLinear(conv_out_dim // params["linear_dim_divider_1"],
                                        conv_out_dim // params["linear_dim_divider_2"]),
            activation(),
            complexLayers.ComplexDropout(params["dropout"]),
            complexLayers.ComplexLinear(conv_out_dim // params["linear_dim_divider_2"], 1)
        )

    def forward(self, x, b=None):
        x = x.type(torch.complex64)
        x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        if b is None:
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x, 1)
            x = torch.cat([x, b], dim=1).type(torch.complex64)
        x = self.lin(x)
        return x.abs()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        return optimizer

    def loss(self, y, y_hat):
        return F.mse_loss(y_hat.flatten(), y) + self.params["l1_loss"] * F.l1_loss(y_hat.flatten(), y)

    def training_step(self, train_batch, batch_idx):
        if self.mode is None:
            x, y = train_batch
            y_hat = self.forward(x)
        else:
            x, y, b = train_batch
            y_hat = self.forward(x, b)
        loss = self.loss(y.type(torch.float32), y_hat).type(torch.float32)
        self.log("Loss", loss)
        return loss

    def predict(self, loader):
        y_hat = []
        for batch in loader:
            if self.mode is None:
                x, y = batch
                y_hat.extend([i.item() for i in self.forward(x).detach()])
            else:
                x, y, b = batch
                y_hat.extend([i.item() for i in self.forward(x, b).detach()])
        return y_hat
