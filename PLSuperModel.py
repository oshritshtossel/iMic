import nni
import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from nni.nas.pytorch import mutables


class SuperModel(pl.LightningModule):
    def __init__(self, params, mode=None, task="reg", weighted=True):
        super().__init__()
        self.params = params
        self.mode = mode
        self.task = task
        self.weighted = weighted
        self.init_done = False
        self.in_dim = params["input_dim"]

        try:
            self.threshold = self.params["threshold"]
        except:
            self.threshold = 5000  # 5000

        if params["activation"] == "relu":
            self.activation = nn.ReLU
        elif params["activation"] == "elu":
            self.activation = nn.ELU
        elif params["activation"] == "tanh":
            self.activation = nn.Tanh

        if task != "reg" and task != "class":
            raise AttributeError("Task can be only be 'reg' or 'class'")

    def forward(self, x, b=None):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        return optimizer

    def loss_reg(self, y, y_hat):
        return F.mse_loss(y_hat.flatten(), y) + self.params["l1_loss"] * F.l1_loss(y_hat.flatten(), y)

    def loss_class(self, y, y_hat):
        weight = torch.ones(len(y))
        weight[y == 1] = self.ones_weight
        weight[y == 0] = self.zeros_weight
        weight = weight.to(self.device)
        if self.num_of_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat.flatten(), y.flatten(), weight=weight) + self.params[
                "l1_loss"] * F.l1_loss(y_hat.flatten(), y)
        else:
            return F.cross_entropy(y_hat, y.long().flatten())

    def on_validation_start(self) -> None:
        self.on_train_start()

    def on_train_start(self) -> None:
        if self.init_done:
            return
        try:
            self.num_of_classes = len(self.train_dataloader.dataloader.dataset.tensors[1].unique())
        except AttributeError:
            self.num_of_classes = len(self.train_dataloader.dataloader[0].dataset.tensors[1].unique())
        self.lin[-1] = nn.Linear(self.lin[-1].in_features,
                                 self.num_of_classes if self.num_of_classes > 2 else 1).to(self.device)

        if self.weighted:
            total, ones = 0, 0
            for _, gt in self.train_dataloader.dataloader:
                ones += gt.sum()
                total += len(gt)
            self.ones_weight = 1 / ones
            self.zeros_weight = 1 / (total - ones)
        else:
            self.ones_weight, self.zeros_weight = 1, 1
        self.init_done = True

    def training_step(self, train_batch, batch_idx):
        if type(train_batch[0]) is not list:
            train_batch = [train_batch]

        loss = torch.tensor(0., device=self.device)
        for i, batch in enumerate(train_batch):
            if self.mode is None:
                x, y = batch
                y_hat = self.forward(x)
            else:
                x, y, b = batch
                y_hat = self.forward(x, b)

            if i > 0:
                aug_loss_factor = 0.1
            else:
                aug_loss_factor = 1
            if self.task == "reg":
                loss += self.loss_reg(y.type(torch.float32), y_hat).type(torch.float32) * aug_loss_factor
            elif self.task == "class":
                # y_hat = torch.sigmoid(y_hat)
                loss += self.loss_class(y.type(torch.float32), y_hat).type(torch.float32) * aug_loss_factor
            else:
                raise AttributeError("task needs to be 'reg' or 'class'")

        self.log("Loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        if self.mode is None:
            x, y = train_batch
            y_hat = self.forward(x)
        else:
            x, y, b = train_batch
            y_hat = self.forward(x, b)
        if self.task == "reg":
            loss = self.loss_reg(y.type(torch.float32), y_hat).type(torch.float32)
        elif self.task == "class":
            # y_hat = torch.sigmoid(y_hat)
            loss = self.loss_class(y.type(torch.float32), y_hat).type(torch.float32)
        else:
            raise AttributeError("task needs to be 'reg' or 'class'")
        self.log("val_loss", loss)
        return loss

    def predict(self, loader):
        y_hat = []
        for batch in loader:
            if self.mode is None:
                x, y = batch
                if self.task == "reg":
                    y_hat.extend([i.item() for i in self.forward(x).detach()])
                elif self.task == "class":
                    if self.num_of_classes == 2:
                        y_hat.extend([torch.sigmoid(i).round().item() for i in self.forward(x).detach()])
                    else:
                        y_hat.extend([torch.softmax(i, 0).numpy() for i in
                                      self.forward(x).detach()])
            else:
                x, y, b = batch
                y_hat.extend([i.item() for i in self.forward(x, b).detach()])

        return y_hat

    def predict_one(self, x):
        x = x.unsqueeze(0)
        if self.mode is None:
            if self.task == "reg":
                y_hat = self.forward(x).round()
            elif self.task == "class":
                y_hat = torch.sigmoid(self.forward(x)).round()
        else:
            x, y, b = batch
            y_hat = torch.sigmoid(self.forward(x, b)).round()
        return y_hat.flatten()
