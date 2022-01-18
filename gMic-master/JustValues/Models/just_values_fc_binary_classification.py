import torch
import torch.nn as nn
import torch.nn.functional as F


class JustValuesOnNodes(nn.Module):
    def __init__(self, data_size, RECEIVED_PARAMS):
        super(JustValuesOnNodes, self).__init__()
        self.data_size = data_size
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.fc1 = nn.Linear(self.data_size, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), 1)

        # if RECEIVED_PARAMS["initialization"] == "xavier":
        #     torch.nn.init.xavier_normal_(self.fc1.weight)
        #     torch.nn.init.xavier_normal_(self.fc2.weight)
        #     torch.nn.init.xavier_normal_(self.fc3.weight)
        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.classifier = nn.Sequential(
            self.fc1,
            self.activation_func_dict[self.activation_func],
            self.dropout,
            self.fc2,
            self.activation_func_dict[self.activation_func],
            )

    def forward(self, x, adjacency_matrix):
        # x = x.view(-1, self.data_size)

        # if self.activation_func == 'relu':
        #     x = F.relu(self.fc1(x))
        #     x = self.dropout(x)
        #     x = F.relu(self.fc2(x))
        # elif self.activation_func == 'elu':
        #     x = F.elu(self.fc1(x))
        #     x = self.dropout(x)
        #     x = F.elu(self.fc2(x))
        # elif self.activation_func == 'tanh':
        #     x = torch.tanh(self.fc1(x))
        #     x = self.dropout(x)
        #     x = torch.tanh(self.fc2(x))

        x = self.classifier(x)
        x = self.fc3(x)
        return x
