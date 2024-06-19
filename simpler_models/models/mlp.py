import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, layers, activation_fn=torch.sigmoid):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(layers[:-1], layers[1:])]
        )
        self.activation_fn = activation_fn

    def forward(self, x):
        x = torch.flatten(x, 1)
        for l in self.layers[:-1]:
            x = l(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
