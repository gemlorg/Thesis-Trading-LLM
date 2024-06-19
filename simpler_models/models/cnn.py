import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_layers,
        conv_out_channels,
        conv_kernel_sizes,
        dense_layers,
        dense_units,
        seq_length,
        activation_fn=torch.relu,
    ):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(conv_layers):
            layer = nn.Conv1d(
                input_channels if i == 0 else conv_out_channels[i - 1],
                conv_out_channels[i],
                kernel_size=conv_kernel_sizes[i],
                stride=1,
                padding=int(conv_kernel_sizes[i] / 2),
            )
            self.conv_layers.append(layer)

        # Dense layers
        self.dense_layers = nn.ModuleList()
        for i in range(dense_layers):
            layer = nn.Linear(
                conv_out_channels[-1] * seq_length if i == 0 else dense_units[i - 1],
                dense_units[i],
            )
            self.dense_layers.append(layer)

        self.output_layer = nn.Linear(dense_units[-1], output_channels)
        self.activation_fn = activation_fn
        self.architecture_string = (
            "Conv_config = ("
            + str(conv_layers)
            + ", "
            + str(conv_out_channels)
            + ", "
            + str(conv_kernel_sizes)
            + ")\n"
            + "Dense_config = ("
            + str(dense_layers)
            + ", "
            + str(dense_units)
            + ")\n"
        )

    def forward(self, x):
        # Applying the convolutional layers
        for conv_layer in self.conv_layers:
            x = self.activation_fn(conv_layer(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Applying the dense layers
        for dense_layer in self.dense_layers:
            x = self.activation_fn(dense_layer(x))

        # Output layer
        x = self.output_layer(x)
        return x

    def __str__(self):
        return self.architecture_string
