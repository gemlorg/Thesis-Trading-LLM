import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
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

        self.output_layer = nn.Linear(dense_units[-1], 1)
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
        output = torch.sigmoid(x)
        return output

    def __str__(self):
        return self.architecture_string


def train(
    model, device, train_loader, optimizer, loss_fn, epoch, log_interval, silent=False
):
    model.train()
    logs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if not silent:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
            logs.append(loss.item())
    return logs


def test(model, device, test_loader, loss_fn, silent=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = (output >= 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if not silent:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return correct / len(test_loader.dataset)


def train_and_evaluate_cnn(
    device,
    model,
    optimizer,
    log_interval,
    epochs,
    train_loader,
    test_loader,
    loss_fn=torch.nn.BCEWithLogitsLoss(),
    silent=False,
):
    train_history = []
    acc_history = []
    for epoch in range(1, epochs + 1):
        train_history.extend(
            train(
                model,
                device,
                train_loader,
                optimizer,
                loss_fn,
                epoch,
                log_interval,
                silent,
            )
        )
        acc_history.append(test(model, device, test_loader, loss_fn, silent))
    # plot_results(train_history, acc_history)
    return train_history, acc_history


def plot_results(train_history, acc_history):
    plt.figure(figsize=(12, 8))
    # plt.plot(train_history, label="train")
    plt.plot(acc_history, label="test")
    plt.legend()
    plt.show()


# Example usage:

# train_loader, test_loader = ...
# input_channels = 3
# conv_layers = 2
# conv_out_channels = [
#     64,
#     128,
# ]
# conv_kernel_sizes = [
#     3,
#     3,
# ]
# dense_layers = 2
# dense_units = [256, 128]

# cnn_model = CNN(
#     input_channels=input_channels,
#     conv_layers=conv_layers,
#     conv_out_channels=conv_out_channels,
#     conv_kernel_sizes=conv_kernel_sizes,
#     dense_layers=dense_layers,
#     dense_units=dense_units,
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cnn_model.to(device)

# optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

# log_interval = 100
# epochs = 10

# train_and_evaluate_cnn(
#     device,
#     cnn_model,
#     optimizer,
#     log_interval,
#     epochs,
#     train_loader,
#     test_loader,
#     silent=False,
# )
