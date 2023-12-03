import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, layers, activation_fn=torch.sigmoid):
        super(Net, self).__init__()
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
        output = self.activation_fn(x)
        return output


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


def test(model, device, test_loader, loss_fn, reduction, silent=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction=reduction).item()
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


# device = torch.device("cuda" if use_cuda else "cpu")
# model = Net().to(device)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# log_interval = 10
# epochs = 5


def train_and_evaluate_mlp(
    device,
    model,
    optimizer,
    loss_fn,
    reduction,
    log_interval,
    epochs,
    silent,
    train_loader,
    test_loader,
):
    # (
    #     device,
    #     model,
    #     optimizer,
    #     loss_fn,
    #     reduction,
    #     log_interval,
    #     epochs,
    #     silent,
    # ) = load_model()
    # (train_loader, test_loader) = load_data()

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
        acc_history.append(test(model, device, test_loader, loss_fn, reduction, silent))
    plot_results(train_history, acc_history)
    return train_history, acc_history


def plot_results(train_history, acc_history):
    plt.figure(figsize=(12, 8))
    plt.plot(train_history, label="train")
    plt.plot(acc_history, label="test")
    plt.legend()
    plt.show()


# from models.mlp import train_and_evaluate_mlp, Net