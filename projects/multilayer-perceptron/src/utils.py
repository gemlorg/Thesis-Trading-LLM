import pandas as pd
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
     
import csv
import utils 
from sklearn.preprocessing import minmax_scale, scale

def add_lags_columns(data, num_lags, exclude_columns):
    lag_columns = []
    for i in range(1, num_lags + 1):
        for col in data.columns:
            if col not in exclude_columns and "lag" not in col and col == "Price":
                lag_col_name = "{}_lag_{}".format(col, i) 
                
                lag_columns.append(data[col].shift(i).rename(lag_col_name)  )

    data = pd.concat([data] + lag_columns, axis=1)
    data = data.dropna()
    return data


def compute_price_deltas(data, price_column):  # TODO: categories instead of 0/1
    # no binary values, but the actual price difference
    # needs to be normalised!!

    data["price_delta"] = (
        (data[price_column] - data["{}_lag_1".format(price_column)]) > 0
    ).astype(int)
    # data["price_delta"] = data[price_column] - data["{}_lag_1".format(price_column)]
    return data


def get_data(num_lags, data_path="../data/google-stock-dataset-Daily.csv", date_format="%Y-%m-%d", date_column="Date", price_column="Price"):
    data = pd.read_csv(data_path)
    data[date_column] = data[date_column].apply(
        lambda x: int(datetime.strptime(x, date_format).timestamp())
    )
    
    # columns_to_keep = [date_column, price_column]
    # data = data[columns_to_keep]
    data = add_lags_columns(data, num_lags, [date_column])
    data = compute_price_deltas(data, price_column)
    data = data.drop([price_column], axis=1)
    return data

def split_data(data, test_size, target_column="price_delta"):
    test_index = int(len(data) * (1 - test_size))

    data_train = data.iloc[:test_index]
    data_test = data.iloc[test_index:]

    X_train = data_train.drop([target_column], axis=1)
    y_train = data_train[target_column]

    X_test = data_test.drop([target_column], axis=1)
    y_test = data_test[target_column]

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    return X_train, y_train, X_test, y_test

def get_layers(in_features, out_features, num_layers, hidden_size):
    layers = []
    layers.append(in_features)
    for _ in range(num_layers):
        layers.append(hidden_size)
    layers.append(out_features)
    print(layers)
    return layers

def create_model_with_layers(in_features, out_features, num_layers, hidden_size):
    layers = []
    layers.append(nn.Linear(in_features, hidden_size))
    layers.append(nn.ReLU())

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_size, out_features))
    layers.append(nn.Sigmoid())

    return layers

def sigmoid(z):
    return 1/(1 + np.exp(-z))



def train(
    model, optimizer, epoch, train_loader, loss_fn, b_size=10,  silent=False, log_interval=1):
    
    model.train()
    logs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        print(data.shape)

        print(target.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if not silent:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            logs.append(loss.item())
    return logs

def test(model, test_loader, loss_fn, silent=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if not silent:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)