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
from sklearn.preprocessing import minmax_scale, scale

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def add_lags_columns(data, num_lags, include_columns):
    lag_columns = []
    for i in range(1, num_lags + 1):
        for col in include_columns:
            if  "lag" not in col :
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

def get_data(data_path, num_lags=10, date_format="%Y-%m-%d", date_column="Date", price_column="Price"):
    data = pd.read_csv(data_path)
    data[date_column] = data[date_column].apply(
        lambda x: int(datetime.strptime(x, date_format).timestamp())
    )
    
    # columns_to_keep = [date_column, price_column]
    # data = data[columns_to_keep]
    data = add_lags_columns(data, num_lags, [price_column])
    data = compute_price_deltas(data, price_column)
    data = data.drop([price_column], axis=1)
    return data


def get_data_loaders(data, cols, target, test_size=0.2, batch_size=32):
    test_index = int(len(data) * (1 - test_size))

    data_train = data.iloc[test_index:]
    data_test = data.iloc[:test_index]

    X_train = data_train.drop(target, axis=1)
    y_train = data_train[target]

    X_test = data_test.drop(target, axis=1)
    y_test = data_test[target]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train[cols].values).float(),
            torch.tensor(y_train.values).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test[cols].values).float(),
            torch.tensor(y_test.values).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader

def get_layers(in_features, out_features, num_layers, hidden_size):
    layers = []
    layers.append(in_features)
    for _ in range(num_layers):
        layers.append(hidden_size)
    layers.append(out_features)
    print(layers)
    return layers