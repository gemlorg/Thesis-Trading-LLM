import pandas as pd
import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn

data_path = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")

num_lags = 50
in_features = num_lags + 1
out_features = 1


def add_lags_columns(data, num_lags, price_column):
    for i in range(1, num_lags + 1):
        data["lag_{}".format(i)] = data[price_column].shift(i)
    data = data.dropna()
    return data


def compute_price_deltas(data, price_column):
    data["price_delta"] = data[price_column] - data["lag_1"]
    data["price_delta"] = (np.sign(data["price_delta"]) + 1) // 2
    return data


def get_data(date_column="datesold", price_column="price"):
    data = pd.read_csv(data_path)
    data[date_column] = data[date_column].apply(
        lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    )
    data = data[(data["propertyType"] == "house") & (data["bedrooms"] == 3)]

    columns_to_keep = [date_column, price_column]
    data = data[columns_to_keep]

    data = add_lags_columns(data, num_lags, price_column)
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


def train_and_evaluate_mlp(
    mlp_model, loss_fn, optimizer, num_epochs, X_train, y_train, X_test, y_test
):
    for n in range(num_epochs):
        y_pred = mlp_model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = mlp_model(X_test)

    accuracy = (y_pred.round() == y_test).float().mean().item()
    return accuracy


data = get_data()

X_train, y_train, X_test, y_test = split_data(data, 0.3)

mlp_model = nn.Sequential(
    nn.Linear(in_features, 30),
    nn.ReLU(),
    nn.Linear(30, 20),
    nn.ReLU(),
    nn.Linear(20, out_features),
    nn.Sigmoid(),
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
num_epochs = 1000

accuracy = train_and_evaluate_mlp(
    mlp_model, loss_fn, optimizer, num_epochs, X_train, y_train, X_test, y_test
)
print("accuracy =", accuracy)
