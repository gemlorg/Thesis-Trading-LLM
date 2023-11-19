import pandas as pd
import os
from datetime import datetime
import torch
import torch.nn as nn
import csv

data_path = os.path.join(
    os.path.dirname(__file__), "../data/google-stock-dataset-Daily.csv"
)
csv_results_path = os.path.join(
    os.path.dirname(__file__), "../results/mlp-results-google.csv"
)

num_lags = 60


def add_lags_columns(data, num_lags, exclude_columns):
    lag_columns = []
    for i in range(1, num_lags + 1):
        for col in data.columns:
            if col not in exclude_columns and "lag" not in col:
                lag_col_name = "{}_lag_{}".format(col, i)
                lag_columns.append(data[col].shift(i).rename(lag_col_name))

    data = pd.concat([data] + lag_columns, axis=1)
    data = data.dropna()
    return data


def compute_price_deltas(data, price_column):  # TODO: categories instead of 0/1
    data["price_delta"] = (
        (data[price_column] - data["{}_lag_1".format(price_column)]) > 0
    ).astype(int)
    return data


def get_data(date_format="%Y-%m-%d", date_column="Date", price_column="Price"):
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


def train_and_evaluate_mlp(
    mlp_model, loss_fn, optimizer, num_epochs, X_train, y_train, X_test, y_test
):
    for epoch in range(num_epochs):
        y_pred = mlp_model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_train_pred = mlp_model(X_train)
    accuracy_train = (y_train_pred.round() == y_train).float().mean().item()

    y_test_pred = mlp_model(X_test)
    accuracy_test = (y_test_pred.round() == y_test).float().mean().item()

    return accuracy_train, accuracy_test


def evaluate_models(
    model_architectures,
    loss_fns,
    learning_rates,
    num_epochs,
    X_train,
    y_train,
    X_test,
    y_test,
):
    results_list = []

    for model_architecture in model_architectures:
        for loss_fn in loss_fns:
            for learning_rate in learning_rates:
                mlp_model = nn.Sequential(*model_architecture)
                optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)

                accuracy_train, accuracy_test = train_and_evaluate_mlp(
                    mlp_model,
                    loss_fn,
                    optimizer,
                    num_epochs,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

                results_list.append(
                    {
                        "Model Architecture": str(model_architecture),
                        "Loss Function": str(loss_fn),
                        "Learning Rate": learning_rate,
                        "Accuracy Train": accuracy_train,
                        "Accuracy Test": accuracy_test,
                    }
                )

                print(
                    "model_architecture = {}, loss_fn = {}, learning_rate = {}, accuracy_train = {}, accuracy_test = {}".format(
                        model_architecture,
                        loss_fn,
                        learning_rate,
                        accuracy_train,
                        accuracy_test,
                    )
                )

    return results_list


def save_results_to_csv(results_list, csv_results_path):
    with open(csv_results_path, "w", newline="") as csvfile:
        fieldnames = [
            "Model Architecture",
            "Loss Function",
            "Learning Rate",
            "Accuracy Train",
            "Accuracy Test",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results_list:
            writer.writerow(result)


data = get_data()
in_features = len(data.columns) - 1
out_features = 1

X_train, y_train, X_test, y_test = split_data(data, 0.3)


model_architectures = [
    [
        nn.Linear(in_features, 30),
        nn.ReLU(),
        nn.Linear(30, 20),
        nn.ReLU(),
        nn.Linear(20, out_features),
        nn.Sigmoid(),
    ],
    [
        nn.Linear(in_features, 50),
        nn.ReLU(),
        nn.Linear(50, 40),
        nn.ReLU(),
        nn.Linear(40, 30),
        nn.ReLU(),
        nn.Linear(30, 20),
        nn.ReLU(),
        nn.Linear(20, out_features),
        nn.Sigmoid(),
    ],
]

num_layers = 10
hidden_size = 20
dynamic_model = create_model_with_layers(
    in_features, out_features, num_layers, hidden_size
)
model_architectures.append(dynamic_model)

loss_fns = [
    nn.BCELoss(),
    nn.CrossEntropyLoss(),
]

learning_rates = [0.001, 0.01, 0.1]
num_epochs = 1000

results_list = evaluate_models(
    model_architectures,
    loss_fns,
    learning_rates,
    num_epochs,
    X_train,
    y_train,
    X_test,
    y_test,
)

save_results_to_csv(results_list, csv_results_path)
