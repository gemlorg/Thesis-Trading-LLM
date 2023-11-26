import pandas as pd
import os
from datetime import datetime
import torch
import torch.nn as nn
import csv
import numpy as np

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

def train_and_evaluate_mlp(
    mlp_model, loss_fn, optimizer, num_epochs, X_train, y_train, X_test, y_test, b_size=10
):
    # X_train = torch.tensor(X_train.values, dtype=torch.float32)
    # y_train = torch.tensor(y_train.values, dtype=torch.float32)
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
    for epoch in range(num_epochs):
        
        
        current_loss = 0.0
        if int(epoch % 1) == 0:
            print(f'Starting epoch {epoch+1}')
            y_train_pred = mlp_model(X_train)
            accuracy_train = (y_train_pred.round() == y_train).float().mean().item()
            
            print("Accuracy train: {}".format(accuracy_train))

        for i, data in enumerate(trainloader, 0):
      
            # Get inputs
            inputs, targets = data
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp_model(inputs)
            
            # Compute loss
            loss = loss_fn(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()

            # print('Loss after mini-batch %5d: %.3f' %
            #             (i + 1, current_loss / 500))
            # current_loss = 0.0
            # if i % 500 == 499:
        
        
        

        # y_pred = mlp_model(X_train)
        # loss = loss_fn(y_pred, y_train)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        y_test_pred = mlp_model(X_test)
        accuracy_test = (y_test_pred.round() == y_test).float().mean().item()

    return accuracy_train, accuracy_test