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

class Net(nn.Module):
    def __init__(self, layers=[784, 128, 128, 10]):
        super(Net, self).__init__()
        # After flattening an image of size 28x28 we have 784 inputs
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(layers[:-1], layers[1:])])
        print(self.layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for l in self.layers[:-1]:
          x = l(x)
          x = F.relu(x)
        x = self.layers[-1](x)
        output = F.log_softmax(x, dim=1)
        return output

class Model:
    def __init__(self, layers, num_lags, num_layers, hidden_size):
        self.network = Net(layers)

        # print(layers)
        self.num_lags = num_lags
        self.num_layers = num_layers
        self.hidden_size = hidden_size

class ModelFactory:
    def __init__(self, data_path, max_num_lags, device):
        self.data = utils.get_data(max_num_lags, data_path)
        self.max_num_lags = max_num_lags
        self.data = self.data.drop(["Unnamed: 0"], axis=1)
        self.device=device

        #don't normalise the target column
        target = "price_delta"
        cols = self.data.drop([target], axis=1).select_dtypes(np.number).columns
        # self.data[target] = utils.sigmoid(self.data[target])
        #maybe should use another method for normalisation
        self.data[cols] = minmax_scale(self.data[cols] )
        self.data[target] = utils.sigmoid(self.data[target])
        # print(self.data.head())


    def get_model(self, num_layers, hidden_size, num_lags=1):
    
        assert(num_lags <= self.max_num_lags)
        in_features = len(self.data.columns) - 1 - self.max_num_lags + num_lags
        out_features = 1
        architecture = utils.get_layers(in_features, out_features, num_layers, hidden_size)
        model = Model(architecture, num_lags, num_layers, hidden_size)
        return model
    

    def train_model(self, model, learning_rate=0.01, num_epochs=100, split=0.7, loss_fn=nn.MSELoss(), b_size=10):
        mlp_model = model.network
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)

        df = self.data.copy()
        df = df.drop(["Price_lag_" + str(i) for i in range(model.num_lags + 1, self.max_num_lags + 1)], axis=1)
        X_train, y_train, X_test, y_test = utils.split_data(self.data, split)
        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        testset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=True)
        results = []
        train_history = []
        acc_history = []
        for epoch in range(1, num_epochs + 1):
            train_history.extend(utils.train(mlp_model, optimizer, epoch, train_loader,loss_fn, silent=True))
            acc_history.append(utils.test(mlp_model, test_loader, loss_fn, silent=False))
        results.append((train_history, acc_history))
        return results
        # plt.figure(figsize=(12,8))
        # plt.plot(results)

        # accuracy_train, accuracy_test = utils.train_and_evaluate_mlp(
        #             mlp_model,
        #             loss_fn,
        #             optimizer,
        #             num_epochs,
        #             X_train,
        #             y_train,
        #             X_test,
        #             y_test,
        #         )
        # print("Accuracy train: {}".format(accuracy_train))
        # print("Accuracy test: {}".format(accuracy_test))

        
        
    

    



data_path = os.path.join(
    os.path.dirname(__file__), "../data/google-stock-dataset-Daily.csv"
)
csv_results_path = os.path.join(
    os.path.dirname(__file__), "../results/mlp-results-google.csv"
)
torch.manual_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

factory = ModelFactory(data_path, 10, device)
model = factory.get_model(num_lags=10, num_layers=3, hidden_size=64)
print(factory.train_model(model, num_epochs=1000))
