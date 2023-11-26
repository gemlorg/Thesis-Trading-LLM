import pandas as pd
import os
from datetime import datetime
import torch
import torch.nn as nn
import csv
import utils 
import numpy as np
from sklearn.preprocessing import minmax_scale, scale

class Model:
    def __init__(self, architecture, num_lags, num_layers, hidden_size):
        self.architecture = architecture
        self.num_lags = num_lags
        self.num_layers = num_layers
        self.hidden_size = hidden_size

class ModelFactory:
    def __init__(self, data_path, max_num_lags):
        self.data = utils.get_data(max_num_lags, data_path)
        self.max_num_lags = max_num_lags
        self.data = self.data.drop(["Unnamed: 0"], axis=1)
        #don't normalise the target column
        target = "price_delta"
        cols = self.data.drop([target], axis=1).select_dtypes(np.number).columns
        # self.data[target] = utils.sigmoid(self.data[target])
        #maybe should use another method for normalisation
        self.data[cols] = minmax_scale(self.data[cols] )
        # print(self.data.head())


    def get_model(self, num_layers, hidden_size, num_lags=1):
        #ATTENTION: after creating a model with a certain number of lags, you can't create a model with a higher number of lags
        # (for optimisation puposes)
        assert(num_lags <= self.max_num_lags)
        in_features = len(self.data.columns) - 1 - self.max_num_lags + num_lags
        out_features = 1
        architecture = utils.create_model_with_layers(in_features, out_features, num_layers, hidden_size)
        # df = self.data.copy()
        # df = df.drop(["Price_lag_" + str(i) for i in range(num_lags + 1, self.max_num_lags + 1)], axis=1)
        # df = df.drop(["Unnamed: 0"], axis=1)
        model = Model(architecture, num_lags, num_layers, hidden_size)
        # print(architecture)

        #later on this can be placed in train_model, but we'll have to create a df whenever training a model.
        # self.max_num_lags = num_lags
        # self.data = self.data.drop(["Price_lag_" + str(i) for i in range(num_lags + 1, self.max_num_lags + 1)], axis=1)
        # print("NEW DATA")
        # print(self.data.head())
        # self.X_train, self.y_train, self.X_test, self.y_test = utils.split_data(self.data, split)
        return model
    
    def train_model(self, model, learning_rate=0.01, num_epochs=1000, split=0.1, loss_fn=nn.BCELoss()):
        mlp_model = nn.Sequential(*model.architecture)
        optimizer = torch.optim.SGD(mlp_model.parameters(), lr=learning_rate)

        df = self.data.copy()
        df = df.drop(["Price_lag_" + str(i) for i in range(model.num_lags + 1, self.max_num_lags + 1)], axis=1)
        X_train, y_train, X_test, y_test = utils.split_data(self.data, split)

        accuracy_train, accuracy_test = utils.train_and_evaluate_mlp(
                    mlp_model,
                    loss_fn,
                    optimizer,
                    num_epochs,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
        print("Accuracy train: {}".format(accuracy_train))
        print("Accuracy test: {}".format(accuracy_test))
        
        
    

    



data_path = os.path.join(
    os.path.dirname(__file__), "../data/google-stock-dataset-Daily.csv"
)
csv_results_path = os.path.join(
    os.path.dirname(__file__), "../results/mlp-results-google.csv"
)
# torch.manual_seed(42)

factory = ModelFactory(data_path, 100)
model = factory.get_model(100, 100, 100)
factory.train_model(model, num_epochs=100)
