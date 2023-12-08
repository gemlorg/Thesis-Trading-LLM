import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import torch
import torch.nn as nn
import csv
import utils
import torch.optim as optim
sys.path.append("..")
import models.mlp as mlp
from sklearn.preprocessing import minmax_scale, scale


data_path = os.path.join(
    os.path.dirname(__file__), "../data/gbpcad_one_hour_202311210827.csv"
)
torch.manual_seed(42)
num_lags = 15

data = utils.get_data(data_path, num_lags, date_column="barTimestamp", price_column="close", date_format="%Y-%m-%d %H:%M:%S")
data = data.iloc[:2000]
data.drop(["id", "provider", "insertTimestamp", "dayOfWeek"], axis=1, inplace=True)
# print(data.head())


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 

data = data.dropna()
#don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
#maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols] )
# print(data.head())

train_loader, test_loader = utils.get_data_loaders(data, cols, target)

# architecture = utils.get_layers(len(cols), len(target), num_layers=3, hidden_size=50)
architecture = [len(cols), 128,16 , len(target)]
# print(architecture)
loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
model = mlp.Net(architecture).to(device)
lr = 0.000004
momentum = 0.4
# print(architecture)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# optimizer = optim.Adam(model.parameters(), lr=lr)
log_interval = 1000
epochs = 2000

# # use mlp

# print(len(test_loader))

mlp.train_and_evaluate_mlp(device, model, optimizer, log_interval, epochs, train_loader, test_loader )