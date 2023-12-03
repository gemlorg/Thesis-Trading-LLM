import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import torch
import torch.nn as nn
import csv
import torch.optim as optim
import utils
sys.path.append("..")
import models.mlp as mlp
from sklearn.preprocessing import minmax_scale, scale

data_path = os.path.join(
    os.path.dirname(__file__), "../data/google-stock-dataset-Daily.csv"
)
csv_results_path = os.path.join(
    os.path.dirname(__file__), "../results/mlp-results-google.csv"
)

num_lags = 40

torch.manual_seed(42)

data = utils.get_data(data_path, num_lags, date_column="Date", price_column="Price")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
data.drop(["Unnamed: 0"], axis=1, inplace=True)
data = data.dropna()
#don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
#maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols] )
# print(data.head())

train_loader, test_loader = utils.get_data_loaders(data, cols, target)

architecture = utils.get_layers(len(cols), len(target), num_layers=2, hidden_size=64)
print(architecture)
loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
model = mlp.Net(architecture).to(device)
lr = 0.00001
momentum = 0.4
# print(architecture)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
log_interval = 100
epochs = 5000

# use mlp



mlp.train_and_evaluate_mlp(device, model, optimizer, log_interval, epochs, train_loader, test_loader )
