import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import torch
import torch.nn as nn
import csv
import utils
import torch.optim as optim

sys.path.append("../..")
import models.cnn as cnn
from sklearn.preprocessing import minmax_scale, scale

data_path = os.path.join(os.path.dirname(__file__), "../../data/raw_sales.csv")
torch.manual_seed(42)
num_lags = 30


data = utils.get_data(
    data_path,
    num_lags,
    date_column="datesold",
    price_column="price",
    date_format="%Y-%m-%d %H:%M:%S",
)
# data = data.iloc[:2000]
# data.drop(["id", "provider", "insertTimestamp", "dayOfWeek"], axis=1, inplace=True)
data = data[data["propertyType"] == "house"][data["bedrooms"] == 3]

data.drop(["postcode", "propertyType", "bedrooms"], axis=1, inplace=True)
# print(data.head())


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data = data.dropna()
# don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
# maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols])
# print(data.head())

train_loader, test_loader = utils.get_data_loaders(data, cols, target)
input_channels = 32
conv_layers = 2
conv_out_channels = [
    64,
    128,
]
conv_kernel_sizes = [
    3,
    5,
]
dense_layers = 2
dense_units = [256, 128]

cnn_model = cnn.CNN(
    input_channels=input_channels,
    conv_layers=conv_layers,
    conv_out_channels=conv_out_channels,
    conv_kernel_sizes=conv_kernel_sizes,
    dense_layers=dense_layers,
    dense_units=dense_units,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)

optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

log_interval = 100
epochs = 10

cnn.train_and_evaluate_cnn(
    device,
    cnn_model,
    optimizer,
    log_interval,
    epochs,
    train_loader,
    test_loader,
    silent=False,
)
