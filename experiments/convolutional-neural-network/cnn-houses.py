import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import utils
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.append("../..")
import models.cnn as cnn
from sklearn.preprocessing import minmax_scale, scale

data_path = os.path.join(os.path.dirname(__file__), "../../data/raw_sales.csv")
torch.manual_seed(42)
num_lags = 10

data = utils.get_data(
    data_path,
    num_lags,
    date_column="datesold",
    price_column="price",
    date_format="%Y-%m-%d %H:%M:%S",
)
data = data[data["propertyType"] == "house"][data["bedrooms"] == 3]

data.drop(["postcode", "propertyType", "bedrooms"], axis=1, inplace=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data = data.dropna()
# don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
# maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols])

train_loader, test_loader = utils.get_data_loaders(data, target, num_lags)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_channels = 2

conv_configs = [
    (2, [64, 128], [3, 5]),
    (2, [128, 256], [3, 5]),
    (3, [32, 64, 128], [3, 5, 3]),
    (3, [64, 128, 256], [3, 5, 7]),
]
dense_configs = [
    (2, [256, 128]),
    (2, [512, 256]),
    (3, [512, 256, 128]),
    (2, [128, 10]),
    (3, [200, 100, 10]),
]
log_interval = 100
epochs = 10


def test_model(
    conv_layers,
    conv_out_channels,
    conv_kernel_sizes,
    dense_layers,
    dense_units,
    name,
    epochs=epochs,
    activation_fn=torch.relu,
    learning_rate=0.001,
):
    cnn_model = cnn.CNN(
        input_channels=input_channels,
        conv_layers=conv_layers,
        conv_out_channels=conv_out_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        dense_layers=dense_layers,
        dense_units=dense_units,
        seq_length=num_lags,
        activation_fn=activation_fn,
    )
    cnn_model.to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

    _, hist = cnn.train_and_evaluate_cnn(
        device,
        cnn_model,
        optimizer,
        log_interval,
        epochs,
        train_loader,
        test_loader,
        silent=False,
    )
    utils.plot_results(cnn_model, hist, "test_" + name)


i = 0

for conv_layers, conv_out_channels, conv_kernel_sizes in conv_configs:
    for dense_layers, dense_units in dense_configs:
        i += 1
        test_model(
            conv_layers,
            conv_out_channels,
            conv_kernel_sizes,
            dense_layers,
            dense_units,
            str(i),
        )

test_model(2, [64, 128], [3, 5], 2, [256, 128], "good_long_train", epochs=50)
test_model(2, [64, 128], [3, 5], 3, [200, 100, 10], "promising_long_train", epochs=50)
test_model(2, [64, 128], [3, 5], 3, [512, 256, 128], "flat_long_train", epochs=30)
test_model(
    2,
    [64, 128],
    [3, 5],
    3,
    [512, 256, 128],
    "flat_very_long_train_w",
    epochs=1000,
    learning_rate=0.05,
)


functions = [
    F.tanh,
    F.sigmoid,
    F.leaky_relu,
    nn.PReLU(num_parameters=1),
    F.softplus,
]
i = 0
for f in functions:
    i += 1
    test_model(2, [64, 128], [3, 5], 2, [256, 128], "f_" + str(i), activation_fn=f)
