import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import torch
import torch.nn as nn
import csv
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
data.drop(["Unnamed: 0"], axis=1, inplace=True)
data = data.dropna()
#don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
# self.data[target] = utils.sigmoid(self.data[target])
#maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols] )
# print(data.head())

train_loader, test_loader = utils.get_data_loaders(data, cols, target)
# print(test_loader)

# use mlp