import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
import torch
import torch.nn as nn
import csv

import torch.optim as optim
sys.path.append("../..")
import models.resNet as resNet
import experiments.utils as utils
from sklearn.preprocessing import minmax_scale, scale


data_path = os.path.join(
    os.path.dirname(__file__), "../../data/gbpcad_one_hour_202311210827.csv"
)
torch.manual_seed(42)
num_lags = 17

data = utils.get_data(data_path, num_lags, date_column="barTimestamp", price_column="close", date_format="%Y-%m-%d %H:%M:%S")
data = data.iloc[:1000]
data.drop(["id", "provider", "insertTimestamp", "dayOfWeek"], axis=1, inplace=True)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data = data.dropna()
#don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
#maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols] )



X_train, X_val, X_test, y_train, y_val, y_test = utils.get_xy_tensors(data, cols, target, test_size=0.2)

t = torch.from_numpy(X_train.to_numpy()).float()

split_tensor = [row.chunk(int(t.shape[1]/3), dim=0) for row in t]
result = [[[part.tolist()] for part in row_parts] for row_parts in split_tensor]
X_train = torch.tensor(result)
X_val = X_train

t = torch.from_numpy(X_test.to_numpy()).float()

split_tensor = [row.chunk(int(t.shape[1]/3), dim=0) for row in t]
result = [[[part.tolist()] for part in row_parts] for row_parts in split_tensor]
X_test = torch.tensor(result)
y_val = torch.from_numpy(y_train.to_numpy()).float()
y_train = y_val
y_test = torch.from_numpy(y_test.to_numpy()).float()


input_size = X_train.shape[1]
dense_units = 256
learning_rate = 0.001
is_pretrained = False
model_instance = resNet.PriceDirectionClassifier(
    input_size=input_size,
    dense_units=dense_units,
    learning_rate=learning_rate,
    resnet_model_name="microsoft/resnet-50",
    resnet_config=None,
    feature_extractor_name= None,
    is_pretrained=is_pretrained
)

for _ in range(10):
    model_instance.train_model(X_train, y_train, X_val, y_val, epochs=1, batch_size=32)
    model_instance.evaluate(X_test, y_test)
model_instance.plot_results()
