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
# print(data)

data = data.dropna()
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
data[cols] = minmax_scale(data[cols] )

X_train, X_test, y_train, y_test = utils.get_xy_tensors(data, cols, target, test_size=0.2)
X_train = utils.to_pixels(X_train, num_channels = 3)
X_test = utils.to_pixels(X_test, num_channels = 3)
X_val = X_test
y_val = y_test


learning_rate = 0.000005
model_instance = resNet.PriceDirectionClassifier(
    learning_rate=learning_rate,
    resnet_model_name="microsoft/resnet-50",
    resnet_config=None,  
    feature_extractor_name= None,
    is_pretrained=True
)

model_instance.train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=10, log_interval=2)
# model_instance.evaluate(X_test, y_test)
model_instance.plot_results("resnet50-gbpcad")


