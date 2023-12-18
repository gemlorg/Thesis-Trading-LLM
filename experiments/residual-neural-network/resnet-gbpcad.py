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
num_entries = 1000
year=2023
num_years = 8
test_size=0.2
data = utils.data_for_year(data, year, num_years, num_entries)
data.drop(["id", "provider", "insertTimestamp", "dayOfWeek"], axis=1, inplace=True)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data = data.dropna()
#don't normalise the target column
target = ["price_delta"]
cols = data.drop(target, axis=1).select_dtypes(np.number).columns
#maybe should use another method for normalisation
data[cols] = minmax_scale(data[cols] )



X_train, X_test, y_train, y_test = utils.get_xy_tensors(data, cols, target, test_size=test_size)
X_train = utils.to_pixels(X_train, num_channels = 3)
X_test = utils.to_pixels(X_test, num_channels = 3)
X_val = X_test
y_val = y_test

dense_units = 256
learning_rate = 0.00001
epochs=200
is_pretrained = False
model_instance = resNet.PriceDirectionClassifier(
    dense_units=dense_units,
    learning_rate=learning_rate,
    resnet_model_name="microsoft/resnet-50",
    resnet_config=None,
    feature_extractor_name= None,
    is_pretrained=is_pretrained
)



model_instance.train_model(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=10)

model_instance.plot_results("./images/resnet-gbpcad_"+str(learning_rate)+"_"+str(epochs), string="GBP/CAD, ResNet50, lr="+str(learning_rate)+", epochs="+str(epochs)+", year=" + str(year) + ", entries=" + str(num_entries) + ", test_size=" + str(test_size))

