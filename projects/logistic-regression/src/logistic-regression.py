import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn import metrics
import datetime
import os
import pandas as pd
import os
from sklearn.svm import SVR
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product

from utils import *



def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test


def train_test_model(data, num_lags):
    for i in range(1, num_lags + 1):
        data["lag_{}".format(i)] = data.price.shift(i)

    data["delta"] = data.price - data.lag_1
    data["delta"] = (np.sign(data["delta"]) + 1) // 2

    data = data.drop("price", axis=1)
    data = data.dropna()

    y = data.delta
    X = data.drop("delta", axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)


    return model.score(X_test, y_test)




file_path = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")
data = pd.read_csv(file_path, parse_dates=["datesold"])
# print(data.head())

data = load_and_preprocess_data(file_path, date_format="%Y-%m-%d %H:%M:%S")



df = pd.DataFrame({"num_lags": [], "accuracy": []})
for num_lags in range(1, 100):
    accuracy = train_test_model(data, num_lags)
    df.loc[len(df)] = [num_lags, accuracy]
df.to_csv("../results/accuracies.csv")
