import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import datetime

file_path = "../data/raw_sales.csv"
data = pd.read_csv(file_path, parse_dates=["datesold"])
data["datesold"] = data["datesold"].map(datetime.datetime.toordinal)

data = data[data["propertyType"] == "house"]
data = data.drop(["postcode", "propertyType"], axis=1)


def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test


def calc_accuracy(prediction, y_test):
    df = prediction * y_test
    return df[df > 0].count() / y_test.count()


def train_test_model(data, num_lags):
    for i in range(1, num_lags + 1):
        data["lag_{}".format(i)] = data.price.shift(i)


    data["delta"] = data.price - data.lag_1
    data["delta"] = (np.sign(data["delta"]) + 1) // 2
    data = data.drop("price", axis=1).dropna()

    y = data.delta
    X = data.drop("delta", axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(random_state = 0)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, prediction)
    accuracy = (cnf_matrix[0][0] + cnf_matrix[1][1]) / y_test.count()
    print("num_lags =", num_lags, " accuracy =", accuracy)


for num_lags in range(1, 50):
    train_test_model(data, num_lags)