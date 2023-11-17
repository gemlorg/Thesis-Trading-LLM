import pandas as pd
import numpy as np
import os
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data_path = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")


def add_lags_columns(data, num_lags, price_column):
    for i in range(1, num_lags + 1):
        data["lag_{}".format(i)] = data[price_column].shift(i)
    data = data.dropna()
    return data


def compute_price_deltas(data, price_column):
    data["price_delta"] = data[price_column] - data["lag_1"]
    data["price_delta"] = np.sign(data["price_delta"])
    return data


def get_data(date_column="datesold", price_column="price"):
    data = pd.read_csv(data_path, parse_dates=[date_column])
    data[date_column] = data[date_column].map(datetime.datetime.toordinal)
    data = data[(data["propertyType"] == "house") & (data["bedrooms"] == 3)]

    columns_to_keep = [date_column, price_column]
    data = data[columns_to_keep]

    data = add_lags_columns(data, 4, price_column)
    data = compute_price_deltas(data, price_column)
    data = data.drop([price_column], axis=1)
    return data


def split_data(data, test_size, target_column="price_delta"):
    test_index = int(len(data) * (1 - test_size))

    data_train = data.iloc[:test_index]
    data_test = data.iloc[test_index:]

    X_train = data_train.drop([target_column], axis=1)
    y_train = data_train[target_column]

    X_test = data_test.drop([target_column], axis=1)
    y_test = data_test[target_column]

    return X_train, y_train, X_test, y_test


def train_and_evaluate_mlp(mlp_model, X_train, y_train, X_test, y_test):
    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


data = get_data()

X_train, y_train, X_test, y_test = split_data(data, 0.3)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(80), activation="logistic", random_state=1
)

accuracy = train_and_evaluate_mlp(mlp_model, X_train, y_train, X_test, y_test)
print("accuracy =", accuracy)
