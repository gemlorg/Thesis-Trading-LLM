import pandas as pd
import os
from sklearn.svm import SVR

csv_file = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")
data = pd.read_csv(csv_file)

# csv_quarter_file = os.path.join(os.path.dirname(__file__), "../data/quarterly_sales.csv")
# data_quarter = pd.read_csv(csv_quarter_file)
# data_quarter = data_quarter.rename(columns={'MA': 'price'})
# data_quarter = data_quarter[(data_quarter["type"] == "house") & (data_quarter["bedrooms"] == 3)]
# data_quarter = data_quarter.drop(["type", "bedrooms"], axis=1)
# data_quarter["saledate"] = pd.to_datetime(data_quarter["saledate"], dayfirst=True).dt.date
# data_quarter["saledate"] = data_quarter["saledate"].apply(lambda x: x.toordinal())


# TODO: train on entire dataset 
data = data[(data["propertyType"] == "house")]
data = data[(data["bedrooms"] == 3)]
data = data.drop(["propertyType", "bedrooms"], axis=1)
# data = data.drop(["postcode"], axis=1)

data["datesold"] = pd.to_datetime(data["datesold"]).dt.date
data["datesold"] = data["datesold"].apply(lambda x: x.toordinal())
# data = pd.get_dummies(data, columns=["propertyType"], prefix="propertyType")

cutoff_date = pd.to_datetime("2017-01-01").date().toordinal()
data_train = data[data["datesold"] < cutoff_date]
# data_test = data_quarter[data_quarter["saledate"] >= cutoff_date]
data_test = data[data["datesold"] >= cutoff_date]
# data_test = data_test.groupby(["datesold"]).mean()

X_train = data_train.drop(["price"], axis=1)
y_train = data_train["price"]

X_test = data_test.drop(["price"], axis=1)
y_test = data_test["price"]

# print(X_train)
# print(y_train.head())

# print(X_test)
# print(y_test.head())

from sklearn.metrics import mean_squared_error, r2_score

svm_model = SVR(kernel="linear", C=1.0)
print("Start training")
svm_model.fit(X_train, y_train)
print("End training")


y_train_pred = svm_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"Training MSE: {mse_train}")
print(f"Training R2: {r2_train}")


y_pred = svm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"R2: {r2}")


# plot decision boundary
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_decision_boundary(X, y, model):
#     xx, yy = np.meshgrid(np.arange(X['datesold'].min() - 1, X['datesold'].max() + 1, 1),
#                          np.arange(X['postcode'].min() - 1, X['postcode'].max() + 1, 1))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.8)
#     plt.scatter(X['datesold'], X['postcode'], c=y, marker='o', edgecolor='k')
#     plt.show()

# plot_decision_boundary(X_train, y_train, svm_model)
