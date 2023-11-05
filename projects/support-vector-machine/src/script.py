import pandas as pd
import os

csv_file = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")
data = pd.read_csv(csv_file)

data["datesold"] = pd.to_datetime(data["datesold"])
# data['year'] = data['datesold'].dt.year
# data['quarter'] = data['datesold'].dt.quarter
data = pd.get_dummies(data, columns=["propertyType"], prefix="propertyType")

data_train = data[data["datesold"] < "2018-01-01"]
data_test = data[data["datesold"] >= "2018-01-01"]

X_train = data_train.drop(["price"], axis=1)
y_train = data_train["price"]

X_test = data_test.drop(["price"], axis=1)
y_test = data_test["price"]

y = data["price"]
X = data.drop(["price"], axis=1)

print(X_train.head())
print(X_test.head())

print(y_train.head())
print(y_test.head())

# from sklearn.svm import SVR
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
# svm_model = SVR(kernel='linear', C=1.0)
# svm_model.fit(X_train, y_train)

# y_pred = svm_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'MSE: {mse}')
# print(f'R2: {r2}')

# # plot decision boundary
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_decision_boundary(X, y, model):
#     xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.01),
#                          np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.01))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.8)
#     plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
#     plt.show()

# plot_decision_boundary(X_train, y_train, svm_model)


# import numpy as np
# from sklearn.svm import SVC
# import matplotlib.pyplot as plt

# accuracy = svm_model.score(X_test, y_test)
# print(f'Model accuracy SVM: {accuracy}')

