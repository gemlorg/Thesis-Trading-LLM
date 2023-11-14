import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
import os


def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test


# let us say that a lag represent an average price throughout the last 4 weeks.
# let us try to predict the price in the next k weeks.


def train_test_model(d, num_lags):
    data = pd.DataFrame(d)
    for i in range(0, num_lags + 1):
        data["lag_{}".format(i)] = data.price.rolling(30).mean()
        data["lag_{}".format(i)] = data["lag_{}".format(i)].shift(1 + 30 * (i+1))

    data = data.dropna()
    y = data["lag_0"] - data["lag_1"]
    y = y.apply(lambda x: (np.sign(x) + 1) // 2)
    X = data.drop(["lag_0", "price"], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

def train_draw_model(d, num_lags, truedate):
    data = pd.DataFrame(d)
    for i in range(0, num_lags + 1):
        data["lag_{}".format(i)] = data.price.rolling(30).mean()
        data["lag_{}".format(i)] = data["lag_{}".format(i)].shift(1 + 30 * (i+1))

    data = data.dropna()
    y = data["lag_0"] - data["lag_1"]
    y = y.apply(lambda x: (np.sign(x) + 1) // 2)
    X = data.drop(["lag_0", "price"], axis=1)

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    data['predicted_delta'] = model.predict(X)
    data["truedate"] = truedate
    # data["predicted_delta"].to_csv("./datapred.csv")


    # Create a new column to store whether the prediction is correct
    data['prediction_correct'] = data['predicted_delta'] == y

    data = data.iloc[::30]

    # Separate the data into correctly predicted and incorrectly predicted points
    correct_predictions = data[data['prediction_correct']]
    incorrect_predictions = data[~data['prediction_correct']]

    

    # Plot the correctly predicted points in green and incorrectly predicted points in red
    
    plt.scatter(correct_predictions['truedate'], correct_predictions['price'], c='green', label='Correct Predictions')
    plt.scatter(incorrect_predictions['truedate'], incorrect_predictions['price'], c='red', label='Incorrect Predictions')

    plt.xlabel('Date Sold')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Logistic Regression Predictions For Monthly Average')

    plt.savefig("../results/data/monthly-predictions-colored.png")



# it would be harder to implement the model had we assumed
#  that the number of bedrooms can be arbitrary
num_bedrooms = 3
file_path = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")
data = pd.read_csv(file_path, parse_dates=["datesold"])
data.sort_values(by="datesold", inplace=True)

truedate = data["datesold"]
data["datesold"] = data["datesold"].map(datetime.datetime.toordinal)


data = data[(data["propertyType"] == "house") & (data["bedrooms"] == num_bedrooms)]
data = data.drop(["postcode", "propertyType", "bedrooms"], axis=1)





# prob shouldn't edit.
df = pd.DataFrame({"num_lags": [], "accuracy": []})
for num_lags in range(1, 30):
    accuracy = train_test_model(data, num_lags)
    df.loc[len(df)] = [num_lags, accuracy]
train_draw_model(data, 20, truedate)
df.to_csv("../results/accuracies_monthly.csv")
