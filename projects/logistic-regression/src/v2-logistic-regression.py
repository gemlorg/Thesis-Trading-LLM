import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import datetime


def timeseries_train_test_split(X, y, test_size):
    test_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test

#let us say that a lag represent an average price throughout the last 4 weeks.
#let us try to predict the price in the next k weeks. 

def train_test_model(data, num_lags):

    for i in range(0, num_lags + 1):
        data["lag_{}".format(i)] = data.price.rolling(30).mean()
        data["lag_{}".format(i)] = data["lag_{}".format(i)].shift(1 + 30 * i)
        
        
    data = data.dropna()


    data["delta"] = data["lag_0"] - data["lag_1"]

    data["delta"] = (np.sign(data["delta"]) + 1) // 2
    


    
    data = data.drop("price", axis=1).dropna()

    y = data.delta
    X = data.drop(["delta", "lag_0"], axis=1)



    X_train, X_test, y_train, y_test = timeseries_train_test_split(
        X, y, test_size=0.3)
    

    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    # prediction = model.predict(X_test, y_test)

    return model.score(X_test, y_test)

#it would be harder to implement the model had we assumed
#  that the number of bedrooms can be arbitrary
num_bedrooms = 3
file_path = "../data/raw_sales.csv"
data = pd.read_csv(file_path, parse_dates=["datesold"])
data.sort_values(by='datesold', inplace=True)

data["datesold"] = data["datesold"].map(datetime.datetime.toordinal)


data = data[(data["propertyType"] == "house") & (data["bedrooms"] == num_bedrooms)]
data = data.drop(["postcode", "propertyType", "bedrooms"], axis=1)


# prob shouldn't edit.
df = pd.DataFrame({"num_lags": [], "accuracy": []})
for num_lags in range(1, 30):
    accuracy = train_test_model(data, num_lags)
    df.loc[len(df)] = [num_lags, accuracy]
df.to_csv("../results/accuracies_monthly.csv")
