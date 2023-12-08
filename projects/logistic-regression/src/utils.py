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

def preprocess_data(df, date_column="datesold", date_format="%Y-%m-%d %H:%M:%S"):
    df = df[(df["propertyType"] == "house") & (df["bedrooms"] == 3)]
    # data = pd.get_dummies(data, columns=["propertyType"], prefix="propertyType")

    columns_to_keep = ["datesold", "price"]
    df = df[columns_to_keep]
    df[date_column] = df[date_column].apply(
        lambda x: int(datetime.strptime(x, date_format).timestamp())
    )
    return df

def load_and_preprocess_data(file_path, date_format="%Y-%m-%d %H:%M:%S"):
    data = pd.read_csv(file_path)
    data = preprocess_data(data, date_format=date_format)
    return data