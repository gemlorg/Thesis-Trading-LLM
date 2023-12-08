import pandas as pd
import os
from sklearn.svm import SVR
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product

csv_file = os.path.join(os.path.dirname(__file__), "../data/raw_sales.csv")
csv_quarter_file = os.path.join(
    os.path.dirname(__file__), "../data/quarterly_sales.csv"
)
csv_results_path = os.path.join(
    os.path.dirname(__file__), "../results/svm_models_results.csv"
)
png_results_path = os.path.join(
    os.path.dirname(__file__), "../results/svm_models_results.png"
)


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


data = load_and_preprocess_data(csv_file, date_format="%Y-%m-%d %H:%M:%S")
data_quarter = load_and_preprocess_data(csv_quarter_file, date_format="%d/%m/%Y")


def split_data_by_date(
    data,
    data_quarter,
    cutoff_date,
    target_column="price",
    date_column="datesold",
):
    data_train = data[data[date_column] < cutoff_date]
    data_test = data_quarter[data_quarter[date_column] >= cutoff_date]
    # data_test_training = data_quarter[data_quarter[date_column] < cutoff_date]

    X_train = data_train.drop([target_column], axis=1)
    y_train = data_train[target_column]

    X_test = data_test.drop([target_column], axis=1)
    y_test = data_test[target_column]

    # X_test_training = data_test_training.drop([target_column], axis=1)
    # y_test_training = data_test_training[target_column]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )  # X_test_training, y_test_training


cutoff_date = int(datetime.strptime("2017-01-01", "%Y-%m-%d").timestamp())
# X_train, y_train, X_test, y_test, X_test_training, y_test_training= split_data_by_date(
#     data, data_quarter, cutoff_date)
X_train, y_train, X_test, y_test = split_data_by_date(data, data_quarter, cutoff_date)


def train_and_evaluate_svm(
    X_train, y_train, X_test, y_test, kernel="rbf", gamma="scale", C=1.0
):
    svm_model = SVR(kernel=kernel, gamma=gamma, C=C)
    print(f"Training SVM with kernel={kernel}, gamma={gamma}, C={C}")
    svm_model.fit(X_train, y_train)

    y_test_pred = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    return mse, r2


def compare_svm_models(X_train, y_train, X_test, y_test, kernels, gammas, Cs):
    results = {"MSE": [], "R2": []}

    for kernel in kernels:
        for gamma in gammas:
            for C in Cs:
                mse, r2 = train_and_evaluate_svm(
                    X_train, y_train, X_test, y_test, kernel, gamma, C
                )
                results["MSE"].append(mse)
                results["R2"].append(r2)

    return results


kernels_to_try = ["rbf", "poly", "sigmoid"]  # "linear" takes too long
gammas_to_try = ["scale"]  # "auto" doesn't work with "poly" without scaling data
Cs_to_try = [0.1, 1.0, 10.0]

results = compare_svm_models(
    X_train, y_train, X_test, y_test, kernels_to_try, gammas_to_try, Cs_to_try
)


def save_results(results, csv_results_path, png_results_path):
    configurations = []
    for kernel, gamma, C in product(kernels_to_try, gammas_to_try, Cs_to_try):
        configurations.extend([{"Kernel": kernel, "Gamma": gamma, "C": C}])

    results_df = pd.DataFrame(results)
    configurations_df = pd.DataFrame(configurations)
    final_results_df = pd.concat([configurations_df, results_df], axis=1)
    final_results_df.to_csv(csv_results_path, index=False)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    for metric, values in results.items():
        ax = axes[0] if metric == "MSE" else axes[1]

        x_values = range(len(values))
        ax.plot(x_values, values, label=metric)
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [f"{config['Kernel']}, C={config['C']}" for config in configurations]
        )

        ax.plot(values, label=metric)
        ax.set_xlabel("Model Configurations")
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()
    plt.savefig(png_results_path)
    plt.show()


save_results(results, csv_results_path, png_results_path)