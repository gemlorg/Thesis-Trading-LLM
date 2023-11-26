import process_data
import train_model

num_lags_list = [1, 5, 10, 13, 25, 40, 50]
n_estimators_list = [20, 50, 100]
max_features_list = [2, 4, 8]
criterion_list = ["gini", "entropy", "log_loss"]

results_list = []

for num_lags in num_lags_list:
    data = process_data.get_data(num_lags)
    X_train, y_train, X_test, y_test = process_data.split_data(data, 0.3)
    for n_estimators in n_estimators_list:
        for max_features in max_features_list:
            for criterion in criterion_list:
                accuracy = train_model.train_and_evaluate(
                    n_estimators,
                    max_features,
                    criterion,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )
                results_list.append(
                    {
                        "num_lags": num_lags,
                        "n_estimators": n_estimators,
                        "max_features": max_features,
                        "criterion": criterion,
                        "accuracy": accuracy,
                    }
                )
                print(
                    "num_lags =",
                    num_lags,
                    "n_estimators =",
                    n_estimators,
                    " max_features =",
                    max_features,
                    " criterion =",
                    criterion,
                    " accuracy =",
                    accuracy,
                )

process_data.save_results_to_csv(results_list)
