import process_data
import train_model

num_lags = 10

data = process_data.get_data(num_lags)
X_train, y_train, X_test, y_test = process_data.split_data(data, 0.3)

n_estimators_list = [20, 50, 100]
max_features_list = [2, 4, 8]
criterion_list = ["gini", "entropy", "log_loss"]

for n_estimators in n_estimators_list:
    for max_features in max_features_list:
        for criterion in criterion_list:
            accuracy = train_model.train_and_evaluate(
                n_estimators, max_features, criterion, X_train, y_train, X_test, y_test
            )
            print(
                "n_estimators =",
                n_estimators,
                " max_features =",
                max_features,
                " criterion =",
                criterion,
                " accuracy =",
                accuracy,
            )
