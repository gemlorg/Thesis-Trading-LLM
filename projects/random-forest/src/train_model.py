from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_and_evaluate(
    n_estimators, max_features, criterion, X_train, y_train, X_test, y_test
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        criterion=criterion,
        random_state=123,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)
    return accuracy
