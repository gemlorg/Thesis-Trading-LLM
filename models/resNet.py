import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ResNetConfig, ResNetModel, AutoFeatureExtractor, BatchFeature
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class PriceDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        dense_units=128,
        learning_rate=0.001,
        resnet_model_name="microsoft/resnet-50",
        resnet_config=None,
        feature_extractor_name=None,
        is_pretrained=False,
    ):
        super(PriceDirectionClassifier, self).__init__()

        # Initializing a ResNet configuration
        if resnet_config is None:
            resnet_config = ResNetConfig()

        # Initializing a ResNet model from the configuration
        self.resnet_model = ResNetModel(resnet_config)

        # Use AutoFeatureExtractor to automatically load the correct image processor
        if feature_extractor_name is None:
            feature_extractor_name = resnet_model_name

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name, do_resize=False, do_rescale=False
        )
        # self.feature_extractor = ResNetFeatureExtractor(model_name="resnet50", do_resize=True, do_center_crop=True)

        # Configuration parameters
        self.input_size = input_size
        self.dense_units = dense_units
        self.learning_rate = learning_rate

        # Dense layers
        # self.dense = nn.Sequential(
        #     nn.Linear(100352, self.dense_units),
        #     nn.ReLU(),
        #     nn.Linear(self.dense_units, 1),
        #     nn.Sigmoid(),
        # )
        self.dense = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

        self.train_history = []
        self.acc_history = []

    def forward(self, data):
        # Processing the data with the feature extractor
        # second_extractor = BatchFeature()
        inputs = self.feature_extractor(data, return_tensors="pt")
        # inputs_v2 = second_extractor(data, return_tensors="pt")


        # Getting the last hidden states from the resnet model
        last_hidden_states = self.resnet_model(**inputs).last_hidden_state

        # Flatten
        x = last_hidden_states.view(last_hidden_states.size(0), -1)

        # Dense layers
        x = self.dense(x)
        return x

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_data = X_train[i : i + batch_size]
                batch_labels = y_train[i : i + batch_size]

                self.optimizer.zero_grad()
                outputs = self(batch_data)

                loss = self.criterion(
                    outputs, batch_labels.view(-1, 1)
                )
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = self.criterion(
                    val_outputs, y_val.view(-1, 1)
                )

            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}"
            )
            self.train_history.append(loss.item())

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            test_outputs = self(X_test)

            test_predictions = (test_outputs > 0.5).float().numpy()

        accuracy = accuracy_score(y_test.numpy(), test_predictions)
        print(f"Accuracy on Test Set: {accuracy}")
        self.acc_history.append(accuracy)
        return accuracy

    def plot_results(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_history, label="train")
        plt.plot(self.acc_history, label="test")
        plt.legend()
        plt.show()

# X_train, X_val, X_test, y_train, y_val, y_test = ...

# Example usage:
# input_size = X_train.shape[1]
# dense_units = 256
# learning_rate = 0.0001
# model_instance = PriceDirectionClassifier(
#     input_size=input_size,
#     dense_units=dense_units,
#     learning_rate=learning_rate,
#     resnet_model_name="microsoft/resnet-50",
#     resnet_config=None,
#     feature_extractor_name=None
# )
# model_instance.train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
# accuracy = model_instance.evaluate(X_test, y_test)
