import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ResNetConfig, ResNetModel, AutoFeatureExtractor, BatchFeature
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import datetime


class ResNet(nn.Module):
    def __init__(
        self,
        input_size=2048,
        dense_units=128,
        learning_rate=0.001,
        resnet_model_name="microsoft/resnet-50",
        resnet_config=None,
        feature_extractor_name=None,
        is_pretrained=False,
    ):
        super(ResNet, self).__init__()

        # Initializing a ResNet configuration
        if resnet_config is None:
            resnet_config = ResNetConfig()

        # Initializing a ResNet model from the configuration
        if is_pretrained:
            self.resnet_model = ResNetModel.from_pretrained(
                resnet_model_name, config=resnet_config
            )
        else:
            self.resnet_model = ResNetModel(resnet_config)

        # Use AutoFeatureExtractor to automatically load the correct image processor
        if feature_extractor_name is None:
            feature_extractor_name = resnet_model_name

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name, do_resize=False, do_rescale=False, do_normalize=False
        )

        # Configuration parameters
        self.input_size = input_size
        self.dense_units = dense_units
        self.learning_rate = learning_rate

        self.dense = nn.Sequential(
            nn.Linear(self.input_size, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

        self.train_history = []
        self.acc_history = []
        self.train_loss_history = []
        self.acc_loss_history = []

    def forward(self, data):
        # Processing the data with the feature extractor
        inputs = self.feature_extractor(data, return_tensors="pt")
        # Getting the last hidden states from the resnet model
        last_hidden_states = self.resnet_model(**inputs).last_hidden_state
        # Flatten
        x = last_hidden_states.view(last_hidden_states.size(0), -1)
        # Dense layers
        x = self.dense(x)
        return x

    def train_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, log_interval=2):
        for epoch in range(epochs):

            train_outputs = []
            for i in range(0, len(X_train), batch_size):
                batch_data = X_train[i: i + batch_size]
                batch_labels = y_train[i: i + batch_size]

                self.optimizer.zero_grad()
                outputs = self(batch_data)
                train_outputs.append(outputs)

                loss = self.criterion(
                    outputs, batch_labels.view(-1, 1)
                )
                loss.backward()
                self.optimizer.step()
            train_outputs = torch.cat(train_outputs)

            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = self.criterion(
                    val_outputs, y_val.view(-1, 1)
                )
                val_predictions = (val_outputs > 0.5).float().numpy()
                val_accuracy = accuracy_score(y_val.numpy(), val_predictions)

                train_predictions = (train_outputs > 0.5).float().numpy()
                train_accuracy = accuracy_score(
                    y_train.numpy(), train_predictions)

            if (epoch) % log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, ({round(train_accuracy * len(y_train))}/{len(y_train)}), Val Accuracy: {val_accuracy:.4f} ({round(val_accuracy * len(y_val))}/{len(y_val)}), Train Loss: {loss.item():.8f}, Val Loss: {val_loss.item():.8f}"
                )
            self.train_history.append(train_accuracy)
            self.acc_history.append(val_accuracy)
            self.train_loss_history.append(loss.item())
            self.acc_loss_history.append(val_loss.item())

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            test_outputs = self(X_test)

            test_predictions = (test_outputs > 0.5).float().numpy()

        accuracy = accuracy_score(y_test.numpy(), test_predictions)
        print(
            f"Accuracy on Test Set: {accuracy} ({round(accuracy * len(y_test))}/{len(y_test)})")
        return accuracy

    def plot_results(self, where, string=""):
        plt.figure(figsize=(11, 5))
        plt.suptitle(string)
        plt.subplot(1, 2, 1)
        plt.title("Accuracy")
        plt.plot(self.train_history, label="train")
        plt.plot(self.acc_history, label="val")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Loss")
        plt.plot(self.train_loss_history, label="train loss")
        plt.plot(self.acc_loss_history, label="val loss")
        plt.legend()
        plt.savefig(where +  "_"+str(datetime.datetime.now())+".png")

        plt.show()
