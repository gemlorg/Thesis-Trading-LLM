import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ResNetConfig, ResNetModel, AutoFeatureExtractor
from sklearn.metrics import accuracy_score
import numpy as np


class PriceDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        dense_units=128,
        learning_rate=0.001,
        resnet_model_name="microsoft/resnet-50",
        resnet_config=None,
        feature_extractor_name=None,
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
            feature_extractor_name
        )

        # Configuration parameters
        self.input_size = input_size
        self.dense_units = dense_units
        self.learning_rate = learning_rate

        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(self.input_size, self.dense_units),
            nn.ReLU(),
            nn.Linear(self.dense_units, 1), 
            nn.Sigmoid(), 
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

    def forward(self, data):
        # Processing the data with the feature extractor
        inputs = self.feature_extractor(inputs=data, return_tensors="pt")

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
                outputs = self(torch.tensor(batch_data, dtype=torch.float32))
                loss = self.criterion(
                    outputs, torch.tensor(batch_labels, dtype=torch.float32).view(-1, 1)
                )
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                val_outputs = self(torch.tensor(X_val, dtype=torch.float32))
                val_loss = self.criterion(
                    val_outputs, torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
                )

            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}"
            )

    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            test_outputs = self(torch.tensor(X_test, dtype=torch.float32))
            test_predictions = (test_outputs > 0.5).float().numpy()

        accuracy = accuracy_score(y_test, test_predictions)
        print(f"Accuracy on Test Set: {accuracy}")
        return accuracy


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
