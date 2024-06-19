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
