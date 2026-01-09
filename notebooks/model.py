"""
Neural network model for gravitational wave detection.
"""

import torch
import torch.nn as nn


class GWClassifier(nn.Module):
    """
    1D Convolutional Neural Network for classifying GW signals.

    Architecture:
    - 3 convolutional layers with ReLU activation and max pooling
    - 2 fully connected layers
    - Sigmoid output for binary classification
    """

    def __init__(self, input_length: int = 300):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        # Calculate size after convolutions and pooling
        # After 3 pooling layers: input_length // 8
        conv_output_size = input_length // 8

        # Fully connected layers
        self.fc1 = nn.Linear(64 * conv_output_size, 64)
        self.fc2 = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, 1, length)
        x = self.pool(self.relu(self.conv1(x)))  # -> (batch, 16, length/2)
        x = self.pool(self.relu(self.conv2(x)))  # -> (batch, 32, length/4)
        x = self.pool(self.relu(self.conv3(x)))  # -> (batch, 64, length/8)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x.squeeze(-1)


def create_model(input_length: int = 300) -> GWClassifier:
    """Create a new GW classifier model."""
    return GWClassifier(input_length=input_length)
