"""Internal Convolutional Neural Network of the Energy Based Model."""

from math import ceil
from torch import nn, sigmoid


class Swish(nn.Module):

    """Swish activation function."""

    def forward(self, input_val):
        """Forward pass."""
        return input_val * sigmoid(input_val)


class CNNModel(nn.Module):

    """Internal Convolutional Neural Network for the Energy Based Model."""

    # pylint: disable=unused-argument
    def __init__(self, img_shape, *args, **kwargs):
        """Initialize the CNN."""
        super().__init__()
        first_stage = img_shape[1] / 2 + 2
        sequence_len = ceil(ceil(ceil(first_stage / 2) / 2) / 2)
        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=5, stride=2, padding=4
            ),  # [16x16] - Larger padding to get 32x32 image
            Swish(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [8x8]
            Swish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # [2x2]
            Swish(),
            nn.Flatten(),
            nn.Linear(sequence_len * 64, 64),
            Swish(),
            nn.Linear(64, 1),
        )

    def forward(self, input_val):
        """Forward pass of the CNN."""
        return self.cnn_layers(input_val.float()).squeeze(dim=-1)
