"""
models/MLP.py
=============
Multi-Layer Perceptron for image classification.

Suitable for MNIST (flattened 784-dim input).
Not recommended for CIFAR-10 — spatial structure is discarded by flattening.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Fully-connected feed-forward network with BatchNorm, ReLU, and Dropout.

    Each hidden layer follows the pattern:
        Linear -> BatchNorm1d -> ReLU -> Dropout

    BatchNorm1d normalises the pre-activation outputs of each linear layer to
    zero mean and unit variance within the batch.  This stabilises training,
    reduces sensitivity to the learning rate, and provides mild regularisation.

    Args:
        input_size:   Dimensionality of the flattened input (e.g. 784 for MNIST).
        hidden_sizes: Number of units in each hidden layer (e.g. [512, 256, 128]).
        num_classes:  Number of output classes.
        dropout:      Dropout probability applied after each hidden ReLU.
    """

    def __init__(
        self,
        input_size:   int,
        hidden_sizes: List[int],
        num_classes:  int,
        dropout:      float = 0.3,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_dim: int = input_size

        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, input_size).

        Returns:
            torch.Tensor: Class logits of shape (B, num_classes).
        """
        x = x.view(x.size(0), -1)   # flatten any image shape to (B, input_size)
        return self.net(x)


class MLP2(nn.Module):
    """
    Minimal MLP variant without BatchNorm or Dropout.

    Useful as a lightweight baseline to isolate the effect of regularisation
    techniques when compared against the full MLP class above.

    Args:
        input_dim:   Flattened input dimensionality.
        hidden_dims: List of hidden layer widths.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim:   int       = 784,
        hidden_dims: List[int] = None,
        num_classes: int       = 10,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: List[nn.Module] = []
        prev_dim: int = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer  = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, *).

        Returns:
            torch.Tensor: Class logits, shape (B, num_classes).
        """
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
