"""
models/VGG.py
=============
VGG architecture adapted for CIFAR-10 (3-channel, 32x32 images).

Reference:
    Simonyan & Zisserman, "Very Deep Convolutional Networks for
    Large-Scale Image Recognition", arXiv:1409.1556, 2014.
"""

from typing import Dict, List, Type, Union

import torch
import torch.nn as nn


# Configuration table: integers are output channel counts; 'M' inserts MaxPool2d.
_CFG: Dict[str, List[Union[int, str]]] = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M",
           512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """
    VGG network with configurable depth for CIFAR-10 classification.

    Architecture overview:
        features   : a stack of Conv-BN-ReLU blocks separated by MaxPool layers.
                     The exact layout is determined by the depth key in _CFG.
        classifier : three fully-connected layers (512 -> 4096 -> 4096 -> num_class).

    Note on input size:
        This implementation is adapted for 32x32 CIFAR-10 images.  After five
        MaxPool2d layers (each halving spatial dimensions), a 32x32 input
        becomes 1x1, so the classifier head receives a 512-dim vector.
        The original VGG paper used 224x224 ImageNet images which produce
        a 7x7 feature map, requiring a 25088-dim FC input instead.

    Args:
        dept:      Depth variant — one of "11", "13", "16", "19".
        norm:      Normalisation layer constructor (default: BatchNorm2d).
        num_class: Number of output classes.
    """

    def __init__(
        self,
        dept:      str                          = "16",
        norm:      Type[nn.Module]              = nn.BatchNorm2d,
        num_class: int                          = 10,
    ) -> None:
        super().__init__()
        self.features   = self._make_layers(dept, norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class),
        )

    def _make_layers(
        self,
        dept: str,
        norm: Type[nn.Module],
    ) -> nn.Sequential:
        """
        Build the convolutional feature extractor from the config table.

        Each integer entry creates a Conv2d(in, out, 3x3, pad=1) + norm + ReLU.
        Each 'M' entry inserts a MaxPool2d(kernel=2, stride=2).

        Args:
            dept: One of "11", "13", "16", "19".
            norm: Normalisation layer class.

        Returns:
            nn.Sequential: The complete feature extractor.
        """
        layers: List[nn.Module] = []
        in_channels: int = 3

        for v in _CFG[dept]:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(v)
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    norm(out_channels),
                    nn.ReLU(inplace=True),
                ]
                in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, 3, 32, 32).

        Returns:
            torch.Tensor: Class logits, shape (B, num_class).
        """
        out = self.features(x)                    # (B, 512, 1, 1)
        out = out.view(out.size(0), -1)            # (B, 512)
        return self.classifier(out)               # (B, num_class)
