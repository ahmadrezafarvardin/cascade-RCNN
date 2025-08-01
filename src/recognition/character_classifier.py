# src/recognition/character_classifier.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from pathlib import Path


class CharacterClassifier(nn.Module):
    """
    ResNet18-based character classifier for mathematical expressions
    """

    def __init__(self, num_classes=16, pretrained=True):
        """
        Initialize character classifier

        Args:
            num_classes: Number of character classes (0-9, +, -, *, /, (, ))
            pretrained: Use pretrained ResNet18 weights
        """
        super(CharacterClassifier, self).__init__()

        # Character mapping
        self.char_to_idx = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "+": 10,
            "-": 11,
            "*": 12,
            "/": 13,
            "(": 14,
            ")": 15,
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Modify first conv layer to accept grayscale images
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def forward(self, x):
        return self.backbone(x)

    def preprocess_image(self, image):
        """Preprocess character image for classification"""
        if isinstance(image, np.ndarray):
            return self.transform(image)
        return image
