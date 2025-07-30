# src/models/backbone_reduced_stride.py
import torch.nn as nn
import torch.nn.functional as F


class SimpleBackboneReducedStride(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 256

        # Reduced stride for better small object detection
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # Remove the last strided conv to reduce total stride from 32 to 16
        self.conv4 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1
        )  # stride=1 instead of 2

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return {"0": x}
