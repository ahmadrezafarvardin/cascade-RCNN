# check_stride.py
import torch
from src.models.backbone import SimpleBackbone


def check_feature_stride():
    backbone = SimpleBackbone()

    # Test input
    test_input = torch.randn(1, 3, 800, 800)

    # Get output
    features = backbone(test_input)

    print("Input shape:", test_input.shape)
    print("Output shape:", features["0"].shape)

    # Calculate stride
    stride_h = test_input.shape[2] / features["0"].shape[2]
    stride_w = test_input.shape[3] / features["0"].shape[3]

    print(f"Feature stride: {stride_h:.1f} x {stride_w:.1f}")
    print(
        f"This means each feature map cell covers {stride_h:.1f}x{stride_w:.1f} pixels in the original image"
    )


if __name__ == "__main__":
    check_feature_stride()
