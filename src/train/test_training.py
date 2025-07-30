# src/train/test_training.py
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.models.anchor_generator import AnchorGenerator
from src.models.heads import RPNHead
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform, collate_fn
from torch.utils.data import DataLoader


def test_model_forward():
    print("Testing model initialization and forward pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    backbone = SimpleBackbone()
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150), aspect_ratios=(0.15, 0.2, 0.3, 0.5)
    )

    model = CascadeRCNN(backbone)
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)
    model.to(device)

    print("✓ Model initialized successfully")

    # Test data loading
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    dataset = MathExpressionDataset(root_dir, "train", get_transform(train=True))
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Test forward pass
    model.train()
    images, targets = next(iter(data_loader))
    images = images.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    print("✓ Training forward pass successful")
    print(f"  Loss components: {list(loss_dict.keys())}")

    # Test inference
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    print("✓ Inference forward pass successful")
    print(f"  Predictions: {len(predictions[0]['boxes'])} boxes")

    print("\nAll tests passed! Training script should work correctly.")


if __name__ == "__main__":
    test_model_forward()
