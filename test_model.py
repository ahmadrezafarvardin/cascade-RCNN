# test_model.py
import torch
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone


def test_cascade_rcnn():
    # Create model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone, num_stages=3)
    model.eval()

    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 800, 800)

    # Test inference
    with torch.no_grad():
        outputs = model(images)

    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Image {i}:")
        print(f"  Boxes shape: {output['boxes'].shape}")
        print(f"  Scores shape: {output['scores'].shape}")
        print(f"  Labels shape: {output['labels'].shape}")

    # Test training
    model.train()
    targets = []
    for i in range(batch_size):
        targets.append(
            {
                "boxes": torch.tensor(
                    [[10, 10, 50, 50], [100, 100, 200, 200]], dtype=torch.float32
                ),
                "labels": torch.tensor([1, 1], dtype=torch.int64),
                "image_id": torch.tensor([i]),
                "area": torch.tensor([1600, 10000], dtype=torch.float32),
                "iscrowd": torch.zeros((2,), dtype=torch.int64),
                "orig_size": torch.tensor([800, 800], dtype=torch.float32),
                "size": torch.tensor([800, 800], dtype=torch.float32),
            }
        )

    losses = model(images, targets)
    print("\nTraining losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")


if __name__ == "__main__":
    test_cascade_rcnn()
