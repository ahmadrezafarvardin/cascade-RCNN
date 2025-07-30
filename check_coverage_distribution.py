# check_coverage_distribution.py
import torch
import numpy as np
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
from torchvision.ops import box_iou
import matplotlib.pyplot as plt


def check_coverage_distribution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with new anchor sizes
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)
    model.to(device)
    model.eval()

    # Load dataset
    root_dir = "dataset"
    dataset = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    all_best_ious = []

    for i in range(min(20, len(dataset))):
        img, target = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        # Get features and anchors
        features = model.backbone(img_tensor)
        image_size = target["orig_size"].tolist()
        anchors = model.anchor_generator(features, [image_size])

        # Check coverage
        gt_boxes = target["boxes"]
        if len(gt_boxes) > 0 and len(anchors[0]) > 0:
            ious = box_iou(gt_boxes, anchors[0].cpu())
            best_iou_per_gt = ious.max(dim=1)[0]
            all_best_ious.extend(best_iou_per_gt.numpy())

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_best_ious, bins=50, alpha=0.7, edgecolor="black")
    plt.axvline(x=0.5, color="red", linestyle="--", label="IoU=0.5 threshold")
    plt.axvline(x=0.3, color="orange", linestyle="--", label="IoU=0.3 threshold")
    plt.xlabel("Best IoU per GT box")
    plt.ylabel("Count")
    plt.title("Distribution of Best Anchor IoUs with GT Boxes")
    plt.legend()
    plt.savefig("iou_distribution.png")

    # Print statistics
    all_best_ious = np.array(all_best_ious)
    print(f"Total GT boxes analyzed: {len(all_best_ious)}")
    print(f"Coverage at different thresholds:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        coverage = (all_best_ious >= threshold).sum() / len(all_best_ious)
        print(
            f"  IoU >= {threshold}: {coverage:.1%} ({(all_best_ious >= threshold).sum()}/{len(all_best_ious)})"
        )


if __name__ == "__main__":
    check_coverage_distribution()
