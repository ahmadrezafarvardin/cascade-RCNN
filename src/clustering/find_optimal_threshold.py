# src/clustering/find_optimal_threshold.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.cascade_rcnn import CascadeRCNN
from models.backbone import SimpleBackbone
from models.anchor_generator import AnchorGenerator
from models.heads import RPNHead


def test_thresholds(model_path, image_path, output_dir):
    """Test different score thresholds on a single image"""

    # Initialize model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150),
        aspect_ratios=(0.15, 0.2, 0.3, 0.5),
    )
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        predictions = model(img_tensor)

    # Test different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, threshold in enumerate(thresholds):
        # Copy image
        img_copy = image.copy()

        if len(predictions) > 0:
            pred = predictions[0]

            # Count detections above threshold
            valid_detections = (pred["scores"] > threshold).sum().item()

            # Draw boxes
            for i in range(len(pred["boxes"])):
                if pred["scores"][i] > threshold:
                    box = pred["boxes"][i].cpu().numpy().astype(int)
                    score = pred["scores"][i].item()
                    x1, y1, x2, y2 = box

                    # Color based on score
                    if score > 0.15:
                        color = (0, 255, 0)
                    elif score > 0.1:
                        color = (255, 255, 0)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Convert to RGB for matplotlib
        img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(img_copy_rgb)
        axes[idx].set_title(f"Threshold: {threshold}\n{valid_detections} detections")
        axes[idx].axis("off")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "threshold_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved threshold comparison to {output_path}")

    # Also save individual images
    for threshold in thresholds:
        img_copy = image.copy()

        if len(predictions) > 0:
            pred = predictions[0]

            for i in range(len(pred["boxes"])):
                if pred["scores"][i] > threshold:
                    box = pred["boxes"][i].cpu().numpy().astype(int)
                    score = pred["scores"][i].item()
                    x1, y1, x2, y2 = box

                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img_copy,
                        f"{score:.3f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        output_path = os.path.join(output_dir, f"threshold_{threshold:.3f}.png")
        cv2.imwrite(output_path, img_copy)


if __name__ == "__main__":
    model_path = "../../results/cascade_rcnn_best.pth"
    dataset_path = "../../dataset"
    output_dir = "../../results/clustering/threshold_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Test on a few different images
    train_images = os.path.join(dataset_path, "train", "images")

    if os.path.exists(train_images):
        # Get a few sample images
        image_files = sorted(os.listdir(train_images))[:5]

        for img_file in image_files:
            image_path = os.path.join(train_images, img_file)
            print(f"\nTesting thresholds on {img_file}")

            img_output_dir = os.path.join(output_dir, img_file.split(".")[0])
            os.makedirs(img_output_dir, exist_ok=True)

            test_thresholds(model_path, image_path, img_output_dir)
