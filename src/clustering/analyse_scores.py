# src/clustering/analyze_scores.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cascade_rcnn import CascadeRCNN
from models.backbone import SimpleBackbone
from models.anchor_generator import AnchorGenerator
from models.heads import RPNHead
import cv2


def analyze_detection_scores(model_path, dataset_path, num_images=20):
    """Analyze the score distribution of model detections"""

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

    # Collect scores
    all_scores = []

    # Get image files
    image_dir = os.path.join(dataset_path, "train", "images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")][:num_images]

    print(f"Analyzing scores from {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            predictions = model(img_tensor)

        if len(predictions) > 0:
            scores = predictions[0]["scores"].cpu().numpy()
            all_scores.extend(scores)

    all_scores = np.array(all_scores)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_scores, bins=50, alpha=0.7, edgecolor="black")
    plt.axvline(x=0.1, color="r", linestyle="--", label="Threshold=0.1")
    plt.axvline(x=0.15, color="g", linestyle="--", label="Threshold=0.15")
    plt.xlabel("Detection Score")
    plt.ylabel("Count")
    plt.title("Distribution of Detection Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = os.path.join("../../results/clustering", "score_distribution.png")
    plt.savefig(output_path)
    print(f"Score distribution saved to {output_path}")

    # Print statistics
    print(f"\nScore Statistics:")
    print(f"Total detections: {len(all_scores)}")
    print(f"Min score: {all_scores.min():.4f}")
    print(f"Max score: {all_scores.max():.4f}")
    print(f"Mean score: {all_scores.mean():.4f}")
    print(f"Median score: {np.median(all_scores):.4f}")

    # Print percentiles
    percentiles = [50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(all_scores, p):.4f}")

    # Suggest threshold
    suggested_threshold = np.percentile(all_scores, 75)  # Keep top 25% of detections
    print(f"\nSuggested threshold (75th percentile): {suggested_threshold:.4f}")


if __name__ == "__main__":
    model_path = "../../results/cascade_rcnn_best.pth"
    dataset_path = "../../dataset"

    analyze_detection_scores(model_path, dataset_path)
