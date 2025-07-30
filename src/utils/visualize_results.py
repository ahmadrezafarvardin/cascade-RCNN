# visualize_results.py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.models.anchor_generator import AnchorGenerator
from src.models.heads import RPNHead
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
import os


def visualize_predictions(model, dataset, device, num_images=5, score_threshold=0.3):
    model.eval()

    fig, axes = plt.subplots(2, num_images, figsize=(20, 8))
    if num_images == 1:
        axes = axes.reshape(2, 1)

    with torch.no_grad():
        for idx in range(num_images):
            # Get image and target
            img, target = dataset[idx]

            # Add batch dimension and move to device
            img_tensor = img.unsqueeze(0).to(device)

            # Get predictions
            predictions = model(img_tensor)[0]

            # Convert image to numpy for visualization
            img_np = img.permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)

            # Plot image with ground truth
            axes[0, idx].imshow(img_np)
            axes[0, idx].set_title(f"Ground Truth - Image {idx}")
            axes[0, idx].axis("off")

            # Draw ground truth boxes in green
            for box in target["boxes"]:
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                )
                axes[0, idx].add_patch(rect)

            # Plot image with predictions
            axes[1, idx].imshow(img_np)
            axes[1, idx].set_title(f"Predictions (thresh={score_threshold})")
            axes[1, idx].axis("off")

            # Draw predicted boxes
            boxes = predictions["boxes"].cpu()
            scores = predictions["scores"].cpu()

            # Filter by score
            keep = scores > score_threshold
            boxes = boxes[keep]
            scores = scores[keep]

            # Apply NMS to reduce overlapping boxes
            if len(boxes) > 0:
                nms_keep = torch.ops.torchvision.nms(boxes, scores, 0.3)
                boxes = boxes[nms_keep]
                scores = scores[nms_keep]

            for box, score in zip(boxes, scores):
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                    alpha=min(1.0, score.item()),
                )
                axes[1, idx].add_patch(rect)
                axes[1, idx].text(
                    box[0],
                    box[1] - 5,
                    f"{score:.2f}",
                    color="red",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

            axes[1, idx].text(
                10,
                30,
                f"{len(boxes)} detections",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

    plt.tight_layout()
    plt.savefig("results/model_predictions.png", dpi=150)
    plt.show()
    print(f"Saved visualization to fixed_model_predictions.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate model architecture
    backbone = SimpleBackbone()
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150), aspect_ratios=(0.15, 0.2, 0.3, 0.5)
    )

    model = CascadeRCNN(backbone)
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    # Load best checkpoint
    checkpoint_path = "results/cascade_rcnn_best_fixed.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model with mAP: {checkpoint['val_metrics']['map_50']:.4f}")

    model.to(device)

    # Load dataset
    root_dir = "dataset"
    dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    # Visualize with different thresholds
    print("Visualizing with threshold 0.3...")
    visualize_predictions(model, dataset_val, device, num_images=3, score_threshold=0.3)
