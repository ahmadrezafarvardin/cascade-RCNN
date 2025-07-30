# visualize_predictions.py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
import os


def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()

    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    if num_images == 1:
        axes = [axes]

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

            # Plot image
            axes[idx].imshow(img_np)
            axes[idx].set_title(f"Image {idx}")
            axes[idx].axis("off")

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
                axes[idx].add_patch(rect)

            # Draw predicted boxes in red (only high confidence ones)
            boxes = predictions["boxes"].cpu()
            scores = predictions["scores"].cpu()

            # Filter by score
            keep = scores > 0.5
            boxes = boxes[keep]
            scores = scores[keep]

            for box, score in zip(boxes, scores):
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                axes[idx].add_patch(rect)
                axes[idx].text(
                    box[0], box[1] - 5, f"{score:.2f}", color="red", fontsize=8
                )

    plt.tight_layout()
    plt.savefig("predictions_visualization.png")
    plt.show()


if __name__ == "__main__":
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    # Load checkpoint if available
    checkpoint_path = "results/cascade_rcnn_char_detection.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded checkpoint")

    model.to(device)

    # Load dataset
    root_dir = "dataset"
    dataset = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    # Visualize
    visualize_predictions(model, dataset, device, num_images=3)
