import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from data.data_preparation import MathExpressionDataset, get_transforms
from models.rcnn import RCNN
from utils.box_utils import calculate_iou, decode_boxes
from utils.region_proposal import selective_search


def debug_model_predictions():
    """Debug what the model is predicting"""

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN().to(device)

    # Load checkpoint
    checkpoint = torch.load("./results/rcnn/checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load a sample from validation set
    val_dataset = MathExpressionDataset(
        root_dir="./", split="val", transform=get_transforms(train=False)
    )

    # Test on first few samples
    for idx in range(min(5, len(val_dataset))):
        print(f"\n{'='*50}")
        print(f"Debugging sample {idx}")
        print("=" * 50)

        image, target, img_name = val_dataset[idx]
        gt_boxes = target["boxes"]

        print(f"Image: {img_name}")
        print(f"Number of GT boxes: {len(gt_boxes)}")

        # Generate proposals
        proposals = selective_search(image, max_proposals=500)
        print(f"Number of proposals: {len(proposals)}")

        # Check IoU between proposals and GT
        max_ious = []
        for gt_box in gt_boxes:
            ious = [calculate_iou(prop, gt_box.numpy()) for prop in proposals]
            if ious:
                max_ious.append(max(ious))

        if max_ious:
            print(f"Max IoU between proposals and GT: {max(max_ious):.4f}")
            print(
                f"Proposals with IoU > 0.5: {sum(1 for iou in max_ious if iou > 0.5)}"
            )

        # Get model predictions
        with torch.no_grad():
            proposals_tensor = torch.tensor(proposals, dtype=torch.float32).to(device)
            image_tensor = image.unsqueeze(0).to(device)

            class_scores, bbox_deltas, _ = model(image_tensor, proposals_tensor)

            # Analyze predictions
            probs = torch.softmax(class_scores, dim=1)
            char_probs = probs[:, 1]

            print(f"\nModel predictions:")
            print(f"Max character probability: {char_probs.max().item():.4f}")
            print(f"Mean character probability: {char_probs.mean().item():.4f}")
            print(f"Std character probability: {char_probs.std().item():.4f}")

            # Check different thresholds
            for thresh in [0.5, 0.3, 0.1, 0.05]:
                n_above = (char_probs > thresh).sum().item()
                print(f"Proposals with prob > {thresh}: {n_above}")

            # Get top predictions
            top_k = 10
            top_scores, top_indices = torch.topk(
                char_probs, min(top_k, len(char_probs))
            )

            print(f"\nTop {top_k} predictions:")
            for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
                proposal = proposals[idx]
                # Check IoU with GT
                ious = [calculate_iou(proposal, gt.numpy()) for gt in gt_boxes]
                max_iou = max(ious) if ious else 0
                print(f"  {i+1}. Score: {score:.4f}, Max IoU with GT: {max_iou:.4f}")

            # Visualize if we have any predictions above threshold
            if (char_probs > 0.1).any():
                visualize_debug(
                    image, proposals, char_probs, gt_boxes, img_name, threshold=0.1
                )


def visualize_debug(image, proposals, scores, gt_boxes, img_name, threshold=0.1):
    """Visualize proposals and their scores"""

    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * std + mean) * 255
    image_np = image_np.astype(np.uint8)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Show GT boxes
    ax1.imshow(image_np)
    ax1.set_title(f"Ground Truth - {img_name}")
    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        ax1.add_patch(rect)
    ax1.axis("off")

    # Right: Show top proposals
    ax2.imshow(image_np)
    ax2.set_title(f"Top Proposals (threshold={threshold})")

    # Get proposals above threshold
    scores_np = scores.cpu().numpy()
    high_score_indices = np.where(scores_np > threshold)[0]

    # Sort by score
    if len(high_score_indices) > 0:
        sorted_indices = high_score_indices[
            np.argsort(scores_np[high_score_indices])[::-1]
        ]

        # Show top 20
        for i, idx in enumerate(sorted_indices[:20]):
            proposal = proposals[idx]
            score = scores_np[idx]

            # Color based on score
            if score > 0.5:
                color = "red"
            elif score > 0.3:
                color = "orange"
            else:
                color = "yellow"

            rect = patches.Rectangle(
                (proposal[0], proposal[1]),
                proposal[2] - proposal[0],
                proposal[3] - proposal[1],
                linewidth=1,
                edgecolor=color,
                facecolor="none",
                alpha=0.7,
            )
            ax2.add_patch(rect)

            # Add score text
            ax2.text(
                proposal[0],
                proposal[1] - 2,
                f"{score:.2f}",
                color=color,
                fontsize=6,
                weight="bold",
            )

    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(f"./results/rcnn/debug_{img_name}", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved debug visualization to ./results/rcnn/debug_{img_name}")


if __name__ == "__main__":
    debug_model_predictions()
