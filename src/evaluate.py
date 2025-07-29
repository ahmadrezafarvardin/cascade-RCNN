import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

# Import our modules
from data.data_preparation import get_transforms, MathExpressionDataset
from models.rcnn import RCNN
from utils.box_utils import calculate_iou, apply_nms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_predictions(image, pred_boxes, gt_boxes=None, save_path=None):
    """
    Visualize predicted and ground truth bounding boxes on an image

    Args:
        image: PIL Image or tensor
        pred_boxes: Predicted bounding boxes in format [x1, y1, x2, y2]
        gt_boxes: Ground truth bounding boxes in format [x1, y1, x2, y2] (optional)
        save_path: Path to save the visualization (optional)

    Returns:
        PIL Image with visualized boxes
    """
    # Convert tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        image = Image.fromarray(
            (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

    # Create a copy of the image for drawing
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Draw ground truth boxes in green if provided
    if gt_boxes is not None:
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

    # Save the image if a path is provided
    if save_path:
        draw_image.save(save_path)

    return draw_image


def evaluate_model(model_path, root_dir, output_dir, num_samples=10, iou_threshold=0.5):
    """
    Evaluate the trained model and visualize results

    Args:
        model_path: Path to the trained model
        root_dir: Root directory of the dataset
        output_dir: Directory to save results
        num_samples: Number of samples to visualize
        iou_threshold: IoU threshold for evaluation

    Returns:
        Evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = RCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create validation dataset
    val_dataset = MathExpressionDataset(
        root_dir=root_dir, split="val", transform=get_transforms(train=False)
    )

    # Select random samples for visualization
    sample_indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    # Evaluation metrics
    all_precisions = []
    all_recalls = []
    all_ious = []

    # Process each sample
    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        # Get sample
        image, target, img_name = val_dataset[idx]

        # Move to device
        image = image.to(device)
        gt_boxes = target["boxes"].to(device)

        # Skip samples without ground truth boxes
        if len(gt_boxes) == 0:
            continue

        # Get predictions
        with torch.no_grad():
            class_scores, bbox_deltas, valid_indices = model(image.unsqueeze(0), gt_boxes.unsqueeze(0))

        # Get predicted class (0: background, 1: character)
        _, pred_classes = torch.max(class_scores, 1)

        # Filter character predictions
        char_indices = (pred_classes == 1).nonzero(as_tuple=True)[0]
        pred_boxes = bbox_deltas[char_indices]

        # Apply NMS to remove duplicate predictions
        keep_indices = apply_nms(
            pred_boxes, class_scores[char_indices, 1], iou_threshold
        )
        final_boxes = pred_boxes[keep_indices]

        # Calculate IoU for each prediction with best matching ground truth
        for pred_box in final_boxes:
            ious = [
                calculate_iou(pred_box.cpu().numpy(), gt_box.cpu().numpy())
                for gt_box in gt_boxes
            ]
            all_ious.append(max(ious) if ious else 0)

        # Calculate precision and recall
        from utils.box_utils import calculate_precision_recall

        precision, recall = calculate_precision_recall(
            final_boxes.cpu().numpy(), gt_boxes.cpu().numpy(), iou_threshold
        )

        all_precisions.append(precision)
        all_recalls.append(recall)

        # Visualize predictions
        save_path = os.path.join(output_dir, f"{img_name}_pred.png")
        visualize_predictions(
            image, final_boxes.cpu().numpy(), gt_boxes.cpu().numpy(), save_path
        )

    # Calculate mean metrics
    mean_precision = np.mean(all_precisions) if all_precisions else 0
    mean_recall = np.mean(all_recalls) if all_recalls else 0
    mean_iou = np.mean(all_ious) if all_ious else 0

    # Calculate F1 score
    f1_score = (
        2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        if (mean_precision + mean_recall) > 0
        else 0
    )

    # Save metrics
    metrics = {
        "precision": mean_precision,
        "recall": mean_recall,
        "iou": mean_iou,
        "f1_score": f1_score,
    }

    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Plot IoU distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_ious, bins=20, alpha=0.7, color="blue")
    plt.axvline(
        x=mean_iou, color="red", linestyle="--", label=f"Mean IoU: {mean_iou:.4f}"
    )
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title("Distribution of IoU Values")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "iou_distribution.png"))

    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    return metrics


if __name__ == "__main__":
    # Evaluation configuration
    config = {
        "model_path": "./results/rcnn/best_model.pth",
        "root_dir": "./",
        "output_dir": "./results/rcnn/evaluation",
        "num_samples": 10,
        "iou_threshold": 0.5,
    }

    # Evaluate model
    metrics = evaluate_model(**config)
