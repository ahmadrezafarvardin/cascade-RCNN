import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from data.data_preparation import get_test_loader, MathExpressionDataset, get_transforms
from models.rcnn import RCNN
from utils.box_utils import calculate_iou, apply_nms, decode_boxes
from utils.region_proposal import selective_search


def evaluate_model(model_path, root_dir, output_dir, num_samples=10):
    """Evaluate trained model and visualize results"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create validation dataset
    val_dataset = MathExpressionDataset(
        root_dir=root_dir, split="val", transform=get_transforms(train=False)
    )

    # Evaluation metrics
    all_precisions = []
    all_recalls = []
    all_ious = []

    # Process samples
    for idx in tqdm(range(min(num_samples, len(val_dataset))), desc="Evaluating"):
        image, target, img_name = val_dataset[idx]

        # Generate proposals
        proposals = selective_search(image, max_proposals=500)

        if len(proposals) == 0:
            continue

        # Convert to tensor
        proposals_tensor = torch.tensor(proposals, dtype=torch.float32).to(device)
        image_tensor = image.unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            class_scores, bbox_deltas, _ = model(image_tensor, proposals_tensor)

        # Get predicted classes
        probs = torch.softmax(class_scores, dim=1)
        char_probs = probs[:, 1]  # Probability of being a character

        # Filter by confidence threshold
        conf_threshold = 0.5
        high_conf_indices = (char_probs > conf_threshold).nonzero(as_tuple=True)[0]

        if len(high_conf_indices) > 0:
            # Get high confidence proposals and their deltas
            high_conf_proposals = proposals_tensor[high_conf_indices]
            high_conf_deltas = bbox_deltas[high_conf_indices]
            high_conf_scores = char_probs[high_conf_indices]

            # Decode bounding boxes
            pred_boxes = decode_boxes(high_conf_proposals, high_conf_deltas)

            # Apply NMS
            keep_indices = apply_nms(pred_boxes, high_conf_scores, threshold=0.3)

            if len(keep_indices) > 0:
                final_boxes = pred_boxes[keep_indices]
                final_scores = high_conf_scores[keep_indices]

                # Calculate metrics
                gt_boxes = target["boxes"]

                # Calculate IoUs
                for pred_box in final_boxes:
                    ious = [
                        calculate_iou(pred_box.cpu().numpy(), gt_box.numpy())
                        for gt_box in gt_boxes
                    ]
                    if ious:
                        all_ious.append(max(ious))

                # Calculate precision/recall
                tp = 0
                for gt_box in gt_boxes:
                    ious = [
                        calculate_iou(pred_box.cpu().numpy(), gt_box.numpy())
                        for pred_box in final_boxes
                    ]
                    if ious and max(ious) >= 0.5:
                        tp += 1

                precision = tp / len(final_boxes) if len(final_boxes) > 0 else 0
                recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0

                all_precisions.append(precision)
                all_recalls.append(recall)

                # Visualize results
                if idx < 5:  # Visualize first 5 samples
                    visualize_predictions(
                        image,
                        final_boxes.cpu().numpy(),
                        gt_boxes.numpy(),
                        final_scores.cpu().numpy(),
                        img_name,
                        output_dir,
                    )

    # Calculate overall metrics
    metrics = {
        "precision": np.mean(all_precisions) if all_precisions else 0,
        "recall": np.mean(all_recalls) if all_recalls else 0,
        "mean_iou": np.mean(all_ious) if all_ious else 0,
    }

    metrics["f1_score"] = (
        2
        * metrics["precision"]
        * metrics["recall"]
        / (metrics["precision"] + metrics["recall"])
        if (metrics["precision"] + metrics["recall"]) > 0
        else 0
    )

    print(f"\nEvaluation Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    return metrics


def visualize_predictions(image, pred_boxes, gt_boxes, scores, img_name, output_dir):
    """Visualize predictions and ground truth"""

    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * std + mean) * 255
    image = image.astype(np.uint8)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Plot ground truth boxes in green
    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label="Ground Truth",
        )
        ax.add_patch(rect)

    # Plot predictions in red
    for box, score in zip(pred_boxes, scores):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label=f"Pred: {score:.2f}",
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1] - 5, f"{score:.2f}", color="red", fontsize=8, weight="bold"
        )

    ax.set_title(f"Predictions for {img_name}")
    ax.axis("off")

    # Save figure
    output_path = os.path.join(output_dir, f"pred_{img_name}")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
