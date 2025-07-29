import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import our modules
from data.data_preparation import create_data_loaders
from models.rcnn import RCNN
from utils.box_utils import calculate_iou, apply_nms, calculate_precision_recall

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_reg_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for images, targets, _ in progress_bar:
        # Move data to device
        images = [img.to(device) for img in images]

        batch_loss = 0
        batch_cls_loss = 0
        batch_reg_loss = 0

        optimizer.zero_grad()

        # Process each image in the batch
        for i, (image, target) in enumerate(zip(images, targets)):
            # Get ground truth boxes
            gt_boxes = target["boxes"].to(device)

            if len(gt_boxes) == 0:
                continue

            # Get predictions
            class_scores, bbox_deltas = model(image.unsqueeze(0), gt_boxes.unsqueeze(0))

            # Ground truth labels (all 1 for characters)
            gt_labels = torch.ones(len(gt_boxes), dtype=torch.long).to(device)

            # Calculate classification loss
            cls_loss = criterion_cls(class_scores, gt_labels)

            # Calculate regression loss (only for positive examples)
            # Here we assume ground truth boxes are already in the right format
            reg_loss = criterion_reg(bbox_deltas, gt_boxes)

            # Total loss
            loss = cls_loss + reg_loss

            # Accumulate loss
            batch_loss += loss.item()
            batch_cls_loss += cls_loss.item()
            batch_reg_loss += reg_loss.item()

            # Backpropagate
            loss.backward()

        # Update weights
        optimizer.step()

        # Update epoch loss
        epoch_loss += batch_loss / len(images)
        epoch_cls_loss += batch_cls_loss / len(images)
        epoch_reg_loss += batch_reg_loss / len(images)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{batch_loss/len(images):.4f}",
                "cls_loss": f"{batch_cls_loss/len(images):.4f}",
                "reg_loss": f"{batch_reg_loss/len(images):.4f}",
            }
        )

    return (
        epoch_loss / len(dataloader),
        epoch_cls_loss / len(dataloader),
        epoch_reg_loss / len(dataloader),
    )


def evaluate(model, dataloader, device, iou_threshold=0.5):
    """Evaluate model on validation set"""
    model.eval()
    all_precisions = []
    all_recalls = []
    all_ious = []

    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            images = [img.to(device) for img in images]

            # Process each image in the batch
            for i, (image, target) in enumerate(zip(images, targets)):
                # Get ground truth boxes
                gt_boxes = target["boxes"].to(device)

                if len(gt_boxes) == 0:
                    continue

                # Get predictions
                class_scores, bbox_deltas = model(
                    image.unsqueeze(0), gt_boxes.unsqueeze(0)
                )

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
                precision, recall = calculate_precision_recall(
                    final_boxes.cpu().numpy(), gt_boxes.cpu().numpy(), iou_threshold
                )

                all_precisions.append(precision)
                all_recalls.append(recall)

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

    return {
        "precision": mean_precision,
        "recall": mean_recall,
        "iou": mean_iou,
        "f1_score": f1_score,
    }


def train_model(root_dir, output_dir, batch_size=4, num_epochs=20, learning_rate=0.001):
    """Train the R-CNN model"""
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        root_dir=root_dir, batch_size=batch_size, num_workers=2
    )

    # Initialize model
    model = RCNN().to(device)

    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # Training history
    history = {"train_loss": [], "val_metrics": []}

    best_f1 = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train one epoch
        train_loss, cls_loss, reg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_reg, device
        )

        # Evaluate model
        val_metrics = evaluate(model, val_loader, device)

        # Update learning rate
        scheduler.step(train_loss)

        # Print metrics
        print(
            f"Train Loss: {train_loss:.4f} (CLS: {cls_loss:.4f}, REG: {reg_loss:.4f})"
        )
        print(
            f"Val Metrics: Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1_score']:.4f}"
        )

        # Save history
        history["train_loss"].append(
            {"total": train_loss, "cls": cls_loss, "reg": reg_loss}
        )
        history["val_metrics"].append(val_metrics)

        # Save best model
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with F1 score: {best_f1:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "history": history,
            },
            os.path.join(output_dir, "checkpoint.pth"),
        )

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    # Training configuration
    config = {
        "root_dir": "./",
        "output_dir": "./results/rcnn",
        "batch_size": 4,
        "num_epochs": 20,
        "learning_rate": 0.001,
    }

    # Train model
    model, history = train_model(**config)
