# train.py
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
from utils.box_utils import (
    calculate_iou,
    apply_nms,
    calculate_precision_recall,
    compute_bbox_targets,
)
from utils.region_proposal import selective_search
from utils.box_utils import decode_boxes

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device):
    model.train()
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_reg_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for images, targets, _ in progress_bar:
        batch_loss = 0
        batch_cls_loss = 0
        batch_reg_loss = 0
        optimizer.zero_grad()

        for image, target in zip(images, targets):
            # Generate region proposals
            proposals = selective_search(image, max_proposals=1000)

            if len(proposals) == 0:
                continue

            # Convert to tensor
            proposals = torch.tensor(proposals, dtype=torch.float32)
            gt_boxes = target["boxes"]

            # Skip if no ground truth boxes
            if len(gt_boxes) == 0:
                continue

            # Compute targets for proposals
            reg_targets, cls_labels = compute_bbox_targets(
                proposals.numpy(), gt_boxes.numpy()
            )

            # Sample positive and negative proposals
            pos_indices = np.where(cls_labels == 1)[0]
            neg_indices = np.where(cls_labels == 0)[0]

            # Balance positive/negative samples
            n_pos = min(len(pos_indices), 32) if len(pos_indices) > 0 else 0
            n_neg = min(len(neg_indices), 96) if len(neg_indices) > 0 else 0

            # Skip if we don't have both positive and negative samples
            if n_pos == 0 or n_neg == 0:
                continue

            # Random sampling
            pos_indices = np.random.choice(
                pos_indices, n_pos, replace=len(pos_indices) < n_pos
            )
            neg_indices = np.random.choice(
                neg_indices, n_neg, replace=len(neg_indices) < n_neg
            )
            all_indices = np.concatenate([pos_indices, neg_indices])

            # Get sampled proposals and targets
            sampled_proposals = proposals[all_indices].to(device)
            sampled_labels = torch.tensor(cls_labels[all_indices], dtype=torch.long).to(
                device
            )
            sampled_reg_targets = torch.tensor(
                reg_targets[all_indices], dtype=torch.float32
            ).to(device)

            # Forward pass
            image_tensor = image.unsqueeze(0).to(device)

            # Get predictions for sampled proposals
            class_scores, bbox_deltas, valid_indices = model(
                image_tensor, sampled_proposals
            )

            # Since we're passing pre-sampled proposals, all should be valid
            assert class_scores.size(0) == len(
                sampled_labels
            ), f"Mismatch: {class_scores.size(0)} predictions vs {len(sampled_labels)} labels"

            # Calculate losses
            cls_loss = criterion_cls(class_scores, sampled_labels)

            # Regression loss only for positive samples
            pos_mask = sampled_labels == 1
            if pos_mask.sum() > 0:
                reg_loss = criterion_reg(
                    bbox_deltas[pos_mask], sampled_reg_targets[pos_mask]
                )
            else:
                reg_loss = torch.tensor(0.0).to(device)

            # Total loss
            loss = cls_loss + 0.5 * reg_loss  # Weight regression loss
            loss.backward()

            batch_loss += loss.item()
            batch_cls_loss += cls_loss.item()
            batch_reg_loss += reg_loss.item()

        optimizer.step()

        # Update epoch losses
        if len(images) > 0:
            epoch_loss += batch_loss / len(images)
            epoch_cls_loss += batch_cls_loss / len(images)
            epoch_reg_loss += batch_reg_loss / len(images)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": (
                    f"{batch_loss/len(images):.4f}" if len(images) > 0 else "0.0000"
                ),
                "cls_loss": (
                    f"{batch_cls_loss/len(images):.4f}" if len(images) > 0 else "0.0000"
                ),
                "reg_loss": (
                    f"{batch_reg_loss/len(images):.4f}" if len(images) > 0 else "0.0000"
                ),
            }
        )

    return (
        epoch_loss / len(dataloader),
        epoch_cls_loss / len(dataloader),
        epoch_reg_loss / len(dataloader),
    )


def evaluate(model, dataloader, device, iou_threshold=0.5):
    """Evaluate model on validation set with debugging"""
    model.eval()
    all_precisions = []
    all_recalls = []
    all_ious = []

    # Debug counters
    total_proposals = 0
    total_high_conf = 0
    total_after_nms = 0

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):
            for image, target in zip(images, targets):
                # Generate proposals
                proposals = selective_search(image, max_proposals=500)
                total_proposals += len(proposals)

                if len(proposals) == 0:
                    continue

                # Get ground truth
                gt_boxes = target["boxes"]
                if len(gt_boxes) == 0:
                    continue

                # Convert to tensors
                proposals_tensor = torch.tensor(proposals, dtype=torch.float32).to(
                    device
                )
                image_tensor = image.unsqueeze(0).to(device)

                # Get predictions
                class_scores, bbox_deltas, _ = model(image_tensor, proposals_tensor)

                # Get probabilities
                probs = torch.softmax(class_scores, dim=1)
                char_probs = probs[:, 1]

                # Filter by confidence
                conf_threshold = 0.3  # Lower threshold for debugging
                high_conf_mask = char_probs > conf_threshold
                high_conf_indices = high_conf_mask.nonzero(as_tuple=True)[0]
                total_high_conf += len(high_conf_indices)

                if len(high_conf_indices) == 0:
                    continue

                # Get high confidence predictions
                high_conf_proposals = proposals_tensor[high_conf_indices]
                high_conf_deltas = bbox_deltas[high_conf_indices]
                high_conf_scores = char_probs[high_conf_indices]

                # Decode boxes
                pred_boxes = decode_boxes(high_conf_proposals, high_conf_deltas)

                # Apply NMS
                keep_indices = apply_nms(pred_boxes, high_conf_scores, threshold=0.3)
                total_after_nms += len(keep_indices)

                if len(keep_indices) == 0:
                    continue

                final_boxes = pred_boxes[keep_indices]

                # Calculate metrics
                precision, recall = calculate_precision_recall(
                    final_boxes.cpu().numpy(), gt_boxes.numpy(), iou_threshold
                )

                if precision > 0 or recall > 0:
                    all_precisions.append(precision)
                    all_recalls.append(recall)

    # Print debug info
    print(f"\nEvaluation Debug Info:")
    print(f"Total proposals generated: {total_proposals}")
    print(f"High confidence predictions: {total_high_conf}")
    print(f"Predictions after NMS: {total_after_nms}")
    print(f"Images with valid predictions: {len(all_precisions)}")

    # Calculate mean metrics
    mean_precision = np.mean(all_precisions) if all_precisions else 0
    mean_recall = np.mean(all_recalls) if all_recalls else 0
    mean_iou = np.mean(all_ious) if all_ious else 0

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


def create_training_proposals(image, gt_boxes, num_proposals=1000):
    """Create proposals that guarantee coverage of ground truth boxes"""

    proposals = []

    # 1. Add all GT boxes (guaranteed 100% recall for training)
    for gt in gt_boxes:
        proposals.append(gt.numpy() if isinstance(gt, torch.Tensor) else gt)

    # 2. Add augmented GT boxes for better learning
    for gt in gt_boxes:
        if isinstance(gt, torch.Tensor):
            gt = gt.numpy()
        x1, y1, x2, y2 = gt
        w, h = x2 - x1, y2 - y1

        # Add jittered versions
        for _ in range(10):  # 10 augmented versions per GT box
            # Random jitter
            dx = np.random.uniform(-w * 0.1, w * 0.1)
            dy = np.random.uniform(-h * 0.1, h * 0.1)
            dw = np.random.uniform(-w * 0.1, w * 0.1)
            dh = np.random.uniform(-h * 0.1, h * 0.1)

            new_x1 = max(0, x1 + dx)
            new_y1 = max(0, y1 + dy)
            new_x2 = min(416, x2 + dx + dw)  # Assuming 416x416 images
            new_y2 = min(416, y2 + dy + dh)

            if new_x2 > new_x1 and new_y2 > new_y1:
                proposals.append([new_x1, new_y1, new_x2, new_y2])

    # 3. Add regular proposals for negative samples
    regular_proposals = selective_search(image, max_proposals=num_proposals)

    # Combine all proposals
    proposals.extend(regular_proposals)

    # Convert to numpy array and limit to num_proposals
    proposals = np.array(proposals, dtype=np.float32)

    # Remove duplicates
    if len(proposals) > 0:
        proposals = np.unique(proposals, axis=0)

    # Randomly sample if we have too many
    if len(proposals) > num_proposals:
        indices = np.random.choice(len(proposals), num_proposals, replace=False)
        proposals = proposals[indices]

    return proposals


def train_one_epoch_hybrid(
    model, dataloader, optimizer, criterion_cls, criterion_reg, device
):
    """Training with guaranteed positive samples using hybrid approach"""
    model.train()
    epoch_loss = 0
    epoch_cls_loss = 0
    epoch_reg_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for images, targets, _ in progress_bar:
        batch_loss = 0
        batch_cls_loss = 0
        batch_reg_loss = 0
        optimizer.zero_grad()

        for image, target in zip(images, targets):
            gt_boxes = target["boxes"]

            # Skip if no ground truth boxes
            if len(gt_boxes) == 0:
                continue

            # Create proposals with guaranteed GT coverage
            proposals = create_training_proposals(image, gt_boxes, num_proposals=1500)

            if len(proposals) == 0:
                continue

            # Convert to tensor
            proposals = torch.tensor(proposals, dtype=torch.float32)

            # Compute targets for proposals
            reg_targets, cls_labels = compute_bbox_targets(
                proposals.numpy(), gt_boxes.numpy()
            )

            # Sample positive and negative proposals
            pos_indices = np.where(cls_labels == 1)[0]
            neg_indices = np.where(cls_labels == 0)[0]

            # Balance positive/negative samples
            n_pos = (
                min(len(pos_indices), 64) if len(pos_indices) > 0 else 0
            )  # More positive samples
            n_neg = min(len(neg_indices), 192) if len(neg_indices) > 0 else 0

            # We should always have positive samples now
            if n_pos == 0:
                print(f"Warning: No positive samples found despite adding GT boxes!")
                continue

            # It's OK if we don't have negative samples (though unlikely)
            if n_neg == 0:
                n_neg = min(64, len(neg_indices)) if len(neg_indices) > 0 else 0

            # Random sampling
            if n_pos > 0:
                pos_indices = np.random.choice(
                    pos_indices, n_pos, replace=len(pos_indices) < n_pos
                )

            if n_neg > 0:
                neg_indices = np.random.choice(
                    neg_indices, n_neg, replace=len(neg_indices) < n_neg
                )
                all_indices = np.concatenate([pos_indices, neg_indices])
            else:
                all_indices = pos_indices

            # Get sampled proposals and targets
            sampled_proposals = proposals[all_indices].to(device)
            sampled_labels = torch.tensor(cls_labels[all_indices], dtype=torch.long).to(
                device
            )
            sampled_reg_targets = torch.tensor(
                reg_targets[all_indices], dtype=torch.float32
            ).to(device)

            # Forward pass
            image_tensor = image.unsqueeze(0).to(device)

            # Get predictions for sampled proposals
            class_scores, bbox_deltas, valid_indices = model(
                image_tensor, sampled_proposals
            )

            # Calculate losses
            cls_loss = criterion_cls(class_scores, sampled_labels)

            # Regression loss only for positive samples
            pos_mask = sampled_labels == 1
            if pos_mask.sum() > 0:
                reg_loss = criterion_reg(
                    bbox_deltas[pos_mask], sampled_reg_targets[pos_mask]
                )
            else:
                reg_loss = torch.tensor(0.0).to(device)

            # Total loss
            loss = cls_loss + reg_loss  # Equal weighting
            loss.backward()

            batch_loss += loss.item()
            batch_cls_loss += cls_loss.item()
            batch_reg_loss += reg_loss.item()

        optimizer.step()

        # Update epoch losses
        if len(images) > 0:
            epoch_loss += batch_loss / len(images)
            epoch_cls_loss += batch_cls_loss / len(images)
            epoch_reg_loss += batch_reg_loss / len(images)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": (
                    f"{batch_loss/len(images):.4f}" if len(images) > 0 else "0.0000"
                ),
                "cls_loss": (
                    f"{batch_cls_loss/len(images):.4f}" if len(images) > 0 else "0.0000"
                ),
                "reg_loss": (
                    f"{batch_reg_loss/len(images):.4f}" if len(images) > 0 else "0.0000"
                ),
            }
        )

    return (
        epoch_loss / len(dataloader),
        epoch_cls_loss / len(dataloader),
        epoch_reg_loss / len(dataloader),
    )


def train_model(
    root_dir,
    output_dir,
    batch_size=4,
    num_epochs=20,
    learning_rate=0.001,
    resume=False,
    use_hybrid=False,
):
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

    # Initialize training variables
    start_epoch = 0
    best_f1 = 0
    history = {"train_loss": [], "val_metrics": []}

    # Resume from checkpoint if requested
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]

        # Find best F1 from history
        if history["val_metrics"]:
            best_f1 = max(m["f1_score"] for m in history["val_metrics"])

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best F1 so far: {best_f1:.4f}")

        # Adjust learning rate if needed
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr > learning_rate:
            print(f"Adjusting learning rate from {current_lr} to {learning_rate}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

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

        # Save training history
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f)

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

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
