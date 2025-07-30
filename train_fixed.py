# train_fixed.py - Memory-efficient version
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import os
from collections import defaultdict

from src.data.dataloader import MathExpressionDataset
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone  # Back to original for memory
from src.models.anchor_generator import AnchorGenerator
from src.models.heads import RPNHead
from src.train.evaluator import evaluate
from src.data.transforms import get_transform, collate_fn


def initialize_model_better(model):
    """Better initialization for classification heads"""
    # Initialize classification heads with bias toward positive class
    for stage in range(model.num_stages):
        # Set bias to favor object detection slightly
        nn.init.constant_(model.roi_heads.predictors[stage].bias[0], 1.0)  # background
        nn.init.constant_(model.roi_heads.predictors[stage].bias[1], -1.0)  # object

        # Smaller weight initialization
        nn.init.normal_(model.roi_heads.predictors[stage].weight, std=0.001)

    # Initialize RPN similarly
    nn.init.constant_(model.rpn.cls_logits.bias, -1.0)

    return model


def train_one_epoch_balanced(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    loss_components = defaultdict(float)

    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Balance losses - reduce classification loss weight
        balanced_losses = {}
        for k, v in loss_dict.items():
            if "classification_loss" in k:
                balanced_losses[k] = v * 0.5  # Reduce classification loss weight
            else:
                balanced_losses[k] = v

        # Sum balanced losses
        losses = sum(loss for loss in balanced_losses.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_loss += losses.item()

        # Track individual loss components
        for k, v in loss_dict.items():
            loss_components[k] += v.item()

        # Print progress
        if i % 10 == 0:
            print(f"Epoch [{epoch}][{i}/{len(data_loader)}] Loss: {losses.item():.4f}")

        # Clear cache periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()

    # Average losses
    avg_loss = total_loss / len(data_loader)
    avg_components = {k: v / len(data_loader) for k, v in loss_components.items()}

    return avg_loss, avg_components


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with original backbone for memory efficiency
    backbone = SimpleBackbone()

    # Create anchor generator with fewer anchors
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150),  # Reduced from 7 to 4 sizes
        aspect_ratios=(0.15, 0.2, 0.3, 0.5),  # Reduced from 9 to 4 ratios
    )

    # Create model
    model = CascadeRCNN(backbone)
    model.anchor_generator = anchor_generator

    # Update RPN with correct number of anchors
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    # Better initialization
    model = initialize_model_better(model)
    model.to(device)

    # Dataset and DataLoader
    root_dir = "dataset"
    dataset_train = MathExpressionDataset(root_dir, "train", get_transform(train=True))
    dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")

    # Reduced batch size for memory
    data_loader_train = DataLoader(
        dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    # Use Adam optimizer for better convergence
    optimizer = Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 20
    best_map = 0

    for epoch in range(num_epochs):
        # Train
        train_loss, loss_components = train_one_epoch_balanced(
            model, optimizer, data_loader_train, device, epoch
        )

        # Clear cache before evaluation
        torch.cuda.empty_cache()

        # Evaluate with lower threshold
        print("\nEvaluating...")
        val_metrics = evaluate(model, data_loader_val, device, score_threshold=0.3)

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print("Loss Components:")
        for k, v in loss_components.items():
            print(f"  {k}: {v:.4f}")
        print(f"Val mAP@0.5: {val_metrics['map_50']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1 Score: {val_metrics['f1_score']:.4f}")
        print("-" * 50)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Save latest checkpoint
        torch.save(checkpoint, "results/cascade_rcnn_fixed.pth")

        # Save best model
        if val_metrics["map_50"] > best_map:
            best_map = val_metrics["map_50"]
            torch.save(checkpoint, "results/cascade_rcnn_best_fixed.pth")
            print(f"New best model saved with mAP: {best_map:.4f}")

    print(f"\nTraining completed! Best mAP@0.5: {best_map:.4f}")


if __name__ == "__main__":
    main()
