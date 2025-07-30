# src/train/trainer.py
from collections import defaultdict
import os
from pathlib import Path
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from src.data.dataloader import MathExpressionDataset
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.train.evaluator import evaluate
from src.data.transforms import get_transform, collate_fn


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    loss_components = defaultdict(float)

    for i, (images, targets) in enumerate(data_loader):
        # Verify sizes are properly set
        for target in targets:
            assert "orig_size" in target, "Original size missing in target"
            assert "size" in target, "Current size missing in target"

        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())

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
            for k, v in loss_dict.items():
                print(f"  {k}: {v.item():.4f}")

    # Average losses
    avg_loss = total_loss / len(data_loader)
    avg_components = {k: v / len(data_loader) for k, v in loss_components.items()}

    return avg_loss, avg_components


root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
dataset_train = MathExpressionDataset(root_dir, "train", get_transform(train=True))
dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))


# In trainer.py main function:
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)
    model.to(device)

    # Dataset and DataLoader
    root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    dataset_train = MathExpressionDataset(root_dir, "train", get_transform(train=True))
    dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    data_loader_train = DataLoader(
        dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, loss_components = train_one_epoch(
            model, optimizer, data_loader_train, device, epoch
        )

        # Evaluation
        val_metrics = evaluate(model, data_loader_val, device)

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print("Loss Components:")
        for k, v in loss_components.items():
            print(f"  {k}: {v:.4f}")
        print(f"Val mAP@0.5: {val_metrics['map_50']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print("-" * 50)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Save model
    torch.save(model.state_dict(), "results/cascade_rcnn_char_detection.pth")
    print("Model saved to results/cascade_rcnn_char_detection.pth")


if __name__ == "__main__":
    main()
