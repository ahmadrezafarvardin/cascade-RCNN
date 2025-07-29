# check_checkpoint.py
import os
import torch
import json


def check_training_progress(output_dir="./results/rcnn"):
    """Check if there's a saved checkpoint and training history"""

    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    history_path = os.path.join(output_dir, "training_history.json")

    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint found at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"  - Last epoch: {checkpoint['epoch'] + 1}")
        print(f"  - Last train loss: {checkpoint['train_loss']:.4f}")
        print(f"  - Last val metrics: {checkpoint['val_metrics']}")
        return True
    else:
        print("✗ No checkpoint found")

    if os.path.exists(history_path):
        print(f"✓ Training history found at: {history_path}")
        with open(history_path, "r") as f:
            history = json.load(f)
        print(f"  - Epochs completed: {len(history['train_loss'])}")

    return False


if __name__ == "__main__":
    check_training_progress()
