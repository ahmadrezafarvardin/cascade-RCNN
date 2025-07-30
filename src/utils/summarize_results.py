# src/utils/summarize_results.py
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import json


def summarize_results():
    # Load checkpoint
    checkpoint_path = "../../results/cascade_rcnn_best_fixed.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        print("=" * 50)
        print("CASCADE R-CNN TRAINING RESULTS SUMMARY")
        print("=" * 50)

        print(f"\nBest Model Checkpoint:")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
        print(f"  Training Loss: {checkpoint['train_loss']:.4f}")

        print(f"\nValidation Metrics:")
        metrics = checkpoint["val_metrics"]
        print(f"  mAP@0.5: {metrics['map_50']:.4f} ({metrics['map_50']*100:.2f}%)")
        print(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

        print("\nModel Configuration:")
        print("  Backbone: SimpleBackbone (4 conv layers)")
        print("  Anchor sizes: [60, 90, 120, 150]")
        print("  Aspect ratios: [0.15, 0.2, 0.3, 0.5]")
        print("  Cascade stages: 3")
        print("  IoU thresholds: [0.3, 0.4, 0.5]")

        print("\nTraining Configuration:")
        print("  Optimizer: Adam (lr=0.0001)")
        print("  Batch size: 1")
        print("  Training samples: 300")
        print("  Validation samples: 200")

        print("=" * 50)
    else:
        print(f"Checkpoint not found at {checkpoint_path}")


if __name__ == "__main__":
    summarize_results()
