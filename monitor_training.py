# monitor_training.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
from torch.utils.data import DataLoader
from src.data.transforms import collate_fn
import os


def check_predictions(model, data_loader, device, num_samples=5):
    """Check what the model is predicting"""
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)

            for j, output in enumerate(outputs):
                print(f"\nImage {i}, Sample {j}:")
                print(f"  Number of predictions: {len(output['boxes'])}")
                if len(output["boxes"]) > 0:
                    scores = output["scores"].cpu().numpy()
                    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"  Predictions with score > 0.1: {(scores > 0.1).sum()}")
                    print(f"  Predictions with score > 0.3: {(scores > 0.3).sum()}")
                    print(f"  Predictions with score > 0.5: {(scores > 0.5).sum()}")

                    # Show top 5 scores
                    top_scores = np.sort(scores)[-5:][::-1]
                    print(f"  Top 5 scores: {top_scores}")


def plot_score_distribution(model, data_loader, device):
    """Plot distribution of prediction scores"""
    model.eval()
    all_scores = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= 10:  # Check first 10 batches
                break

            images = images.to(device)
            outputs = model(images)

            for output in outputs:
                if len(output["scores"]) > 0:
                    all_scores.extend(output["scores"].cpu().numpy())

    if all_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Prediction Score")
        plt.ylabel("Count")
        plt.title("Distribution of Prediction Scores")
        plt.axvline(x=0.5, color="red", linestyle="--", label="Threshold=0.5")
        plt.legend()
        plt.savefig("score_distribution.png")
        print(f"\nTotal predictions: {len(all_scores)}")
        print(f"Predictions > 0.5: {sum(s > 0.5 for s in all_scores)}")
        print(f"Mean score: {np.mean(all_scores):.4f}")
        print(f"Max score: {np.max(all_scores):.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    # Load latest checkpoint
    checkpoint_path = "results/cascade_rcnn_latest.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        print(f"Training loss: {checkpoint['train_loss']:.4f}")
        print(f"Validation metrics: {checkpoint['val_metrics']}")

    model.to(device)

    # Load validation data
    root_dir = "dataset"
    dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # Check predictions
    print("\nChecking model predictions...")
    check_predictions(model, data_loader_val, device)

    # Plot score distribution
    print("\nPlotting score distribution...")
    plot_score_distribution(model, data_loader_val, device)
