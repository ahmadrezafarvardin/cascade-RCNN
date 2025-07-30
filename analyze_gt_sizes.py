# analyze_gt_sizes.py
import torch
import numpy as np
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
import matplotlib.pyplot as plt


def analyze_gt_box_sizes():
    # Load dataset
    root_dir = "dataset"
    dataset = MathExpressionDataset(root_dir, "train", get_transform(train=False))

    all_widths = []
    all_heights = []
    all_areas = []

    # Collect box sizes from first 50 images
    for i in range(min(50, len(dataset))):
        _, target = dataset[i]
        boxes = target["boxes"]

        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height

            all_widths.append(width.item())
            all_heights.append(height.item())
            all_areas.append(area.item())

    # Print statistics
    print("Ground Truth Box Statistics:")
    print(f"Number of boxes analyzed: {len(all_widths)}")
    print(f"\nWidth stats:")
    print(f"  Min: {np.min(all_widths):.1f}")
    print(f"  Max: {np.max(all_widths):.1f}")
    print(f"  Mean: {np.mean(all_widths):.1f}")
    print(f"  Median: {np.median(all_widths):.1f}")
    print(f"  25th percentile: {np.percentile(all_widths, 25):.1f}")
    print(f"  75th percentile: {np.percentile(all_widths, 75):.1f}")

    print(f"\nHeight stats:")
    print(f"  Min: {np.min(all_heights):.1f}")
    print(f"  Max: {np.max(all_heights):.1f}")
    print(f"  Mean: {np.mean(all_heights):.1f}")
    print(f"  Median: {np.median(all_heights):.1f}")
    print(f"  25th percentile: {np.percentile(all_heights, 25):.1f}")
    print(f"  75th percentile: {np.percentile(all_heights, 75):.1f}")

    print(f"\nArea stats:")
    print(f"  Min: {np.min(all_areas):.1f}")
    print(f"  Max: {np.max(all_areas):.1f}")
    print(f"  Mean: {np.mean(all_areas):.1f}")
    print(f"  Median: {np.median(all_areas):.1f}")

    # Plot distributions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.hist(all_widths, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Count")
    ax1.set_title("GT Box Width Distribution")
    ax1.axvline(
        np.median(all_widths),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(all_widths):.1f}",
    )
    ax1.legend()

    ax2.hist(all_heights, bins=50, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("GT Box Height Distribution")
    ax2.axvline(
        np.median(all_heights),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(all_heights):.1f}",
    )
    ax2.legend()

    ax3.scatter(all_widths, all_heights, alpha=0.5)
    ax3.set_xlabel("Width (pixels)")
    ax3.set_ylabel("Height (pixels)")
    ax3.set_title("Width vs Height")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gt_box_sizes.png")
    plt.show()

    # Suggest anchor sizes based on the data
    print("\nSuggested anchor sizes based on data:")
    sizes = []
    for percentile in [25, 50, 75]:
        size = int(np.sqrt(np.percentile(all_areas, percentile)))
        sizes.append(size)
    print(f"  Sizes: {sizes}")

    # Also check aspect ratios
    aspect_ratios = [w / h for w, h in zip(all_widths, all_heights) if h > 0]
    print(f"\nAspect ratio stats:")
    print(f"  Min: {np.min(aspect_ratios):.2f}")
    print(f"  Max: {np.max(aspect_ratios):.2f}")
    print(f"  Mean: {np.mean(aspect_ratios):.2f}")
    print(f"  Median: {np.median(aspect_ratios):.2f}")


if __name__ == "__main__":
    analyze_gt_box_sizes()
