import numpy as np
import matplotlib.pyplot as plt
from data.data_preparation import MathExpressionDataset, get_transforms
from PIL import Image


def analyze_ground_truth_boxes():
    """Analyze the size distribution of ground truth boxes"""

    # Load validation dataset without transforms to get original sizes
    val_dataset = MathExpressionDataset(
        root_dir="./", split="val", transform=None  # No transform to get original sizes
    )

    all_widths = []
    all_heights = []
    all_areas = []
    image_sizes = []

    print("Analyzing ground truth boxes...")

    for idx in range(min(50, len(val_dataset))):
        image, target, img_name = val_dataset[idx]

        # Get image size
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        else:
            img_h, img_w = image.shape[:2]

        image_sizes.append((img_w, img_h))

        gt_boxes = target["boxes"]

        for box in gt_boxes:
            if isinstance(box, torch.Tensor):
                box = box.numpy()

            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height

            all_widths.append(width)
            all_heights.append(height)
            all_areas.append(area)

    # Calculate statistics
    print(f"\nAnalyzed {len(all_widths)} boxes from {idx+1} images")
    print(f"\nImage sizes: {set(image_sizes)}")

    print(f"\nBox Width Statistics:")
    print(f"  Min: {np.min(all_widths):.1f}")
    print(f"  Max: {np.max(all_widths):.1f}")
    print(f"  Mean: {np.mean(all_widths):.1f}")
    print(f"  Median: {np.median(all_widths):.1f}")
    print(f"  Std: {np.std(all_widths):.1f}")

    print(f"\nBox Height Statistics:")
    print(f"  Min: {np.min(all_heights):.1f}")
    print(f"  Max: {np.max(all_heights):.1f}")
    print(f"  Mean: {np.mean(all_heights):.1f}")
    print(f"  Median: {np.median(all_heights):.1f}")
    print(f"  Std: {np.std(all_heights):.1f}")

    print(f"\nAspect Ratio (W/H) Statistics:")
    aspect_ratios = np.array(all_widths) / np.array(all_heights)
    print(f"  Min: {np.min(aspect_ratios):.2f}")
    print(f"  Max: {np.max(aspect_ratios):.2f}")
    print(f"  Mean: {np.mean(aspect_ratios):.2f}")
    print(f"  Median: {np.median(aspect_ratios):.2f}")

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(all_widths, bins=50, edgecolor="black")
    axes[0, 0].set_title("Box Width Distribution")
    axes[0, 0].set_xlabel("Width (pixels)")
    axes[0, 0].axvline(
        np.mean(all_widths),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_widths):.1f}",
    )
    axes[0, 0].legend()

    axes[0, 1].hist(all_heights, bins=50, edgecolor="black")
    axes[0, 1].set_title("Box Height Distribution")
    axes[0, 1].set_xlabel("Height (pixels)")
    axes[0, 1].axvline(
        np.mean(all_heights),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_heights):.1f}",
    )
    axes[0, 1].legend()

    axes[1, 0].hist(aspect_ratios, bins=50, edgecolor="black")
    axes[1, 0].set_title("Aspect Ratio (W/H) Distribution")
    axes[1, 0].set_xlabel("Aspect Ratio")
    axes[1, 0].axvline(
        np.mean(aspect_ratios),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(aspect_ratios):.2f}",
    )
    axes[1, 0].legend()

    axes[1, 1].scatter(all_widths, all_heights, alpha=0.5)
    axes[1, 1].set_title("Width vs Height")
    axes[1, 1].set_xlabel("Width (pixels)")
    axes[1, 1].set_ylabel("Height (pixels)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./results/rcnn/gt_box_analysis.png", dpi=150)
    plt.show()

    return {
        "widths": all_widths,
        "heights": all_heights,
        "aspect_ratios": aspect_ratios.tolist(),
        "image_sizes": image_sizes,
    }


def test_specific_proposals(stats):
    """Generate proposals based on actual GT statistics"""
    from utils.box_utils import calculate_iou

    # Use the statistics to generate better proposals
    width_mean = np.mean(stats["widths"])
    height_mean = np.mean(stats["heights"])
    width_std = np.std(stats["widths"])
    height_std = np.std(stats["heights"])

    print(f"\nGenerating proposals based on GT statistics...")
    print(f"Target width: {width_mean:.1f} ± {width_std:.1f}")
    print(f"Target height: {height_mean:.1f} ± {height_std:.1f}")

    # Load one image to test
    val_dataset = MathExpressionDataset(root_dir="./", split="val", transform=None)

    image, target, img_name = val_dataset[0]
    gt_boxes = target["boxes"]

    if isinstance(image, Image.Image):
        img_w, img_h = image.size
    else:
        img_h, img_w = image.shape[:2]

    # Generate proposals with sizes based on statistics
    proposals = []

    # Use actual GT box size ranges
    min_w = int(np.percentile(stats["widths"], 10))
    max_w = int(np.percentile(stats["widths"], 90))
    min_h = int(np.percentile(stats["heights"], 10))
    max_h = int(np.percentile(stats["heights"], 90))

    print(f"\nUsing size ranges:")
    print(f"Width: {min_w} - {max_w}")
    print(f"Height: {min_h} - {max_h}")

    # Dense sliding window with appropriate sizes
    stride = 5
    for w in range(min_w, max_w + 1, 5):
        for h in range(min_h, max_h + 1, 5):
            for y in range(0, img_h - h + 1, stride):
                for x in range(0, img_w - w + 1, stride):
                    proposals.append([x, y, x + w, y + h])

    proposals = np.array(proposals[:2000])  # Limit for testing

    # Check coverage
    matched = 0
    for gt_box in gt_boxes:
        ious = [calculate_iou(prop, gt_box.numpy()) for prop in proposals]
        if ious and max(ious) >= 0.5:
            matched += 1
            print(f"GT box {gt_box.numpy()} matched with IoU: {max(ious):.3f}")

    print(
        f"\nCoverage with targeted proposals: {matched}/{len(gt_boxes)} = {matched/len(gt_boxes)*100:.1f}%"
    )


if __name__ == "__main__":
    import torch

    stats = analyze_ground_truth_boxes()
    test_specific_proposals(stats)
