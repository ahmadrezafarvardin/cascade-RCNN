import numpy as np
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def analyze_scaled_gt_boxes():
    """Analyze GT boxes after scaling to 416x416"""

    from data.data_preparation import MathExpressionDataset, get_transforms

    dataset = MathExpressionDataset(
        root_dir="./", split="train", transform=get_transforms(train=False)
    )

    all_boxes = []

    print("Collecting scaled GT boxes...")
    for idx in range(min(200, len(dataset))):
        _, target, _ = dataset[idx]
        boxes = target["boxes"].numpy()

        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w > 0 and h > 0:
                all_boxes.append([w, h])

    all_boxes = np.array(all_boxes)

    print(f"\nAnalyzed {len(all_boxes)} boxes in 416x416 images")
    print(
        f"Width: min={all_boxes[:, 0].min():.1f}, max={all_boxes[:, 0].max():.1f}, mean={all_boxes[:, 0].mean():.1f}"
    )
    print(
        f"Height: min={all_boxes[:, 1].min():.1f}, max={all_boxes[:, 1].max():.1f}, mean={all_boxes[:, 1].mean():.1f}"
    )

    # K-means clustering to find optimal anchor sizes
    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_boxes)

    anchors = kmeans.cluster_centers_
    anchors = anchors[anchors[:, 0].argsort()]  # Sort by width

    print(f"\nOptimal anchor sizes (width, height):")
    for i, (w, h) in enumerate(anchors):
        print(f"  Anchor {i+1}: {w:.1f} x {h:.1f} (ratio: {w/h:.2f})")

    # Visualize
    plt.figure(figsize=(10, 8))
    plt.scatter(all_boxes[:, 0], all_boxes[:, 1], alpha=0.5, s=1)
    plt.scatter(
        anchors[:, 0], anchors[:, 1], color="red", s=100, marker="x", linewidths=3
    )
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("GT Box Sizes in 416x416 Images (with K-means centers)")
    plt.grid(True, alpha=0.3)
    plt.savefig("./results/rcnn/scaled_gt_analysis.png", dpi=150)
    plt.show()

    return anchors


if __name__ == "__main__":
    anchors = analyze_scaled_gt_boxes()
