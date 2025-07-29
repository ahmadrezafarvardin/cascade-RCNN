import json
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def explore_dataset(root_dir, split="train", num_samples=3, save_dir=None):
    """
    Explore the dataset to understand its structure

    Args:
        root_dir: Project root directory
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to explore
        save_dir: Directory to save results
    """
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f"{split}_exploration_log.txt")
        with open(log_file, "w") as f:
            f.write(f"Dataset Exploration: {split} split\n")
            f.write("=" * 50 + "\n")

    images_dir = os.path.join(root_dir, "dataset", split, "images")
    labels_dir = os.path.join(root_dir, "dataset", split, "labels")

    # List image files
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    )

    print(f"Total {split} images: {len(image_files)}")
    if save_dir:
        with open(log_file, "a") as f:
            f.write(f"Total {split} images: {len(image_files)}\n\n")

    # Select random samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    # Store dataset statistics
    dataset_stats = {"total_images": len(image_files), "samples": []}

    for img_file in samples:
        img_path = os.path.join(images_dir, img_file)
        image = Image.open(img_path)

        # Print image information
        print(f"\nImage: {img_file}")
        print(f"Size: {image.size}")
        print(f"Mode: {image.mode}")

        sample_info = {"filename": img_file, "size": image.size, "mode": image.mode}

        # Log to file if saving
        if save_dir:
            with open(log_file, "a") as f:
                f.write(f"\nImage: {img_file}\n")
                f.write(f"Size: {image.size}\n")
                f.write(f"Mode: {image.mode}\n")

        # Check corresponding label file
        json_name = os.path.splitext(img_file)[0] + ".json"
        json_path = os.path.join(labels_dir, json_name)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Print annotation structure
            print("Annotation keys:", annotation.keys())

            sample_info["annotation_keys"] = list(annotation.keys())

            # Check annotations format
            if "annotations" in annotation:
                print(f"Number of annotations: {len(annotation['annotations'])}")
                if len(annotation["annotations"]) > 0:
                    print(f"First annotation format: {annotation['annotations'][0]}")

                sample_info["num_annotations"] = len(annotation["annotations"])
                if len(annotation["annotations"]) > 0:
                    sample_info["first_annotation_format"] = annotation["annotations"][
                        0
                    ]

            # Check for expression
            if "expression" in annotation:
                print(f"Expression: {annotation['expression']}")
                sample_info["expression"] = annotation["expression"]

            # Log to file if saving
            if save_dir:
                with open(log_file, "a") as f:
                    f.write(f"Annotation keys: {annotation.keys()}\n")
                    if "annotations" in annotation:
                        f.write(
                            f"Number of annotations: {len(annotation['annotations'])}\n"
                        )
                        if len(annotation["annotations"]) > 0:
                            f.write(
                                f"First annotation format: {annotation['annotations'][0]}\n"
                            )
                    if "expression" in annotation:
                        f.write(f"Expression: {annotation['expression']}\n")

            # Visualize image with bounding boxes
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(np.array(image))

            # Draw bounding boxes
            if "annotations" in annotation:
                for box in annotation["annotations"]:
                    # Check format of box
                    if len(box) == 4:
                        x, y, w, h = box
                        rect = patches.Rectangle(
                            (x, y), w, h, linewidth=1, edgecolor="r", facecolor="none"
                        )
                        ax.add_patch(rect)

            plt.title(f"{img_file} - {len(annotation.get('annotations', []))} boxes")
            plt.axis("off")
            plt.tight_layout()

            # Save visualization
            if save_dir:
                viz_path = os.path.join(
                    save_dir, f"{split}_{os.path.splitext(img_file)[0]}_viz.png"
                )
                plt.savefig(viz_path, bbox_inches="tight")
                print(f"Saved visualization to {viz_path}")

            plt.show()
        else:
            print(f"No label file found for {img_file}")
            sample_info["has_label"] = False

            if save_dir:
                with open(log_file, "a") as f:
                    f.write(f"No label file found for {img_file}\n")

        dataset_stats["samples"].append(sample_info)

    # Save dataset statistics as JSON
    if save_dir:
        stats_path = os.path.join(save_dir, f"{split}_dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump(dataset_stats, f, indent=2)
        print(f"Saved dataset statistics to {stats_path}")

    return dataset_stats


if __name__ == "__main__":
    # Set your project root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Create results directory for exploration
    results_dir = os.path.join(root_dir, "results", "dataset_exploration")
    os.makedirs(results_dir, exist_ok=True)

    # Explore train set
    print("====== EXPLORING TRAIN SET ======")
    explore_dataset(root_dir, "train", save_dir=results_dir)

    # Explore validation set
    print("\n====== EXPLORING VALIDATION SET ======")
    explore_dataset(root_dir, "val", save_dir=results_dir)
