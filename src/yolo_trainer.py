# src/yolo_trainer.py
from ultralytics import YOLO
import os
import json
import shutil
from PIL import Image
import numpy as np


def convert_to_yolo_format():
    """Convert dataset to YOLO format with proper normalization"""
    for split in ["train", "val"]:
        # Create directories
        os.makedirs(f"dataset_yolo/{split}/images", exist_ok=True)
        os.makedirs(f"dataset_yolo/{split}/labels", exist_ok=True)

        img_dir = f"dataset/{split}/images"
        label_dir = f"dataset/{split}/labels"

        skipped_images = []

        for img_name in os.listdir(img_dir):
            if not img_name.endswith(".png"):
                continue

            # Load image to get dimensions
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Copy image
            shutil.copy(img_path, f"dataset_yolo/{split}/images/{img_name}")

            # Convert labels
            label_path = os.path.join(label_dir, img_name.replace(".png", ".json"))
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    label_data = json.load(f)

                # Convert to YOLO format with validation
                yolo_labels = []
                valid_image = True

                for ann in label_data.get("annotations", []):
                    bbox = ann["boundingBox"]

                    # Calculate YOLO format coordinates
                    x_min = bbox["x"]
                    y_min = bbox["y"]
                    x_max = x_min + bbox["width"]
                    y_max = y_min + bbox["height"]

                    # Clip coordinates to image boundaries
                    x_min = max(0, min(x_min, img_w))
                    y_min = max(0, min(y_min, img_h))
                    x_max = max(0, min(x_max, img_w))
                    y_max = max(0, min(y_max, img_h))

                    # Skip if box is invalid after clipping
                    if x_max <= x_min or y_max <= y_min:
                        print(f"Skipping invalid box in {img_name}: {bbox}")
                        continue

                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x_min + x_max) / 2 / img_w
                    y_center = (y_min + y_max) / 2 / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h

                    # Validate normalized coordinates
                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 < width <= 1
                        and 0 < height <= 1
                    ):
                        print(
                            f"Invalid normalized coords in {img_name}: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})"
                        )
                        valid_image = False
                        break

                    yolo_labels.append(
                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

                if valid_image and yolo_labels:
                    # Save YOLO labels
                    with open(
                        f'dataset_yolo/{split}/labels/{img_name.replace(".png", ".txt")}',
                        "w",
                    ) as f:
                        f.write("\n".join(yolo_labels))
                else:
                    # Remove the copied image if labels are invalid
                    os.remove(f"dataset_yolo/{split}/images/{img_name}")
                    skipped_images.append(img_name)

        print(
            f"{split} set: Skipped {len(skipped_images)} images with invalid annotations"
        )
        if skipped_images:
            print(
                f"Skipped images: {skipped_images[:5]}{'...' if len(skipped_images) > 5 else ''}"
            )


def train_yolo():
    """Train YOLOv8 model with memory optimization"""
    # Create data.yaml
    data_yaml = """
path: dataset_yolo
train: train/images
val: val/images

nc: 1
names: ['character']
"""
    with open("dataset_yolo/data.yaml", "w") as f:
        f.write(data_yaml)

    # Use model and batch size for GTX 1650
    model = YOLO("yolov8n.pt")  # Use nano model for less memory

    results = model.train(
        data="dataset_yolo/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,  # Reduced batch size
        device=0,
        workers=4,  # Reduced workers
        amp=False,  # Disable AMP as it's causing issues
        name="character_detection",
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # Memory optimization
        cache=False,  # Don't cache images in RAM
        rect=False,  # Don't use rectangular training
        mosaic=0.5,  # Reduce mosaic augmentation
        mixup=0.0,  # Disable mixup
        copy_paste=0.0,  # Disable copy-paste
    )

    return model


def validate_dataset():
    """Validate the converted dataset"""
    for split in ["train", "val"]:
        img_dir = f"dataset_yolo/{split}/images"
        label_dir = f"dataset_yolo/{split}/labels"

        total_images = len([f for f in os.listdir(img_dir) if f.endswith(".png")])
        total_labels = len([f for f in os.listdir(label_dir) if f.endswith(".txt")])

        print(f"\n{split} set statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Total label files: {total_labels}")

        # Check a few samples
        sample_count = 0
        total_boxes = 0
        for label_file in os.listdir(label_dir)[:5]:
            with open(os.path.join(label_dir, label_file), "r") as f:
                lines = f.readlines()
                total_boxes += len(lines)
                sample_count += 1

        if sample_count > 0:
            print(f"  Average boxes per image (sample): {total_boxes/sample_count:.1f}")


if __name__ == "__main__":
    # Convert and validate dataset
    print("Converting dataset to YOLO format...")
    convert_to_yolo_format()

    print("\nValidating converted dataset...")
    validate_dataset()

    print("\nStarting training...")
    model = train_yolo()

    # Validate
    print("\nRunning validation...")
    metrics = model.val()
    print(f"\nValidation metrics:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")

    # Test on a few images
    print("\nTesting on sample images...")
    test_images = os.listdir("dataset/val/images")[:3]
    for img_name in test_images:
        img_path = f"dataset/val/images/{img_name}"
        results = model(img_path)
        # Save results
        for r in results:
            r.save(filename=f"results/yolo_test_{img_name}")
