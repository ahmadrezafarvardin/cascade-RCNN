# src/recognition/train_recognition_improved.py
import sys
from pathlib import Path
import torch
import json
import numpy as np
from typing import Dict, List, Tuple
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.character_classifier import CharacterClassifier
from src.recognition.semi_supervised_trainer import (
    SemiSupervisedTrainer,
    CharacterDataset,
)
from src.recognition.expression_recognizer import ExpressionRecognizer
from ultralytics import YOLO


def verify_yolo_model(yolo_model_path):
    """Verify that YOLO model is trained for character detection"""
    print(f"Loading YOLO model from: {yolo_model_path}")
    model = YOLO(yolo_model_path)

    # Test on a sample image
    test_img = "dataset/val/images/391.png"
    if Path(test_img).exists():
        results = model(test_img, verbose=False)
        for r in results:
            if r.boxes is not None:
                print(f"YOLO detected {len(r.boxes)} objects")
                # Check class names if available
                if hasattr(model, "names"):
                    print(f"Model classes: {model.names}")

    return model


def extract_characters_from_clustering_results(
    clustering_dir: str = "results/clustering_complete",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load character images and cluster labels from clustering results
    """
    characters = []
    cluster_labels = []

    # Load sample characters
    sample_dir = Path(clustering_dir) / "sample_characters_yolo"
    if sample_dir.exists():
        for img_file in sorted(sample_dir.glob("*.png")):
            img = cv2.imread(str(img_file))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                characters.append(img_rgb)

    # Load cluster labels
    results_file = Path(clustering_dir) / "clustering_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
            if "labels" in data:
                cluster_labels = np.array(data["labels"][: len(characters)])

    print(f"Loaded {len(characters)} characters from clustering results")
    return characters, cluster_labels


def create_manual_labeled_samples() -> Tuple[List[np.ndarray], List[int]]:
    """
    Create a small set of manually labeled samples for each character class
    """
    print("Creating manual labeled samples...")

    images = []
    labels = []

    # Character set with better font variations
    characters = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "+",
        "-",
        "*",
        "/",
        "(",
        ")",
    ]
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]

    for idx, char in enumerate(characters):
        for font in fonts:
            for size_factor in [0.8, 1.0, 1.2]:
                # Create character image
                img = np.ones((64, 64, 3), dtype=np.uint8) * 255

                font_scale = 1.5 * size_factor
                thickness = 2

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    char, font, font_scale, thickness
                )

                # Center the text
                x = (64 - text_width) // 2
                y = (64 + text_height) // 2

                # Draw text
                cv2.putText(img, char, (x, y), font, font_scale, (0, 0, 0), thickness)

                # Add slight rotation
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                    img = cv2.warpAffine(img, M, (64, 64), borderValue=255)

                images.append(img)
                labels.append(idx)

    print(f"Created {len(images)} manual labeled samples")
    return images, labels


def augment_character_image(img):
    """Apply data augmentation to character image"""
    augmented = img.copy()

    # Random brightness
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=factor, beta=0)

    # Random noise
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 5, augmented.shape).astype(np.uint8)
        augmented = cv2.add(augmented, noise)

    # Random blur
    if np.random.rand() > 0.5:
        kernel_size = np.random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)

    return augmented


def train_improved_model(
    dataset_path: str = "dataset",
    output_dir: str = "results/recognition",
    yolo_model_path: str = None,
    clustering_dir: str = "results/clustering_complete",
    epochs: int = 100,
    batch_size: int = 32,
):
    """
    Train character recognition model with improved data handling
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Verify and load correct YOLO model
    if not yolo_model_path:
        # Try to find character detection YOLO model
        possible_paths = [
            "results/yolo_runs/detect/character_detection/weights/best.pt",
            "results/yolo_runs/detect/character_detection/weights/last.pt",
            "src/models/yolo/character_detection_best.pt",
        ]

        for path in possible_paths:
            if Path(path).exists():
                yolo_model_path = path
                break
        else:
            print("WARNING: No character detection YOLO model found!")
            print("Using default YOLO model - results may be poor")
            yolo_model_path = "yolov8n.pt"

    yolo_model = verify_yolo_model(yolo_model_path)

    # Step 2: Create labeled dataset
    labeled_images = []
    labeled_targets = []

    # Add manual labeled samples
    manual_images, manual_labels = create_manual_labeled_samples()
    labeled_images.extend(manual_images)
    labeled_targets.extend(manual_labels)

    # Step 3: Load unlabeled data from clustering
    unlabeled_images = []
    cluster_labels = None

    if Path(clustering_dir).exists():
        cluster_chars, cluster_labels = extract_characters_from_clustering_results(
            clustering_dir
        )
        unlabeled_images.extend(cluster_chars)

    # If not enough unlabeled data, extract more
    if len(unlabeled_images) < 500:
        print(f"\nExtracting additional unlabeled characters...")
        from src.clustering.feature_extraction import YOLOFeatureExtractor

        extractor = YOLOFeatureExtractor(yolo_model_path)

        for split in ["train", "val"]:
            img_dir = Path(dataset_path) / split / "images"
            for img_path in list(img_dir.glob("*.png"))[:50]:
                chars = extractor.extract_detected_characters(
                    str(img_path), conf_threshold=0.3
                )
                unlabeled_images.extend(chars)

                if len(unlabeled_images) >= 1000:
                    break

    print(f"\nDataset summary:")
    print(f"  Labeled samples: {len(labeled_images)}")
    print(f"  Unlabeled samples: {len(unlabeled_images)}")

    # Step 4: Apply data augmentation
    print("\nApplying data augmentation...")
    augmented_images = []
    augmented_labels = []

    for img, label in zip(labeled_images, labeled_targets):
        # Add original
        augmented_images.append(img)
        augmented_labels.append(label)

        # Add augmented versions
        for _ in range(2):
            aug_img = augment_character_image(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    labeled_images = augmented_images
    labeled_targets = augmented_labels
    print(f"  After augmentation: {len(labeled_images)} labeled samples")

    # Step 5: Initialize and train model
    print("\nInitializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterClassifier(num_classes=16, pretrained=True)
    trainer = SemiSupervisedTrainer(model, device)

    print("\nStarting semi-supervised training...")
    trainer.train_semi_supervised(
        labeled_images=labeled_images,
        labeled_targets=labeled_targets,
        unlabeled_images=unlabeled_images,
        cluster_labels=cluster_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        pseudo_label_weight=0.3,  # Lower weight for pseudo-labels
        confidence_threshold=0.90,  # Higher threshold
    )

    # Step 6: Save model
    model_path = Path(output_dir) / "character_classifier_improved.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_idx": model.char_to_idx,
            "idx_to_char": model.idx_to_char,
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")

    # Step 7: Test on sample characters
    print("\nTesting on sample characters...")
    test_character_recognition(model, yolo_model, dataset_path, output_dir)

    return model, yolo_model


def test_character_recognition(model, yolo_model, dataset_path, output_dir):
    """Test character recognition on sample images"""
    model.eval()

    # Test on a few validation images
    val_dir = Path(dataset_path) / "val" / "images"
    test_images = list(val_dir.glob("*.png"))[:5]

    fig, axes = plt.subplots(len(test_images), 1, figsize=(15, 4 * len(test_images)))
    if len(test_images) == 1:
        axes = [axes]

    recognizer = ExpressionRecognizer(model, yolo_model)

    for idx, img_path in enumerate(test_images):
        # Detect and classify characters
        results = yolo_model(str(img_path), conf=0.3, verbose=False)

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw detections and predictions
        expression = ""
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Extract character
                    char_img = img_rgb[y1:y2, x1:x2]

                    # Classify
                    with torch.no_grad():
                        char_tensor = model.preprocess_image(char_img).unsqueeze(0)
                        char_tensor = char_tensor.to(next(model.parameters()).device)

                        output = model(char_tensor)
                        prob = torch.softmax(output, dim=1)
                        conf, pred = torch.max(prob, 1)

                        if conf.item() > 0.5:
                            char = model.idx_to_char[pred.item()]
                            expression += char

                            # Draw box and label
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                img_rgb,
                                f"{char}:{conf.item():.2f}",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"Detected: {expression}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / "character_detection_test.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(f"Test visualization saved to {output_dir}/character_detection_test.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train improved character recognition model"
    )
    parser.add_argument("--dataset", default="dataset", help="Dataset path")
    parser.add_argument(
        "--output", default="results/recognition", help="Output directory"
    )
    parser.add_argument("--yolo-model", default=None, help="YOLO model path")
    parser.add_argument(
        "--clustering",
        default="results/clustering_complete",
        help="Clustering results directory",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    train_improved_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        yolo_model_path=args.yolo_model,
        clustering_dir=args.clustering,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )