# src/recognition/train_recognition.py
import sys
from pathlib import Path
import torch
import json
import numpy as np
from typing import Dict, List, Tuple
import cv2
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.character_classifier import CharacterClassifier
from src.recognition.semi_supervised_trainer import (
    SemiSupervisedTrainer,
    CharacterDataset,
)
from src.recognition.expression_recognizer import ExpressionRecognizer
from src.models.yolo import YOLOCharacterDetector
from src.clustering.feature_extraction import YOLOFeatureExtractor


def load_expression_labels(dataset_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load expression labels from dataset

    Returns:
        train_expressions: Dictionary of training image -> expression
        val_expressions: Dictionary of validation image -> expression
    """
    train_expressions = {}
    val_expressions = {}

    # Load training expressions
    train_labels_dir = Path(dataset_path) / "train" / "labels"
    for label_file in train_labels_dir.glob("*.json"):
        with open(label_file, "r") as f:
            data = json.load(f)
            if "expression" in data:
                img_name = label_file.stem + ".png"
                train_expressions[img_name] = data["expression"]

    # Load validation expressions
    val_labels_dir = Path(dataset_path) / "val" / "labels"
    for label_file in val_labels_dir.glob("*.json"):
        with open(label_file, "r") as f:
            data = json.load(f)
            if "expression" in data:
                img_name = label_file.stem + ".png"
                val_expressions[img_name] = data["expression"]

    print(f"Loaded {len(train_expressions)} training expressions")
    print(f"Loaded {len(val_expressions)} validation expressions")

    return train_expressions, val_expressions


def create_labeled_dataset_from_expressions(
    expression_labels: Dict[str, str], dataset_path: str, yolo_model_path: str = None
) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Create labeled dataset from expression annotations

    Returns:
        images: List of character images
        labels: List of character labels
        metadata: List of metadata for each character
    """
    # Initialize YOLO for character detection
    extractor = YOLOFeatureExtractor(yolo_model_path)

    images = []
    labels = []
    metadata = []

    # Character mapping
    char_to_idx = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "+": 10,
        "-": 11,
        "*": 12,
        "/": 13,
        "(": 14,
        ")": 15,
    }

    for img_name, expression in expression_labels.items():
        # Find image path
        img_path = None
        for split in ["train", "val"]:
            potential_path = Path(dataset_path) / split / "images" / img_name
            if potential_path.exists():
                img_path = str(potential_path)
                break

        if not img_path:
            continue

        # Detect characters
        detected_chars = extractor.extract_detected_characters(
            img_path, conf_threshold=0.25
        )

        if len(detected_chars) == len(expression):
            # Perfect match - assign labels directly
            for char_img, char_label in zip(detected_chars, expression):
                if char_label in char_to_idx:
                    images.append(char_img)
                    labels.append(char_to_idx[char_label])
                    metadata.append({"source_image": img_name, "character": char_label})
        else:
            # Mismatch - skip or use more sophisticated alignment
            print(
                f"Mismatch in {img_name}: detected {len(detected_chars)}, expected {len(expression)}"
            )

    print(f"Created labeled dataset with {len(images)} characters")
    return images, labels, metadata


def train_semi_supervised_model(
    dataset_path: str = "dataset",
    output_dir: str = "results/recognition",
    yolo_model_path: str = None,
    clustering_results_path: str = None,
    epochs: int = 50,
    batch_size: int = 32,
):
    """
    Train character recognition model using semi-supervised learning
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load expression labels
    print("Loading expression labels...")
    train_expressions, val_expressions = load_expression_labels(dataset_path)

    # Step 2: Create labeled dataset from expressions
    print("\nCreating labeled dataset from expressions...")
    labeled_images, labeled_targets, labeled_metadata = (
        create_labeled_dataset_from_expressions(
            train_expressions, dataset_path, yolo_model_path
        )
    )

    # Step 3: Load unlabeled data (all detected characters)
    print("\nLoading unlabeled character data...")
    # Load from clustering results if available
    if clustering_results_path and os.path.exists(clustering_results_path):
        features_data = np.load(clustering_results_path)
        # Load sample character images
        sample_dir = Path(clustering_results_path).parent / "sample_characters_yolo"
        unlabeled_images = []

        for img_file in sorted(os.listdir(sample_dir)):
            if img_file.endswith(".png"):
                img = cv2.imread(str(sample_dir / img_file))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    unlabeled_images.append(img_rgb)
    else:
        # Extract characters from all images
        print("Extracting characters from dataset...")
        extractor = YOLOFeatureExtractor(yolo_model_path)
        unlabeled_images = []

        for split in ["train", "val"]:
            img_dir = Path(dataset_path) / split / "images"
            for img_file in list(img_dir.glob("*.png"))[:100]:  # Limit for efficiency
                chars = extractor.extract_detected_characters(str(img_file))
                unlabeled_images.extend(chars)

    print(f"Loaded {len(unlabeled_images)} unlabeled characters")

    # Step 4: Load clustering results if available
    cluster_labels = None
    if clustering_results_path:
        try:
            with open(
                Path(clustering_results_path).parent / "clustering_results.json", "r"
            ) as f:
                clustering_data = json.load(f)
                cluster_labels = np.array(clustering_data["labels"])
                print(
                    f"Loaded clustering results with {clustering_data['n_clusters']} clusters"
                )
        except:
            print("Could not load clustering results")

    # Step 5: Initialize model and trainer
    print("\nInitializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterClassifier(num_classes=16, pretrained=True)
    trainer = SemiSupervisedTrainer(model, device)

    # Step 6: Train model
    print("\nStarting semi-supervised training...")
    trainer.train_semi_supervised(
        labeled_images=labeled_images,
        labeled_targets=labeled_targets,
        unlabeled_images=unlabeled_images,
        cluster_labels=cluster_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001,
        pseudo_label_weight=0.5,
        confidence_threshold=0.9,
    )

    # Step 7: Save model
    model_path = Path(output_dir) / "character_classifier.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_idx": model.char_to_idx,
            "idx_to_char": model.idx_to_char,
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")

    # Step 8: Evaluate on validation set
    print("\nEvaluating on validation set...")
    evaluate_model(model, val_expressions, dataset_path, yolo_model_path, output_dir)

    return model


def evaluate_model(
    model: CharacterClassifier,
    val_expressions: Dict[str, str],
    dataset_path: str,
    yolo_model_path: str,
    output_dir: str,
):
    """Evaluate model on validation expressions"""
    # Initialize recognizer
    yolo_model = YOLOCharacterDetector(yolo_model_path) if yolo_model_path else None
    recognizer = ExpressionRecognizer(model, yolo_model.model)

    predictions = {}

    for img_name in val_expressions:
        # Find image path
        img_path = Path(dataset_path) / "val" / "images" / img_name
        if img_path.exists():
            pred_expression = recognizer.recognize_expression(str(img_path))
            predictions[img_name] = pred_expression
            print(
                f"{img_name}: True='{val_expressions[img_name]}', Pred='{pred_expression}'"
            )

    # Calculate metrics
    metrics = recognizer.evaluate_predictions(predictions, val_expressions)

    print("\nValidation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results = {
        "predictions": predictions,
        "ground_truth": val_expressions,
        "metrics": metrics,
    }

    with open(Path(output_dir) / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)


def generate_test_predictions(
    model_path: str,
    test_dir: str,
    output_file: str = "test_predictions.csv",
    yolo_model_path: str = None,
):
    """Generate predictions for test set"""
    # Load model
    checkpoint = torch.load(model_path)
    model = CharacterClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize recognizer
    yolo_model = YOLOCharacterDetector(yolo_model_path) if yolo_model_path else None
    recognizer = ExpressionRecognizer(model, yolo_model.model)

    # Process test images
    predictions = []
    test_images = sorted(Path(test_dir).glob("*.png"))

    for img_path in test_images:
        expression = recognizer.recognize_expression(str(img_path))
        predictions.append({"image": img_path.name, "expression": expression})
        print(f"{img_path.name}: {expression}")

    # Save predictions
    import csv

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "expression"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train character recognition model")
    parser.add_argument("--dataset", default="dataset", help="Dataset path")
    parser.add_argument(
        "--output", default="results/recognition", help="Output directory"
    )
    parser.add_argument("--yolo-model", default=None, help="YOLO model path")
    parser.add_argument("--clustering", default=None, help="Clustering results path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Generate test predictions")
    parser.add_argument(
        "--model", default=None, help="Trained model path (for testing)"
    )

    args = parser.parse_args()

    if args.test:
        if not args.model:
            args.model = "results/recognition/character_classifier.pth"
        generate_test_predictions(
            args.model, "dataset/test/images", "test_predictions.csv", args.yolo_model
        )
    else:
        # Use best YOLO model if not specified
        if not args.yolo_model:
            best_yolo = Path(
                "results/yolo_runs/detect/character_detection/weights/best.pt"
            )
            if best_yolo.exists():
                args.yolo_model = str(best_yolo)

        # Use clustering results if not specified
        if not args.clustering:
            clustering_path = Path(
                "results/clustering_complete/character_features_yolo.npz"
            )
            if clustering_path.exists():
                args.clustering = str(clustering_path)

        train_semi_supervised_model(
            dataset_path=args.dataset,
            output_dir=args.output,
            yolo_model_path=args.yolo_model,
            clustering_results_path=args.clustering,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
