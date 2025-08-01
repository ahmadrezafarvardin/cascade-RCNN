# src/recognition/evaluate_and_predict.py
import sys
from pathlib import Path
import torch
import json
import numpy as np
import cv2
from tqdm import tqdm
import csv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.character_classifier import CharacterClassifier
from src.recognition.expression_recognizer import ExpressionRecognizer
from ultralytics import YOLO


def evaluate_on_validation(
    model_path="results/recognition/character_classifier_improved.pth",
    yolo_model_path="results/yolo_runs/detect/character_detection2/weights/best.pt",
    dataset_path="dataset",
    output_dir="results/recognition",
):
    """Evaluate the trained model on validation expressions"""

    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load character classifier
    checkpoint = torch.load(model_path, map_location=device)
    model = CharacterClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load YOLO model
    yolo_model = YOLO(yolo_model_path)

    # Initialize recognizer
    recognizer = ExpressionRecognizer(model, yolo_model)

    # Load validation expressions
    val_expressions = {}
    val_labels_dir = Path(dataset_path) / "val" / "labels"

    for label_file in val_labels_dir.glob("*.json"):
        with open(label_file, "r") as f:
            data = json.load(f)
            if "expression" in data:
                img_name = label_file.stem + ".png"
                val_expressions[img_name] = data["expression"]

    print(f"\nEvaluating on {len(val_expressions)} validation expressions...")

    # Evaluate with different confidence thresholds
    best_threshold = 0.5
    best_score = float("inf")

    for conf_threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        predictions = {}

        for img_name in tqdm(val_expressions, desc=f"Conf={conf_threshold}"):
            img_path = Path(dataset_path) / "val" / "images" / img_name
            if img_path.exists():
                pred_expr = recognizer.recognize_expression(
                    str(img_path), conf_threshold=conf_threshold
                )
                predictions[img_name] = pred_expr

        # Calculate metrics
        metrics = recognizer.evaluate_predictions(predictions, val_expressions)

        print(f"\nConfidence threshold: {conf_threshold}")
        print(f"  Avg Levenshtein: {metrics['avg_levenshtein_distance']:.2f}")
        print(f"  Exact match: {metrics['exact_match_accuracy']:.2%}")

        if metrics["avg_levenshtein_distance"] < best_score:
            best_score = metrics["avg_levenshtein_distance"]
            best_threshold = conf_threshold

    print(f"\nBest confidence threshold: {best_threshold}")

    # Generate final predictions with best threshold
    final_predictions = {}
    sample_results = []

    for i, img_name in enumerate(tqdm(val_expressions, desc="Final evaluation")):
        img_path = Path(dataset_path) / "val" / "images" / img_name
        if img_path.exists():
            pred_expr = recognizer.recognize_expression(
                str(img_path), conf_threshold=best_threshold
            )
            final_predictions[img_name] = pred_expr

            if i < 10:  # Save first 10 for display
                sample_results.append(
                    {
                        "image": img_name,
                        "true": val_expressions[img_name],
                        "pred": pred_expr,
                    }
                )
                print(
                    f"{img_name}: True='{val_expressions[img_name]}', Pred='{pred_expr}'"
                )

    # Final metrics
    final_metrics = recognizer.evaluate_predictions(final_predictions, val_expressions)

    print("\nFinal Validation Metrics:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results = {
        "best_threshold": best_threshold,
        "metrics": final_metrics,
        "sample_results": sample_results,
        "predictions": final_predictions,
    }

    with open(Path(output_dir) / "validation_results_final.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_threshold


def generate_test_predictions(
    model_path="results/recognition/character_classifier_improved.pth",
    yolo_model_path="results/yolo_runs/detect/character_detection2/weights/best.pt",
    test_dir="dataset/test/images",
    output_file="expression_predictions.csv",
    conf_threshold=0.5,
):
    """Generate predictions for test set"""

    print("\nGenerating test predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    checkpoint = torch.load(model_path, map_location=device)
    model = CharacterClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    yolo_model = YOLO(yolo_model_path)
    recognizer = ExpressionRecognizer(model, yolo_model)

    # Process test images
    test_images = sorted(Path(test_dir).glob("*.png"))
    predictions = []

    print(f"Processing {len(test_images)} test images...")

    for img_path in tqdm(test_images, desc="Generating predictions"):
        try:
            expression = recognizer.recognize_expression(
                str(img_path), conf_threshold=conf_threshold
            )

            # If no expression detected, use a default
            if not expression:
                expression = "0"

            predictions.append({"image": img_path.name, "expression": expression})

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            predictions.append({"image": img_path.name, "expression": "0"})

    # Save predictions in competition format
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "expression"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nPredictions saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")

    # Show sample predictions
    print("\nSample predictions:")
    for pred in predictions[:10]:
        print(f"  {pred['image']}: {pred['expression']}")


def visualize_expression_recognition(
    model_path="results/recognition/character_classifier_improved.pth",
    yolo_model_path="results/yolo_runs/detect/character_detection2/weights/best.pt",
    dataset_path="dataset",
    num_samples=5,
):
    """Visualize complete expression recognition pipeline"""

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    checkpoint = torch.load(model_path, map_location=device)
    model = CharacterClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    yolo_model = YOLO(yolo_model_path)

    # Get sample images
    val_dir = Path(dataset_path) / "val" / "images"
    sample_images = list(val_dir.glob("*.png"))[:num_samples]

    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx, img_path in enumerate(sample_images):
        # Detect characters
        results = yolo_model(str(img_path), conf=0.5, verbose=False)

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(img_rgb)

        # Process detections
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append(
                        {"bbox": [x1, y1, x2 - x1, y2 - y1], "x_center": (x1 + x2) / 2}
                    )

        # Sort by x-coordinate
        detections = sorted(detections, key=lambda d: d["x_center"])

        # Classify and visualize
        expression = ""
        for det in detections:
            x, y, w, h = det["bbox"]

            # Extract character
            char_img = img_rgb[int(y) : int(y + h), int(x) : int(x + w)]

            # Classify
            with torch.no_grad():
                char_tensor = model.preprocess_image(char_img).unsqueeze(0).to(device)
                output = model(char_tensor)
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

                if conf.item() > 0.5:
                    char = model.idx_to_char[pred.item()]
                    expression += char

                    # Draw bounding box
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    axes[idx].add_patch(rect)

                    # Add label
                    axes[idx].text(
                        x,
                        y - 5,
                        f"{char}",
                        fontsize=12,
                        color="lime",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="black", alpha=0.7
                        ),
                    )

        axes[idx].set_title(f"Detected Expression: {expression}", fontsize=16)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        "results/recognition/expression_recognition_demo.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and generate predictions")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate on validation set"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Generate test predictions"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize recognition"
    )
    parser.add_argument(
        "--model", default="results/recognition/character_classifier_improved.pth"
    )
    parser.add_argument(
        "--yolo",
        default="results/yolo_runs/detect/character_detection2/weights/best.pt",
    )
    parser.add_argument("--output", default="expression_predictions.csv")

    args = parser.parse_args()

    if args.evaluate:
        best_threshold = evaluate_on_validation(
            model_path=args.model, yolo_model_path=args.yolo
        )

        if args.predict:
            generate_test_predictions(
                model_path=args.model,
                yolo_model_path=args.yolo,
                output_file=args.output,
                conf_threshold=best_threshold,
            )

    elif args.predict:
        generate_test_predictions(
            model_path=args.model, yolo_model_path=args.yolo, output_file=args.output
        )

    if args.visualize:
        visualize_expression_recognition(
            model_path=args.model, yolo_model_path=args.yolo
        )
