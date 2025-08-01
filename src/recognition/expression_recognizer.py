# src/recognition/expression_recognizer.py
import numpy as np
from typing import List, Dict, Tuple
import cv2
from pathlib import Path
import json
from character_classifier import CharacterClassifier
import torch


class ExpressionRecognizer:
    """
    Recognize mathematical expressions from detected and classified characters
    """

    def __init__(self, character_classifier: CharacterClassifier, yolo_model=None):
        self.classifier = character_classifier
        self.yolo_model = yolo_model

    def recognize_expression(self, image_path: str, conf_threshold: float = 0.5) -> str:
        """
        Recognize mathematical expression from an image

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detection

        Returns:
            expression: Recognized mathematical expression string
        """
        # Step 1: Detect characters using YOLO
        detections = self._detect_characters(image_path, conf_threshold)

        if not detections:
            return ""

        # Step 2: Sort characters by position (left to right)
        sorted_chars = self._sort_characters_by_position(detections)

        # Step 3: Classify each character
        expression = ""
        for char_info in sorted_chars:
            char_img = char_info["image"]

            # Preprocess and classify
            char_tensor = self.classifier.preprocess_image(char_img).unsqueeze(0)
            char_tensor = char_tensor.to(next(self.classifier.parameters()).device)

            with torch.no_grad():
                output = self.classifier(char_tensor)
                prob = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(prob, 1)

                if confidence.item() > conf_threshold:
                    char = self.classifier.idx_to_char[predicted.item()]
                    expression += char

        return expression

    def _detect_characters(self, image_path: str, conf_threshold: float) -> List[Dict]:
        """Detect characters in image using YOLO"""
        results = self.yolo_model(image_path, conf=conf_threshold)

        image = cv2.imread(image_path)
        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    char_img = image[y1:y2, x1:x2]

                    detections.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "image": cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB),
                            "confidence": box.conf[0].item(),
                        }
                    )

        return detections

    def _sort_characters_by_position(self, detections: List[Dict]) -> List[Dict]:
        # Sort by x-coordinate (left edge of bounding box)
        return sorted(detections, key=lambda d: d["bbox"][0])

    def evaluate_predictions(
        self, predictions: Dict[str, str], ground_truth: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Evaluate predictions using various metrics including Levenshtein distance

        Args:
            predictions: Dictionary of image_name -> predicted expression
            ground_truth: Dictionary of image_name -> true expression

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        from Levenshtein import distance as levenshtein_distance

        total_distance = 0
        total_chars = 0
        correct_expressions = 0

        for img_name in ground_truth:
            if img_name in predictions:
                pred = predictions[img_name]
                true = ground_truth[img_name]

                # Levenshtein distance
                dist = levenshtein_distance(pred, true)
                total_distance += dist
                total_chars += len(true)

                # Exact match
                if pred == true:
                    correct_expressions += 1

        metrics = {
            "avg_levenshtein_distance": total_distance / len(ground_truth),
            "normalized_levenshtein": (
                total_distance / total_chars if total_chars > 0 else 0
            ),
            "exact_match_accuracy": correct_expressions / len(ground_truth),
            "total_expressions": len(ground_truth),
        }

        return metrics
