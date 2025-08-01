# src/clustering/feature_extraction.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import os
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo import YOLOCharacterDetector
from ultralytics import YOLO


class YOLOFeatureExtractor:
    def __init__(self, yolo_model_path=None, device="cuda"):
        """
        Initialize feature extractor with YOLO model

        Args:
            yolo_model_path: Path to YOLO model weights (.pt file)
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if yolo_model_path:
            if Path(yolo_model_path).exists():
                self.model = YOLO(yolo_model_path)
            else:
                # Try to find in default locations
                default_paths = [
                    "results/yolo_runs/detect/character_detection/weights/best.pt",
                    "results/yolo_runs/detect/character_detection/weights/last.pt",
                    "src/models/yolo/yolov8n.pt",
                    "src/models/yolo/yolov8m.pt",
                ]
                for path in default_paths:
                    if Path(path).exists():
                        self.model = YOLO(path)
                        print(f"Loaded YOLO model from: {path}")
                        break
                else:
                    raise FileNotFoundError(
                        f"YOLO model not found. Tried: {default_paths}"
                    )
        else:
            # Use default pretrained model
            self.model = YOLO("yolov8n.pt")

        # For CNN feature extraction, we'll use a pretrained ResNet
        self.feature_model = None
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize a CNN model for feature extraction"""
        try:
            import torchvision.models as models

            # Use ResNet18 for feature extraction
            self.feature_model = models.resnet18(pretrained=True)
            self.feature_model.eval()
            self.feature_model.to(self.device)
            # Remove the final classification layer
            self.feature_model = torch.nn.Sequential(
                *list(self.feature_model.children())[:-1]
            )
        except Exception as e:
            print(f"Warning: Could not initialize feature extraction model: {e}")

    def extract_detected_characters(
        self, image_path: str, conf_threshold: float = 0.25
    ) -> List[np.ndarray]:
        """
        Extract character regions from an image using YOLO

        Args:
            image_path: Path to input image
            conf_threshold: Minimum confidence score for detections

        Returns:
            List of cropped character images
        """
        # Run YOLO inference
        results = self.model(image_path, conf=conf_threshold, verbose=False)

        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract character regions
        characters = []

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                print(
                    f"  Found {len(boxes)} characters (max conf: {boxes.conf.max().item():.3f})"
                )

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

                    if x2 > x1 + 5 and y2 > y1 + 5:  # Minimum size check
                        char_img = image_rgb[y1:y2, x1:x2]
                        characters.append(char_img)

        return characters

    def extract_raw_pixels(
        self, images: List[np.ndarray], size: Tuple[int, int] = (28, 28)
    ) -> np.ndarray:
        """
        Resize and flatten images to create raw pixel features

        Args:
            images: List of character images
            size: Target size for resizing

        Returns:
            Array of flattened pixel features
        """
        if len(images) == 0:
            return np.array([])

        features = []
        for img in images:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Resize
            resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0

            # Flatten
            features.append(normalized.flatten())

        return np.array(features)

    def extract_hog_features(
        self,
        images: List[np.ndarray],
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
    ) -> np.ndarray:
        """
        Extract HOG (Histogram of Oriented Gradients) features

        Args:
            images: List of character images
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell
            cells_per_block: Number of cells in each block

        Returns:
            Array of HOG features
        """
        if len(images) == 0:
            return np.array([])

        features = []
        target_size = (64, 64)  # HOG works better with larger images

        for img in images:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Resize to consistent size
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

            # Extract HOG features
            fd = hog(
                resized,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,
                feature_vector=True,
            )

            features.append(fd)

        return np.array(features)

    def extract_cnn_features(
        self,
        images: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract CNN features using a pretrained ResNet

        Args:
            images: List of character images
            batch_size: Batch size for processing

        Returns:
            Array of CNN features
        """
        if self.feature_model is None:
            print("Warning: Feature model not initialized, skipping CNN features")
            return np.array([])

        if len(images) == 0:
            return np.array([])

        features = []
        target_size = (224, 224)  # ResNet input size

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_tensors = []

            for img in batch_images:
                # Ensure RGB
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]

                # Resize
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to tensor and normalize (ImageNet normalization)
                img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
                # Normalize with ImageNet stats
                normalize = torch.nn.functional.normalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std

                batch_tensors.append(img_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                # Extract features
                feat = self.feature_model(batch_tensor)
                feat = feat.squeeze()
                features.extend(feat.cpu().numpy())

            # Clear cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return np.array(features)

    def extract_statistical_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract simple statistical features from character images

        Args:
            images: List of character images

        Returns:
            Array of statistical features
        """
        if len(images) == 0:
            return np.array([])

        features = []

        for img in images:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Normalize
            gray = gray.astype(np.float32) / 255.0

            # Extract features
            feat = [
                np.mean(gray),
                np.std(gray),
                np.min(gray),
                np.max(gray),
                np.percentile(gray, 25),
                np.percentile(gray, 75),
                (
                    gray.shape[0] / gray.shape[1] if gray.shape[1] > 0 else 1.0
                ),  # aspect ratio
                (
                    np.sum(gray > 0.5) / gray.size if gray.size > 0 else 0.0
                ),  # fraction of bright pixels
            ]

            features.append(feat)

        return np.array(features)

    def combine_features(
        self, feature_dict: Dict[str, np.ndarray], normalize: bool = True
    ) -> np.ndarray:
        """
        Combine multiple feature types

        Args:
            feature_dict: Dictionary of feature arrays
            normalize: Whether to normalize each feature type

        Returns:
            Combined feature array
        """
        combined = []

        for name, features in feature_dict.items():
            if features is None or len(features) == 0:
                continue

            if normalize:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)

            combined.append(features)

        return np.hstack(combined) if combined else np.array([])

    def save_features(
        self, features: Dict[str, np.ndarray], metadata: Dict, output_path: str
    ):
        """
        Save extracted features and metadata

        Args:
            features: Dictionary of feature arrays
            metadata: Metadata about the features
            output_path: Path to save features
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save features
        np.savez(output_path, **features)

        # Save metadata
        metadata_path = output_path.replace(".npz", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_features(self, features_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Load saved features and metadata

        Args:
            features_path: Path to saved features

        Returns:
            Tuple of (features dict, metadata dict)
        """
        # Load features
        features = dict(np.load(features_path))

        # Load metadata
        metadata_path = features_path.replace(".npz", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return features, metadata


def extract_features_from_dataset_yolo(
    model_path: Optional[str] = None,
    dataset_path: str = "dataset",
    output_dir: str = "results/clustering",
    max_images: Optional[int] = None,
    conf_threshold: float = 0.25,
):
    """
    Extract features from all images in a dataset using YOLO

    Args:
        model_path: Path to YOLO model (None to use default)
        dataset_path: Path to dataset directory
        output_dir: Directory to save extracted features
        max_images: Maximum number of images to process (None for all)
        conf_threshold: Minimum confidence score for detections
    """
    # Initialize feature extractor
    extractor = YOLOFeatureExtractor(model_path)

    # Process images
    all_characters = []
    image_paths = []
    character_metadata = []

    # Get image files
    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_path, split, "images")
        if os.path.exists(split_dir):
            for img_file in sorted(os.listdir(split_dir)):
                if img_file.endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(split_dir, img_file))

    print(f"Found {len(image_paths)} images in dataset")

    # Limit number of images if specified
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Processing {len(image_paths)} images...")
    print(f"Using confidence threshold: {conf_threshold}")

    # Extract characters from each image
    for img_path in tqdm(image_paths, desc="Extracting characters"):
        try:
            characters = extractor.extract_detected_characters(
                img_path, conf_threshold=conf_threshold
            )
            if len(characters) > 0:
                all_characters.extend(characters)
                # Store metadata for each character
                for char in characters:
                    character_metadata.append(
                        {
                            "source_image": os.path.basename(img_path),
                            "shape": char.shape,
                        }
                    )
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

    print(f"\nTotal characters extracted: {len(all_characters)}")

    if len(all_characters) == 0:
        print("No characters detected! Check your model and detection threshold.")
        return None, None

    # Extract different feature types
    print("\nExtracting features...")
    features = {}

    print("  - Extracting raw pixel features...")
    features["raw_pixels"] = extractor.extract_raw_pixels(all_characters)

    print("  - Extracting HOG features...")
    features["hog"] = extractor.extract_hog_features(all_characters)

    print("  - Extracting CNN features...")
    features["cnn"] = extractor.extract_cnn_features(all_characters)

    print("  - Extracting statistical features...")
    features["statistical"] = extractor.extract_statistical_features(all_characters)

    # Save metadata
    metadata = {
        "num_characters": len(all_characters),
        "num_images": len(image_paths),
        "feature_types": list(features.keys()),
        "feature_shapes": {k: v.shape for k, v in features.items() if v.size > 0},
        "model_path": model_path or "default",
        "dataset_path": dataset_path,
        "conf_threshold": conf_threshold,
        "character_metadata": character_metadata[:100],  # Save first 100 for reference
    }

    # Save features
    output_path = os.path.join(output_dir, "character_features_yolo.npz")
    extractor.save_features(features, metadata, output_path)

    print(f"\nFeatures saved to {output_path}")
    print("\nFeature shapes:")
    for name, feat in features.items():
        if feat.size > 0:
            print(f"  {name}: {feat.shape}")

    # Save some sample character images for visualization
    sample_dir = os.path.join(output_dir, "sample_characters_yolo")
    os.makedirs(sample_dir, exist_ok=True)

    num_samples = min(50, len(all_characters))
    if num_samples > 0:
        indices = np.random.choice(len(all_characters), num_samples, replace=False)

        print(f"\nSaving {num_samples} sample character images...")
        for i, idx in enumerate(indices):
            char_img = all_characters[idx]
            # Convert RGB to BGR for OpenCV
            char_bgr = cv2.cvtColor(char_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(sample_dir, f"char_{i:03d}.png"), char_bgr)

    return features, metadata


if __name__ == "__main__":
    # Example usage with YOLO

    # Option 1: Use trained YOLO model
    model_path = "results/yolo_runs/detect/character_detection/weights/best.pt"

    # Option 2: Use default pretrained model
    # model_path = None

    dataset_path = "dataset"
    output_dir = "results/clustering"

    # Extract features
    extract_features_from_dataset_yolo(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_images=100,  # Process first 100 images
        conf_threshold=0.25,  # YOLO confidence threshold
    )
