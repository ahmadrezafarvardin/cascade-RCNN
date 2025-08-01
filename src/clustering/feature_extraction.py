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


class FeatureExtractor:
    def __init__(self, cascade_rcnn_model=None, device="cuda"):
        """
        Initialize feature extractor

        Args:
            cascade_rcnn_model: Trained Cascade R-CNN model
            device: Device to run model on
        """
        self.model = cascade_rcnn_model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def extract_detected_characters(
        self, image_path: str, score_threshold: float = 0.1
    ) -> List[np.ndarray]:
        """
        Extract character regions from an image using the trained model

        Args:
            image_path: Path to input image
            score_threshold: Minimum confidence score for detections

        Returns:
            List of cropped character images
        """
        if self.model is None:
            raise ValueError("Model not provided for character detection")

        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Get predictions
        try:
            with torch.no_grad():
                predictions = self.model(img_tensor)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Clear cache and try CPU
                torch.cuda.empty_cache()
                gc.collect()
                print(
                    f"  GPU OOM, falling back to CPU for {os.path.basename(image_path)}"
                )
                img_tensor = img_tensor.cpu()
                self.model.cpu()
                with torch.no_grad():
                    predictions = self.model(img_tensor)
                # Move model back to GPU for next image
                self.model.to(self.device)
            else:
                raise e

        # Extract character regions
        characters = []
        if len(predictions) > 0:
            pred = predictions[0]
            # Get number of detections above threshold
            valid_detections = (pred["scores"] > score_threshold).sum().item()
            if valid_detections > 0:
                print(
                    f"  Found {valid_detections} characters (max score: {pred['scores'].max().item():.3f})"
                )

            for idx in range(len(pred["boxes"])):
                if pred["scores"][idx] > score_threshold:
                    box = pred["boxes"][idx].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box

                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

                    if x2 > x1 + 5 and y2 > y1 + 5:  # Minimum size check
                        char_img = image_rgb[y1:y2, x1:x2]
                        characters.append(char_img)

        # Clear GPU cache after each image
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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
        layer_name: str = "conv3",
        pool_output: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract CNN features from your trained model's backbone

        Args:
            images: List of character images
            layer_name: Which conv layer to extract features from ('conv1', 'conv2', 'conv3', 'conv4')
            pool_output: Whether to apply global average pooling to the feature maps
            batch_size: Batch size for processing

        Returns:
            Array of CNN features
        """
        if self.model is None:
            raise ValueError("Model not provided for CNN feature extraction")

        if len(images) == 0:
            return np.array([])

        self.model.eval()
        features = []

        # Target size for CNN input
        target_size = (64, 64)

        # Process in batches to avoid memory issues
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

                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
                batch_tensors.append(img_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                # Extract features from backbone
                x = F.relu(self.model.backbone.conv1(batch_tensor))
                x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

                if layer_name == "conv1":
                    feat = x
                else:
                    x = F.relu(self.model.backbone.conv2(x))
                    if layer_name == "conv2":
                        feat = x
                    else:
                        x = F.relu(self.model.backbone.conv3(x))
                        if layer_name == "conv3":
                            feat = x
                        else:
                            x = F.relu(self.model.backbone.conv4(x))
                            feat = x

                # Apply global average pooling if requested
                if pool_output:
                    feat = F.adaptive_avg_pool2d(feat, (1, 1))
                    feat = feat.squeeze()
                else:
                    feat = feat.flatten(1)  # Flatten all dimensions except batch

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


# Helper function to extract features from a dataset
def extract_features_from_dataset(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    max_images: Optional[int] = None,
    score_threshold: float = 0.1,
):  # Lowered threshold
    """
    Extract features from all images in a dataset

    Args:
        model_path: Path to trained Cascade R-CNN model
        dataset_path: Path to dataset directory
        output_dir: Directory to save extracted features
        max_images: Maximum number of images to process (None for all)
        score_threshold: Minimum confidence score for detections
    """
    # Load model
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.cascade_rcnn import CascadeRCNN
    from models.backbone import SimpleBackbone
    from models.anchor_generator import AnchorGenerator
    from models.heads import RPNHead

    # Initialize model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    # Create anchor generator with same config as training
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150),
        aspect_ratios=(0.15, 0.2, 0.3, 0.5),
    )
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize feature extractor
    extractor = FeatureExtractor(model)

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
    print(f"Using score threshold: {score_threshold}")

    # Extract characters from each image
    for img_path in tqdm(image_paths, desc="Extracting characters"):
        try:
            characters = extractor.extract_detected_characters(
                img_path, score_threshold=score_threshold
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
    features["cnn_conv3"] = extractor.extract_cnn_features(
        all_characters, layer_name="conv3"
    )

    print("  - Extracting statistical features...")
    features["statistical"] = extractor.extract_statistical_features(all_characters)

    # Save metadata
    metadata = {
        "num_characters": len(all_characters),
        "num_images": len(image_paths),
        "feature_types": list(features.keys()),
        "feature_shapes": {k: v.shape for k, v in features.items()},
        "model_path": model_path,
        "dataset_path": dataset_path,
        "score_threshold": score_threshold,
        "character_metadata": character_metadata[:100],  # Save first 100 for reference
    }

    # Save features
    output_path = os.path.join(output_dir, "character_features.npz")
    extractor.save_features(features, metadata, output_path)

    print(f"\nFeatures saved to {output_path}")
    print("\nFeature shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    # Save some sample character images for visualization
    sample_dir = os.path.join(output_dir, "sample_characters")
    os.makedirs(sample_dir, exist_ok=True)

    num_samples = min(50, len(all_characters))
    indices = np.random.choice(len(all_characters), num_samples, replace=False)

    print(f"\nSaving {num_samples} sample character images...")
    for i, idx in enumerate(indices):
        char_img = all_characters[idx]
        # Convert RGB to BGR for OpenCV
        char_bgr = cv2.cvtColor(char_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sample_dir, f"char_{i:03d}.png"), char_bgr)

    return features, metadata


if __name__ == "__main__":
    # Example usage
    model_path = "../../results/cascade_rcnn_best.pth"
    dataset_path = "../../dataset"
    output_dir = "../../results/clustering"

    # Process with lower threshold to get more detections
    extract_features_from_dataset(
        model_path,
        dataset_path,
        output_dir,
        max_images=100,  # Process first 100 images
        score_threshold=0.0028,  # Very low threshold based on test results
    )
# # src/clustering/feature_extraction.py
# import numpy as np
# import cv2
# import torch
# import torch.nn.functional as F
# from skimage.feature import hog
# from sklearn.preprocessing import StandardScaler
# import os
# import json
# from typing import List, Dict, Tuple, Optional
# from tqdm import tqdm


# class FeatureExtractor:
#     def __init__(self, cascade_rcnn_model=None, device="cuda"):
#         """
#         Initialize feature extractor

#         Args:
#             cascade_rcnn_model: Trained Cascade R-CNN model
#             device: Device to run model on
#         """
#         self.model = cascade_rcnn_model
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")

#         if self.model is not None:
#             self.model.to(self.device)
#             self.model.eval()

#     def extract_detected_characters(
#         self, image_path: str, score_threshold: float = 0.5
#     ) -> List[np.ndarray]:
#         """
#         Extract character regions from an image using the trained model

#         Args:
#             image_path: Path to input image
#             score_threshold: Minimum confidence score for detections

#         Returns:
#             List of cropped character images
#         """
#         if self.model is None:
#             raise ValueError("Model not provided for character detection")

#         # Load and preprocess image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Warning: Could not load image {image_path}")
#             return []

#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Convert to tensor
#         img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)

#         # Get predictions
#         with torch.no_grad():
#             predictions = self.model(img_tensor)

#         # Extract character regions
#         characters = []
#         if len(predictions) > 0:
#             pred = predictions[0]
#             for idx in range(len(pred["boxes"])):
#                 if pred["scores"][idx] > score_threshold:
#                     box = pred["boxes"][idx].cpu().numpy().astype(int)
#                     x1, y1, x2, y2 = box

#                     # Ensure valid coordinates
#                     x1, y1 = max(0, x1), max(0, y1)
#                     x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

#                     if x2 > x1 and y2 > y1:
#                         char_img = image_rgb[y1:y2, x1:x2]
#                         characters.append(char_img)

#         return characters

#     def extract_raw_pixels(
#         self, images: List[np.ndarray], size: Tuple[int, int] = (28, 28)
#     ) -> np.ndarray:
#         """
#         Resize and flatten images to create raw pixel features

#         Args:
#             images: List of character images
#             size: Target size for resizing

#         Returns:
#             Array of flattened pixel features
#         """
#         if len(images) == 0:
#             return np.array([])

#         features = []
#         for img in images:
#             # Convert to grayscale if needed
#             if len(img.shape) == 3:
#                 gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             else:
#                 gray = img

#             # Resize
#             resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

#             # Normalize to [0, 1]
#             normalized = resized.astype(np.float32) / 255.0

#             # Flatten
#             features.append(normalized.flatten())

#         return np.array(features)

#     def extract_hog_features(
#         self,
#         images: List[np.ndarray],
#         orientations: int = 9,
#         pixels_per_cell: Tuple[int, int] = (8, 8),
#         cells_per_block: Tuple[int, int] = (2, 2),
#     ) -> np.ndarray:
#         """
#         Extract HOG (Histogram of Oriented Gradients) features

#         Args:
#             images: List of character images
#             orientations: Number of orientation bins
#             pixels_per_cell: Size of a cell
#             cells_per_block: Number of cells in each block

#         Returns:
#             Array of HOG features
#         """
#         if len(images) == 0:
#             return np.array([])

#         features = []
#         target_size = (64, 64)  # HOG works better with larger images

#         for img in images:
#             # Convert to grayscale if needed
#             if len(img.shape) == 3:
#                 gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             else:
#                 gray = img

#             # Resize to consistent size
#             resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

#             # Extract HOG features
#             fd = hog(
#                 resized,
#                 orientations=orientations,
#                 pixels_per_cell=pixels_per_cell,
#                 cells_per_block=cells_per_block,
#                 visualize=False,
#                 feature_vector=True,
#             )

#             features.append(fd)

#         return np.array(features)

#     def extract_cnn_features(
#         self,
#         images: List[np.ndarray],
#         layer_name: str = "conv3",
#         pool_output: bool = True,
#     ) -> np.ndarray:
#         """
#         Extract CNN features from your trained model's backbone

#         Args:
#             images: List of character images
#             layer_name: Which conv layer to extract features from ('conv1', 'conv2', 'conv3', 'conv4')
#             pool_output: Whether to apply global average pooling to the feature maps

#         Returns:
#             Array of CNN features
#         """
#         if self.model is None:
#             raise ValueError("Model not provided for CNN feature extraction")

#         if len(images) == 0:
#             return np.array([])

#         self.model.eval()
#         features = []

#         # Target size for CNN input
#         target_size = (64, 64)

#         with torch.no_grad():
#             for img in images:
#                 # Ensure RGB
#                 if len(img.shape) == 2:
#                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#                 elif img.shape[2] == 4:
#                     img = img[:, :, :3]

#                 # Resize
#                 resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

#                 # Convert to tensor and normalize
#                 img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
#                 img_tensor = img_tensor.unsqueeze(0).to(self.device)

#                 # Extract features from backbone
#                 x = F.relu(self.model.backbone.conv1(img_tensor))
#                 x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

#                 if layer_name == "conv1":
#                     feat = x
#                 else:
#                     x = F.relu(self.model.backbone.conv2(x))
#                     if layer_name == "conv2":
#                         feat = x
#                     else:
#                         x = F.relu(self.model.backbone.conv3(x))
#                         if layer_name == "conv3":
#                             feat = x
#                         else:
#                             x = F.relu(self.model.backbone.conv4(x))
#                             feat = x

#                 # Apply global average pooling if requested
#                 if pool_output:
#                     feat = F.adaptive_avg_pool2d(feat, (1, 1))
#                     feat = feat.squeeze()
#                 else:
#                     feat = feat.flatten()

#                 features.append(feat.cpu().numpy())

#         return np.array(features)

#     def extract_statistical_features(self, images: List[np.ndarray]) -> np.ndarray:
#         """
#         Extract simple statistical features from character images

#         Args:
#             images: List of character images

#         Returns:
#             Array of statistical features
#         """
#         if len(images) == 0:
#             return np.array([])

#         features = []

#         for img in images:
#             # Convert to grayscale if needed
#             if len(img.shape) == 3:
#                 gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             else:
#                 gray = img

#             # Normalize
#             gray = gray.astype(np.float32) / 255.0

#             # Extract features
#             feat = [
#                 np.mean(gray),
#                 np.std(gray),
#                 np.min(gray),
#                 np.max(gray),
#                 np.percentile(gray, 25),
#                 np.percentile(gray, 75),
#                 (
#                     gray.shape[0] / gray.shape[1] if gray.shape[1] > 0 else 1.0
#                 ),  # aspect ratio
#                 (
#                     np.sum(gray > 0.5) / gray.size if gray.size > 0 else 0.0
#                 ),  # fraction of bright pixels
#             ]

#             features.append(feat)

#         return np.array(features)

#     def combine_features(
#         self, feature_dict: Dict[str, np.ndarray], normalize: bool = True
#     ) -> np.ndarray:
#         """
#         Combine multiple feature types

#         Args:
#             feature_dict: Dictionary of feature arrays
#             normalize: Whether to normalize each feature type

#         Returns:
#             Combined feature array
#         """
#         combined = []

#         for name, features in feature_dict.items():
#             if features is None or len(features) == 0:
#                 continue

#             if normalize:
#                 scaler = StandardScaler()
#                 features = scaler.fit_transform(features)

#             combined.append(features)

#         return np.hstack(combined) if combined else np.array([])

#     def save_features(
#         self, features: Dict[str, np.ndarray], metadata: Dict, output_path: str
#     ):
#         """
#         Save extracted features and metadata

#         Args:
#             features: Dictionary of feature arrays
#             metadata: Metadata about the features
#             output_path: Path to save features
#         """
#         # Create output directory
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Save features
#         np.savez(output_path, **features)

#         # Save metadata
#         metadata_path = output_path.replace(".npz", "_metadata.json")
#         with open(metadata_path, "w") as f:
#             json.dump(metadata, f, indent=2)

#     def load_features(self, features_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
#         """
#         Load saved features and metadata

#         Args:
#             features_path: Path to saved features

#         Returns:
#             Tuple of (features dict, metadata dict)
#         """
#         # Load features
#         features = dict(np.load(features_path))

#         # Load metadata
#         metadata_path = features_path.replace(".npz", "_metadata.json")
#         if os.path.exists(metadata_path):
#             with open(metadata_path, "r") as f:
#                 metadata = json.load(f)
#         else:
#             metadata = {}

#         return features, metadata


# # Helper function to extract features from a dataset
# def extract_features_from_dataset(
#     model_path: str,
#     dataset_path: str,
#     output_dir: str,
#     max_images: Optional[int] = None,
#     score_threshold: float = 0.3,
# ):
#     """
#     Extract features from all images in a dataset

#     Args:
#         model_path: Path to trained Cascade R-CNN model
#         dataset_path: Path to dataset directory
#         output_dir: Directory to save extracted features
#         max_images: Maximum number of images to process (None for all)
#         score_threshold: Minimum confidence score for detections
#     """
#     # Load model
#     import sys

#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#     from models.cascade_rcnn import CascadeRCNN
#     from models.backbone import SimpleBackbone
#     from models.anchor_generator import AnchorGenerator
#     from models.heads import RPNHead

#     # Initialize model
#     backbone = SimpleBackbone()
#     model = CascadeRCNN(backbone)

#     # Create anchor generator with same config as training
#     anchor_generator = AnchorGenerator(
#         sizes=(60, 90, 120, 150),
#         aspect_ratios=(0.15, 0.2, 0.3, 0.5),
#     )
#     model.anchor_generator = anchor_generator
#     model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location="cpu")
#     model.load_state_dict(checkpoint["model_state_dict"])

#     # Initialize feature extractor
#     extractor = FeatureExtractor(model)

#     # Process images
#     all_characters = []
#     image_paths = []
#     character_metadata = []

#     # Get image files - FIXED: Look in the correct subdirectory
#     for split in ["train", "val"]:
#         split_dir = os.path.join(
#             dataset_path, split, "images"
#         )  # Added 'images' subdirectory
#         if os.path.exists(split_dir):
#             for img_file in sorted(os.listdir(split_dir)):
#                 if img_file.endswith((".png", ".jpg", ".jpeg")):
#                     image_paths.append(os.path.join(split_dir, img_file))

#     print(f"Found {len(image_paths)} images in dataset")

#     # Limit number of images if specified
#     if max_images:
#         image_paths = image_paths[:max_images]

#     print(f"Processing {len(image_paths)} images...")

#     # Extract characters from each image
#     for img_path in tqdm(image_paths, desc="Extracting characters"):
#         try:
#             characters = extractor.extract_detected_characters(
#                 img_path, score_threshold=score_threshold
#             )
#             if len(characters) > 0:
#                 all_characters.extend(characters)
#                 # Store metadata for each character
#                 for char in characters:
#                     character_metadata.append(
#                         {
#                             "source_image": os.path.basename(img_path),
#                             "shape": char.shape,
#                         }
#                     )
#                 print(
#                     f"  Extracted {len(characters)} characters from {os.path.basename(img_path)}"
#                 )
#         except Exception as e:
#             print(f"  Error processing {img_path}: {e}")

#     print(f"\nTotal characters extracted: {len(all_characters)}")

#     if len(all_characters) == 0:
#         print("No characters detected! Check your model and detection threshold.")
#         return None, None

#     # Extract different feature types
#     print("\nExtracting features...")
#     features = {}

#     print("  - Extracting raw pixel features...")
#     features["raw_pixels"] = extractor.extract_raw_pixels(all_characters)

#     print("  - Extracting HOG features...")
#     features["hog"] = extractor.extract_hog_features(all_characters)

#     print("  - Extracting CNN features...")
#     features["cnn_conv3"] = extractor.extract_cnn_features(
#         all_characters, layer_name="conv3"
#     )

#     print("  - Extracting statistical features...")
#     features["statistical"] = extractor.extract_statistical_features(all_characters)

#     # Save metadata
#     metadata = {
#         "num_characters": len(all_characters),
#         "num_images": len(image_paths),
#         "feature_types": list(features.keys()),
#         "feature_shapes": {k: v.shape for k, v in features.items()},
#         "model_path": model_path,
#         "dataset_path": dataset_path,
#         "score_threshold": score_threshold,
#         "character_metadata": character_metadata[:100],  # Save first 100 for reference
#     }

#     # Save features
#     output_path = os.path.join(output_dir, "character_features.npz")
#     extractor.save_features(features, metadata, output_path)

#     print(f"\nFeatures saved to {output_path}")
#     print("\nFeature shapes:")
#     for name, feat in features.items():
#         print(f"  {name}: {feat.shape}")

#     # Save some sample character images for visualization
#     sample_dir = os.path.join(output_dir, "sample_characters")
#     os.makedirs(sample_dir, exist_ok=True)

#     num_samples = min(50, len(all_characters))
#     indices = np.random.choice(len(all_characters), num_samples, replace=False)

#     print(f"\nSaving {num_samples} sample character images...")
#     for i, idx in enumerate(indices):
#         char_img = all_characters[idx]
#         # Convert RGB to BGR for OpenCV
#         char_bgr = cv2.cvtColor(char_img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(os.path.join(sample_dir, f"char_{i:03d}.png"), char_bgr)

#     return features, metadata


# if __name__ == "__main__":
#     # Example usage
#     model_path = "../../results/cascade_rcnn_best.pth"
#     dataset_path = "../../dataset"
#     output_dir = "../../results/clustering"

#     # Process more images and use lower threshold to get more detections
#     extract_features_from_dataset(
#         model_path,
#         dataset_path,
#         output_dir,
#         max_images=50,  # Process first 50 images
#         score_threshold=0.2,  # Lower threshold to get more detections
#     )
