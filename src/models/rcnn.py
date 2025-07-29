import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Simple CNN backbone for feature extraction"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        return x


class RegionClassifier(nn.Module):
    """Classifier for determining if region contains a character"""

    def __init__(self, feature_size=512, roi_size=7):
        super(RegionClassifier, self).__init__()
        # Calculate flattened size after ROI pooling
        flat_size = feature_size * roi_size * roi_size

        # Classification layers
        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)  # 2 classes: background, character

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc3(x)
        return x


class BBoxRegressor(nn.Module):
    """Regressor for refining bounding box coordinates"""

    def __init__(self, feature_size=512, roi_size=7):
        super(BBoxRegressor, self).__init__()
        # Calculate flattened size after ROI pooling
        flat_size = feature_size * roi_size * roi_size

        # Regression layers
        self.fc1 = nn.Linear(flat_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 4)  # 4 coordinates: [dx, dy, dw, dh]

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc3(x)
        return x


class RCNN(nn.Module):
    """R-CNN model for character localization"""

    def __init__(self, roi_size=7):
        super(RCNN, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.roi_size = roi_size
        self.classifier = RegionClassifier(512, roi_size)
        self.bbox_regressor = BBoxRegressor(512, roi_size)

    def forward(self, image, rois):
        # Extract features from the entire image
        features = self.feature_extractor(image)

        # Apply ROI pooling to extract fixed-size features for each ROI
        roi_features = self.roi_pooling(features, rois, self.roi_size, image)

        # Check if we have any valid ROIs
        if roi_features.size(0) == 0:
            # Return empty tensors with correct shapes if no valid ROIs
            return (
                torch.zeros((0, 2), device=features.device),
                torch.zeros((0, 4), device=features.device),
                torch.tensor([], dtype=torch.long, device=features.device),
            )

        # Classify each ROI
        class_scores = self.classifier(roi_features)

        # Predict bounding box refinements for each ROI
        bbox_deltas = self.bbox_regressor(roi_features)

        # Return all indices since we're not filtering in roi_pooling anymore
        valid_indices = torch.arange(roi_features.size(0), device=features.device)

        return class_scores, bbox_deltas, valid_indices

    def roi_pooling(self, features, rois, output_size, image):
        """
        Simple ROI pooling implementation

        Args:
            features: Feature maps from the backbone [batch_size, channels, height, width]
            rois: Region proposals [batch_size, num_rois, 4] or [num_rois, 4]
            output_size: Size of the output feature map after ROI pooling
            image: Original input image tensor

        Returns:
            Pooled features [num_rois, channels, output_size, output_size]
        """
        # Handle both batched and unbatched ROIs
        if rois.dim() == 2:
            rois = rois.unsqueeze(0)

        batch_size = features.size(0)
        num_channels = features.size(1)
        feature_h = features.size(2)
        feature_w = features.size(3)

        # Get image dimensions
        img_h = image.size(2)
        img_w = image.size(3)

        # Scale factors
        scale_h = feature_h / img_h
        scale_w = feature_w / img_w

        pooled_features = []

        for i in range(batch_size):
            batch_rois = rois[i] if rois.size(0) > 1 else rois[0]

            for roi in batch_rois:
                # Scale ROI coordinates to feature map size
                x1 = int(roi[0] * scale_w)
                y1 = int(roi[1] * scale_h)
                x2 = int(roi[2] * scale_w)
                y2 = int(roi[3] * scale_h)

                # Ensure valid coordinates
                x1 = max(0, min(x1, feature_w - 1))
                y1 = max(0, min(y1, feature_h - 1))
                x2 = max(x1 + 1, min(x2, feature_w))
                y2 = max(y1 + 1, min(y2, feature_h))

                # Extract ROI from feature map
                roi_features = features[i, :, y1:y2, x1:x2]

                # Apply adaptive pooling to get fixed size output
                if roi_features.numel() > 0:
                    roi_features = F.adaptive_max_pool2d(roi_features, output_size)
                else:
                    # If ROI is invalid, create zero features
                    roi_features = torch.zeros(
                        num_channels, output_size, output_size, device=features.device
                    )

                pooled_features.append(roi_features)

        # Stack all ROIs
        if pooled_features:
            return torch.stack(pooled_features)
        else:
            return torch.zeros(
                0, num_channels, output_size, output_size, device=features.device
            )
