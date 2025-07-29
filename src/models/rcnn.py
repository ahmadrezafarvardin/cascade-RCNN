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
        self.fc3 = nn.Linear(256, 4)  # 4 coordinates: [x1, y1, x2, y2]

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

        # Classify each ROI
        class_scores = self.classifier(roi_features)

        # Predict bounding box refinements for each ROI
        bbox_deltas = self.bbox_regressor(roi_features)

        return class_scores, bbox_deltas

    def roi_pooling(self, features, rois, output_size, image):
        """
        Simple ROI pooling implementation

        Args:
            features: Feature maps from the backbone [batch_size, channels, height, width]
            rois: Region proposals [batch_size, num_rois, 4] with format [x1, y1, x2, y2]
            output_size: Size of the output feature map after ROI pooling

        Returns:
            Pooled features for each ROI [batch_size * num_rois, channels, output_size, output_size]
        """
        batch_size = features.size(0)
        num_channels = features.size(1)

        # Initialize output tensor
        pooled_features = []

        for i in range(batch_size):
            batch_rois = rois[i]
            for roi in batch_rois:
                x1, y1, x2, y2 = roi

                # Convert to integers and ensure within feature map bounds
                x1 = max(0, int(x1 * features.size(3) / image.size(3)))
                y1 = max(0, int(y1 * features.size(2) / image.size(2)))
                x2 = min(
                    features.size(3) - 1, int(x2 * features.size(3) / image.size(3))
                )
                y2 = min(
                    features.size(2) - 1, int(y2 * features.size(2) / image.size(2))
                )

                # Skip invalid ROIs
                if x1 >= x2 or y1 >= y2:
                    continue

                # Extract ROI from feature map
                roi_features = features[i, :, y1:y2, x1:x2]

                # Apply adaptive pooling to get fixed size output
                roi_features = F.adaptive_max_pool2d(roi_features, output_size)

                pooled_features.append(roi_features)

        # Stack all ROIs
        if not pooled_features:
            return torch.zeros(
                0, num_channels, output_size, output_size, device=features.device
            )

        return torch.stack(pooled_features)
