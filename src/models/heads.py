# heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, box_iou


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors

        # 3x3 convolution for feature processing
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        # 1x1 convolutions for classification and regression
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        # Initialize weights
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Dict[str, Tensor] - feature maps from backbone
        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                - List of classification logits for each feature level
                - List of bbox regression predictions for each feature level
        """
        logits = []
        bbox_reg = []

        # Process each feature map
        for feature in x.values():
            t = F.relu(self.conv(feature))

            # Get predictions
            logits_per_level = self.cls_logits(t)
            bbox_reg_per_level = self.bbox_pred(t)

            # Reshape: [N, A, H, W] -> [N, H, W, A] -> [N, H*W*A]
            N, _, H, W = logits_per_level.shape
            A = self.num_anchors

            # Reshape classification logits
            logits_per_level = logits_per_level.permute(0, 2, 3, 1)  # [N, H, W, A]
            logits_per_level = logits_per_level.reshape(N, -1)  # [N, H*W*A]

            # Reshape bbox predictions: [N, A*4, H, W] -> [N, H, W, A*4] -> [N, H*W*A, 4]
            bbox_reg_per_level = bbox_reg_per_level.permute(
                0, 2, 3, 1
            )  # [N, H, W, A*4]
            bbox_reg_per_level = bbox_reg_per_level.reshape(
                N, H, W, A, 4
            )  # [N, H, W, A, 4]
            bbox_reg_per_level = bbox_reg_per_level.reshape(N, -1, 4)  # [N, H*W*A, 4]

            logits.append(logits_per_level)
            bbox_reg.append(bbox_reg_per_level)

        return logits, bbox_reg


class CascadeROIHeads(nn.Module):
    def __init__(
        self, in_channels, num_stages=3, num_classes=2, stage_loss_weights=None
    ):
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.stage_loss_weights = stage_loss_weights or [1.0] * num_stages

        # IoU thresholds for each stage
        self.stage_iou_thresholds = [0.3, 0.4, 0.5][:num_stages]

        # ROI Pooling
        self.roi_pool = RoIAlign(output_size=7, spatial_scale=1.0, sampling_ratio=2)

        # Heads for each stage
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channels * 7 * 7, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                )
                for _ in range(num_stages)
            ]
        )

        # Predictors for each stage
        self.predictors = nn.ModuleList(
            [nn.Linear(1024, num_classes) for _ in range(num_stages)]
        )

        # Bounding box regression for each stage
        self.bbox_pred = nn.ModuleList(
            [nn.Linear(1024, 4) for _ in range(num_stages)]  # 4 coordinates
        )

    def forward(self, features, proposals, image_shapes, targets=None, stage=0):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Convert proposals to proper format for ROI pooling
        if isinstance(proposals, list):
            # Convert list of proposals to single tensor with batch indices
            roi_proposals = []
            for i, props in enumerate(proposals):
                if len(props) == 0:
                    # Handle empty proposals
                    props = torch.zeros((1, 4), device=features["0"].device)
                batch_idx = torch.full(
                    (props.shape[0], 1), i, dtype=props.dtype, device=props.device
                )
                roi_proposals.append(torch.cat([batch_idx, props], dim=1))
            proposals_concat = torch.cat(roi_proposals, dim=0)
        else:
            proposals_concat = proposals

        # Extract the main feature map
        feature_map = features["0"]

        # Normalize proposals to [0, 1] range
        proposals_normalized = self.normalize_proposals(
            proposals_concat, feature_map.shape[-2:]
        )

        # ROI Pooling
        box_features = self.roi_pool(feature_map, proposals_normalized)
        box_features = box_features.view(box_features.size(0), -1)

        # Head forward
        head_output = self.heads[stage](box_features)

        # Classification scores
        class_logits = self.predictors[stage](head_output)

        # Bounding box regression
        pred_boxes = self.bbox_pred[stage](head_output)

        # For training, calculate losses
        if self.training:
            losses = self.compute_loss(
                class_logits, pred_boxes, proposals, targets, stage
            )
            return None, losses
        else:
            # Apply predictions and return formatted output
            predictions = self.postprocess_predictions(
                class_logits, pred_boxes, proposals, image_shapes
            )
            return predictions, {}

    def normalize_proposals(self, proposals, feature_shape):
        """Normalize proposals to match feature map scale"""
        # Assuming proposals are in format [batch_idx, x1, y1, x2, y2]
        normalized = proposals.clone()
        # Keep batch indices unchanged
        # Scale coordinates to feature map size
        normalized[:, 1] = proposals[:, 1] / 800 * feature_shape[1]  # x1
        normalized[:, 2] = proposals[:, 2] / 800 * feature_shape[0]  # y1
        normalized[:, 3] = proposals[:, 3] / 800 * feature_shape[1]  # x2
        normalized[:, 4] = proposals[:, 4] / 800 * feature_shape[0]  # y2
        return normalized

    # def compute_loss(self, class_logits, pred_boxes, proposals, targets, stage):
    #     """Compute classification and regression losses for ROI heads"""
    #     # Get IoU threshold for this stage
    #     iou_threshold = self.stage_iou_thresholds[stage]

    #     # Match proposals to ground truth
    #     matched_idxs, labels, regression_targets = self.match_proposals_to_targets(
    #         proposals, targets, iou_threshold
    #     )

    #     # Classification loss
    #     classification_loss = F.cross_entropy(class_logits, labels)

    #     # Regression loss (only for positive samples)
    #     positive_mask = labels > 0
    #     if positive_mask.sum() > 0:
    #         regression_loss = F.smooth_l1_loss(
    #             pred_boxes[positive_mask], regression_targets[positive_mask]
    #         )
    #     else:
    #         regression_loss = torch.tensor(0.0, device=class_logits.device)

    #     return {
    #         "classification_loss": classification_loss * self.stage_loss_weights[stage],
    #         "bbox_regression_loss": regression_loss * self.stage_loss_weights[stage],
    #     }
    def compute_loss(self, class_logits, pred_boxes, proposals, targets, stage):
        """Compute classification and regression losses for ROI heads"""
        # Get IoU threshold for this stage
        iou_threshold = self.stage_iou_thresholds[stage]

        # Match proposals to ground truth
        matched_idxs, labels, regression_targets = self.match_proposals_to_targets(
            proposals, targets, iou_threshold
        )

        # # Debug: print statistics
        # num_positive = (labels > 0).sum().item()
        # num_negative = (labels == 0).sum().item()
        # total_samples = len(labels)

        # if stage == 0:  # Only print for first stage to avoid clutter
        #     print(
        #         f"Stage {stage} - Positive: {num_positive}, Negative: {num_negative}, Total: {total_samples}"
        #     )

        # Classification loss
        classification_loss = F.cross_entropy(class_logits, labels)

        # Regression loss (only for positive samples)
        positive_mask = labels > 0
        if positive_mask.sum() > 0:
            regression_loss = F.smooth_l1_loss(
                pred_boxes[positive_mask], regression_targets[positive_mask]
            )
        else:
            regression_loss = torch.tensor(0.0, device=class_logits.device)

        return {
            "classification_loss": classification_loss * self.stage_loss_weights[stage],
            "bbox_regression_loss": regression_loss * self.stage_loss_weights[stage],
        }

    def match_proposals_to_targets(self, proposals, targets, iou_threshold):
        """Match proposals to ground truth boxes"""
        device = (
            proposals[0].device if isinstance(proposals, list) else proposals.device
        )

        all_matched_idxs = []
        all_labels = []
        all_regression_targets = []

        # Process each image
        num_images = len(targets)
        for i in range(num_images):
            if isinstance(proposals, list):
                proposals_i = proposals[i]
            else:
                # Extract proposals for this image from concatenated tensor
                mask = proposals[:, 0] == i
                proposals_i = proposals[mask, 1:]

            gt_boxes = targets[i]["boxes"]

            if len(proposals_i) == 0:
                # No proposals for this image
                matched_idxs = torch.zeros((0,), dtype=torch.long, device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)
                regression_targets = torch.zeros(
                    (0, 4), dtype=torch.float32, device=device
                )
            elif len(gt_boxes) == 0:
                # No ground truth for this image
                matched_idxs = torch.full(
                    (len(proposals_i),), -1, dtype=torch.long, device=device
                )
                labels = torch.zeros(
                    (len(proposals_i),), dtype=torch.long, device=device
                )
                regression_targets = torch.zeros(
                    (len(proposals_i), 4), dtype=torch.float32, device=device
                )
            else:
                # Compute IoU between proposals and ground truth
                ious = box_iou(proposals_i, gt_boxes)

                # Match each proposal to the ground truth with highest IoU
                matched_vals, matched_idxs = ious.max(dim=1)

                # Label proposals
                labels = torch.zeros(len(proposals_i), dtype=torch.long, device=device)
                labels[matched_vals >= iou_threshold] = 1  # Positive samples
                labels[matched_vals < 0.3] = 0  # Negative samples
                # Proposals with IoU in [0.3, iou_threshold) are ignored (label remains 0)

                # Compute regression targets for positive samples
                regression_targets = self.encode_boxes(
                    proposals_i, gt_boxes[matched_idxs]
                )

            all_matched_idxs.append(matched_idxs)
            all_labels.append(labels)
            all_regression_targets.append(regression_targets)

        return (
            torch.cat(all_matched_idxs),
            torch.cat(all_labels),
            torch.cat(all_regression_targets),
        )

    def encode_boxes(self, proposals, gt_boxes):
        """Encode ground truth boxes relative to proposals"""
        # Compute widths and heights
        proposals_widths = proposals[:, 2] - proposals[:, 0]
        proposals_heights = proposals[:, 3] - proposals[:, 1]
        proposals_ctr_x = proposals[:, 0] + 0.5 * proposals_widths
        proposals_ctr_y = proposals[:, 1] + 0.5 * proposals_heights

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

        # Avoid division by zero
        eps = 1e-6
        proposals_widths = torch.clamp(proposals_widths, min=eps)
        proposals_heights = torch.clamp(proposals_heights, min=eps)

        # Encode
        targets = torch.zeros_like(proposals)
        targets[:, 0] = (gt_ctr_x - proposals_ctr_x) / proposals_widths
        targets[:, 1] = (gt_ctr_y - proposals_ctr_y) / proposals_heights
        targets[:, 2] = torch.log(gt_widths / proposals_widths)
        targets[:, 3] = torch.log(gt_heights / proposals_heights)

        return targets

    def postprocess_predictions(
        self, class_logits, pred_boxes, proposals, image_shapes
    ):
        """Convert predictions to final detections"""
        # Apply softmax to get class probabilities
        class_probs = F.softmax(class_logits, dim=-1)

        # Decode boxes
        if isinstance(proposals, list):
            # Handle list of proposals
            predictions = []
            start_idx = 0
            for i, props in enumerate(proposals):
                end_idx = start_idx + len(props)
                scores_i = class_probs[start_idx:end_idx]
                boxes_i = self.decode_boxes(pred_boxes[start_idx:end_idx], props)
                predictions.append(
                    {
                        "boxes": boxes_i,
                        "scores": scores_i[:, 1],  # Score for positive class
                        "labels": torch.ones(
                            len(boxes_i), dtype=torch.long, device=boxes_i.device
                        ),
                    }
                )
                start_idx = end_idx
        else:
            # Handle concatenated proposals
            predictions = [
                {
                    "boxes": self.decode_boxes(pred_boxes, proposals[:, 1:]),
                    "scores": class_probs[:, 1],
                    "labels": torch.ones(
                        len(class_probs), dtype=torch.long, device=class_probs.device
                    ),
                }
            ]

        return predictions

    def decode_boxes(self, box_deltas, proposals):
        """Decode box regression predictions"""
        # Get proposal parameters
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        ctr_x = proposals[:, 0] + 0.5 * widths
        ctr_y = proposals[:, 1] + 0.5 * heights

        # Apply deltas
        dx = box_deltas[:, 0]
        dy = box_deltas[:, 1]
        dw = box_deltas[:, 2]
        dh = box_deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw.clamp(max=4)) * widths
        pred_h = torch.exp(dh.clamp(max=4)) * heights

        # Convert to x1, y1, x2, y2 format
        pred_boxes = torch.zeros_like(box_deltas)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes
