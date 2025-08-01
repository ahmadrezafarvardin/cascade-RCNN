# cascade_rcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from .heads import RPNHead, CascadeROIHeads
from .anchor_generator import AnchorGenerator


class CascadeRCNN(nn.Module):
    def __init__(self, backbone, num_stages=3):
        super().__init__()
        self.backbone = backbone
        self.num_stages = num_stages

        # Anchor Generator (must be created before RPN)
        self.anchor_generator = AnchorGenerator()

        # RPN Head (num_anchors must match anchor generator)
        self.rpn = RPNHead(
            backbone.out_channels, num_anchors=self.anchor_generator.num_anchors
        )

        # ROI Heads
        self.roi_heads = CascadeROIHeads(
            in_channels=backbone.out_channels, num_stages=num_stages
        )

    def forward(self, images, targets=None):
        # Feature extraction
        features = self.backbone(images)

        # Get image sizes
        if targets is not None:
            image_sizes = [t["orig_size"].tolist() for t in targets]
        else:
            image_sizes = [images.shape[-2:]] * images.shape[0]

        # RPN forward pass
        logits, bbox_reg = self.rpn(features)

        # Generate anchors
        anchors = self.anchor_generator(features, image_sizes)

        # Process RPN outputs
        proposals, rpn_losses = self.process_rpn_outputs(
            logits, bbox_reg, anchors, image_sizes, targets
        )

        # ROI Heads forward pass
        detector_losses = {}

        # During training, we need to keep track of proposals for each stage
        current_proposals = proposals

        for stage in range(self.num_stages):
            stage_predictions, stage_losses = self.roi_heads(
                features, current_proposals, image_sizes, targets, stage
            )

            # Add stage losses
            for k, v in stage_losses.items():
                detector_losses[f"stage{stage}_{k}"] = v

            # For cascade training, we need to update proposals even during training
            if stage < self.num_stages - 1:  # Not the last stage
                if self.training:
                    # During training, we need to get refined proposals for the next stage
                    # We'll create a modified forward pass for this
                    with torch.no_grad():
                        refined_proposals = self.get_refined_proposals(
                            features, current_proposals, stage
                        )
                    current_proposals = refined_proposals
                else:
                    # During inference, use predictions from current stage
                    refined_proposals = []
                    for pred in stage_predictions:
                        # Keep only high-scoring detections
                        keep = pred["scores"] > 0.05
                        if keep.sum() > 0:
                            refined_proposals.append(pred["boxes"][keep])
                        else:
                            # If no high-scoring detections, keep original proposals
                            refined_proposals.append(
                                current_proposals[len(refined_proposals)]
                            )
                    current_proposals = refined_proposals

        if self.training:
            return {**rpn_losses, **detector_losses}
        else:
            # Return predictions from the last stage
            return stage_predictions

    def get_refined_proposals(self, features, proposals, stage):
        """Get refined proposals for the next cascade stage during training"""
        # Convert proposals to proper format
        if isinstance(proposals, list):
            roi_proposals = []
            for i, props in enumerate(proposals):
                if len(props) == 0:
                    props = torch.zeros((1, 4), device=features["0"].device)
                batch_idx = torch.full(
                    (props.shape[0], 1), i, dtype=props.dtype, device=props.device
                )
                roi_proposals.append(torch.cat([batch_idx, props], dim=1))
            proposals_concat = torch.cat(roi_proposals, dim=0)
        else:
            proposals_concat = proposals

        # Get feature map
        feature_map = features["0"]

        # Normalize proposals
        proposals_normalized = self.roi_heads.normalize_proposals(
            proposals_concat, feature_map.shape[-2:]
        )

        # ROI pooling
        box_features = self.roi_heads.roi_pool(feature_map, proposals_normalized)
        box_features = box_features.view(box_features.size(0), -1)

        # Forward through the stage head
        head_output = self.roi_heads.heads[stage](box_features)

        # Get box predictions
        pred_boxes = self.roi_heads.bbox_pred[stage](head_output)

        # Decode boxes
        if isinstance(proposals, list):
            refined_proposals = []
            start_idx = 0
            for i, props in enumerate(proposals):
                end_idx = start_idx + len(props)
                refined_boxes = self.roi_heads.decode_boxes(
                    pred_boxes[start_idx:end_idx], props
                )
                refined_proposals.append(refined_boxes)
                start_idx = end_idx
        else:
            refined_proposals = [
                self.roi_heads.decode_boxes(pred_boxes, proposals_concat[:, 1:])
            ]

        return refined_proposals

    def process_rpn_outputs(self, logits, bbox_reg, anchors, image_sizes, targets):
        """Process RPN outputs to generate proposals and calculate losses"""
        proposals = []
        losses = {}

        batch_size = len(image_sizes)

        # Initialize losses
        if self.training:
            total_class_loss = 0
            total_reg_loss = 0
            num_samples = 0

        # Process each image in the batch
        for i in range(batch_size):
            # Get predictions for this image
            objectness = logits[0][i]  # Shape: [H*W*A]
            pred_boxes = bbox_reg[0][i]  # Shape: [H*W*A, 4]

            # Get anchors for this image
            anchors_i = anchors[i]  # Shape: [H*W*A, 4]

            # Calculate losses if training
            if self.training and targets is not None:
                losses_i = self.compute_rpn_loss(
                    objectness, pred_boxes, anchors_i, targets[i]
                )
                total_class_loss += losses_i["rpn_class_loss"]
                total_reg_loss += losses_i["rpn_regression_loss"]
                num_samples += 1

            # Decode boxes from RPN predictions
            proposals_i = self.decode_boxes(pred_boxes, anchors_i)

            # Clip to image boundaries
            proposals_i = box_ops.clip_boxes_to_image(proposals_i, image_sizes[i])

            # Remove small boxes
            keep = box_ops.remove_small_boxes(proposals_i, min_size=1)
            proposals_i = proposals_i[keep]
            objectness_i = objectness[keep]

            # Apply NMS with lower threshold for more diverse proposals
            keep = torch.ops.torchvision.nms(
                proposals_i, objectness_i.sigmoid(), 0.5
            )  # Lowered from 0.7

            # Keep top proposals
            if self.training:
                # During training, keep more proposals
                keep = keep[: min(2000, len(keep))]
            else:
                # During inference, keep fewer but higher quality proposals
                keep = keep[: min(1000, len(keep))]

            proposals_i = proposals_i[keep]

            proposals.append(proposals_i)

        if self.training and num_samples > 0:
            losses = {
                "rpn_class_loss": total_class_loss / num_samples,
                "rpn_regression_loss": total_reg_loss / num_samples,
            }

        return proposals, losses

    def decode_boxes(self, pred_boxes, anchors):
        """Convert RPN predictions to box coordinates"""
        # pred_boxes shape: [num_anchors, 4]
        # anchors shape: [num_anchors, 4]

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = pred_boxes[:, 0]
        dy = pred_boxes[:, 1]
        dw = pred_boxes[:, 2]
        dh = pred_boxes[:, 3]

        # Apply transformations
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw.clamp(max=4)) * widths  # Clamp to prevent overflow
        pred_h = torch.exp(dh.clamp(max=4)) * heights

        # Convert to x1, y1, x2, y2 format
        pred_boxes_decoded = torch.zeros_like(pred_boxes)
        pred_boxes_decoded[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes_decoded[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes_decoded[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes_decoded[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes_decoded

    def compute_rpn_loss(self, objectness, pred_boxes, anchors, target):
        """Compute RPN classification and regression losses."""
        gt_boxes = target["boxes"]
        device = objectness.device

        # Match anchors to ground truth boxes
        matched_gt_boxes, labels = self.match_anchors_to_gt(anchors, gt_boxes)

        # Compute classification loss
        classification_loss = self.compute_classification_loss(objectness, labels)

        # Compute regression loss
        regression_loss = self.compute_regression_loss(
            pred_boxes, anchors, matched_gt_boxes, labels
        )

        return {
            "rpn_class_loss": classification_loss,
            "rpn_regression_loss": regression_loss,
        }

    def match_anchors_to_gt(
        self, anchors, gt_boxes, positive_thresh=0.7, negative_thresh=0.3
    ):
        """Match each anchor to the best ground truth box."""
        if gt_boxes.numel() == 0:
            return torch.zeros_like(anchors), torch.full(
                (anchors.shape[0],), -1, device=anchors.device
            )

        # Compute IoU between all anchors and gt boxes
        ious = box_ops.box_iou(anchors, gt_boxes)

        # Match each anchor to its best gt box
        max_ious, best_gt_idx = ious.max(dim=1)

        # Create labels
        labels = torch.full(
            (anchors.shape[0],), -1, dtype=torch.float32, device=anchors.device
        )

        # Positive anchors
        positive_mask = max_ious >= positive_thresh

        # Ensure each gt box has at least one positive anchor
        gt_best_anchor = ious.argmax(dim=0)
        positive_mask[gt_best_anchor] = True

        # Negative anchors
        negative_mask = max_ious < negative_thresh

        labels[positive_mask] = 1
        labels[negative_mask] = 0

        # Get matched gt boxes
        matched_gt_boxes = gt_boxes[best_gt_idx]
        matched_gt_boxes[labels != 1] = 0

        return matched_gt_boxes, labels

    def compute_classification_loss(self, objectness, labels):
        """Compute binary cross-entropy loss for object/background classification."""
        mask = labels >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=objectness.device)

        return F.binary_cross_entropy_with_logits(
            objectness[mask], labels[mask].float(), reduction="mean"
        )

    def compute_regression_loss(self, pred_boxes, anchors, matched_gt_boxes, labels):
        """Compute smooth L1 loss for box regression."""
        positive_mask = labels == 1
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Encode ground truth boxes
        gt_deltas = self.encode_boxes(
            matched_gt_boxes[positive_mask], anchors[positive_mask]
        )

        return F.smooth_l1_loss(pred_boxes[positive_mask], gt_deltas, reduction="mean")

    def encode_boxes(self, boxes, anchors):
        """Encode box coordinates relative to anchors."""
        anchors_width = anchors[:, 2] - anchors[:, 0]
        anchors_height = anchors[:, 3] - anchors[:, 1]
        anchors_ctr_x = anchors[:, 0] + 0.5 * anchors_width
        anchors_ctr_y = anchors[:, 1] + 0.5 * anchors_height

        boxes_width = boxes[:, 2] - boxes[:, 0]
        boxes_height = boxes[:, 3] - boxes[:, 1]
        boxes_ctr_x = boxes[:, 0] + 0.5 * boxes_width
        boxes_ctr_y = boxes[:, 1] + 0.5 * boxes_height

        # Avoid division by zero
        eps = 1e-6
        anchors_width = torch.clamp(anchors_width, min=eps)
        anchors_height = torch.clamp(anchors_height, min=eps)

        deltas = torch.zeros_like(boxes)
        deltas[:, 0] = (boxes_ctr_x - anchors_ctr_x) / anchors_width
        deltas[:, 1] = (boxes_ctr_y - anchors_ctr_y) / anchors_height
        deltas[:, 2] = torch.log(boxes_width / anchors_width)
        deltas[:, 3] = torch.log(boxes_height / anchors_height)

        return deltas
