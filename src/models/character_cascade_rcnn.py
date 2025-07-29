# models/character_cascade_rcnn.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class CharacterCascadeRCNN(nn.Module):
    """
    Cascade R-CNN optimized for character detection with extreme aspect ratios
    """

    def __init__(self, num_classes=2):  # background + character
        super().__init__()

        # 1. FPN Backbone - Essential for multi-scale features
        self.backbone = resnet_fpn_backbone(
            "resnet50",
            pretrained=True,
            returned_layers=[2, 3, 4, 5],  # Use all FPN levels
            extra_blocks="pool",  # Add extra FPN level for tiny objects
        )

        # 2. Custom Anchor Generator based on your K-means analysis
        anchor_sizes = (
            # P2 level - for tiny characters
            ((8,), (12,), (16,)),
            # P3 level
            ((16,), (20,), (24,)),
            # P4 level
            ((24,), (32,), (40,)),
            # P5 level
            ((48,), (64,), (80,)),
            # P6 level - for large characters
            ((96,), (128,), (160,)),
        )

        # Aspect ratios based on your data (W/H ratios)
        aspect_ratios = ((0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5),) * len(anchor_sizes)

        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

        # 3. RoI Pooler with higher resolution for small objects
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3", "4"],
            output_size=14,  # Higher than standard 7
            sampling_ratio=4,  # More sampling points
        )

        # 4. Cascade stages with increasing IoU thresholds
        self.cascade_stages = nn.ModuleList(
            [
                self._make_stage(256, num_classes, iou_threshold=0.5),
                self._make_stage(256, num_classes, iou_threshold=0.6),
                self._make_stage(256, num_classes, iou_threshold=0.7),
            ]
        )

        # 5. RPN with custom settings
        self.rpn = self._make_rpn()

    def _make_stage(self, in_features, num_classes, iou_threshold):
        """Create a cascade stage"""
        return {
            "box_head": TwoMLPHead(
                in_channels=in_features * 14 * 14,  # 14x14 RoI size
                representation_size=1024,
            ),
            "box_predictor": FastRCNNPredictor(1024, num_classes),
            "iou_threshold": iou_threshold,
        }

    def _make_rpn(self):
        """RPN with settings optimized for small objects"""
        return RegionProposalNetwork(
            self.anchor_generator,
            256,  # out_channels from FPN
            # RPN parameters tuned for small objects
            pre_nms_top_n_train=4000,  # More proposals
            pre_nms_top_n_test=2000,
            post_nms_top_n_train=2000,
            post_nms_top_n_test=1000,
            nms_thresh=0.7,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=512,  # More samples
            positive_fraction=0.33,  # More positive samples
            score_thresh=0.0,
        )
