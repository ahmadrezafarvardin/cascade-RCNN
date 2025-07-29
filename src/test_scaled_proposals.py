import numpy as np
import torch
from data.data_preparation import MathExpressionDataset, get_transforms
from utils.region_proposal import selective_search
from utils.box_utils import calculate_iou


def test_scaled_proposals():
    """Test proposals with properly scaled boxes"""

    # Load with transforms (images will be 416x416)
    val_dataset = MathExpressionDataset(
        root_dir="./", split="val", transform=get_transforms(train=False)
    )

    total_gt = 0
    total_matched = 0

    for idx in range(min(10, len(val_dataset))):
        image, target, img_name = val_dataset[idx]
        gt_boxes = target["boxes"]

        print(f"\nImage {idx} ({img_name}):")
        print(f"  Transformed size: {image.shape}")
        print(f"  GT boxes: {len(gt_boxes)}")

        if len(gt_boxes) > 0:
            # Calculate box sizes
            widths = (gt_boxes[:, 2] - gt_boxes[:, 0]).numpy()
            heights = (gt_boxes[:, 3] - gt_boxes[:, 1]).numpy()
            print(f"  Box width range: [{widths.min():.1f}, {widths.max():.1f}]")
            print(f"  Box height range: [{heights.min():.1f}, {heights.max():.1f}]")

        # Generate proposals
        proposals = selective_search(image, max_proposals=1500)
        print(f"  Proposals: {len(proposals)}")

        # Check coverage
        matched = 0
        best_ious = []
        for gt_box in gt_boxes:
            ious = [calculate_iou(prop, gt_box.numpy()) for prop in proposals]
            if ious:
                max_iou = max(ious)
                best_ious.append(max_iou)
                if max_iou >= 0.5:
                    matched += 1

        coverage = matched / len(gt_boxes) if len(gt_boxes) > 0 else 0
        print(f"  Coverage (IoU >= 0.5): {coverage:.1%} ({matched}/{len(gt_boxes)})")

        if best_ious:
            print(
                f"  Best IoUs: mean={np.mean(best_ious):.3f}, max={np.max(best_ious):.3f}"
            )

        total_gt += len(gt_boxes)
        total_matched += matched

    print(
        f"\nOverall coverage: {total_matched/total_gt:.1%} ({total_matched}/{total_gt})"
    )


if __name__ == "__main__":
    test_scaled_proposals()
