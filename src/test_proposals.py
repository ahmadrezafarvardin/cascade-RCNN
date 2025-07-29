import numpy as np
from data.data_preparation import MathExpressionDataset, get_transforms
from utils.region_proposal import selective_search, visualize_proposals_coverage
from utils.box_utils import calculate_iou


def test_proposal_quality():
    """Test the quality of region proposals"""

    # Load validation dataset
    val_dataset = MathExpressionDataset(
        root_dir="./", split="val", transform=get_transforms(train=False)
    )

    total_gt_boxes = 0
    total_matched = 0
    iou_threshold = 0.5

    for idx in range(min(10, len(val_dataset))):
        image, target, img_name = val_dataset[idx]
        gt_boxes = target["boxes"]

        # Generate proposals
        proposals = selective_search(image, max_proposals=1000)

        print(f"\nImage {idx} ({img_name}):")
        print(f"  GT boxes: {len(gt_boxes)}")
        print(f"  Proposals: {len(proposals)}")

        # Check coverage
        matched = 0
        for gt_box in gt_boxes:
            ious = [calculate_iou(prop, gt_box.numpy()) for prop in proposals]
            if ious and max(ious) >= iou_threshold:
                matched += 1

        coverage = matched / len(gt_boxes) if len(gt_boxes) > 0 else 0
        print(
            f"  Coverage (IoU >= {iou_threshold}): {coverage:.2%} ({matched}/{len(gt_boxes)})"
        )

        total_gt_boxes += len(gt_boxes)
        total_matched += matched

        # Visualize first few
        if idx < 3:
            visualize_proposals_coverage(
                image,
                proposals,
                gt_boxes,
                save_path=f"./results/rcnn/proposal_test_{img_name}",
            )

    print(
        f"\nOverall coverage: {total_matched/total_gt_boxes:.2%} ({total_matched}/{total_gt_boxes})"
    )


if __name__ == "__main__":
    test_proposal_quality()
