import numpy as np
from data.data_preparation import MathExpressionDataset, get_transforms
from utils.box_utils import calculate_iou


def generate_targeted_proposals(image_shape, gt_box):
    """Generate proposals specifically targeting a ground truth box"""
    h, w = image_shape[1:3]
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
    gt_w = gt_x2 - gt_x1
    gt_h = gt_y2 - gt_y1

    proposals = []

    # Generate proposals around the GT box with slight variations
    for dx in range(-5, 6, 1):
        for dy in range(-5, 6, 1):
            for dw in range(-3, 4, 1):
                for dh in range(-5, 6, 1):
                    x1 = gt_x1 + dx
                    y1 = gt_y1 + dy
                    x2 = gt_x2 + dx + dw
                    y2 = gt_y2 + dy + dh

                    if (
                        x1 >= 0
                        and y1 >= 0
                        and x2 <= w
                        and y2 <= h
                        and x2 > x1
                        and y2 > y1
                    ):
                        proposals.append([x1, y1, x2, y2])

    return np.array(proposals)


def test_targeted():
    """Test if we can generate good proposals when we know the target"""

    val_dataset = MathExpressionDataset(
        root_dir="./", split="val", transform=get_transforms(train=False)
    )

    image, target, img_name = val_dataset[0]
    gt_boxes = target["boxes"].numpy()

    print(f"Testing targeted proposals for {img_name}")

    for i, gt_box in enumerate(gt_boxes):
        proposals = generate_targeted_proposals(image.shape, gt_box)

        ious = [calculate_iou(prop, gt_box) for prop in proposals]
        max_iou = max(ious) if ious else 0
        high_iou_count = sum(1 for iou in ious if iou >= 0.5)

        print(f"  GT Box {i}: {gt_box}")
        print(f"    Generated {len(proposals)} proposals")
        print(f"    Max IoU: {max_iou:.3f}")
        print(f"    Proposals with IoU >= 0.5: {high_iou_count}")


if __name__ == "__main__":
    test_targeted()
