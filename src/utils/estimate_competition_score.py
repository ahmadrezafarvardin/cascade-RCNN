# estimate_competition_score.py
import torch
import numpy as np
from torchvision.ops import box_iou
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.models.anchor_generator import AnchorGenerator
from src.models.heads import RPNHead
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform, collate_fn
from torch.utils.data import DataLoader


def estimate_competition_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (same as submission generator)
    backbone = SimpleBackbone()
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150), aspect_ratios=(0.15, 0.2, 0.3, 0.5)
    )

    model = CascadeRCNN(backbone)
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    checkpoint = torch.load("results/cascade_rcnn_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load validation data
    dataset_val = MathExpressionDataset("dataset", "val", get_transform(train=False))
    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # Competition thresholds
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    f1_scores = []

    print("Estimating competition performance on validation set...")

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in data_loader_val:
            images = images.to(device)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # Same filtering as submission
                keep = output["scores"] > 0.3
                pred_boxes = output["boxes"][keep]

                if len(pred_boxes) > 0:
                    scores = output["scores"][keep]
                    nms_keep = torch.ops.torchvision.nms(pred_boxes, scores, 0.3)
                    pred_boxes = pred_boxes[nms_keep]

                all_predictions.append(pred_boxes.cpu())
                all_ground_truths.append(target["boxes"].cpu())

    # Calculate F1 for each threshold
    for threshold in iou_thresholds:
        tp, fp, fn = 0, 0, 0

        for pred_boxes, gt_boxes in zip(all_predictions, all_ground_truths):
            if len(pred_boxes) == 0:
                fn += len(gt_boxes)
                continue

            if len(gt_boxes) == 0:
                fp += len(pred_boxes)
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            matched_gt = set()

            for i in range(len(pred_boxes)):
                max_iou, max_idx = ious[i].max(dim=0)
                if max_iou >= threshold and max_idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(max_idx.item())
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1)

        print(
            f"IoU≥{threshold}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
        )

    avg_f1 = np.mean(f1_scores)
    print(f"\nEstimated Competition Score (Average F1): {avg_f1:.4f}")


if __name__ == "__main__":
    estimate_competition_performance()
