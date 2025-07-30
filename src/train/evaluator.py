# evaluator.py
import torch
from collections import defaultdict


def calculate_iou(boxA, boxB):
    # Determine intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute areas of both boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    metrics = defaultdict(list)

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()

                # Calculate IoUs
                ious = []
                for gt_box in gt_boxes:
                    for pred_box in pred_boxes:
                        ious.append(calculate_iou(gt_box, pred_box))

                # Calculate precision and recall
                true_positives = sum(iou > iou_threshold for iou in ious)
                precision = (
                    true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
                )
                recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0

                metrics["precision"].append(precision)
                metrics["recall"].append(recall)

    # Aggregate metrics
    avg_precision = sum(metrics["precision"]) / len(metrics["precision"])
    avg_recall = sum(metrics["recall"]) / len(metrics["recall"])

    return {"map_50": avg_precision, "recall": avg_recall}
