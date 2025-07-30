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
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    metrics = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move images to device - images is already a batched tensor
            images = images.to(device)

            # Get predictions
            outputs = model(images)

            # Process each image in the batch
            for i, (target, output) in enumerate(zip(targets, outputs)):
                gt_boxes = target["boxes"].cpu().numpy()

                # Skip if no predictions or no ground truth
                if len(output["boxes"]) == 0 or len(gt_boxes) == 0:
                    metrics["precision"].append(0.0)
                    metrics["recall"].append(0.0)
                    continue

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                # Filter predictions by score
                score_threshold = 0.5
                keep = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                if len(pred_boxes) == 0:
                    metrics["precision"].append(0.0)
                    metrics["recall"].append(0.0)
                    continue

                # Calculate IoUs between all predictions and ground truth
                true_positives = 0
                matched_gt = set()

                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou > iou_threshold:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)

                # Calculate precision and recall
                precision = (
                    true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
                )
                recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0

                metrics["precision"].append(precision)
                metrics["recall"].append(recall)

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Evaluating: [{batch_idx}/{len(data_loader)}]")

    # Aggregate metrics
    avg_precision = (
        sum(metrics["precision"]) / len(metrics["precision"])
        if metrics["precision"]
        else 0
    )
    avg_recall = (
        sum(metrics["recall"]) / len(metrics["recall"]) if metrics["recall"] else 0
    )

    # Calculate F1 score
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)

    return {"map_50": avg_precision, "recall": avg_recall, "f1_score": f1_score}
