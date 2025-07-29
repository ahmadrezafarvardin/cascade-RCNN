import torch
import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: Bounding box in format [x1, y1, x2, y2]
        box2: Bounding box in format [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    # Get the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


def generate_proposals(
    image_size, scales=[64, 128, 256], aspect_ratios=[0.5, 1, 2], stride=16
):
    """
    Generate region proposals using a sliding window approach

    Args:
        image_size: Size of the image (height, width)
        scales: List of scales for the proposals
        aspect_ratios: List of aspect ratios for the proposals
        stride: Stride for sliding window

    Returns:
        List of proposals in format [x1, y1, x2, y2]
    """
    height, width = image_size
    proposals = []

    for scale in scales:
        for ratio in aspect_ratios:
            # Calculate width and height of the proposal
            w = int(scale * np.sqrt(ratio))
            h = int(scale / np.sqrt(ratio))

            # Slide window over the image
            for y in range(0, height - h + 1, stride):
                for x in range(0, width - w + 1, stride):
                    proposals.append([x, y, x + w, y + h])

    return torch.tensor(proposals, dtype=torch.float32)


def filter_proposals(proposals, scores, threshold=0.5, max_proposals=300):
    """
    Filter proposals based on confidence scores

    Args:
        proposals: Tensor of proposals in format [x1, y1, x2, y2]
        scores: Confidence scores for each proposal
        threshold: Confidence threshold
        max_proposals: Maximum number of proposals to keep

    Returns:
        Filtered proposals and their scores
    """
    # Filter by threshold
    mask = scores > threshold
    filtered_proposals = proposals[mask]
    filtered_scores = scores[mask]

    # Sort by score and take top max_proposals
    if len(filtered_scores) > max_proposals:
        _, indices = torch.sort(filtered_scores, descending=True)
        indices = indices[:max_proposals]
        filtered_proposals = filtered_proposals[indices]
        filtered_scores = filtered_scores[indices]

    return filtered_proposals, filtered_scores


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to avoid duplicate detections

    Args:
        boxes: Tensor of boxes in format [x1, y1, x2, y2]
        scores: Confidence scores for each box
        iou_threshold: IoU threshold for considering boxes as duplicates

    Returns:
        Indices of boxes to keep
    """
    # Sort boxes by score
    _, indices = torch.sort(scores, descending=True)
    boxes = boxes[indices]

    keep = []
    while indices.size(0) > 0:
        # Keep the box with highest score
        keep.append(indices[0].item())

        # Calculate IoU of this box with all remaining boxes
        ious = torch.tensor([calculate_iou(boxes[0], box) for box in boxes[1:]])

        # Find boxes with IoU less than threshold
        mask = ious < iou_threshold
        indices = indices[1:][mask]
        boxes = boxes[1:][mask]

    return keep


def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculate precision and recall for object detection

    Args:
        pred_boxes: Predicted bounding boxes
        true_boxes: Ground truth bounding boxes
        iou_threshold: IoU threshold for considering a prediction correct

    Returns:
        precision, recall
    """
    # Initialize counters
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Mark ground truth boxes as unmatched initially
    matched_gt = [False] * len(true_boxes)

    # Check each predicted box
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        # Find the best matching ground truth box
        for i, gt_box in enumerate(true_boxes):
            if not matched_gt[i]:  # Only consider unmatched ground truth boxes
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        # Check if the best match is good enough
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt[best_gt_idx] = True  # Mark this ground truth box as matched
        else:
            fp += 1

    # Count unmatched ground truth boxes as false negatives
    fn = matched_gt.count(False)

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall
