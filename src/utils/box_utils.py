import numpy as np
import torch


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_iou_matrix(boxes1, boxes2):
    """Calculate IoU matrix between two sets of boxes"""
    n1 = len(boxes1)
    n2 = len(boxes2)
    iou_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            iou_matrix[i, j] = calculate_iou(boxes1[i], boxes2[j])

    return iou_matrix


def compute_bbox_targets(proposals, gt_boxes):
    """
    Compute regression targets and labels for proposals

    Args:
        proposals: Array of proposal boxes [N, 4]
        gt_boxes: Array of ground truth boxes [M, 4]

    Returns:
        targets: Regression targets [N, 4]
        labels: Binary labels (0: background, 1: object) [N]
    """
    if len(gt_boxes) == 0:
        return np.zeros((len(proposals), 4)), np.zeros(len(proposals), dtype=np.int64)

    # Calculate IoU between all proposals and ground truth boxes
    ious = calculate_iou_matrix(proposals, gt_boxes)

    # Find best matching GT box for each proposal
    max_ious = ious.max(axis=1)
    max_indices = ious.argmax(axis=1)

    # Assign labels (1 for IoU >= 0.5, 0 otherwise)
    labels = (max_ious >= 0.5).astype(np.int64)

    # Get corresponding GT boxes for each proposal
    target_boxes = gt_boxes[max_indices]

    # Compute regression targets
    targets = encode_boxes(proposals, target_boxes)

    return targets, labels


def encode_boxes(proposals, gt_boxes):
    """
    Encode ground truth boxes w.r.t proposals
    """
    # Prevent division by zero
    eps = 1e-6

    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = proposals[:, 2] - proposals[:, 0] + eps
    ph = proposals[:, 3] - proposals[:, 1] + eps

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = np.log(gw / pw)
    dh = np.log(gh / ph)

    return np.stack([dx, dy, dw, dh], axis=1)


def decode_boxes(proposals, deltas):
    """
    Decode predicted box deltas to get final boxes
    """
    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    gx = dx * pw + px
    gy = dy * ph + py
    gw = torch.exp(dw) * pw
    gh = torch.exp(dh) * ph

    x1 = gx - gw / 2
    y1 = gy - gh / 2
    x2 = gx + gw / 2
    y2 = gy + gh / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def apply_nms(boxes, scores, threshold=0.5):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []

    # Convert to numpy if tensor
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Calculate precision and recall"""
    if len(pred_boxes) == 0:
        return 0, 0

    if len(gt_boxes) == 0:
        return 0, 1

    # Calculate IoU matrix
    ious = calculate_iou_matrix(pred_boxes, gt_boxes)

    # Find matches
    matched_gt = set()
    matched_pred = 0

    for i in range(len(pred_boxes)):
        max_iou = 0
        max_j = -1
        for j in range(len(gt_boxes)):
            if j not in matched_gt and ious[i, j] > max_iou:
                max_iou = ious[i, j]
                max_j = j

        if max_iou >= iou_threshold:
            matched_pred += 1
            matched_gt.add(max_j)

    precision = matched_pred / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = len(matched_gt) / len(gt_boxes) if len(gt_boxes) > 0 else 0

    return precision, recall
