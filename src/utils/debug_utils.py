# debug_utils.py
import torch
import numpy as np

from utils.box_utils import calculate_iou


def debug_predictions(model, image, proposals, gt_boxes, device):
    """Debug what the model is predicting"""

    model.eval()
    with torch.no_grad():
        # Get predictions
        proposals_tensor = torch.tensor(proposals, dtype=torch.float32).to(device)
        image_tensor = image.unsqueeze(0).to(device)

        class_scores, bbox_deltas, _ = model(image_tensor, proposals_tensor)

        # Analyze predictions
        probs = torch.softmax(class_scores, dim=1)
        char_probs = probs[:, 1]

        print(f"\nDebug Info:")
        print(f"Number of proposals: {len(proposals)}")
        print(f"Number of GT boxes: {len(gt_boxes)}")
        print(f"Max character probability: {char_probs.max().item():.4f}")
        print(f"Mean character probability: {char_probs.mean().item():.4f}")
        print(f"Proposals with prob > 0.5: {(char_probs > 0.5).sum().item()}")
        print(f"Proposals with prob > 0.3: {(char_probs > 0.3).sum().item()}")
        print(f"Proposals with prob > 0.1: {(char_probs > 0.1).sum().item()}")

        # Check IoU of proposals with GT
        ious = []
        for prop in proposals[:100]:  # Check first 100 proposals
            for gt in gt_boxes:
                iou = calculate_iou(prop, gt.numpy())
                ious.append(iou)

        if ious:
            print(f"Max IoU between proposals and GT: {max(ious):.4f}")
            print(
                f"Number of proposals with IoU > 0.5: {sum(1 for iou in ious if iou > 0.5)}"
            )
