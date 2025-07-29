# src/train_hybrid.py
import numpy as np
from utils.region_proposal import selective_search
def create_training_proposals(image, gt_boxes, num_proposals=1000):
    """Create proposals that guarantee coverage"""

    proposals = []

    # 1. Add all GT boxes (guaranteed 100% recall)
    for gt in gt_boxes:
        proposals.append(gt)

    # 2. Add augmented GT boxes
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt
        w, h = x2 - x1, y2 - y1

        # Variations in position and size
        for dx in [-4, -2, 0, 2, 4]:
            for dy in [-4, -2, 0, 2, 4]:
                for scale in [0.9, 1.0, 1.1]:
                    new_w, new_h = w * scale, h * scale
                    new_x1 = x1 + dx
                    new_y1 = y1 + dy
                    new_x2 = new_x1 + new_w
                    new_y2 = new_y1 + new_h

                    if (
                        0 <= new_x1 < 416
                        and 0 <= new_y1 < 416
                        and new_x2 < 416
                        and new_y2 < 416
                    ):
                        proposals.append([new_x1, new_y1, new_x2, new_y2])

    # 3. Add random proposals for negative samples
    random_proposals = selective_search(
        image, max_proposals=num_proposals - len(proposals)
    )
    proposals.extend(random_proposals)

    return np.array(proposals[:num_proposals])
