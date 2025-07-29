import numpy as np
import torch
from PIL import Image


def generate_character_proposals(
    image, min_size=4, max_size=400, stride=2, max_proposals=3000
):
    """
    Generate region proposals specifically for narrow character boxes
    """
    if isinstance(image, torch.Tensor):
        h, w = image.shape[1:3]
    else:
        h, w = 416, 416

    proposals = []

    # Focus on narrow boxes based on your data:
    # Widths: mostly 4-25 pixels
    # Heights: 10-92 pixels (but can go up to 380)

    # Strategy 1: Very narrow boxes (critical for your dataset)
    narrow_widths = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    for width in narrow_widths:
        # For narrow boxes, heights are typically 3-20x the width
        height_ratios = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20]
        for ratio in height_ratios:
            height = int(width * ratio)
            if height < 10 or height > h * 0.9:
                continue

            # Dense sampling for narrow boxes
            x_stride = max(1, width // 4)
            y_stride = max(2, height // 10)

            for y in range(0, h - height + 1, y_stride):
                for x in range(0, w - width + 1, x_stride):
                    proposals.append([x, y, x + width, y + height])

    # Strategy 2: Fixed heights with narrow widths
    target_heights = [
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        150,
        180,
        200,
        250,
        300,
    ]
    for height in target_heights:
        if height > h * 0.9:
            continue

        # For each height, use narrow widths
        width_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        for ratio in width_ratios:
            width = max(4, int(height * ratio))
            if width > 50:  # Based on your data, widths rarely exceed 50
                continue

            y_stride = max(2, height // 10)
            x_stride = max(1, width // 2)

            for y in range(0, h - height + 1, y_stride):
                for x in range(0, w - width + 1, x_stride):
                    proposals.append([x, y, x + width, y + height])

    # Strategy 3: Ultra-narrow boxes (width 4-10) with various heights
    for width in range(4, 11):
        for height in range(20, min(300, h), 10):
            # Sample more densely in regions where characters typically appear
            for y in range(
                100, min(350, h - height), 4
            ):  # Characters seem to be in middle-lower region
                for x in range(0, w - width + 1, 3):
                    proposals.append([x, y, x + width, y + height])

    # Strategy 4: Add jittered boxes around typical positions
    # Based on your debug output, boxes are often around x=60-320, y=130-290
    for _ in range(500):
        # Random narrow box
        width = np.random.randint(4, 30)
        height = np.random.randint(max(10, width * 2), min(150, width * 20))

        # Bias towards middle-right region where characters appear
        x = np.random.randint(0, max(1, w - width))
        y = np.random.randint(max(0, h // 4), max(1, h - height))

        proposals.append([x, y, x + width, y + height])

    proposals = np.array(proposals, dtype=np.float32)

    # Remove invalid boxes
    if len(proposals) > 0:
        valid_mask = (proposals[:, 2] > proposals[:, 0]) & (
            proposals[:, 3] > proposals[:, 1]
        )
        proposals = proposals[valid_mask]

        # Remove duplicates
        proposals = np.unique(proposals, axis=0)

    # Limit proposals with smart sampling
    if len(proposals) > max_proposals:
        # Keep a mix of different sizes
        n_keep = max_proposals

        # Sort by width to ensure we keep narrow boxes
        widths = proposals[:, 2] - proposals[:, 0]
        width_order = np.argsort(widths)

        # Keep the narrowest 50%
        narrow_indices = width_order[: len(width_order) // 2]

        # Random sample from the rest
        other_indices = width_order[len(width_order) // 2 :]
        if len(other_indices) > n_keep // 2:
            other_indices = np.random.choice(other_indices, n_keep // 2, replace=False)

        keep_indices = np.concatenate([narrow_indices[: n_keep // 2], other_indices])
        proposals = proposals[keep_indices[:n_keep]]

    return proposals


def selective_search(image, min_size=4, max_proposals=3000):
    """
    Main function for region proposal generation
    """
    return generate_character_proposals(
        image, min_size=min_size, max_size=400, stride=2, max_proposals=max_proposals
    )
