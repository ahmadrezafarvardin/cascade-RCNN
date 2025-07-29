import numpy as np
import torch
from PIL import Image

# K-means derived anchor sizes from your dataset
ANCHOR_SIZES = [
    (18.7, 126.6),
    (20.0, 60.4),
    (21.1, 102.8),
    (21.3, 37.8),
    (21.8, 16.9),
    (22.1, 82.3),
    (22.8, 317.7),
    (23.8, 210.6),
    (24.0, 152.7),
    (24.3, 178.9),
    (25.9, 248.4),
    (36.9, 114.5),
]


def generate_character_proposals(image, stride=4, max_proposals=5000):
    """
    Generate region proposals using K-means derived anchors
    """
    if isinstance(image, torch.Tensor):
        h, w = image.shape[1:3]
    else:
        h, w = 416, 416

    proposals = []

    # Strategy 1: Use exact K-means anchors with dense sliding window
    for anchor_w, anchor_h in ANCHOR_SIZES:
        anchor_w = int(anchor_w)
        anchor_h = int(anchor_h)

        # Skip if anchor is too large for image
        if anchor_w >= w or anchor_h >= h:
            continue

        # Dense sliding window
        x_stride = max(1, min(stride, anchor_w // 4))
        y_stride = max(1, min(stride, anchor_h // 8))

        for y in range(0, h - anchor_h + 1, y_stride):
            for x in range(0, w - anchor_w + 1, x_stride):
                proposals.append([x, y, x + anchor_w, y + anchor_h])

    # Strategy 2: Add variations around each anchor size
    for anchor_w, anchor_h in ANCHOR_SIZES:
        # Scale variations: 0.8x, 0.9x, 1.0x, 1.1x, 1.2x
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            w_scaled = int(anchor_w * scale)
            h_scaled = int(anchor_h * scale)

            if w_scaled < 4 or h_scaled < 4 or w_scaled >= w or h_scaled >= h:
                continue

            # Less dense for variations
            x_stride = max(2, min(stride * 2, w_scaled // 3))
            y_stride = max(2, min(stride * 2, h_scaled // 6))

            for y in range(0, h - h_scaled + 1, y_stride):
                for x in range(0, w - w_scaled + 1, x_stride):
                    proposals.append([x, y, x + w_scaled, y + h_scaled])

    # Strategy 3: Add intermediate sizes between anchors
    for i in range(len(ANCHOR_SIZES) - 1):
        w1, h1 = ANCHOR_SIZES[i]
        w2, h2 = ANCHOR_SIZES[i + 1]

        # Interpolate between adjacent anchors
        for alpha in [0.33, 0.67]:
            w_interp = int(w1 * (1 - alpha) + w2 * alpha)
            h_interp = int(h1 * (1 - alpha) + h2 * alpha)

            if w_interp >= w or h_interp >= h:
                continue

            x_stride = max(2, min(stride * 2, w_interp // 3))
            y_stride = max(2, min(stride * 2, h_interp // 6))

            for y in range(0, h - h_interp + 1, y_stride):
                for x in range(0, w - w_interp + 1, x_stride):
                    proposals.append([x, y, x + w_interp, y + h_interp])

    # Strategy 4: Focus on problematic sizes from test output
    # Very narrow boxes (4-10 pixels wide) with various heights
    for width in range(4, 11):
        for height_mult in [3, 5, 8, 10, 15, 20, 25, 30]:
            height = width * height_mult
            if height >= h:
                continue

            # Very dense sampling for narrow boxes
            for y in range(0, h - height + 1, 2):
                for x in range(0, w - width + 1, 1):
                    proposals.append([x, y, x + width, y + height])

    # Strategy 5: Add jittered versions of anchors
    n_jitter = 500
    for _ in range(n_jitter):
        # Pick a random anchor
        anchor_w, anchor_h = ANCHOR_SIZES[np.random.randint(len(ANCHOR_SIZES))]

        # Add random jitter
        w_jitter = int(anchor_w + np.random.uniform(-5, 5))
        h_jitter = int(anchor_h + np.random.uniform(-20, 20))

        w_jitter = max(4, min(w_jitter, w - 1))
        h_jitter = max(4, min(h_jitter, h - 1))

        x = np.random.randint(0, max(1, w - w_jitter))
        y = np.random.randint(0, max(1, h - h_jitter))

        proposals.append([x, y, x + w_jitter, y + h_jitter])

        # Add shifted versions
        for dx in [-2, 2]:
            for dy in [-2, 2]:
                new_x = max(0, min(x + dx, w - w_jitter))
                new_y = max(0, min(y + dy, h - h_jitter))
                proposals.append([new_x, new_y, new_x + w_jitter, new_y + h_jitter])

    proposals = np.array(proposals, dtype=np.float32)

    # Remove invalid and duplicate proposals
    if len(proposals) > 0:
        # Remove invalid
        valid_mask = (proposals[:, 2] > proposals[:, 0]) & (
            proposals[:, 3] > proposals[:, 1]
        )
        proposals = proposals[valid_mask]

        # Remove exact duplicates
        proposals = np.unique(proposals, axis=0)

        # If too many, sample intelligently
        if len(proposals) > max_proposals:
            # Prioritize proposals that match anchor sizes
            scores = []
            for prop in proposals:
                w = prop[2] - prop[0]
                h = prop[3] - prop[1]

                # Score based on similarity to anchors
                min_dist = float("inf")
                for anchor_w, anchor_h in ANCHOR_SIZES:
                    dist = abs(w - anchor_w) + abs(h - anchor_h)
                    min_dist = min(min_dist, dist)

                scores.append(1.0 / (1.0 + min_dist))

            scores = np.array(scores)
            # Keep top scoring proposals
            top_indices = np.argsort(scores)[-max_proposals:]
            proposals = proposals[top_indices]

    return proposals


def selective_search(image, min_size=4, max_proposals=5000):
    """
    Main function for region proposal generation
    """
    return generate_character_proposals(image, stride=4, max_proposals=max_proposals)
