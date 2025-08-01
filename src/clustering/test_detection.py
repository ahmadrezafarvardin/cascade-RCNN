# src/clustering/test_detection.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from models.cascade_rcnn import CascadeRCNN
from models.backbone import SimpleBackbone
from models.anchor_generator import AnchorGenerator
from models.heads import RPNHead


def test_model_detection(model_path, image_path, output_path):
    """Test if the model can detect characters in a single image"""

    # Initialize model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    # Create anchor generator with same config as training
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150),
        aspect_ratios=(0.15, 0.2, 0.3, 0.5),
    )
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Get predictions
    with torch.no_grad():
        predictions = model(img_tensor)

    # Draw predictions
    if len(predictions) > 0:
        pred = predictions[0]
        print(f"Number of detections: {len(pred['boxes'])}")
        print(f"Scores: {pred['scores'].tolist()}")

        # Draw boxes on image
        for idx in range(len(pred["boxes"])):
            score = pred["scores"][idx].item()
            if score > 0.1:  # Very low threshold for testing
                box = pred["boxes"][idx].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put score
                cv2.putText(
                    image,
                    f"{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Save result
        cv2.imwrite(output_path, image)
        print(f"Saved visualization to {output_path}")
    else:
        print("No predictions returned!")


if __name__ == "__main__":
    model_path = "../../results/cascade_rcnn_best.pth"

    # Test on first training image
    dataset_path = "../../dataset"
    train_images = os.path.join(dataset_path, "train", "images")

    if os.path.exists(train_images):
        first_image = sorted(os.listdir(train_images))[0]
        image_path = os.path.join(train_images, first_image)
        output_path = "../../results/clustering/test_detection.png"

        print(f"Testing on image: {image_path}")
        test_model_detection(model_path, image_path, output_path)
    else:
        print(f"Could not find training images at {train_images}")
