# src/utils/generate_submission.py
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import pandas as pd
from pathlib import Path
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.models.anchor_generator import AnchorGenerator
from src.models.heads import RPNHead
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with exact same configuration as training
    backbone = SimpleBackbone()
    anchor_generator = AnchorGenerator(
        sizes=(60, 90, 120, 150), aspect_ratios=(0.15, 0.2, 0.3, 0.5)
    )

    model = CascadeRCNN(backbone)
    model.anchor_generator = anchor_generator
    model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

    # Use the checkpoint that exists
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "cascade_rcnn_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        # Try from current directory
        checkpoint_path = "results/cascade_rcnn_best.pth"

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Print model info
    if "val_metrics" in checkpoint:
        print(
            f"Loaded model - mAP: {checkpoint['val_metrics']['map_50']:.4f}, Recall: {checkpoint['val_metrics']['recall']:.4f}"
        )
    print(f"Model from epoch: {checkpoint.get('epoch', 'unknown') + 1}")

    model.to(device)
    model.eval()

    # Load test dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    print(f"\nLoading test dataset from {dataset_path}...")
    test_dataset = MathExpressionDataset(
        dataset_path, "test", get_transform(train=False)
    )
    print(f"Found {len(test_dataset)} test images")

    # Generate predictions
    results = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Get image
            img, _, img_name = test_dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)

            # Get predictions
            outputs = model(img_tensor)[0]

            # Filter predictions by score
            score_threshold = 0.3
            keep = outputs["scores"] > score_threshold
            boxes = outputs["boxes"][keep]
            scores = outputs["scores"][keep]

            # Apply NMS to reduce overlapping boxes
            if len(boxes) > 0:
                nms_keep = torch.ops.torchvision.nms(boxes, scores, 0.3)
                boxes = boxes[nms_keep]

            # Extract image ID from filename
            image_id = int(Path(img_name).stem)

            # Add each detection to results
            for box in boxes:
                results.append(
                    {
                        "image_id": image_id,
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1]),
                    }
                )

            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_dataset)} images...")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "output.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n✓ Submission generated successfully!")
    print(f"  Total predictions: {len(results)}")
    print(f"  Images with predictions: {df['image_id'].nunique()}")
    if len(test_dataset) > 0:
        print(
            f"  Average predictions per image: {len(results) / len(test_dataset):.2f}"
        )
    print(f"  Saved to: {output_path}")

    # Show sample of the output
    print("\nFirst 5 rows of output.csv:")
    print(df.head())

    # Verify format
    print("\nColumn names:", df.columns.tolist())
    print("Data types:\n", df.dtypes)


if __name__ == "__main__":
    generate_submission()
