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

    # Track statistics
    total_boxes_before_clipping = 0
    total_boxes_clipped = 0
    total_invalid_boxes = 0
    image_sizes = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # Get image
            img, _, img_name = test_dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)

            # Get image dimensions
            img_height, img_width = img.shape[-2:]
            image_sizes.append((img_width, img_height))

            # Get predictions
            outputs = model(img_tensor)[0]

            # Count boxes before clipping
            total_boxes_before_clipping += len(outputs["boxes"])

            # Clip boxes to image bounds
            boxes = outputs["boxes"].clone()
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=img_width)  # x1
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=img_height)  # y1
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=img_width)  # x2
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=img_height)  # y2

            # Check if any boxes were clipped
            clipped = (boxes != outputs["boxes"]).any(dim=1)
            total_boxes_clipped += clipped.sum().item()

            # Update outputs with clipped boxes
            outputs["boxes"] = boxes

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
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1

                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    total_invalid_boxes += 1
                    continue

                # Skip very small boxes (likely noise)
                if width < 5 or height < 5:
                    total_invalid_boxes += 1
                    continue

                # Additional check: skip boxes that are too large relative to typical character size
                # This is a heuristic - adjust based on your dataset
                if height > img_height * 0.8 or width > img_width * 0.8:
                    total_invalid_boxes += 1
                    continue

                results.append(
                    {
                        "image_id": image_id,
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height,
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

    # Print clipping statistics
    print(f"\nBox statistics:")
    print(f"  Total boxes before filtering: {total_boxes_before_clipping}")
    print(f"  Boxes clipped to image bounds: {total_boxes_clipped}")
    print(f"  Invalid/tiny/oversized boxes removed: {total_invalid_boxes}")

    # Print image size statistics
    widths = [w for w, h in image_sizes]
    heights = [h for w, h in image_sizes]
    print(f"\nImage size statistics:")
    print(f"  Width range: [{min(widths)}, {max(widths)}]")
    print(f"  Height range: [{min(heights)}, {max(heights)}]")
    print(f"  Most common size: {max(set(image_sizes), key=image_sizes.count)}")

    # Show sample of the output
    print("\nFirst 5 rows of output.csv:")
    print(df.head())

    # Verify format and check for issues
    print("\nColumn names:", df.columns.tolist())
    print("Data types:\n", df.dtypes)

    # Additional validation
    if len(df) > 0:
        print("\nBox coordinate ranges:")
        print(f"  x: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
        print(f"  y: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
        print(f"  width: [{df['width'].min():.1f}, {df['width'].max():.1f}]")
        print(f"  height: [{df['height'].min():.1f}, {df['height'].max():.1f}]")

        # Check for any remaining issues
        print(f"\nNegative x values: {(df['x'] < 0).sum()}")
        print(f"Negative y values: {(df['y'] < 0).sum()}")
        print(f"Non-positive widths: {(df['width'] <= 0).sum()}")
        print(f"Non-positive heights: {(df['height'] <= 0).sum()}")

        # Show distribution of box sizes
        print(f"\nBox size distribution:")
        print(f"  Boxes with height > 400: {(df['height'] > 400).sum()}")
        print(f"  Boxes with width > 100: {(df['width'] > 100).sum()}")
        print(f"  Average box area: {(df['width'] * df['height']).mean():.1f}")

        # Show sample predictions for one image
        sample_image_id = df["image_id"].iloc[0]
        sample_df = df[df["image_id"] == sample_image_id].head()
        print(f"\nSample predictions for image {sample_image_id}:")
        print(sample_df)


if __name__ == "__main__":
    generate_submission()

# # src/utils/generate_submission.py
# import sys
# import os

# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# )

# import torch
# import pandas as pd
# from pathlib import Path
# from src.models.cascade_rcnn import CascadeRCNN
# from src.models.backbone import SimpleBackbone
# from src.models.anchor_generator import AnchorGenerator
# from src.models.heads import RPNHead
# from src.data.dataloader import MathExpressionDataset
# from src.data.transforms import get_transform


# def generate_submission():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Initialize model with exact same configuration as training
#     backbone = SimpleBackbone()
#     anchor_generator = AnchorGenerator(
#         sizes=(60, 90, 120, 150), aspect_ratios=(0.15, 0.2, 0.3, 0.5)
#     )

#     model = CascadeRCNN(backbone)
#     model.anchor_generator = anchor_generator
#     model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

#     # Use the checkpoint that exists
#     checkpoint_path = os.path.join(
#         os.path.dirname(__file__), "..", "..", "results", "cascade_rcnn_best.pth"
#     )

#     if not os.path.exists(checkpoint_path):
#         # Try from current directory
#         checkpoint_path = "results/cascade_rcnn_best.pth"

#     print(f"Loading model from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])

#     # Print model info
#     if "val_metrics" in checkpoint:
#         print(
#             f"Loaded model - mAP: {checkpoint['val_metrics']['map_50']:.4f}, Recall: {checkpoint['val_metrics']['recall']:.4f}"
#         )
#     print(f"Model from epoch: {checkpoint.get('epoch', 'unknown') + 1}")

#     model.to(device)
#     model.eval()

#     # Load test dataset
#     dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
#     print(f"\nLoading test dataset from {dataset_path}...")
#     test_dataset = MathExpressionDataset(
#         dataset_path, "test", get_transform(train=False)
#     )
#     print(f"Found {len(test_dataset)} test images")

#     # Generate predictions
#     results = []

#     # Track statistics
#     total_boxes_before_clipping = 0
#     total_boxes_clipped = 0
#     total_invalid_boxes = 0

#     with torch.no_grad():
#         for idx in range(len(test_dataset)):
#             # Get image
#             img, _, img_name = test_dataset[idx]
#             img_tensor = img.unsqueeze(0).to(device)

#             # Get image dimensions
#             img_height, img_width = img.shape[-2:]

#             # Get predictions
#             outputs = model(img_tensor)[0]

#             # Count boxes before clipping
#             total_boxes_before_clipping += len(outputs["boxes"])

#             # Clip boxes to image bounds
#             boxes = outputs["boxes"].clone()
#             boxes[:, 0] = boxes[:, 0].clamp(min=0, max=img_width)  # x1
#             boxes[:, 1] = boxes[:, 1].clamp(min=0, max=img_height)  # y1
#             boxes[:, 2] = boxes[:, 2].clamp(min=0, max=img_width)  # x2
#             boxes[:, 3] = boxes[:, 3].clamp(min=0, max=img_height)  # y2

#             # Check if any boxes were clipped
#             clipped = (boxes != outputs["boxes"]).any(dim=1)
#             total_boxes_clipped += clipped.sum().item()

#             # Update outputs with clipped boxes
#             outputs["boxes"] = boxes

#             # Filter predictions by score
#             score_threshold = 0.3
#             keep = outputs["scores"] > score_threshold
#             boxes = outputs["boxes"][keep]
#             scores = outputs["scores"][keep]

#             # Apply NMS to reduce overlapping boxes
#             if len(boxes) > 0:
#                 nms_keep = torch.ops.torchvision.nms(boxes, scores, 0.3)
#                 boxes = boxes[nms_keep]

#             # Extract image ID from filename
#             image_id = int(Path(img_name).stem)

#             # Add each detection to results
#             for box in boxes:
#                 x1, y1, x2, y2 = box.tolist()
#                 width = x2 - x1
#                 height = y2 - y1

#                 # Skip invalid boxes
#                 if width <= 0 or height <= 0:
#                     total_invalid_boxes += 1
#                     continue

#                 # Skip very small boxes (likely noise)
#                 if width < 5 or height < 5:
#                     total_invalid_boxes += 1
#                     continue

#                 results.append(
#                     {
#                         "image_id": image_id,
#                         "x": x1,
#                         "y": y1,
#                         "width": width,
#                         "height": height,
#                     }
#                 )

#             # Progress update
#             if (idx + 1) % 50 == 0:
#                 print(f"Processed {idx + 1}/{len(test_dataset)} images...")

#     # Create DataFrame and save to CSV
#     df = pd.DataFrame(results)
#     output_path = os.path.join(os.path.dirname(__file__), "..", "..", "output.csv")
#     df.to_csv(output_path, index=False)

#     # Print summary
#     print(f"\n✓ Submission generated successfully!")
#     print(f"  Total predictions: {len(results)}")
#     print(f"  Images with predictions: {df['image_id'].nunique()}")
#     if len(test_dataset) > 0:
#         print(
#             f"  Average predictions per image: {len(results) / len(test_dataset):.2f}"
#         )
#     print(f"  Saved to: {output_path}")

#     # Print clipping statistics
#     print(f"\nBox statistics:")
#     print(f"  Total boxes before filtering: {total_boxes_before_clipping}")
#     print(f"  Boxes clipped to image bounds: {total_boxes_clipped}")
#     print(f"  Invalid/tiny boxes removed: {total_invalid_boxes}")

#     # Show sample of the output
#     print("\nFirst 5 rows of output.csv:")
#     print(df.head())

#     # Verify format and check for issues
#     print("\nColumn names:", df.columns.tolist())
#     print("Data types:\n", df.dtypes)

#     # Additional validation
#     if len(df) > 0:
#         print("\nBox coordinate ranges:")
#         print(f"  x: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
#         print(f"  y: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
#         print(f"  width: [{df['width'].min():.1f}, {df['width'].max():.1f}]")
#         print(f"  height: [{df['height'].min():.1f}, {df['height'].max():.1f}]")

#         # Check for any remaining issues
#         print(f"\nNegative x values: {(df['x'] < 0).sum()}")
#         print(f"Negative y values: {(df['y'] < 0).sum()}")
#         print(f"Non-positive widths: {(df['width'] <= 0).sum()}")
#         print(f"Non-positive heights: {(df['height'] <= 0).sum()}")

#         # Show sample predictions for one image
#         sample_image_id = df["image_id"].iloc[0]
#         sample_df = df[df["image_id"] == sample_image_id].head()
#         print(f"\nSample predictions for image {sample_image_id}:")
#         print(sample_df)


# if __name__ == "__main__":
#     generate_submission()


# # src/utils/generate_submission.py
# import sys
# import os

# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# )

# import torch
# import pandas as pd
# from pathlib import Path
# from src.models.cascade_rcnn import CascadeRCNN
# from src.models.backbone import SimpleBackbone
# from src.models.anchor_generator import AnchorGenerator
# from src.models.heads import RPNHead
# from src.data.dataloader import MathExpressionDataset
# from src.data.transforms import get_transform


# def generate_submission():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Initialize model with exact same configuration as training
#     backbone = SimpleBackbone()
#     anchor_generator = AnchorGenerator(
#         sizes=(60, 90, 120, 150), aspect_ratios=(0.15, 0.2, 0.3, 0.5)
#     )

#     model = CascadeRCNN(backbone)
#     model.anchor_generator = anchor_generator
#     model.rpn = RPNHead(backbone.out_channels, num_anchors=anchor_generator.num_anchors)

#     # Use the checkpoint that exists
#     checkpoint_path = os.path.join(
#         os.path.dirname(__file__), "..", "..", "results", "cascade_rcnn_best.pth"
#     )

#     if not os.path.exists(checkpoint_path):
#         # Try from current directory
#         checkpoint_path = "results/cascade_rcnn_best.pth"

#     print(f"Loading model from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])

#     # Print model info
#     if "val_metrics" in checkpoint:
#         print(
#             f"Loaded model - mAP: {checkpoint['val_metrics']['map_50']:.4f}, Recall: {checkpoint['val_metrics']['recall']:.4f}"
#         )
#     print(f"Model from epoch: {checkpoint.get('epoch', 'unknown') + 1}")

#     model.to(device)
#     model.eval()

#     # Load test dataset
#     dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
#     print(f"\nLoading test dataset from {dataset_path}...")
#     test_dataset = MathExpressionDataset(
#         dataset_path, "test", get_transform(train=False)
#     )
#     print(f"Found {len(test_dataset)} test images")

#     # Generate predictions
#     results = []

#     with torch.no_grad():
#         for idx in range(len(test_dataset)):
#             # Get image
#             img, _, img_name = test_dataset[idx]
#             img_tensor = img.unsqueeze(0).to(device)

#             # Get predictions
#             outputs = model(img_tensor)[0]

#             # Filter predictions by score
#             score_threshold = 0.3
#             keep = outputs["scores"] > score_threshold
#             boxes = outputs["boxes"][keep]
#             scores = outputs["scores"][keep]

#             # Apply NMS to reduce overlapping boxes
#             if len(boxes) > 0:
#                 nms_keep = torch.ops.torchvision.nms(boxes, scores, 0.3)
#                 boxes = boxes[nms_keep]

#             # Extract image ID from filename
#             image_id = int(Path(img_name).stem)

#             # Add each detection to results
#             for box in boxes:
#                 results.append(
#                     {
#                         "image_id": image_id,
#                         "x": float(box[0]),
#                         "y": float(box[1]),
#                         "width": float(box[2] - box[0]),
#                         "height": float(box[3] - box[1]),
#                     }
#                 )

#             # Progress update
#             if (idx + 1) % 50 == 0:
#                 print(f"Processed {idx + 1}/{len(test_dataset)} images...")

#     # Create DataFrame and save to CSV
#     df = pd.DataFrame(results)
#     output_path = os.path.join(os.path.dirname(__file__), "..", "..", "output.csv")
#     df.to_csv(output_path, index=False)

#     # Print summary
#     print(f"\n✓ Submission generated successfully!")
#     print(f"  Total predictions: {len(results)}")
#     print(f"  Images with predictions: {df['image_id'].nunique()}")
#     if len(test_dataset) > 0:
#         print(
#             f"  Average predictions per image: {len(results) / len(test_dataset):.2f}"
#         )
#     print(f"  Saved to: {output_path}")

#     # Show sample of the output
#     print("\nFirst 5 rows of output.csv:")
#     print(df.head())

#     # Verify format
#     print("\nColumn names:", df.columns.tolist())
#     print("Data types:\n", df.dtypes)


# if __name__ == "__main__":
#     generate_submission()
