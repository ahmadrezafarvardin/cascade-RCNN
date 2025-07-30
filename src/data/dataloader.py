# dataloader
import json
import os
from pathlib import Path
from PIL import Image
import torch


class MathExpressionDataset:
    def __init__(self, root_dir, split="train", transforms=None):
        """
        Args:
            root_dir (str): Path to the dataset root (should contain train/val/test folders)
            split (str): One of 'train', 'val', or 'test'
            transforms: Optional transforms to be applied
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms

        # Set up paths
        self.images_dir = self.root_dir / split / "images"
        if split != "test":
            self.labels_dir = self.root_dir / split / "labels"

        # Verify paths exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if split != "test" and not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Get sorted list of image files
        self.image_files = sorted(
            [
                f.name
                for f in self.images_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.images_dir / img_name
        img = Image.open(img_path).convert("RGB")

        # Store original image dimensions (height, width)
        orig_size = torch.tensor([img.height, img.width], dtype=torch.float32)

        if self.split == "test":
            target = {"orig_size": orig_size, "size": orig_size}  # For compatibility
            if self.transforms:
                img, target = self.transforms(img, target)
            return img, target, img_name

        # Load annotations
        json_name = f"{Path(img_name).stem}.json"
        json_path = self.labels_dir / json_name

        with open(json_path) as f:
            annotations = json.load(f)

        # Process boxes
        boxes = []
        for ann in annotations["annotations"]:
            box = ann["boundingBox"]
            boxes.append(
                [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]]
            )

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "orig_size": orig_size,
            "size": orig_size,  # For compatibility
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
