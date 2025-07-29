# data_preparation.py
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MathExpressionDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Dataset for character localization in mathematical expressions.

        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Define paths
        self.images_dir = os.path.join(root_dir, "dataset", split, "images")
        if split != "test":
            self.labels_dir = os.path.join(root_dir, "dataset", split, "labels")

        # Get image file names
        self.image_files = sorted(
            [
                f
                for f in os.listdir(self.images_dir)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get original image dimensions
        orig_width, orig_height = image.size

        # Process annotations if not in test set
        if self.split != "test":
            # Load corresponding JSON file
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(self.labels_dir, json_name)

            with open(json_path, "r") as f:
                annotation = json.load(f)

            # Extract bounding boxes
            boxes = []
            for box_data in annotation.get("annotations", []):
                # Extract the nested bounding box
                bbox = box_data.get("boundingBox", {})
                x = bbox.get("x", 0)
                y = bbox.get("y", 0)
                w = bbox.get("width", 0)
                h = bbox.get("height", 0)

                # Convert to [x1, y1, x2, y2] format
                boxes.append([x, y, x + w, y + h])

            # Convert to tensor
            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)

            # Apply transforms to image
            if self.transform:
                image = self.transform(image)

                # Scale boxes to match transformed image size
                # After transform, image is (3, 416, 416)
                new_h, new_w = image.shape[1], image.shape[2]

                # Scale factors
                scale_x = new_w / orig_width
                scale_y = new_h / orig_height

                # Scale all boxes
                if len(boxes) > 0:
                    boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
                    boxes[:, [1, 3]] *= scale_y  # Scale y coordinates

                    # Ensure boxes are within image bounds
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

            # Create target dictionary
            target = {
                "boxes": boxes,
                "labels": torch.ones(len(boxes), dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": (
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    if len(boxes) > 0
                    else torch.zeros(0)
                ),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
                "orig_size": torch.tensor([orig_height, orig_width]),
            }

            # Get expression if available
            if "expression" in annotation:
                target["expression"] = annotation["expression"]

            return image, target, img_name

        else:
            # For test set, only return the image
            if self.transform:
                image = self.transform(image)

            return image, img_name


def get_transforms(train=True, target_size=(416, 416)):
    """
    Get image transformations for training or validation.

    Args:
        train (bool): Whether to use training or validation transforms
        target_size (tuple): Target size for resizing

    Returns:
        transforms.Compose: Composed transformations
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def collate_fn(batch):
    """
    Custom collate function for batching samples with varying number of bounding boxes.

    Args:
        batch (list): List of tuples (image, target, img_name)

    Returns:
        tuple: (images, targets, img_names)
    """
    images = []
    targets = []
    img_names = []

    for item in batch:
        if len(item) == 3:  # train/val set
            img, target, img_name = item
            images.append(img)
            targets.append(target)
            img_names.append(img_name)
        else:  # test set
            img, img_name = item
            images.append(img)
            img_names.append(img_name)

    if len(batch[0]) == 3:
        return images, targets, img_names
    else:
        return images, img_names


def create_data_loaders(root_dir, batch_size=8, num_workers=2, target_size=(416, 416)):
    """
    Create data loaders for training and validation.

    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker threads for data loading
        target_size (tuple): Target size for resizing images

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = MathExpressionDataset(
        root_dir=root_dir,
        split="train",
        transform=get_transforms(train=True, target_size=target_size),
    )

    val_dataset = MathExpressionDataset(
        root_dir=root_dir,
        split="val",
        transform=get_transforms(train=False, target_size=target_size),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_test_loader(root_dir, batch_size=8, num_workers=2, target_size=(416, 416)):
    """
    Create data loader for test set.

    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker threads for data loading
        target_size (tuple): Target size for resizing images

    Returns:
        DataLoader: Test data loader
    """
    test_dataset = MathExpressionDataset(
        root_dir=root_dir,
        split="test",
        transform=get_transforms(train=False, target_size=target_size),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return test_loader
