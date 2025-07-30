# transforms.py
import random
import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        # Convert PIL Image to tensor and ensure it's float32
        image = F.to_tensor(image).float()
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.choice(self.sizes)
        return Resize(size, self.max_size)(image, target)


class Resize:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, image, target):
        # Store original size if not already present
        if "orig_size" not in target:
            target["orig_size"] = torch.tensor(image.shape[-2:], dtype=torch.float32)

        # Resize image
        image = F.resize(image, self.size, antialias=True)

        # Scale boxes if present
        if "boxes" in target:
            orig_h, orig_w = target["orig_size"]
            scale_h = self.size[0] / orig_h
            scale_w = self.size[1] / orig_w

            boxes = target["boxes"]
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h
            target["boxes"] = boxes

        # Update current size
        target["size"] = torch.tensor(self.size, dtype=torch.float32)
        return image, target


def get_transform(train=True):
    transforms = []
    transforms.append(ToTensor())
    transforms.append(Resize((800, 800)))  # Fixed size for all images
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return Compose(transforms)


def collate_fn(batch):
    # Since images are now all the same size, we can stack them
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
