# anchor_generator.py - Memory efficient version
import torch


class AnchorGenerator:
    def __init__(
        self,
        sizes=(60, 90, 120, 150),  # Reduced to 4 sizes
        aspect_ratios=(0.15, 0.2, 0.3, 0.5),
    ):  # Reduced to 4 ratios
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        # Number of anchors per location
        self.num_anchors = len(sizes) * len(aspect_ratios)

    def __call__(self, features, image_sizes):
        """Generate anchors for all feature maps"""
        anchors = []

        # For single feature map output from backbone
        feature_map = list(features.values())[0]

        for i in range(len(image_sizes)):
            anchors_in_image = self.generate_anchors_for_image(
                feature_map.shape[-2:], image_sizes[i], feature_map.device
            )
            anchors.append(anchors_in_image)

        return anchors

    def generate_anchors_for_image(self, feature_shape, image_size, device):
        """Generate anchors for a single image"""
        grid_height, grid_width = feature_shape
        img_height, img_width = image_size

        # Calculate stride
        stride_h = img_height / grid_height
        stride_w = img_width / grid_width

        # Generate anchor centers
        shifts_x = (
            torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_w
            + stride_w / 2
        )
        shifts_y = (
            torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_h
            + stride_h / 2
        )

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2).reshape(-1, 4)

        # Generate base anchors - note that aspect_ratio is width/height
        base_anchors = []
        for size in self.sizes:
            for ratio in self.aspect_ratios:
                # For tall boxes (ratio < 1), we want width < height
                w = size * (ratio**0.5)
                h = size / (ratio**0.5)
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)

        # Generate all anchors
        anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
        anchors = anchors.reshape(-1, 4)

        return anchors


# # anchor_generator.py
# import torch


# class AnchorGenerator:
#     def __init__(
#         self, sizes=(72, 95, 118), aspect_ratios=(0.2, 0.3, 0.5)
#     ):  # Based on your data!
#         self.sizes = sizes
#         self.aspect_ratios = aspect_ratios
#         # Number of anchors per location
#         self.num_anchors = len(sizes) * len(aspect_ratios)

#     def __call__(self, features, image_sizes):
#         """Generate anchors for all feature maps"""
#         anchors = []

#         # For single feature map output from backbone
#         feature_map = list(features.values())[0]

#         for i in range(len(image_sizes)):
#             anchors_in_image = self.generate_anchors_for_image(
#                 feature_map.shape[-2:], image_sizes[i], feature_map.device
#             )
#             anchors.append(anchors_in_image)

#         return anchors

#     def generate_anchors_for_image(self, feature_shape, image_size, device):
#         """Generate anchors for a single image"""
#         grid_height, grid_width = feature_shape
#         img_height, img_width = image_size

#         # Calculate stride
#         stride_h = img_height / grid_height
#         stride_w = img_width / grid_width

#         # Generate anchor centers
#         shifts_x = (
#             torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_w
#             + stride_w / 2
#         )
#         shifts_y = (
#             torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_h
#             + stride_h / 2
#         )

#         shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
#         shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2).reshape(-1, 4)

#         # Generate base anchors - note that aspect_ratio is width/height
#         base_anchors = []
#         for size in self.sizes:
#             for ratio in self.aspect_ratios:
#                 # For tall boxes (ratio < 1), we want width < height
#                 w = size * (ratio**0.5)
#                 h = size / (ratio**0.5)
#                 base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

#         base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)

#         # Generate all anchors
#         anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
#         anchors = anchors.reshape(-1, 4)

#         return anchors
