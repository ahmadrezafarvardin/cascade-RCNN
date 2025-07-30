# debug_proposals.py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
from torchvision.ops import box_iou
import os


def visualize_proposals_and_anchors(model, dataset, device, image_idx=0):
    model.eval()

    # Get image and target
    img, target = dataset[image_idx]
    img_tensor = img.unsqueeze(0).to(device)

    # Get features
    features = model.backbone(img_tensor)

    # Get image size
    image_size = target["orig_size"].tolist()

    # Generate anchors
    anchors = model.anchor_generator(features, [image_size])

    # Get RPN outputs
    logits, bbox_reg = model.rpn(features)

    # Process to get proposals
    with torch.no_grad():
        proposals, _ = model.process_rpn_outputs(
            logits, bbox_reg, anchors, [image_size], None
        )

    # Convert image for visualization
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Show anchors
    axes[0].imshow(img_np)
    axes[0].set_title(
        f"Anchors (showing {min(50, len(anchors[0]))} of {len(anchors[0])})"
    )

    # Sample anchors to show
    anchor_indices = np.random.choice(
        len(anchors[0]), min(50, len(anchors[0])), replace=False
    )
    for idx in anchor_indices:
        anchor = anchors[0][idx].cpu()
        rect = patches.Rectangle(
            (anchor[0], anchor[1]),
            anchor[2] - anchor[0],
            anchor[3] - anchor[1],
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
            alpha=0.3,
        )
        axes[0].add_patch(rect)

    # 2. Show proposals
    axes[1].imshow(img_np)
    axes[1].set_title(f"RPN Proposals ({len(proposals[0])} proposals)")

    # Show top proposals
    for i, proposal in enumerate(proposals[0][:30].cpu()):
        rect = patches.Rectangle(
            (proposal[0], proposal[1]),
            proposal[2] - proposal[0],
            proposal[3] - proposal[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            alpha=0.7,
        )
        axes[1].add_patch(rect)

    # 3. Show GT boxes and check IoU with proposals
    axes[2].imshow(img_np)
    axes[2].set_title("GT Boxes (green) vs Best Proposals (red)")

    gt_boxes = target["boxes"]
    proposals_cpu = proposals[0].cpu()

    # Draw GT boxes
    for gt_box in gt_boxes:
        rect = patches.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=3,
            edgecolor="green",
            facecolor="none",
        )
        axes[2].add_patch(rect)

    # For each GT box, find best matching proposal
    if len(proposals_cpu) > 0 and len(gt_boxes) > 0:
        ious = box_iou(gt_boxes, proposals_cpu)
        best_proposal_per_gt = ious.argmax(dim=1)
        best_iou_per_gt = ious.max(dim=1)[0]

        for i, (best_idx, best_iou) in enumerate(
            zip(best_proposal_per_gt, best_iou_per_gt)
        ):
            best_proposal = proposals_cpu[best_idx]
            rect = patches.Rectangle(
                (best_proposal[0], best_proposal[1]),
                best_proposal[2] - best_proposal[0],
                best_proposal[3] - best_proposal[1],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            axes[2].add_patch(rect)
            axes[2].text(
                gt_boxes[i][0],
                gt_boxes[i][1] - 5,
                f"IoU: {best_iou:.2f}",
                color="white",
                backgroundcolor="black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig("debug_proposals.png", dpi=150)
    plt.show()

    # Print statistics
    print(f"\nProposal Statistics:")
    print(f"Number of anchors: {len(anchors[0])}")
    print(f"Number of proposals after NMS: {len(proposals[0])}")
    print(f"Number of GT boxes: {len(gt_boxes)}")

    if len(proposals_cpu) > 0 and len(gt_boxes) > 0:
        print(f"\nIoU Statistics:")
        print(f"Best IoU per GT box: {best_iou_per_gt.numpy()}")
        print(f"Mean best IoU: {best_iou_per_gt.mean():.3f}")
        print(
            f"GT boxes with IoU > 0.5: {(best_iou_per_gt > 0.5).sum().item()}/{len(gt_boxes)}"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    checkpoint_path = "results/cascade_rcnn_latest.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    # Load dataset
    root_dir = "dataset"
    dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    # Visualize for multiple images
    for i in range(3):
        print(f"\n{'='*50}")
        print(f"Analyzing Image {i}")
        visualize_proposals_and_anchors(model, dataset_val, device, image_idx=i)
