# verify_fixes.py
import torch
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
from torchvision.ops import box_iou


def check_anchor_coverage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with new anchor sizes
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)
    model.to(device)
    model.eval()

    # Load dataset
    root_dir = "dataset"
    dataset = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    total_gt_boxes = 0
    total_covered = 0

    for i in range(min(10, len(dataset))):
        img, target = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        # Get features and anchors
        features = model.backbone(img_tensor)
        image_size = target["orig_size"].tolist()
        anchors = model.anchor_generator(features, [image_size])

        # Check coverage
        gt_boxes = target["boxes"]
        if len(gt_boxes) > 0 and len(anchors[0]) > 0:
            ious = box_iou(gt_boxes, anchors[0].cpu())
            best_iou_per_gt = ious.max(dim=1)[0]

            total_gt_boxes += len(gt_boxes)
            total_covered += (best_iou_per_gt > 0.5).sum().item()

            print(
                f"Image {i}: {(best_iou_per_gt > 0.5).sum().item()}/{len(gt_boxes)} GT boxes covered"
            )
            print(f"  Best IoUs: {best_iou_per_gt.numpy()}")

    print(
        f"\nOverall coverage: {total_covered}/{total_gt_boxes} = {total_covered/total_gt_boxes:.2%}"
    )


if __name__ == "__main__":
    check_anchor_coverage()
