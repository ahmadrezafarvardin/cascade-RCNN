# debug_model_outputs.py
import torch
import torch.nn.functional as F
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
import os


def debug_model_outputs(model, dataset, device):
    model.eval()

    # Get a sample
    img, target = dataset[0]
    img_tensor = img.unsqueeze(0).to(device)

    # Hook to capture intermediate outputs
    intermediate_outputs = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            intermediate_outputs[name] = output.detach().cpu()

        return hook_fn

    # Register hooks
    handles = []
    for stage in range(model.num_stages):
        handles.append(
            model.roi_heads.predictors[stage].register_forward_hook(
                make_hook(f"stage{stage}_logits")
            )
        )
        handles.append(
            model.roi_heads.bbox_pred[stage].register_forward_hook(
                make_hook(f"stage{stage}_bbox")
            )
        )

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Analyze outputs
    print("Model Output Analysis:")
    print("=" * 50)

    for stage in range(model.num_stages):
        if f"stage{stage}_logits" in intermediate_outputs:
            logits = intermediate_outputs[f"stage{stage}_logits"]
            probs = F.softmax(logits, dim=1)

            print(f"\nStage {stage}:")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"  Logits mean: {logits.mean():.4f}")
            print(f"  Background logits mean: {logits[:, 0].mean():.4f}")
            print(f"  Object logits mean: {logits[:, 1].mean():.4f}")
            print(
                f"  Object probability range: [{probs[:, 1].min():.4f}, {probs[:, 1].max():.4f}]"
            )
            print(f"  Object probability mean: {probs[:, 1].mean():.4f}")

            # Check if logits are all zeros
            if torch.allclose(logits, torch.zeros_like(logits), atol=1e-4):
                print("  WARNING: Logits are all near zero!")

    # Check final outputs
    print(f"\nFinal outputs:")
    print(f"  Number of detections: {len(outputs[0]['boxes'])}")
    print(
        f"  Score range: [{outputs[0]['scores'].min():.4f}, {outputs[0]['scores'].max():.4f}]"
    )

    # Check weight statistics
    print("\nWeight Statistics:")
    for stage in range(model.num_stages):
        predictor = model.roi_heads.predictors[stage]
        weight_norm = predictor.weight.norm().item()
        bias_norm = predictor.bias.norm().item() if predictor.bias is not None else 0
        print(
            f"  Stage {stage} predictor - Weight norm: {weight_norm:.4f}, Bias norm: {bias_norm:.4f}"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)

    checkpoint_path = "results/cascade_rcnn_latest.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    root_dir = "dataset"
    dataset_val = MathExpressionDataset(root_dir, "val", get_transform(train=False))

    debug_model_outputs(model, dataset_val, device)
