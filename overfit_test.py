# overfit_test.py
import torch
from torch.optim import Adam
from src.models.cascade_rcnn import CascadeRCNN
from src.models.backbone import SimpleBackbone
from src.data.dataloader import MathExpressionDataset
from src.data.transforms import get_transform
import matplotlib.pyplot as plt


def overfit_single_image(num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create fresh model
    backbone = SimpleBackbone()
    model = CascadeRCNN(backbone)
    model.to(device)
    model.train()

    # Load single sample
    root_dir = "dataset"
    dataset = MathExpressionDataset(root_dir, "train", get_transform(train=False))
    img, target = dataset[0]

    # Prepare batch
    images = img.unsqueeze(0).to(device)
    targets = [
        {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in target.items()
        }
    ]

    # Optimizer with high learning rate
    optimizer = Adam(model.parameters(), lr=0.001)

    losses = []

    print("Overfitting on single image...")
    for i in range(num_iterations):
        optimizer.zero_grad()

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if i % 10 == 0:
            print(f"Iteration {i}: Loss = {total_loss.item():.4f}")
            for k, v in loss_dict.items():
                print(f"  {k}: {v.item():.4f}")

    # Test on same image
    model.eval()
    with torch.no_grad():
        predictions = model(images)[0]

    print(f"\nFinal predictions:")
    print(f"Number of boxes: {len(predictions['boxes'])}")
    if len(predictions["scores"]) > 0:
        print(
            f"Score range: [{predictions['scores'].min():.4f}, {predictions['scores'].max():.4f}]"
        )
        print(f"Top 5 scores: {predictions['scores'][:5].cpu().numpy()}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Overfitting Loss Curve")
    plt.savefig("overfit_loss.png")

    return model, predictions


if __name__ == "__main__":
    model, predictions = overfit_single_image(num_iterations=200)
