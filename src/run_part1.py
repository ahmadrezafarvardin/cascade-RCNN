# run_part1.py
import os
import argparse
import torch
from train import train_model
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate R-CNN model")
    parser.add_argument(
        "--root_dir", type=str, default="./", help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/rcnn",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model for evaluation",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.eval_only:
        # Train model
        print("Training model...")
        model, history = train_model(
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )

        model_path = os.path.join(args.output_dir, "best_model.pth")
    else:
        # Use provided model path
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(args.output_dir, "best_model.pth")

    # Evaluate model
    print("Evaluating model...")
    eval_output_dir = os.path.join(args.output_dir, "evaluation")
    metrics = evaluate_model(
        model_path=model_path,
        root_dir=args.root_dir,
        output_dir=eval_output_dir,
        num_samples=10,
    )

    print("Done!")


if __name__ == "__main__":
    main()
