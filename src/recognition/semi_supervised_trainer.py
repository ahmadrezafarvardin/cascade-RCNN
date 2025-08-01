# src/recognition/semi_supervised_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os
from character_classifier import CharacterClassifier


class CharacterDataset(Dataset):
    """Dataset for character images with optional labels"""

    def __init__(
        self,
        images: List[np.ndarray],
        labels: Optional[List[int]] = None,
        transform=None,
        pseudo_labels: Optional[List[int]] = None,
    ):
        self.images = images
        self.labels = labels
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        # Use real label if available, otherwise use pseudo-label
        if self.labels is not None and self.labels[idx] != -1:
            label = self.labels[idx]
            is_pseudo = False
        elif self.pseudo_labels is not None and self.pseudo_labels[idx] != -1:
            label = self.pseudo_labels[idx]
            is_pseudo = True
        else:
            label = -1  # Unlabeled
            is_pseudo = False

        return image, label, is_pseudo


class SemiSupervisedTrainer:
    """
    Semi-supervised trainer using pseudo-labeling
    """

    def __init__(self, model: CharacterClassifier, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def extract_labeled_data_from_expressions(
        self,
        character_images: List[np.ndarray],
        character_metadata: List[Dict],
        expression_labels: Dict[str, str],
    ) -> Tuple[List[int], Dict]:
        """
        Extract labeled data from expression annotations

        Args:
            character_images: List of detected character images
            character_metadata: Metadata for each character (source image, position)
            expression_labels: Dictionary mapping image names to expression strings

        Returns:
            labels: List of labels (-1 for unlabeled)
            label_stats: Statistics about labeled data
        """
        labels = [-1] * len(character_images)
        labeled_count = defaultdict(int)

        # For each character, check if its source image has an expression label
        for idx, metadata in enumerate(character_metadata):
            source_image = metadata["source_image"]

            if source_image in expression_labels:
                # This is a simplified approach - in practice, you'd need to
                # align character positions with expression characters
                # For now, we'll use clustering results to assign labels
                pass

        # Use a small subset of manually labeled data or cluster centers
        # This is where you'd incorporate the labeled expressions

        return labels, dict(labeled_count)

    def generate_pseudo_labels(
        self,
        model: CharacterClassifier,
        unlabeled_data: DataLoader,
        confidence_threshold: float = 0.9,
    ) -> List[int]:
        """
        Generate pseudo-labels for unlabeled data

        Args:
            model: Trained model
            unlabeled_data: DataLoader with unlabeled samples
            confidence_threshold: Minimum confidence for pseudo-labeling

        Returns:
            pseudo_labels: List of pseudo-labels (-1 if below threshold)
        """
        model.eval()
        pseudo_labels = []

        with torch.no_grad():
            for images, _, _ in tqdm(unlabeled_data, desc="Generating pseudo-labels"):
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                max_probs, predictions = torch.max(probs, dim=1)

                # Only assign pseudo-labels for high-confidence predictions
                for prob, pred in zip(max_probs, predictions):
                    if prob >= confidence_threshold:
                        pseudo_labels.append(pred.item())
                    else:
                        pseudo_labels.append(-1)

        return pseudo_labels

    def train_semi_supervised(
        self,
        labeled_images: List[np.ndarray],
        labeled_targets: List[int],
        unlabeled_images: List[np.ndarray],
        cluster_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        pseudo_label_weight: float = 0.5,
        confidence_threshold: float = 0.9,
    ):
        """
        Train model using semi-supervised learning with pseudo-labeling

        Args:
            labeled_images: Images with labels
            labeled_targets: Ground truth labels
            unlabeled_images: Images without labels
            cluster_labels: Cluster assignments from Section 2 (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            pseudo_label_weight: Weight for pseudo-labeled samples
            confidence_threshold: Confidence threshold for pseudo-labeling
        """
        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Combine all data
        all_images = labeled_images + unlabeled_images
        all_labels = labeled_targets + [-1] * len(unlabeled_images)

        # Initialize with cluster-based pseudo-labels if available
        if cluster_labels is not None:
            # Map clusters to most common character in labeled data
            cluster_to_char = self._map_clusters_to_characters(
                labeled_images, labeled_targets, cluster_labels[: len(labeled_images)]
            )

            # Assign initial pseudo-labels based on clusters
            for i, cluster in enumerate(cluster_labels[len(labeled_images) :]):
                if cluster in cluster_to_char:
                    all_labels[len(labeled_images) + i] = cluster_to_char[cluster]

        # Create dataset
        dataset = CharacterDataset(
            all_images, all_labels, transform=self.model.transform
        )

        # Training loop with pseudo-labeling
        for epoch in range(epochs):
            # Create data loader
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training phase
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, labels, is_pseudo in tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{epochs}"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Calculate loss (weighted for pseudo-labels)
                loss = 0
                for i, (output, label, pseudo) in enumerate(
                    zip(outputs, labels, is_pseudo)
                ):
                    if label != -1:
                        weight = pseudo_label_weight if pseudo else 1.0
                        loss += weight * self.criterion(
                            output.unsqueeze(0), label.unsqueeze(0)
                        )

                if loss > 0:
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                mask = labels != -1
                correct += (predicted[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

            # Update learning rate
            scheduler.step()

            # Print statistics
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total if total > 0 else 0
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

            # Generate new pseudo-labels every few epochs
            if (epoch + 1) % 5 == 0:
                print("Updating pseudo-labels...")
                unlabeled_dataset = CharacterDataset(
                    unlabeled_images, transform=self.model.transform
                )
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size)

                new_pseudo_labels = self.generate_pseudo_labels(
                    self.model, unlabeled_loader, confidence_threshold
                )

                # Update dataset with new pseudo-labels
                for i, pseudo_label in enumerate(new_pseudo_labels):
                    if pseudo_label != -1:
                        dataset.labels[len(labeled_images) + i] = pseudo_label

                print(
                    f"Assigned {sum(1 for l in new_pseudo_labels if l != -1)} new pseudo-labels"
                )

    def _map_clusters_to_characters(
        self,
        labeled_images: List[np.ndarray],
        labels: List[int],
        cluster_assignments: np.ndarray,
    ) -> Dict[int, int]:
        """
        Map cluster IDs to character classes based on labeled data
        """
        cluster_to_char = {}
        cluster_votes = defaultdict(lambda: defaultdict(int))

        # Count votes for each cluster
        for label, cluster in zip(labels, cluster_assignments):
            if label != -1 and cluster != -1:
                cluster_votes[cluster][label] += 1

        # Assign most common character to each cluster
        for cluster, votes in cluster_votes.items():
            if votes:
                cluster_to_char[cluster] = max(votes.items(), key=lambda x: x[1])[0]

        return cluster_to_char
