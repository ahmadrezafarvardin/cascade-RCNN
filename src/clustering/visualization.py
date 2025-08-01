# src/clustering/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
from typing import List, Optional, Tuple
import os
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings("ignore")


class ClusterVisualizer:
    """
    Class for visualizing clustering results
    """

    def __init__(self, output_dir: str = "results/clustering_yolo"):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style - use available style
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except:
            plt.style.use("seaborn-darkgrid")
        sns.set_palette("husl")

    def plot_cluster_samples(
        self,
        images: List[np.ndarray],
        labels: np.ndarray,
        n_samples_per_cluster: int = 10,
        save_name: str = "cluster_samples.png",
    ):
        """
        Plot sample images from each cluster

        Args:
            images: List of character images
            labels: Cluster labels
            n_samples_per_cluster: Number of samples to show per cluster
            save_name: Filename to save plot
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])

        if n_clusters == 0:
            print("No clusters found (all points are outliers)")
            return

        # Calculate grid size
        n_cols = n_samples_per_cluster
        n_rows = n_clusters + (1 if -1 in unique_labels else 0)  # Add row for outliers

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        row_idx = 0

        # Plot samples from each cluster
        for label in sorted(unique_labels):
            if label == -1:
                continue

            # Get indices for this cluster
            cluster_indices = np.where(labels == label)[0]

            # Sample indices
            if len(cluster_indices) > n_samples_per_cluster:
                sample_indices = np.random.choice(
                    cluster_indices, n_samples_per_cluster, replace=False
                )
            else:
                sample_indices = cluster_indices

            # Plot samples
            for col_idx, idx in enumerate(sample_indices):
                if col_idx >= n_cols:
                    break

                ax = axes[row_idx, col_idx]

                # Get image
                img = images[idx]
                if len(img.shape) == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap="gray")

                ax.axis("off")

                if col_idx == 0:
                    ax.set_ylabel(
                        f"Cluster {label}",
                        fontsize=10,
                        rotation=0,
                        labelpad=30,
                        ha="right",
                    )

            # Clear remaining axes in row
            for col_idx in range(len(sample_indices), n_cols):
                axes[row_idx, col_idx].axis("off")

            row_idx += 1

        # Plot outliers if any
        if -1 in unique_labels:
            outlier_indices = np.where(labels == -1)[0]

            if len(outlier_indices) > 0:
                sample_indices = np.random.choice(
                    outlier_indices,
                    min(len(outlier_indices), n_samples_per_cluster),
                    replace=False,
                )

                for col_idx, idx in enumerate(sample_indices):
                    if col_idx >= n_cols:
                        break

                    ax = axes[row_idx, col_idx]
                    img = images[idx]

                    if len(img.shape) == 3:
                        ax.imshow(img)
                    else:
                        ax.imshow(img, cmap="gray")

                    ax.axis("off")

                    if col_idx == 0:
                        ax.set_ylabel(
                            "Outliers", fontsize=10, rotation=0, labelpad=30, ha="right"
                        )

                # Clear remaining axes
                for col_idx in range(len(sample_indices), n_cols):
                    axes[row_idx, col_idx].axis("off")

        plt.suptitle(
            f"Character Samples by Cluster ({n_clusters} clusters)", fontsize=14
        )
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Cluster samples saved to {save_path}")

    def plot_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        perplexity: int = 30,
        save_name: str = "tsne_visualization.png",
    ):
        """
        Create t-SNE visualization of clusters

        Args:
            features: Feature matrix
            labels: Cluster labels
            perplexity: t-SNE perplexity parameter
            save_name: Filename to save plot
        """
        print("Computing t-SNE embedding...")

        # Reduce dimensionality if needed
        if features.shape[1] > 50:
            print(f"Reducing dimensions from {features.shape[1]} to 50 using PCA...")
            pca = PCA(n_components=50, random_state=42)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features

        # Compute t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(features) - 1),
            random_state=42,
            max_iter=1000,  # Changed from n_iter to max_iter
        )
        embeddings = tsne.fit_transform(features_reduced)

        # Create plot
        plt.figure(figsize=(10, 8))

        # Plot each cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                # Plot outliers differently
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c="black",
                    marker="x",
                    s=50,
                    alpha=0.5,
                    label="Outliers",
                )
            else:
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[color],
                    s=50,
                    alpha=0.7,
                    label=f"Cluster {label}",
                )

        plt.title("t-SNE Visualization of Character Clusters", fontsize=14)
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"t-SNE visualization saved to {save_path}")

    def plot_pca(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_name: str = "pca_visualization.png",
    ):
        """
        Create PCA visualization of clusters

        Args:
            features: Feature matrix
            labels: Cluster labels
            save_name: Filename to save plot
        """
        print("Computing PCA...")

        # Compute PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings = pca.fit_transform(features)

        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_

        # Create plot
        plt.figure(figsize=(10, 8))

        # Plot each cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            if label == -1:
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c="black",
                    marker="x",
                    s=50,
                    alpha=0.5,
                    label="Outliers",
                )
            else:
                plt.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=[color],
                    s=50,
                    alpha=0.7,
                    label=f"Cluster {label}",
                )

        plt.title(
            f"PCA Visualization of Character Clusters\n"
            + f"(Explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%})",
            fontsize=14,
        )
        plt.xlabel(f"PC1 ({explained_var[0]:.2%})")
        plt.ylabel(f"PC2 ({explained_var[1]:.2%})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"PCA visualization saved to {save_path}")

    def plot_dendrogram(
        self,
        features: np.ndarray,
        method: str = "ward",
        save_name: str = "dendrogram.png",
    ):
        """
        Plot hierarchical clustering dendrogram

        Args:
            features: Feature matrix
            method: Linkage method
            save_name: Filename to save plot
        """
        print("Computing dendrogram...")

        # Sample if too many points
        if len(features) > 500:
            print(f"Sampling 500 points from {len(features)} for dendrogram...")
            indices = np.random.choice(len(features), 500, replace=False)
            features_sample = features[indices]
        else:
            features_sample = features

        # Compute linkage
        Z = linkage(features_sample, method=method)

        # Create plot
        plt.figure(figsize=(12, 8))

        dendrogram(
            Z,
            truncate_mode="level",
            p=10,  # Show only last 10 merges
            show_leaf_counts=True,
            leaf_font_size=10,
        )

        plt.title(f"Hierarchical Clustering Dendrogram ({method} linkage)", fontsize=14)
        plt.xlabel("Cluster Size")
        plt.ylabel("Distance")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Dendrogram saved to {save_path}")

    def plot_cluster_statistics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_name: str = "cluster_statistics.png",
    ):
        """
        Plot statistics about clusters

        Args:
            features: Feature matrix
            labels: Cluster labels
            save_name: Filename to save plot
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])

        # Calculate statistics
        cluster_sizes = []
        cluster_labels_list = []

        for label in unique_labels:
            if label != -1:
                size = np.sum(labels == label)
                cluster_sizes.append(size)
                cluster_labels_list.append(f"Cluster {label}")

        if -1 in unique_labels:
            outlier_size = np.sum(labels == -1)
            cluster_sizes.append(outlier_size)
            cluster_labels_list.append("Outliers")

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart of cluster sizes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cluster_sizes)))
        ax1.pie(
            cluster_sizes, labels=cluster_labels_list, colors=colors, autopct="%1.1f%%"
        )
        ax1.set_title("Cluster Size Distribution", fontsize=12)

        # Bar chart of cluster sizes
        bars = ax2.bar(cluster_labels_list, cluster_sizes, color=colors)
        ax2.set_title("Number of Characters per Cluster", fontsize=12)
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Number of Characters")
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(size)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Cluster statistics saved to {save_path}")

    def create_cluster_report(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        images: Optional[List[np.ndarray]] = None,
        method_name: str = "clustering",
    ):
        """
        Create a comprehensive cluster visualization report

        Args:
            features: Feature matrix
            labels: Cluster labels
            images: Optional list of character images
            method_name: Name of clustering method
        """
        print(f"\nCreating comprehensive report for {method_name}...")

        # Create subdirectory for this report
        report_dir = os.path.join(self.output_dir, f"{method_name}_report")
        os.makedirs(report_dir, exist_ok=True)

        # Update output directory temporarily
        original_dir = self.output_dir
        self.output_dir = report_dir

        # 1. t-SNE visualization
        self.plot_tsne(features, labels, save_name="1_tsne.png")

        # 2. PCA visualization
        self.plot_pca(features, labels, save_name="2_pca.png")

        # 3. Cluster samples (if images provided)
        if images is not None and len(images) >= len(labels):
            self.plot_cluster_samples(
                images[: len(labels)], labels, save_name="3_cluster_samples.png"
            )

        # 4. Cluster statistics
        self.plot_cluster_statistics(features, labels, save_name="4_cluster_stats.png")

        # Restore original output directory
        self.output_dir = original_dir

        print(f"Report saved to {report_dir}")


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import os

    print("Testing ClusterVisualizer with synthetic data...")

    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=300, n_features=50, centers=5, cluster_std=1.0, random_state=42
    )

    # Perform clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X)

    # Create synthetic "images" (just random arrays for testing)
    images = []
    for i in range(len(X)):
        # Create a simple pattern based on cluster
        img = np.ones((32, 32, 3), dtype=np.uint8) * 255
        color_value = int(255 * (labels[i] + 1) / 6)
        img[:, :, labels[i] % 3] = color_value
        images.append(img)

    # Initialize visualizer
    visualizer = ClusterVisualizer(output_dir="results/clustering_test")

    # Test individual visualizations
    print("\nTesting individual visualizations...")
    visualizer.plot_cluster_samples(images, labels)
    visualizer.plot_tsne(X, labels)
    visualizer.plot_pca(X, labels)
    visualizer.plot_cluster_statistics(X, labels)

    # Test comprehensive report
    print("\nTesting comprehensive report...")
    visualizer.create_cluster_report(X, labels, images, "kmeans_test")

    print("\nAll visualization tests completed successfully!")
