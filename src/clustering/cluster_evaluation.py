# src/clustering/cluster_evaluation.py
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class ClusterEvaluator:
    """
    Class for evaluating clustering results
    """

    def __init__(self):
        """Initialize the evaluator"""
        self.metrics_history = []

    def silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette coefficient

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        # Remove outliers for silhouette calculation
        mask = labels != -1
        if np.sum(mask) < 2:
            return -1.0

        n_clusters = len(np.unique(labels[mask]))
        if n_clusters < 2 or n_clusters >= len(labels[mask]):
            return -1.0

        try:
            score = silhouette_score(features[mask], labels[mask])
            return score
        except:
            return -1.0

    def davies_bouldin_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin index

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Davies-Bouldin score (lower is better)
        """
        # Remove outliers
        mask = labels != -1
        if np.sum(mask) < 2:
            return float("inf")

        n_clusters = len(np.unique(labels[mask]))
        if n_clusters < 2:
            return float("inf")

        try:
            score = davies_bouldin_score(features[mask], labels[mask])
            return score
        except:
            return float("inf")

    def calinski_harabasz_score(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Calculate Calinski-Harabasz index

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Calinski-Harabasz score (higher is better)
        """
        # Remove outliers
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0

        n_clusters = len(np.unique(labels[mask]))
        if n_clusters < 2:
            return 0.0

        try:
            score = calinski_harabasz_score(features[mask], labels[mask])
            return score
        except:
            return 0.0

    def evaluate_all_metrics(
        self, features: np.ndarray, labels: np.ndarray, method_name: str = "Unknown"
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics

        Args:
            features: Feature matrix
            labels: Cluster labels
            method_name: Name of clustering method

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "method": method_name,
            "n_clusters": len(np.unique(labels[labels != -1])),
            "n_outliers": np.sum(labels == -1),
            "silhouette": self.silhouette_score(features, labels),
            "davies_bouldin": self.davies_bouldin_score(features, labels),
            "calinski_harabasz": self.calinski_harabasz_score(features, labels),
        }

        # Calculate cluster size statistics
        unique_labels = np.unique(labels)
        cluster_sizes = [np.sum(labels == l) for l in unique_labels if l != -1]

        if cluster_sizes:
            metrics["avg_cluster_size"] = np.mean(cluster_sizes)
            metrics["std_cluster_size"] = np.std(cluster_sizes)
            metrics["min_cluster_size"] = np.min(cluster_sizes)
            metrics["max_cluster_size"] = np.max(cluster_sizes)

        self.metrics_history.append(metrics)
        return metrics

    def plot_silhouette_analysis(
        self, features: np.ndarray, labels: np.ndarray, save_path: Optional[str] = None
    ):
        """
        Create silhouette plot for each cluster

        Args:
            features: Feature matrix
            labels: Cluster labels
            save_path: Path to save plot
        """
        # Remove outliers
        mask = labels != -1
        if np.sum(mask) < 2:
            print("Not enough samples for silhouette analysis")
            return

        features_clean = features[mask]
        labels_clean = labels[mask]

        n_clusters = len(np.unique(labels_clean))

        # Calculate silhouette scores for each sample
        silhouette_vals = silhouette_samples(features_clean, labels_clean)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_lower = 10
        for i in range(n_clusters):
            # Get silhouette scores for cluster i
            cluster_silhouette_vals = silhouette_vals[labels_clean == i]
            cluster_silhouette_vals.sort()

            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label clusters
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax.set_title("Silhouette Plot for Various Clusters", fontsize=14)
        ax.set_xlabel("Silhouette Coefficient Values", fontsize=12)
        ax.set_ylabel("Cluster Label", fontsize=12)

        # Add average silhouette score line
        avg_score = np.mean(silhouette_vals)
        ax.axvline(
            x=avg_score, color="red", linestyle="--", label=f"Average: {avg_score:.3f}"
        )
        ax.legend()

        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(features_clean) + (n_clusters + 1) * 10])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def compare_methods(self, save_path: Optional[str] = None):
        """
        Compare different clustering methods based on metrics history

        Args:
            save_path: Path to save comparison plot
        """
        if not self.metrics_history:
            print("No metrics history available")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        methods = [m["method"] for m in self.metrics_history]

        # Silhouette scores
        ax = axes[0, 0]
        silhouette_scores = [m["silhouette"] for m in self.metrics_history]
        bars = ax.bar(methods, silhouette_scores)
        ax.set_title("Silhouette Score (Higher is Better)", fontsize=12)
        ax.set_ylabel("Score")
        ax.set_ylim(-0.1, 1.0)

        # Color bars based on score
        for bar, score in zip(bars, silhouette_scores):
            if score > 0.5:
                bar.set_color("green")
            elif score > 0.25:
                bar.set_color("orange")
            else:
                bar.set_color("red")

        # Davies-Bouldin scores
        ax = axes[0, 1]
        db_scores = [m["davies_bouldin"] for m in self.metrics_history]
        bars = ax.bar(methods, db_scores)
        ax.set_title("Davies-Bouldin Index (Lower is Better)", fontsize=12)
        ax.set_ylabel("Score")

        # Calinski-Harabasz scores
        ax = axes[1, 0]
        ch_scores = [m["calinski_harabasz"] for m in self.metrics_history]
        ax.bar(methods, ch_scores)
        ax.set_title("Calinski-Harabasz Index (Higher is Better)", fontsize=12)
        ax.set_ylabel("Score")

        # Number of clusters
        ax = axes[1, 1]
        n_clusters = [m["n_clusters"] for m in self.metrics_history]
        n_outliers = [m["n_outliers"] for m in self.metrics_history]

        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width / 2, n_clusters, width, label="Clusters")
        ax.bar(x + width / 2, n_outliers, width, label="Outliers")
        ax.set_title("Number of Clusters and Outliers", fontsize=12)
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def print_evaluation_summary(self, metrics: Dict[str, float]):
        """
        Print a formatted summary of evaluation metrics

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 50)
        print(f"Clustering Evaluation Summary - {metrics['method']}")
        print("=" * 50)
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Number of outliers: {metrics['n_outliers']}")
        print(f"\nQuality Metrics:")
        print(
            f"  Silhouette Score: {metrics['silhouette']:.3f} (range: [-1, 1], higher is better)"
        )
        print(
            f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.3f} (lower is better)"
        )
        print(
            f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.1f} (higher is better)"
        )

        if "avg_cluster_size" in metrics:
            print(f"\nCluster Size Statistics:")
            print(f"  Average size: {metrics['avg_cluster_size']:.1f}")
            print(f"  Std deviation: {metrics['std_cluster_size']:.1f}")
            print(f"  Min size: {metrics['min_cluster_size']}")
            print(f"  Max size: {metrics['max_cluster_size']}")

        # Provide interpretation
        print("\nInterpretation:")
        if metrics["silhouette"] > 0.5:
            print("  ✓ Good cluster separation (high silhouette score)")
        elif metrics["silhouette"] > 0.25:
            print("  ~ Moderate cluster separation")
        else:
            print("  ✗ Poor cluster separation (low silhouette score)")

        if metrics["davies_bouldin"] < 1.0:
            print("  ✓ Clusters are well-separated (low DB index)")
        else:
            print("  ✗ Clusters may be overlapping (high DB index)")

        print("=" * 50 + "\n")


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans, DBSCAN
    import os

    print("Testing ClusterEvaluator with synthetic data...")

    # Generate synthetic data
    X, y_true = make_blobs(
        n_samples=300, n_features=20, centers=4, cluster_std=1.0, random_state=42
    )

    # Initialize evaluator
    evaluator = ClusterEvaluator()

    # Test with K-Means
    print("\nEvaluating K-Means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    kmeans_metrics = evaluator.evaluate_all_metrics(X, kmeans_labels, "K-Means")
    evaluator.print_evaluation_summary(kmeans_metrics)

    # Test with DBSCAN
    print("\nEvaluating DBSCAN clustering...")
    dbscan = DBSCAN(eps=3.0, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    dbscan_metrics = evaluator.evaluate_all_metrics(X, dbscan_labels, "DBSCAN")
    evaluator.print_evaluation_summary(dbscan_metrics)

    # Create output directory
    os.makedirs("results/clustering_yolo", exist_ok=True)

    # Test silhouette analysis plot
    print("\nCreating silhouette analysis plot...")
    evaluator.plot_silhouette_analysis(
        X, kmeans_labels, save_path="results/clustering_yolo/silhouette_analysis.png"
    )

    # Compare methods
    print("\nComparing clustering methods...")
    evaluator.compare_methods(
        save_path="results/clustering_yolo/methods_comparison.png"
    )

    print("\nAll tests completed successfully!")
