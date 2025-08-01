# src/clustering/clustering_methods.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import warnings

warnings.filterwarnings("ignore")


class CharacterClusterer:
    """
    Class for clustering character features using various algorithms
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the clusterer

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.labels_ = None
        self.model_ = None

    def preprocess_features(
        self, features: np.ndarray, scale: bool = True
    ) -> np.ndarray:
        """
        Preprocess features before clustering

        Args:
            features: Feature matrix (n_samples, n_features)
            scale: Whether to standardize features

        Returns:
            Preprocessed features
        """
        # Remove any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize features if requested
        if scale:
            features = self.scaler.fit_transform(features)

        return features

    def kmeans_clustering(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        n_init: int = 10,
        max_iter: int = 300,
        scale: bool = True,
    ) -> np.ndarray:
        """
        Perform K-Means clustering

        Args:
            features: Feature matrix
            n_clusters: Number of clusters
            n_init: Number of times to run K-means with different seeds
            max_iter: Maximum iterations
            scale: Whether to standardize features

        Returns:
            Cluster labels
        """
        print(f"Running K-Means clustering with {n_clusters} clusters...")

        # Preprocess features
        features = self.preprocess_features(features, scale)

        # Initialize and fit K-Means
        self.model_ = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state,
            verbose=0,
        )

        self.labels_ = self.model_.fit_predict(features)

        # Calculate inertia (within-cluster sum of squares)
        inertia = self.model_.inertia_
        print(f"K-Means complete. Inertia: {inertia:.2f}")

        return self.labels_

    def dbscan_clustering(
        self,
        features: np.ndarray,
        eps: Optional[float] = None,
        min_samples: int = 5,
        scale: bool = True,
        auto_eps: bool = True,
    ) -> np.ndarray:
        """
        Perform DBSCAN clustering

        Args:
            features: Feature matrix
            eps: Maximum distance between samples in a neighborhood
            min_samples: Minimum samples in a neighborhood
            scale: Whether to standardize features
            auto_eps: Automatically determine eps using k-distance graph

        Returns:
            Cluster labels (-1 indicates outliers)
        """
        print("Running DBSCAN clustering...")

        # Preprocess features
        features = self.preprocess_features(features, scale)

        # Automatically determine eps if requested
        if auto_eps and eps is None:
            eps = self._estimate_eps(features, min_samples)
            print(f"Automatically determined eps: {eps:.3f}")
        elif eps is None:
            eps = 0.5  # Default value

        # Initialize and fit DBSCAN
        self.model_ = DBSCAN(
            eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1
        )

        self.labels_ = self.model_.fit_predict(features)

        # Report results
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_outliers = list(self.labels_).count(-1)

        print(f"DBSCAN complete. Found {n_clusters} clusters and {n_outliers} outliers")

        return self.labels_

    def hierarchical_clustering(
        self,
        features: np.ndarray,
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        linkage: str = "ward",
        scale: bool = True,
    ) -> np.ndarray:
        """
        Perform Hierarchical/Agglomerative clustering

        Args:
            features: Feature matrix
            n_clusters: Number of clusters (if None, use distance_threshold)
            distance_threshold: Distance threshold for clustering
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            scale: Whether to standardize features

        Returns:
            Cluster labels
        """
        print(f"Running Hierarchical clustering with {linkage} linkage...")

        # Preprocess features
        features = self.preprocess_features(features, scale)

        # Reduce dimensionality if too high (for computational efficiency)
        if features.shape[1] > 100:
            print(
                f"Reducing dimensionality from {features.shape[1]} to 100 using PCA..."
            )
            pca = PCA(n_components=100, random_state=self.random_state)
            features = pca.fit_transform(features)

        # Initialize and fit Agglomerative Clustering
        self.model_ = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage,
            compute_distances=True,  # For dendrogram
        )

        self.labels_ = self.model_.fit_predict(features)

        n_clusters_found = len(set(self.labels_))
        print(f"Hierarchical clustering complete. Found {n_clusters_found} clusters")

        return self.labels_

    def find_optimal_clusters_elbow(
        self,
        features: np.ndarray,
        k_range: range = range(2, 21),
        scale: bool = True,
        plot: bool = True,
    ) -> Dict[str, any]:
        """
        Find optimal number of clusters using elbow method for K-Means

        Args:
            features: Feature matrix
            k_range: Range of k values to test
            scale: Whether to standardize features
            plot: Whether to plot the elbow curve

        Returns:
            Dictionary with k values, inertias, and suggested k
        """
        print("Finding optimal clusters using elbow method...")

        # Preprocess features
        features = self.preprocess_features(features, scale)

        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(features)
            inertias.append(kmeans.inertia_)

            # Calculate silhouette score
            if k < len(features):
                sil_score = silhouette_score(features, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(-1)

        # Find elbow point (using second derivative)
        optimal_k = self._find_elbow_point(list(k_range), inertias)

        # Plot if requested
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Elbow plot
            ax1.plot(k_range, inertias, "bo-")
            ax1.axvline(
                x=optimal_k, color="r", linestyle="--", label=f"Elbow at k={optimal_k}"
            )
            ax1.set_xlabel("Number of Clusters (k)")
            ax1.set_ylabel("Inertia")
            ax1.set_title("Elbow Method")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Silhouette plot
            ax2.plot(k_range, silhouette_scores, "go-")
            best_k_sil = list(k_range)[np.argmax(silhouette_scores)]
            ax2.axvline(
                x=best_k_sil,
                color="r",
                linestyle="--",
                label=f"Best silhouette at k={best_k_sil}",
            )
            ax2.set_xlabel("Number of Clusters (k)")
            ax2.set_ylabel("Silhouette Score")
            ax2.set_title("Silhouette Score vs k")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "results/clustering_yolo/elbow_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        return {
            "k_values": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "optimal_k_elbow": optimal_k,
            "optimal_k_silhouette": list(k_range)[np.argmax(silhouette_scores)],
        }

    def find_optimal_clusters_silhouette(
        self, features: np.ndarray, k_range: range = range(2, 21), scale: bool = True
    ) -> Tuple[int, List[float]]:
        """
        Find optimal number of clusters using silhouette score

        Args:
            features: Feature matrix
            k_range: Range of k values to test
            scale: Whether to standardize features

        Returns:
            Tuple of (optimal_k, silhouette_scores)
        """
        print("Finding optimal clusters using silhouette score...")

        # Preprocess features
        features = self.preprocess_features(features, scale)

        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(features)

            if k < len(features):
                score = silhouette_score(features, labels)
                silhouette_scores.append(score)
                print(f"  k={k}: silhouette score = {score:.3f}")
            else:
                silhouette_scores.append(-1)

        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        print(f"\nOptimal k based on silhouette score: {optimal_k}")

        return optimal_k, silhouette_scores

    def find_optimal_clusters_davies_bouldin(
        self, features: np.ndarray, k_range: range = range(2, 21), scale: bool = True
    ) -> Tuple[int, List[float]]:
        """
        Find optimal number of clusters using Davies-Bouldin index

        Args:
            features: Feature matrix
            k_range: Range of k values to test
            scale: Whether to standardize features

        Returns:
            Tuple of (optimal_k, davies_bouldin_scores)
        """
        print("Finding optimal clusters using Davies-Bouldin index...")

        # Preprocess features
        features = self.preprocess_features(features, scale)

        db_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(features)

            if k < len(features):
                score = davies_bouldin_score(features, labels)
                db_scores.append(score)
                print(f"  k={k}: Davies-Bouldin index = {score:.3f}")
            else:
                db_scores.append(float("inf"))

        optimal_k = list(k_range)[np.argmin(db_scores)]
        print(f"\nOptimal k based on Davies-Bouldin index: {optimal_k}")

        return optimal_k, db_scores

    def _estimate_eps(self, features: np.ndarray, k: int) -> float:
        """
        Estimate eps parameter for DBSCAN using k-distance graph

        Args:
            features: Feature matrix
            k: Number of nearest neighbors

        Returns:
            Estimated eps value
        """
        from sklearn.neighbors import NearestNeighbors

        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(features)

        # Get distances to k-th nearest neighbor
        distances, _ = nn.kneighbors(features)
        k_distances = distances[:, -1]

        # Sort distances
        k_distances = np.sort(k_distances)

        # Find elbow point in k-distance graph
        # Use 90th percentile as a robust estimate
        eps = np.percentile(k_distances, 90)

        return eps

    def _find_elbow_point(self, x: List[float], y: List[float]) -> float:
        """
        Find elbow point in a curve using the kneedle algorithm

        Args:
            x: x values
            y: y values

        Returns:
            x value at elbow point
        """
        # Normalize data
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Calculate differences
        diffs = []
        for i in range(1, len(x_norm) - 1):
            # Calculate angle
            v1 = np.array([x_norm[i] - x_norm[0], y_norm[i] - y_norm[0]])
            v2 = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])

            # Calculate distance from point to line
            diff = np.abs(np.cross(v1, v2)) / np.linalg.norm(v2)
            diffs.append(diff)

        # Find maximum difference
        if diffs:
            elbow_idx = np.argmax(diffs) + 1
            return x[elbow_idx]
        else:
            return x[len(x) // 2]  # Default to middle

    def get_cluster_summary(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Get summary statistics for clusters

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary with cluster statistics
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])  # Exclude outliers

        summary = {
            "n_clusters": n_clusters,
            "n_outliers": np.sum(labels == -1),
            "cluster_sizes": {},
            "cluster_centers": {},
            "cluster_variances": {},
        }

        for label in unique_labels:
            if label == -1:  # Skip outliers
                continue

            cluster_mask = labels == label
            cluster_features = features[cluster_mask]

            summary["cluster_sizes"][int(label)] = int(np.sum(cluster_mask))
            summary["cluster_centers"][int(label)] = np.mean(
                cluster_features, axis=0
            ).tolist()[
                :10
            ]  # First 10 dims
            summary["cluster_variances"][int(label)] = float(
                np.mean(np.var(cluster_features, axis=0))
            )

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data for testing
    from sklearn.datasets import make_blobs
    import os

    print("Testing CharacterClusterer with synthetic data...")

    # Generate synthetic character features
    n_samples = 500
    n_features = 50
    n_clusters_true = 5

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters_true,
        cluster_std=1.0,
        random_state=42,
    )

    # Initialize clusterer
    clusterer = CharacterClusterer(random_state=42)

    # Test K-Means clustering
    print("\n" + "=" * 50)
    print("Testing K-Means Clustering")
    print("=" * 50)
    kmeans_labels = clusterer.kmeans_clustering(X, n_clusters=5)
    print(f"Unique labels: {np.unique(kmeans_labels)}")

    # Test optimal cluster finding
    print("\n" + "=" * 50)
    print("Finding Optimal Clusters")
    print("=" * 50)

    # Create output directory
    os.makedirs("results/clustering_yolo", exist_ok=True)

    optimal_results = clusterer.find_optimal_clusters_elbow(
        X, k_range=range(2, 11), plot=True
    )
    print(f"Optimal k (elbow): {optimal_results['optimal_k_elbow']}")
    print(f"Optimal k (silhouette): {optimal_results['optimal_k_silhouette']}")

    # Test DBSCAN clustering
    print("\n" + "=" * 50)
    print("Testing DBSCAN Clustering")
    print("=" * 50)
    dbscan_labels = clusterer.dbscan_clustering(X, auto_eps=True)
    print(f"Unique labels: {np.unique(dbscan_labels)}")

    # Test Hierarchical clustering
    print("\n" + "=" * 50)
    print("Testing Hierarchical Clustering")
    print("=" * 50)
    hier_labels = clusterer.hierarchical_clustering(X, n_clusters=5)
    print(f"Unique labels: {np.unique(hier_labels)}")

    # Get cluster summary
    print("\n" + "=" * 50)
    print("Cluster Summary for K-Means")
    print("=" * 50)
    summary = clusterer.get_cluster_summary(X, kmeans_labels)
    print(f"Number of clusters: {summary['n_clusters']}")
    print(f"Cluster sizes: {summary['cluster_sizes']}")

    print("\nAll tests completed successfully!")
