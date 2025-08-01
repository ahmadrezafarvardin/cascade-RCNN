# src/clustering/run_complete_pipeline.py
"""
Complete clustering pipeline example
Demonstrates the full workflow from feature extraction to visualization
"""

import sys
from pathlib import Path
import numpy as np
import json
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import (
    extract_features_from_dataset_yolo,
    CharacterClusterer,
    ClusterEvaluator,
    ClusterVisualizer,
)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def run_complete_pipeline(
    model_path=None,
    dataset_path="dataset",
    output_dir="results/clustering_complete",
    max_images=200,
    conf_threshold=0.25,
):
    """
    Run the complete clustering pipeline
    """
    print("=" * 60)
    print("COMPLETE CHARACTER CLUSTERING PIPELINE")
    print("=" * 60)

    # Step 1: Feature Extraction
    print("\n" + "=" * 60)
    print("STEP 1: FEATURE EXTRACTION")
    print("=" * 60)

    features, metadata = extract_features_from_dataset_yolo(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_images=max_images,
        conf_threshold=conf_threshold,
    )

    if features is None or all(f.size == 0 for f in features.values()):
        print("Feature extraction failed!")
        return

    # Step 2: Prepare features for clustering
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE PREPARATION")
    print("=" * 60)

    # Combine HOG and statistical features
    feature_list = []
    feature_names = []

    if "hog" in features and features["hog"].size > 0:
        feature_list.append(features["hog"])
        feature_names.append("hog")

    if "statistical" in features and features["statistical"].size > 0:
        feature_list.append(features["statistical"])
        feature_names.append("statistical")

    combined_features = np.hstack(feature_list)
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Using features: {', '.join(feature_names)}")

    # Step 3: Find optimal number of clusters
    print("\n" + "=" * 60)
    print("STEP 3: OPTIMAL CLUSTER SELECTION")
    print("=" * 60)

    clusterer = CharacterClusterer()

    # Use elbow method
    elbow_results = clusterer.find_optimal_clusters_elbow(
        combined_features, k_range=range(5, 21), plot=True
    )

    optimal_k = elbow_results["optimal_k_silhouette"]
    print(f"Optimal k (silhouette): {optimal_k}")
    print(f"Optimal k (elbow): {elbow_results['optimal_k_elbow']}")

    # Step 4: Perform clustering with different methods
    print("\n" + "=" * 60)
    print("STEP 4: CLUSTERING")
    print("=" * 60)

    evaluator = ClusterEvaluator()
    visualizer = ClusterVisualizer(output_dir=output_dir)

    # Load sample images for visualization
    import cv2
    import os

    sample_dir = os.path.join(output_dir, "sample_characters_yolo")
    sample_images = []
    if os.path.exists(sample_dir):
        for img_file in sorted(os.listdir(sample_dir)):
            if img_file.endswith(".png"):
                img = cv2.imread(os.path.join(sample_dir, img_file))
                if img is not None:
                    sample_images.append(img)

    # Method 1: K-Means
    print("\n--- K-Means Clustering ---")
    kmeans_labels = clusterer.kmeans_clustering(combined_features, n_clusters=optimal_k)
    kmeans_metrics = evaluator.evaluate_all_metrics(
        combined_features, kmeans_labels, "K-Means"
    )
    evaluator.print_evaluation_summary(kmeans_metrics)

    if len(sample_images) >= len(kmeans_labels):
        visualizer.create_cluster_report(
            combined_features,
            kmeans_labels,
            sample_images[: len(kmeans_labels)],
            "kmeans",
        )

    # Method 2: DBSCAN
    print("\n--- DBSCAN Clustering ---")
    dbscan_labels = clusterer.dbscan_clustering(combined_features, auto_eps=True)
    dbscan_metrics = evaluator.evaluate_all_metrics(
        combined_features, dbscan_labels, "DBSCAN"
    )
    evaluator.print_evaluation_summary(dbscan_metrics)

    if len(sample_images) >= len(dbscan_labels):
        visualizer.create_cluster_report(
            combined_features,
            dbscan_labels,
            sample_images[: len(dbscan_labels)],
            "dbscan",
        )

    # Method 3: Hierarchical
    print("\n--- Hierarchical Clustering ---")
    hier_labels = clusterer.hierarchical_clustering(
        combined_features, n_clusters=optimal_k
    )
    hier_metrics = evaluator.evaluate_all_metrics(
        combined_features, hier_labels, "Hierarchical"
    )
    evaluator.print_evaluation_summary(hier_metrics)

    if len(sample_images) >= len(hier_labels):
        visualizer.create_cluster_report(
            combined_features,
            hier_labels,
            sample_images[: len(hier_labels)],
            "hierarchical",
        )

    # Plot dendrogram for hierarchical
    visualizer.plot_dendrogram(combined_features)

    # Step 5: Compare methods
    print("\n" + "=" * 60)
    print("STEP 5: METHOD COMPARISON")
    print("=" * 60)

    evaluator.compare_methods(
        save_path=os.path.join(output_dir, "methods_comparison.png")
    )

    # # Save final results
    # final_results = {
    #     "feature_extraction": {
    #         "n_images_processed": metadata["num_images"],
    #         "n_characters_extracted": metadata["num_characters"],
    #         "feature_types": feature_names,
    #         "combined_feature_shape": combined_features.shape,
    #     },
    #     "optimal_clusters": {
    #         "elbow_method": elbow_results["optimal_k_elbow"],
    #         "silhouette_method": optimal_k,
    #     },
    #     "clustering_results": {
    #         "kmeans": kmeans_metrics,
    #         "dbscan": dbscan_metrics,
    #         "hierarchical": hier_metrics,
    #     },
    # }
    # Save final results - convert numpy types to native Python types
    final_results = {
        "feature_extraction": {
            "n_images_processed": int(metadata["num_images"]),
            "n_characters_extracted": int(metadata["num_characters"]),
            "feature_types": feature_names,
            "combined_feature_shape": [int(x) for x in combined_features.shape],
        },
        "optimal_clusters": {
            "elbow_method": int(elbow_results["optimal_k_elbow"]),
            "silhouette_method": int(optimal_k),
        },
        "clustering_results": {
            "kmeans": convert_to_serializable(kmeans_metrics),
            "dbscan": convert_to_serializable(dbscan_metrics),
            "hierarchical": convert_to_serializable(hier_metrics),
        },
    }

    results_path = os.path.join(output_dir, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFinal results saved to {results_path}")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

    # Print summary
    print("\nSUMMARY:")
    print(f"- Processed {metadata['num_images']} images")
    print(f"- Extracted {metadata['num_characters']} characters")
    print(f"- Optimal clusters: {optimal_k}")
    print(f"- Best method (by silhouette): ", end="")

    best_method = max(
        [
            ("K-Means", kmeans_metrics["silhouette"]),
            ("DBSCAN", dbscan_metrics["silhouette"]),
            ("Hierarchical", hier_metrics["silhouette"]),
        ],
        key=lambda x: x[1],
    )
    print(f"{best_method[0]} (score: {best_method[1]:.3f})")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete character clustering pipeline"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset", help="Path to dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/clustering_complete",
        help="Output directory",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="YOLO confidence threshold"
    )

    args = parser.parse_args()

    # Use best YOLO weights if available and not specified
    if args.model is None:
        best_weights = Path(
            "results/yolo_runs/detect/character_detection/weights/best.pt"
        )
        if best_weights.exists():
            args.model = str(best_weights)
            print(f"Using best YOLO weights: {args.model}")

    run_complete_pipeline(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_images=args.max_images,
        conf_threshold=args.conf,
    )
