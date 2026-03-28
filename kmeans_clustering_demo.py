"""
KMeans Clustering Demo with Example

This script demonstrates KMeans clustering:
- Generates a synthetic 2D dataset with 3 clusters using make_blobs
- Fits KMeans with k=3
- Visualizes the data points colored by cluster labels
- Prints centroids and inertia (within-cluster sum of squares)

Requirements: scikit-learn, numpy, matplotlib
Run: python kmeans_clustering_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic dataset
# 300 samples, 3 centers, random state for reproducibility
X, true_labels = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.0)

# Step 2: Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Step 3: Print results
print("KMeans Clustering Results:")
print(f"Centroids:\\n{kmeans.cluster_centers_}")
print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")

# Step 4: Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data (true labels)
ax1.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
ax1.set_title('Original Data (True Clusters)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Clustered data
ax2.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3, label='Centroids')
ax2.set_title('KMeans Clusters (k=3)')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend()

plt.tight_layout()
plt.savefig('kmeans_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\nPlot saved as 'kmeans_demo.png'")
print("\\nKMeans snippet ready for reuse:")
print("""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(your_data)
print(kmeans.cluster_centers_)
""")

