# MARK: imports 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import time


# MARK: 1. Data Preprocessing
print("Mark 1")
start_time = time.time()
# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32') / 255.0 
y = mnist.target.astype('int')

X_sample = X[:10000]
y_sample = y[:10000]
print(f"Sample size: {len(X_sample)} images")

# MARK: 2. K-Means Clustering
print("Mark 2")
start_time = time.time()
# determining optimal k using the Elbow Method
inertia = []
silhouette_scores = []
k_values = range(2, 16)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_sample)
    inertia.append(kmeans.inertia_)
    silhouette = silhouette_score(X_sample, kmeans.labels_)
    silhouette_scores.append(silhouette)
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette:.3f}")


# MARK: 3. Plotting
print("Mark 3")
# plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for different k')
plt.show()

print("\nStep 4: Performing final clustering with optimal k...")
start_time = time.time()

# choosing an optimal k
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_sample)

# MARK: 4. PCA
print("Mark 4")
start_time = time.time()
# using PCA to reduce high-dim data to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)
print(f"PCA completed in {time.time() - start_time:.2f} seconds")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

print("\nStep 6: Creating visualizations...")

# plot clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='tab10', s=10)
plt.colorbar(scatter, label="Cluster Label") 
plt.title('K-Means Clustering of MNIST (PCA-reduced data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# visualize cluster centers (mean images)
cluster_centers = kmeans.cluster_centers_

fig, axes = plt.subplots(1, optimal_k, figsize=(20, 4))
for i, ax in enumerate(axes.flat):
    center_image = cluster_centers[i].reshape(28, 28)
    ax.imshow(center_image, cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')
plt.suptitle('Cluster Centers as Images')
plt.show()


