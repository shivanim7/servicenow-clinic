from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def determine_optimal_clusters(embeddings, max_k=100):
    """Calculates WCSS and Silhouette Scores for determining optimal K."""
    wcss = []
    silhouette_scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))
    return wcss, silhouette_scores

def plot_elbow(wcss, max_k=100):
    """Plots the Elbow Method graph."""
    plt.plot(range(2, max_k), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.show()

def plot_silhouette(silhouette_scores, max_k=100):
    """Plots Silhouette Score graph."""
    plt.plot(range(2, max_k), silhouette_scores, marker='o')
    plt.title('Silhouette Score vs. Number of Clusters (K)')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()

def perform_clustering(embeddings, num_clusters):
    """Performs K-Means clustering."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels

def visualize_clusters(embeddings, labels):
    """Visualizes clusters using PCA."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.title('K-Means Clustering of Sentence Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
