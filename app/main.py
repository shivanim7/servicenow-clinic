import pandas as pd
from embedding import load_embedding_model, encode_text
from clustering import (
    determine_optimal_clusters,
    plot_elbow,
    plot_silhouette,
    perform_clustering,
    visualize_clusters,
)
from decision_tree import train_decision_tree, reduce_features

# Load Data
df = pd.read_csv("../IMDB.csv")
df = df.sample(frac=0.05)
text_col = "review"
label_col = "sentiment"

# Step 1: Load embeddings
embedding_model = load_embedding_model()
embeddings = encode_text(embedding_model, df[text_col].tolist())

# Step 2: Clustering on full embeddings
wcss, silhouette_scores = determine_optimal_clusters(embeddings)
plot_elbow(wcss)
plot_silhouette(silhouette_scores)

num_clusters = 5
kmeans, labels = perform_clustering(embeddings, num_clusters)
df["cluster"] = labels
visualize_clusters(embeddings, labels)

# Step 3: Decision Tree for Feature Selection
model, accuracy, X_train, X_test, y_train, y_test = train_decision_tree(embeddings, df[label_col])
print(f"Initial Accuracy: {accuracy}")
embeddings_small, zero_importance_indices = reduce_features(model, embeddings)

# Step 4: Clustering on reduced embeddings
wcss_small, silhouette_scores_small = determine_optimal_clusters(embeddings_small)
plot_elbow(wcss_small)
plot_silhouette(silhouette_scores_small)

kmeans_small, labels_small = perform_clustering(embeddings_small, num_clusters)
df["cluster_small"] = labels_small
visualize_clusters(embeddings_small, labels_small)

# Output Results
print(df.head())
