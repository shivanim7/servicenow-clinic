from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train_decision_tree(embeddings, labels, test_size=0.3):
    """Splits the data, trains a Decision Tree, and returns the model and accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=8)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, X_train, X_test, y_train, y_test

def reduce_features(model, embeddings, threshold=0.025):
    """Reduces embeddings by removing features with low importance."""
    feature_importances = model.feature_importances_
    zero_importance_indices = np.where(feature_importances < threshold)[0]
    embeddings_small = np.delete(embeddings, zero_importance_indices, axis=1)
    return embeddings_small, zero_importance_indices
