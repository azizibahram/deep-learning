from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
import numpy as np


__all__ = ["KMeans"]


class KMeans(BaseEstimator, TransformerMixin, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None):
        self.labels_, self.cluster_centers_ = self.kmeans(
            X, self.n_clusters, self.max_iter)
        return self

    def transform(self, X):
        # Convert the DataFrame to a NumPy array
        X = X.to_numpy()

        distances = np.linalg.norm(
            X[:, np.newaxis] - self.cluster_centers_, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def predict(self, X):
        return self.transform(X)

    def kmeans(self, X, n_clusters=8, max_iter=300):
        # Convert the DataFrame to a NumPy array
        X = X.to_numpy()

        # Initialize cluster centers
        centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

        for _ in range(max_iter):
            # Assign data points to closest cluster center
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.array([X[labels == i].mean(axis=0)
                                    for i in range(n_clusters)])

            # Check for convergence
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

            return labels, centers
