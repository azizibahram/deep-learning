import numpy as np

__all__ = ["KNeighborsClassifier"]


def most_common(lst):
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# from sklearn.neighbors import KDTree
# import numpy as np
# from collections import Counter
# from scipy.stats import mode
# from sklearn.base import BaseEstimator, ClassifierMixin

# __all__ = ["KNeighborsClassifier"]


# class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, k=3):
#         self.k = k

#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y
#         self.tree = KDTree(X)

#     def predict(self, X):
#         y_pred = [self._predict(x) for x in X]
#         return np.array(y_pred)

#     def _predict(self, x):
#         # Find indices of the k nearest neighbors
#         k_idx = self.tree.query([x], k=self.k, return_distance=False)[0]

#         # Extract the labels of the k nearest neighbors
#         k_neighbor_labels = [self.y_train[i] for i in k_idx]

#         # Return the most common class label
#         most_common = Counter(k_neighbor_labels).most_common(1)
#         return most_common[0][0]

#     def set_params(self, **params):
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self

#     def get_params(self, deep=True):
#         return {"k": self.k}
