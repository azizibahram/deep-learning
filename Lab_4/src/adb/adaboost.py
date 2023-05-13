import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from src.dt.decision_tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Callable


__all__ = ["AdaBoost"]


class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, M=10):
        self.M = M

    def fit(self, X, y):
        # Initialize the weights
        w = np.ones(len(y)) / len(y)

        self.classifiers_ = []
        self.alphas_ = []

        for m in range(self.M):
            # Train a weak classifier
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, sample_weight=w)

            # Calculate the weighted error rate
            y_pred = clf.predict(X)
            err = w.dot(y_pred != y) / w.sum()

            # Update the weights of the samples
            alpha = np.log((1 - err) / err)
            w *= np.exp(alpha * (y_pred != y))

            # Store the classifier and its weight
            self.classifiers_.append(clf)
            self.alphas_.append(alpha)

        return self

    def predict(self, X):
        # Make predictions using each classifier
        y_pred = np.zeros(len(X))
        for clf, alpha in zip(self.classifiers_, self.alphas_):
            y_pred += alpha * clf.predict(X)

        # Take a weighted vote of the predictions
        y_pred = np.sign(y_pred)

        return y_pred
