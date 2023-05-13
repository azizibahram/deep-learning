from typing import Callable

import numpy as np
from sklearn.metrics import mean_squared_error

from src.gd.gradient_descent_base import (
    GradientDescentBase,
)

__all__ = ["LinearRegression"]


class LinearRegression:
    def __init__(
        self,
        descent: GradientDescentBase,
        tolerance: float,
        max_iter: int,
        loss: Callable = mean_squared_error,
    ):
        self._optimizer = descent
        self._tolerance = tolerance
        self._max_iter = max_iter
        self._loss = loss
        self._loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        _X = LinearRegression.__add_ones_column(X)
        if self._optimizer.get_weights() is None:
            self._optimizer.set_weights(np.zeros(_X.shape[1]))

        for i in range(self._max_iter):
            self.__update_loss_history(X, y)
            prev_w = self._optimizer.get_weights()
            self._optimizer.update_weights(_X, y, i)

            if (
                np.linalg.norm(self._optimizer.get_weights() - prev_w) ** 2
                < self._tolerance
            ):
                break
        return self

    def predict(self, X: np.ndarray):
        val = LinearRegression.__add_ones_column(
            X) @ self._optimizer.get_weights()
        return val

    def get_weights(self):
        return self._optimizer.get_weights()

    def __update_loss_history(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        self._loss_history.append(self._loss(y, y_pred))

    @staticmethod
    def __add_ones_column(X: np.ndarray):
        return np.append(np.ones(X.shape[0])[:, None], X, 1)
