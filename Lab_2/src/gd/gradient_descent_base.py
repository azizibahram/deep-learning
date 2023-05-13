from typing import Callable

import numpy as np

from abc import abstractmethod, ABC

__all__ = ["GradientDescentBase"]


class GradientDescentBase(ABC):
    def __init__(
        self,
        w_start: np.ndarray,
        learning_rate: Callable,
        regularization: Callable,
        alpha: float,
    ):
        self._learning_rate = learning_rate
        self._w = w_start
        self._regularization = regularization
        self._alpha = alpha

    def get_weights(self):
        return self._w

    def set_weights(self, w: np.ndarray):
        self._w = w

    def update_weights(
        self, X: np.ndarray, y: np.ndarray, iteration: int
    ) -> np.ndarray:
        self._w = (
            self._w
            - self._learning_rate(iteration) *
            self._calculate_gradient(X, y, self._w)
            + self._alpha * self._regularization(self._w)
        )
        return self._w

    @abstractmethod
    def _calculate_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        pass
