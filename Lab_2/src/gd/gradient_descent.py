from typing import Callable

import numpy as np

from src.gd.gradient_descent_base import (
    GradientDescentBase,
)

__all__ = ["GradientDescent"]


class GradientDescent(GradientDescentBase):
    def __init__(
        self,
        w_start: np.ndarray = None,
        learning_rate: Callable = lambda k: 0.001,
        gradient: Callable = lambda X, y, w: (2 / y.size)
        * np.dot(X.T, np.dot(X, w) - y),
        regularization: Callable = lambda w: 0.0,
        alpha: float = 1e-2,
    ):
        super().__init__(
            w_start=w_start,
            learning_rate=learning_rate,
            regularization=regularization,
            alpha=alpha,
        )
        self._gradient = gradient

    def _calculate_gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        return self._gradient(X, y, w)
