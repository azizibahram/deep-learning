from typing import Callable

import numpy as np
import pandas as pd

from src.gd.gradient_descent_base import (
    GradientDescentBase,
)

__all__ = ["StochasticGradientDescent"]


class StochasticGradientDescent(GradientDescentBase):
    def __init__(
        self,
        w_start: np.ndarray = None,
        learning_rate: Callable = lambda k: 0.001,
        regularization: Callable = lambda w: 0.0,
        alpha: float = 1e-2,
    ):
        super().__init__(
            w_start=w_start,
            learning_rate=learning_rate,
            regularization=regularization,
            alpha=alpha,
        )

    def _calculate_gradient(self, X, y, w):
        sample_size = y.size
        index = np.random.randint(0, sample_size)
        if isinstance(y, pd.core.series.Series):
            y = y.to_numpy()

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.to_numpy()

        X_batch = X[index]
        y_batch = y[index]

        return 2 * np.dot(X_batch.T, np.dot(X_batch, w) - y_batch)
