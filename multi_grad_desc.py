import numpy as np
from typing import Tuple
from scipy.stats import skewnorm

# limit np print precision to 2
np.set_printoptions(precision=2)


def multi_gradient_descent(X, y, lr, epochs):
    m, n = X.shape

    # print("X:\n", X)

    w = np.zeros(n)

    # print("weights:\n", w)

    b = 0
    for _ in range(epochs):
        y_hat = w @ X.T + b

        # print("y_hat:\n", y_hat)

        error = y_hat - y

        # print("error:\n", error)

        dw = error @ X / m

        # print("dw:\n", dw)

        db = np.sum(error) / m
        w = w - lr * dw

        # print("new w:\n", w)

        b = b - lr * db
    return w, b


def get_bounded_normal(shape: Tuple[int, int], bound: Tuple[int, int] = (0, 1)):
    """return 1d normal distribution bounded between 0 and 1"""
    dist = np.random.normal(0, 1, shape)
    return np.interp(dist, (dist.min(), dist.max()), bound)


def get_bounded_skewed(shape: Tuple[int, int], bound: Tuple[int, int] = (0, 1), skew_param: int = 100):
    """return 1d skewed distribution bounded between 0 and 1"""
    dist = skewnorm.rvs(skew_param, size=shape)
    return np.interp(dist, (dist.min(), dist.max()), bound)
