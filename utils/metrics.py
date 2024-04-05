import numpy as np


def mean_square_error(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2)


def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.average((y_true - y_pred) ** 2))


def accuracy(y_true, y_pred):
    return np.average(y_true == y_pred)


def accuracy_for_regression_iris(y_true, y_pred):
    return np.average((y_true + 0.5 > y_pred) & (y_true - 0.5 < y_pred))
