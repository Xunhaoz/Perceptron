import numpy as np


def mean_square_error(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2)


def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.average((y_true - y_pred) ** 2))


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))


def accuracy(y_true, y_pred):
    return np.average(y_true == y_pred)


def accuracy_for_regression_iris(y_true, y_pred):
    return np.average((y_true + 0.5 > y_pred) & (y_true - 0.5 < y_pred))


def accuracy_for_classification_iris(y_true, y_pred):
    return np.average(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
