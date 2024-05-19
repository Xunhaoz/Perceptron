import numpy as np


def mean_square_error_loss(y_true, y_pred):
    return - 2 * (y_true - y_pred)


def cross_entropy_loss(y_true, y_pred):
    # y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return y_pred - y_true
