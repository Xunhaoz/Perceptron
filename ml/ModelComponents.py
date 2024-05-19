import numpy as np
import copy


class Linear:
    def __init__(self, input_dim, output_dim):
        self.weight = np.random.rand(input_dim, output_dim)
        self.bias = np.random.rand(1, output_dim)
        self.last_X = None

    def forward(self, X):
        self.last_X = copy.deepcopy(X)
        return np.dot(X, self.weight) + self.bias

    def backward(self, gradient, learning_rate):
        gradient_weights = np.dot(self.last_X.T, gradient) / gradient.shape[0]
        gradient_bias = np.average(gradient, axis=0, keepdims=True)
        gradient = np.dot(gradient, self.weight.T)
        self.weight -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias
        return gradient


class Sigmoid:
    def __init__(self):
        self.last_X = None
        self.next_X = None

    def forward(self, X):
        self.last_X = copy.deepcopy(X)
        self.next_X = 1 / (1 + np.exp(-np.clip(X, -500, 500)))
        return self.next_X

    def backward(self, gradient, learning_rate):
        return gradient * self.next_X * (1 - self.next_X)


class Relu:
    def __init__(self):
        self.last_X = None

    def forward(self, X):
        self.last_X = X
        return np.maximum(0, X)

    def backward(self, gradient, learning_rate):
        gradient[self.last_X < 0] = 0
        return gradient


class Softmax:
    def __init__(self):
        self.last_X = None

    def forward(self, X):
        self.last_X = X
        shift_X = X - np.max(X, axis=-1, keepdims=True)
        softmax_output = np.exp(shift_X) / np.sum(np.exp(shift_X), axis=-1, keepdims=True)
        return softmax_output

    def backward(self, gradient, learning_rate):
        return gradient * self.last_X


class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, gradient, learning_rate=None):
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient, learning_rate)
        return gradient
