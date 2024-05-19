import copy

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from utils.DataLoader import DataLoader
from utils.metrics import root_mean_square_error, accuracy_for_regression_iris, cross_entropy
from utils.loss import cross_entropy_loss
from ml.ModelComponents import Linear, Model, Relu, Softmax


def train(data_loader):
    avg_cross_entropy, avg_root_mean_square_error, avg_accuracy, = 0, 0, 0
    for X, y in copy.deepcopy(data_loader):
        pred = model.forward(X)

        model.backward(cross_entropy_loss(y, pred), learning_rate=LEARNING_RATE)

        avg_cross_entropy += cross_entropy(y, pred)
        avg_root_mean_square_error += root_mean_square_error(y, pred)
        avg_accuracy += accuracy_for_regression_iris(y, pred)

    return avg_cross_entropy / data_loader.__len__(), \
           avg_root_mean_square_error / data_loader.__len__(), \
           avg_accuracy / data_loader.__len__()


LEARNING_RATE = 0.01
BATCH_SIZE = 2
EPOCH = 100

model = Model([
    Linear(4, 10),
    Relu(),
    Linear(10, 3)
])

data = pd.concat([
    pd.read_csv('dataset/iris_in.csv', header=None),
    pd.read_csv('dataset/iris_out.csv', header=None)], axis=1
)

train_loader = DataLoader(data[:75], batch_size=BATCH_SIZE, one_hot_encode=True)
test_loader = DataLoader(data[-75:], batch_size=BATCH_SIZE, one_hot_encode=True)

train_ce = []
train_rmse = []
train_acc = []
for epoch in tqdm(range(EPOCH)):
    ce, rmse, acc = train(train_loader)
    train_ce.append(ce)
    train_rmse.append(rmse)
    train_acc.append(acc)

mse, rmse, acc = train(test_loader)
print(f"Test CE: {mse}, RMSE: {rmse}, Accuracy: {acc}")

df = pd.DataFrame({
    'cross_entropy': train_ce[5:],
    'root_mean_square_error': train_rmse[5:],
    'avg_accuracy': train_acc[5:],
})
df.to_csv("training_log.csv", index=False)

df.plot()
plt.show()
