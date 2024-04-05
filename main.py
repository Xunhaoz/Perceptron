import copy
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.DataLoader import DataLoader
from utils.metrics import mean_square_error, root_mean_square_error, accuracy_for_regression_iris
from utils.loss import mean_square_error_loss
from ml.ModelComponents import Linear, Model, Relu

LEARNING_RATE = 0.0001
BATCH_SIZE = 4

model = Model([
    Linear(4, 8),
    Relu(),
    Linear(8, 8),
    Relu(),
    Linear(8, 1),
])

iris = pd.read_csv('dataset/iris.csv')
train, test = train_test_split(iris, train_size=0.8)
train_loader = DataLoader(train, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)


def train(data_loader, train=True):
    avg_mean_square_error, avg_root_mean_square_error, avg_accuracy = 0, 0, 0
    for X, y in copy.deepcopy(data_loader):
        pred = model.forward(X)

        if train:
            model.backward(mean_square_error_loss(y, pred), learning_rate=LEARNING_RATE)

        avg_mean_square_error += mean_square_error(y, pred)
        avg_root_mean_square_error += root_mean_square_error(y, pred)
        avg_accuracy += accuracy_for_regression_iris(y, pred)

    return avg_mean_square_error / data_loader.__len__(), \
           avg_root_mean_square_error / data_loader.__len__(), \
           avg_accuracy / data_loader.__len__()


train_mse, train_rmse, train_acc = [], [], []
valid_mse, valid_rmse, valid_acc = [], [], []
for epoch in tqdm(range(100)):
    mse, rmse, acc = train(train_loader, train=True)
    train_mse.append(mse)
    train_rmse.append(rmse)
    train_acc.append(acc)

    mse, rmse, acc = train(test_loader, train=True)
    valid_mse.append(mse)
    valid_rmse.append(rmse)
    valid_acc.append(acc)

df = pd.DataFrame({
    'mean_square_error': train_mse,
    'root_mean_square_error': train_rmse,
    'avg_accuracy': train_acc,
    'valid_mean_square_error': valid_mse,
    'valid_root_mean_square_error': valid_rmse,
    'valid_avg_accuracy': valid_acc
})
df.to_csv("training_log.csv", index=False)

import matplotlib.pyplot as plt

df[2:].plot()
plt.show()

print(df)
