import copy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from utils.DataLoader import DataLoader
from utils.metrics import mean_square_error, root_mean_square_error, accuracy_for_regression_iris
from utils.loss import mean_square_error_loss
from ml.ModelComponents import Linear, Model, Relu


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


LEARNING_RATE = 0.0001
BATCH_SIZE = 1

model = Model([
    Linear(4, 8),
    Relu(),
    Linear(8, 8),
    Relu(),
    Linear(8, 1),
])

data = pd.concat([
    pd.read_csv('dataset/iris_in.csv', header=None),
    pd.read_csv('dataset/iris_out.csv', header=None)], axis=1
)

train_loader = DataLoader(data[:75], batch_size=BATCH_SIZE)
test_loader = DataLoader(data[-75:], batch_size=BATCH_SIZE)

train_mse = []
train_rmse = []
train_acc = []
for epoch in tqdm(range(100)):
    mse, rmse, acc = train(train_loader, train=True)
    train_mse.append(mse)
    train_rmse.append(rmse)
    train_acc.append(acc)

mse, rmse, acc = train(test_loader, train=True)
print(f"Test MSE: {mse}, RMSE: {rmse}, Accuracy: {acc}")

df = pd.DataFrame({
    'mean_square_error': train_mse[5:],
    'root_mean_square_error': train_rmse[5:],
    'avg_accuracy': train_acc[5:],
})
df.to_csv("training_log.csv", index=False)

df.plot()
plt.show()
