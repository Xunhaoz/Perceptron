import numpy as np


class DataLoader:
    def __init__(self, dataframe, batch_size=5, shuffle=True, one_hot_encode=False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_batch_size = batch_size
        self.index = np.arange(len(dataframe))
        if shuffle:
            np.random.shuffle(self.index)
        self.current_index = 0
        self.one_hot_encode = one_hot_encode

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.index):
            raise StopIteration

        batch_index = self.index[self.current_index:self.current_index + self.batch_size]
        batch_data = self.dataframe.iloc[batch_index].reset_index(drop=True)
        self.current_batch_size = len(batch_data)
        self.current_index += self.batch_size

        if self.one_hot_encode:
            return batch_data.values[:, :4], self.transform_one_hot_encode(batch_data.values[:, 4:])
        else:
            return batch_data.values[:, :4], batch_data.values[:, 4:]

    def __len__(self):
        return len(self.dataframe) // self.batch_size + (1 if len(self.dataframe) % self.batch_size != 0 else 0)

    def get_current_batch_size(self):
        return self.current_batch_size

    def transform_one_hot_encode(self, y):
        one_hot = np.zeros((len(y), 3))
        for i in range(len(y)):
            one_hot[i][int(y[i]) - 1] = 1
        return one_hot
