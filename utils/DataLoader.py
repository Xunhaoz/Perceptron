import numpy as np


class DataLoader:
    def __init__(self, dataframe, batch_size=5, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_batch_size = batch_size
        self.index = np.arange(len(dataframe))
        if shuffle:
            np.random.shuffle(self.index)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.index):
            raise StopIteration

        batch_index = self.index[self.current_index:self.current_index + self.batch_size]
        batch_data = self.dataframe.iloc[batch_index].reset_index(drop=True)
        self.current_batch_size = len(batch_data)
        self.current_index += self.batch_size

        return batch_data.values[:, :4], batch_data.values[:, 4:]

    def __len__(self):
        return len(self.dataframe) // self.batch_size + (1 if len(self.dataframe) % self.batch_size != 0 else 0)

    def get_current_batch_size(self):
        return self.current_batch_size
