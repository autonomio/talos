from tensorflow.keras.utils import Sequence


class SequenceGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):

        import numpy as np

        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
