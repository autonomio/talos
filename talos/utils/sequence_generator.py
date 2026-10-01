"""A lazy factory avoids importing a deep-learning backend with Talos."""
import numpy as np


class SequenceGenerator:
    def __new__(cls, x_set=None, y_set=None, batch_size=32, *, x=None, y=None, backend='keras'):
        x_set = x_set if x_set is not None else x
        y_set = y_set if y_set is not None else y
        if x_set is None or y_set is None or batch_size < 1:
            raise ValueError('Provide x/y data and a positive batch_size.')
        if backend in ('tensorflow', 'tf', 'tf.keras'):
            from tensorflow.keras.utils import Sequence
        else:
            from keras.utils import Sequence
        class Batches(Sequence):
            def __init__(self):
                super().__init__()
                self.x, self.y, self.batch_size = x_set, y_set, batch_size

            def __len__(self):
                return int(np.ceil(len(self.x) / float(self.batch_size)))

            def __getitem__(self, idx):
                return (self.x[idx * self.batch_size:(idx + 1) * self.batch_size],
                        self.y[idx * self.batch_size:(idx + 1) * self.batch_size])
        return Batches()
