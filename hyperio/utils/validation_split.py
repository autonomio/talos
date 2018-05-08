import numpy as np


def validation_split(self):

    '''VALIDATION SPLIT OF X AND Y
    Based on the Hyperio() parameter val_split
    both 'x' and 'y' are split.

    '''

    if self.shuffle == True:
        random_shuffle(self)

    len_x = len(self.x)
    limit = int(len_x * (1 - self.val_split))

    self.x_train = self.x[:limit]
    self.y_train = self.y[:limit]

    self.x_val = self.x[limit:]
    self.y_val = self.y[limit:]

    return self


def random_shuffle(self):

    random_index = np.arange(len(self.x))
    np.random.shuffle(random_index)

    self.x = self.x[random_index]
    self.y = self.y[random_index]
