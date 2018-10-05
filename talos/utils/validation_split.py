import numpy as np
from wrangle import shuffle


def validation_split(self):
    """Defines the attributes `x_train`, `y_train`, `x_val` and `y_val`.
    The validation (cross-validation, aka development) sets are determined
    by the attribute val_split, which is a number in (0, 1) which determines
    the proportion of the input data to be allocated for cross-validation."""

    if self.custom_val_split:
        self.x_train = self.x
        self.y_train = self.y
        # self.x/y_val are already set

    else:
        if self.shuffle:
            random_shuffle(self)

        limit = int(len(self.x) * (1 - self.val_split))

        self.x_train = self.x[:limit]
        self.y_train = self.y[:limit]

        self.x_val = self.x[limit:]
        self.y_val = self.y[limit:]

    return self


def random_shuffle(self):
    """Randomly shuffles the datasets. If self.seed is set, seed the generator
    to ensure that the results are reproducible."""

    random_index = np.arange(len(self.x))

    if self.seed is not None:
        np.random.seed(self.seed)

    np.random.shuffle(random_index)

    self.x = self.x[random_index]
    self.y = self.y[random_index]


def kfold(x, y, folds=10, shuffled=True):

    if shuffled is True:
        x, y = shuffle(x, y)

    out_x = []
    out_y = []

    x_len = len(x)
    step = int(x_len / folds)

    lo = 0
    hi = step

    for i in range(folds):
        out_x.append(x[lo:hi])
        out_y.append(y[lo:hi])

        lo += step
        hi += step

    return out_x, out_y
