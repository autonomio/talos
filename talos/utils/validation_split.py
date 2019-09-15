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
        # shuffle the data before splitting
        random_shuffle(self)

        # deduce the midway point for input data
        limit = int(len(self.x) * (1 - self.val_split))

        self.x_train = self.x[:limit]
        self.y_train = self.y[:limit]

        self.x_val = self.x[limit:]
        self.y_val = self.y[limit:]

    return self


def random_shuffle(self):

    """Randomly shuffles the datasets.
    If self.seed is set, seed the generator
    to ensure that the results are reproducible."""

    def randomize(x):

        '''
        Helper function to support the case
        where x consist of a list of arrays.
        '''

        import numpy as np

        if self.seed is not None:
            np.random.seed(self.seed)

        ix = np.arange(len(x))
        np.random.shuffle(ix)

        return ix

    if isinstance(self.x, list):

        ix = randomize(self.x[0])
        out = []

        for a in self.x:
            out.append(a[ix])
        self.x = out

    else:
        ix = randomize(self.x)
        self.x = self.x[ix]

    self.y = self.y[ix]


def kfold(x, y, folds=10, shuffled=True):

    import wrangle as wr

    if shuffled is True:
        x, y = wr.array_random_shuffle(x, y)

    out_x = []
    out_y = []

    x_len = len(x)
    step = int(x_len / folds)

    lo = 0
    hi = step

    for _i in range(folds):
        out_x.append(x[lo:hi])
        out_y.append(y[lo:hi])

        lo += step
        hi += step

    return out_x, out_y
