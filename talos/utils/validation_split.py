def validation_split(self):

    '''Defines the attributes `x_train`, `y_train`, `x_val` and `y_val`.
    The validation sets are determined by the attribute val_split,
    which is a number in (0, 1) which determines the proportion of
    the input data to be allocated for cross-validation.'''

    # data input is list but multi_input is not set to True
    if isinstance(self.x, list) and self.multi_input is False:

        raise TypeError("For multi-input x, set multi_input to True")

    # If split is done in `Scan()` do nothing
    if self.custom_val_split:

        self.x_train = self.x
        self.y_train = self.y

        return self

    # Otherwise start by shuffling
    import wrangle
    self.x, self.y = wrangle.array_random_shuffle(x=self.x,
                                                  y=self.y,
                                                  multi_input=self.multi_input)

    # deduce the midway point for input data
    limit = int(len(self.y) * (1 - self.val_split))

    # handle the case where x is multi-input
    if self.multi_input:

        self.x_train = []
        self.x_val = []

        for ar in self.x:
            self.x_train.append(ar[:limit])
            self.x_val.append(ar[limit:])

    # handle the case where x is not multi-input
    else:

        self.x_train = self.x[:limit]
        self.x_val = self.x[limit:]

    # handle y data same for both cases
    self.y_train = self.y[:limit]
    self.y_val = self.y[limit:]

    return self


def kfold(x, y, folds=10, shuffled=True, multi_input=False):

    import wrangle

    # data input is list but multi_input is not set to True
    if isinstance(x, list) and multi_input is False:
        raise TypeError("For multi-input x, set multi_input to True")

    if shuffled is True:
        x, y = wrangle.array_random_shuffle(x, y, multi_input)

    out_x = []
    out_y = []

    # establish the fold size
    y_len = len(y)
    step = int(y_len / folds)

    lo = 0
    hi = step

    # create folds one by one
    for _i in range(folds):

        # handle the case for multi-input model
        if multi_input:
            fold_x = []
            for ar in x:
                fold_x.append(ar[lo:hi])
            out_x.append(fold_x)

        # handle the case where model is not multi-input
        else:
            out_x.append(x[lo:hi])

        out_y.append(y[lo:hi])

        lo += step
        hi += step

    return out_x, out_y
