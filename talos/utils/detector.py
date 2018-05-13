from numpy import median, unique, mean


def prediction_type(self):

    try:
        y_cols = self.y.shape[1]
    except IndexError:
        y_cols = 1
    y_max = self.y.max()
    y_uniques = len(unique(self.y))

    if y_cols > 1:
        self._y_type = 'category'
        self._y_range = y_cols
        self._y_format = 'onehot'
    else:
        if y_max == 1:
            self._y_type = 'binary'
            self._y_range = y_cols
            self._y_format = 'single'
        elif mean(self.y) == median(self.y):
            self._y_type = 'category'
            self._y_range = y_uniques
            self._y_format = 'single'
        else:
            self._y_type = 'continuous'
            self._y_num = self.y.max() - self.y.min()
            self._y_format = 'single'

    return self
