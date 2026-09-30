"""Aligned, nonmutating legacy data splits."""
import numpy as np


def _take(values, indices, nested=False):
    if nested or isinstance(values, dict):
        if isinstance(values, dict):
            return {key: _take(value, indices) for key, value in values.items()}
        return [_take(value, indices) for value in values]
    if hasattr(values, 'iloc'):
        return values.iloc[indices]
    return values[indices]


def _size(values, nested=False):
    if isinstance(values, dict):
        return len(next(iter(values.values())))
    return len(values[0]) if nested else len(values)


def validation_split(self):
    if isinstance(self.x, list) and not self.multi_input:
        raise TypeError('For multi-input x, set multi_input to True')
    if self.custom_val_split:
        self.x_train, self.y_train = self.x, self.y
        return self
    if not 0 < self.val_split < 1:
        raise ValueError('val_split must lie strictly between zero and one.')
    nested_y = isinstance(self.y, list)
    size = _size(self.y, nested_y)
    indices = np.random.default_rng(getattr(self, 'seed', None)).permutation(size)
    split = int(size * (1 - self.val_split))
    if split == 0 or split == size:
        raise ValueError('The validation split must contain training and validation rows.')
    self.x_train = _take(self.x, indices[:split], self.multi_input)
    self.x_val = _take(self.x, indices[split:], self.multi_input)
    self.y_train = _take(self.y, indices[:split], nested_y)
    self.y_val = _take(self.y, indices[split:], nested_y)
    return self


def kfold(x, y, folds=10, shuffled=True, multi_input=False, seed=None):
    if isinstance(x, list) and not multi_input:
        raise TypeError('For multi-input x, set multi_input to True')
    nested_y = isinstance(y, list)
    size = _size(y, nested_y)
    if not isinstance(folds, int) or folds < 1 or folds > size:
        raise ValueError('folds must be an integer between one and the number of rows.')
    indices = np.arange(size)
    if shuffled:
        indices = np.random.default_rng(seed).permutation(indices)
    parts = np.array_split(indices, folds)
    return ([_take(x, part, multi_input) for part in parts],
            [_take(y, part, nested_y) for part in parts])
