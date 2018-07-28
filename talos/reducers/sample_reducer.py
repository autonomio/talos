from numpy import random


def sample_reducer(self):

    random.shuffle(self.param_grid)
    _d_out = int(len(self.param_grid) * self.grid_downsample)

    return self.param_grid[:_d_out]
