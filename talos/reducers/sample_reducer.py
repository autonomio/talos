from numpy import random


def sample_reducer(self):

    if self.grid_downsample is None:

        return self.param_grid

    else:
        random.shuffle(self.param_grid)
        _d_out = int(len(self.param_grid) * self.grid_downsample)

        return self.param_grid[:_d_out]
