from numpy import random


def sample_reducer(self):

    '''Reduces the sample before starting the experiment based
    on 'grid_downsample' paramater from Scan()'''

    random.shuffle(self.param_grid)
    _d_out = int(len(self.param_grid) * self.main_self.grid_downsample)

    return self.param_grid[:_d_out]
