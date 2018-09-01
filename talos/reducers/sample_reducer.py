from numpy import random, vstack
from sklearn.model_selection import train_test_split

from ..samplers.lhs_sudoku import sudoku, lhs


def sample_reducer(self):

    '''Sample Reducer (Helper)

    NOTE: The Scan() object  is in self.main_self because
    the object being passed here is ParamGrid() object where
    the Scan() object is attached as self.main_self.

    Utilize 'grid_downsample', 'shuffle', and 'random_method'
    to reduce the param_grid before starting the experiment.
    This is the simplest method in Talos for dealing with curse
    of dimensionality.

    Options are uniform random, stratified random, latin hypercube
    sampling, and latin hypercube with sudoku style constraint.

    Returns the reduced param_grid as numpy array.

    '''
    # calculate the size of the downsample
    _d_out = int(len(self.param_grid) * self.main_self.grid_downsample)

    # initialize with random shuffle if needed
    if self.main_self.shuffle is True:
        random.shuffle(self.param_grid)

    # creates a uniform random sample
    if self.main_self.random_method == 'uniform':

        random.shuffle(self.param_grid)
        return self.param_grid[:_d_out]

    # creates a stratified sample
    elif self.main_self.random_method == 'stratified':

        size = self.main_self.grid_downsample / 2
        train, test = train_test_split(self.param_grid,
                                       train_size=size,
                                       test_size=size,
                                       stratify=None)
        return vstack((train, test))

    # creates a latin hypercube sample
    elif self.main_self.random_method == 'lhs':
        
        size = lhs.sample(2, _d_out)[:, 1]
        return self.param_grid.take(size, axis=0)

    # creates a latin hypercube sample with sudoku constraint
    elif self.main_self.random_method == 'lhs_sudoku':

        out = sudoku.sample(1, 1, _d_out)
        size = [i[0] for i in out[0]]
        return self.param_grid.take(size, axis=0)
