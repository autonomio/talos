from numpy import random, vstack
from sklearn.model_selection import train_test_split

import chances

from ..utils.exceptions import TalosDataError


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

    random_method = self.main_self.random_method

    # calculate the size of the downsample
    n = int(len(self.param_grid) * self.main_self.grid_downsample)

    #if n > 1: 
    #    raise TalosDataError("Grid < 1: Incease grid_downsample")

    # initialize with random shuffle if needed
    if self.main_self.shuffle is True:
        random.shuffle(self.param_grid)

    # creates a stratified sample
    if random_method == 'stratified':
        size = self.main_self.grid_downsample / 2
        train, test = train_test_split(self.param_grid,
                                       train_size=size,
                                       test_size=size,
                                       stratify=None)
        return vstack((train, test))

    # Initialize Randomizer()
    r = chances.Randomizer(len(self.param_grid), n)

    # use the user selected method
    if random_method == 'sobol':
        out = r.sobol()
    elif random_method == 'quantum':
        out = r.quantum()
    elif random_method == 'halton':
        out = r.halton()
    elif random_method == 'korobov_matrix':
        out = r.korobov_matrix()
    elif random_method == 'latin_sudoku':
        out = r.latin_sudoku()
    elif random_method == 'latin_matrix':
        out = r.latin_matrix()
    elif random_method == 'latin_improved':
        out = r.latin_improved()
    elif random_method == 'uniform_mersenne':
        out = r.uniform_mersenne()
    elif random_method == 'uniform_crypto':
        out = r.uniform_crypto()
    elif random_method == 'ambience':
        out = r.ambience()

    return self.param_grid.take(out, axis=0)
