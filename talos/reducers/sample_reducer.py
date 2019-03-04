import chances

from ..utils.exceptions import TalosDataError


def sample_reducer(self, length, max_value):

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
    n = int(max_value * self.main_self.grid_downsample)

    # throw an error if
    if n < 1:
        raise TalosDataError("No permutations in grid. Incease grid_downsample")

    # Initialize Randomizer()
    r = chances.Randomizer(max_value, length)

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
    else:
        print('check random_method, no eligble method found. Using uniform mersenne.')
        out = r.uniform_mersenne()

    return out
