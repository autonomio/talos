def sample_reducer(limit, max_value, random_method):

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

    import chances as ch

    # calculate the size of the downsample
    if isinstance(limit, float):
        n = int(max_value * limit)
    if isinstance(limit, int):
        n = limit

    max_value = int(max_value)

    # throw an error if
    from ..utils.exceptions import TalosDataError
    if n < 1:
        raise TalosDataError("Limiters lead to < 1 permutations.")

    # Initialize Randomizer()
    r = ch.Randomizer(max_value, n)

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
        print('No eligble random_method found. Using uniform_mersenne.')
        out = r.uniform_mersenne()

    return out
