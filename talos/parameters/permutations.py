from numpy import array
from itertools import product


def param_grid(self):

    '''CREATE THE PARAMETER PERMUTATIONS
    '''

    ls = [list(self.p[key]) for key in self.p.keys()]
    _pg_out = array(list(product(*ls)))

    return _pg_out
