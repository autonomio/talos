from __future__ import division, print_function, absolute_import
import numpy as np


def sample(dims, n):

    '''Latin Hypercube Sampling

    Parameters:
    dims : int, >= 1
           number of dimensions
    n : int, >= 1
        number of samples
    '''

    # sanity check input
    if not isinstance(dims, int) or dims < 1:
        raise ValueError("dims must be int >= 1, got %g" % (dims))
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be int >= 1, got %g" % (n))

    S = np.empty((n, dims), dtype=int, order="C")

    S[:, 0] = range(n)
    for j in range(1, dims):
        tmp = np.array(range(n), dtype=int)
        np.random.shuffle(tmp)
        S[:, j] = tmp

    return S
