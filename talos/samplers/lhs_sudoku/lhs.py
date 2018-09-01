# -*- coding: utf-8 -*-
#
"""Classical latin hypercube sampler."""

from __future__ import division, print_function, absolute_import

import numpy as np

def sample(N,k):
    """Generate a latin hypercube sample in `N` dimensions.

Parameters:
    N : int, >= 1
        number of dimensions
    k : int, >= 1
        number of bins per axis

Return value:
    rank-2 np.array
        first index indexes the sample number, the second indexes the axis
        (i.e. each row is an N-dimensional sample).

**Notes:**

In the result, range(k) on the first axis is paired with a random permutation
of range(k) on each subsequent axis.
"""
    # sanity check input
    if not isinstance(N, int)  or  N < 1:
        raise ValueError("N must be int >= 1, got %g" % (N))
    if not isinstance(k, int)  or  k < 1:
        raise ValueError("k must be int >= 1, got %g" % (k))

    S = np.empty( (k,N), dtype=int, order="C" )

    S[:,0] = range(k)
    for j in range(1, N):
        tmp = np.array( range(k), dtype=int )
        np.random.shuffle(tmp)
        S[:,j] = tmp

    return S

