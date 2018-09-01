# -*- coding: utf-8 -*-
#
"""Trivial combinatorial sampler."""

from __future__ import division, print_function, absolute_import

from itertools import chain

import numpy as np

# http://stackoverflow.com/questions/17973507/why-is-converting-a-long-2d-list-to-numpy-array-so-slow
def __longlist2array(longlist):
    flat = np.fromiter(chain.from_iterable(longlist), np.array(longlist[0][0]).dtype, -1)
    return flat.reshape((len(longlist), -1))


def sample(N,k):
    """Generate all `k` ** `N` combinations of range(`k`) "outer-raised" to power `N`.

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

The possibilities are enumerated by generating a nested list comprehension::

    [ [j1] for j1 in range(k) ]
    [ [j1,j2] for j1 in range(k) for j2 in range(k) ]
    [ [j1,j2,j3] for j1 in range(k) for j2 in range(k) for j3 in range(k) ]
    # ...

I.e., in general,::

    [ [j1, ..., jN] for j1 in range(k) ... for jN in range(k) ]

This is then eval'd and the resulting list, converted to an np.array, is returned.
"""
    # sanity check input
    if not isinstance(N, int)  or  N < 1:
        raise ValueError("N must be int >= 1, got %g" % (N))
    if not isinstance(k, int)  or  k < 1:
        raise ValueError("k must be int >= 1, got %g" % (k))

    code = "[ ["
    code += "j0"  # no comma (this may be the only one)
    for j in range(1,N):
        code += ", j%d" % j
    code += "]"
    for j in range(N):
        code += " for j%d in range(%d)" % (j, k)
    code += " ]"

    return __longlist2array(eval(code))

