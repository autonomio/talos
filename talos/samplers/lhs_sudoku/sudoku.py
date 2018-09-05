# -*- coding: utf-8 -*-
"""Latin hypercube sampler with a sudoku-like constraint."""

from __future__ import division, print_function, absolute_import

import numpy as np


def sample(N, k, n):

    """Create a coarsely `N`-dimensionally stratified latin hypercube sample
       (LHS) of range(`k` * `m`) in `N` dimensions.

Parameters:
    N : int, >= 1
        number of dimensions
    k : int, >= 1
        number of large subdivisions (sudoku boxes, "subspaces") per dimension
    n : int, >= 1
        number of samples to place in each subspace

Return value:
    tuple (`S`, `m`), where:
        S : (`k` * `m`)-by-`N` rank-2 np.array
            where each row is an `N`-tuple of integers in
            range(1, `k` * `m` + 1).

        m : int, >= 1
            number of bins per parameter in one subspace (i.e. sample slots
            per axis in one box).

            `m` = `n` * (`k` ** (`N` - 1)), but is provided as output for
            convenience.

**Examples:**

    `N` = 2 dimensions, k = 3 subspaces per axis, `n` = 1 sample per subspace.
    `m` will be `n` * (`k` ** (`N` - 1)) = 1 * 3**(2-1) = 3.
    Plot the result and show progress messages::

        S,m = sample(2, 3, 1, visualize=True, verbose=True)

    For comparison with the previous example, try this classical
    Latin hypercube that has 9 samples in total, plotting the result.
    We choose 9, because in the previous example, `k` * `m` = 3*3 = 9::

        S,m = sample(2, 1, 9, visualize=True)

**Notes:**

    If `k` = 1, the algorithm reduces to classical Latin hypercube sampling.

    If `N` = 1, the algorithm simply produces a
    random permutation of range(`k`).

    Let `m` = `n` * (`k` ** (`N` - 1)) denote the number
    of bins for one variable
    in one subspace. The total number of samples is always exactly `k` * `m'.
    Each component of a sample can take on values 0, 1, ..., (`k` * `m` - 1).
"""
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be int >= 1, got %g" % (N))
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be int >= 1, got %g" % (k))
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be int >= 1, got %g" % (n))

    m = n * k ** (N-1)

    I = np.empty([N, k, m], dtype=int, order="C")
    Iidx = np.zeros([N, k], dtype=int, order="C")

    for i in range(N):
        for j in range(k):
            tmp = np.array(range(m), dtype=int)
            np.random.shuffle(tmp)
            I[i, j, :] = tmp

    L = k * m
    Ns = k ** N

    S = np.empty([L, N], dtype=int, order="C")
    out_idx = 0

    I_lin = np.reshape(I, -1)
    Iidx_lin = np.reshape(Iidx, -1)

    rgN = np.arange(N, dtype=int)

    while L > 0:
        for j in range(Ns):
            pj = np.array((j // (k ** rgN)) % k, dtype=int)
            i = np.array(k*rgN + pj, dtype=int)
            indices = Iidx_lin[i]
            idx_first = np.array(k*m*rgN + m*pj + indices, dtype=int)
            s = I_lin[idx_first]
            Iidx_lin[i] += 1
            a = pj * m
            S[out_idx, :] = a + s
            out_idx += 1

        L -= Ns

    return (S, m)
