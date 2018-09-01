# -*- coding: utf-8 -*-
"""Latin hypercube sampler with a sudoku-like constraint."""

from __future__ import division, print_function, absolute_import

import numpy as np

def sample(N,k,n, visualize=False, showdiag=False, verbose=False):
    """Create a coarsely `N`-dimensionally stratified latin hypercube sample (LHS) of range(`k` * `m`) in `N` dimensions.

Parameters:
    N : int, >= 1
        number of dimensions
    k : int, >= 1
        number of large subdivisions (sudoku boxes, "subspaces") per dimension
    n : int, >= 1
        number of samples to place in each subspace
    visualize : bool (optional)
        If True, the results (projected into two dimensions pairwise)
        are plotted using Matplotlib when the sampling is finished.
    showdiag : bool (optional)
        If True, and `N` >= 3, show also one-dimensional projection
        of the result onto each axis.

        Implies "visualize".

        This should produce a straight line with no holes onto
        each subplot that is on the diagonal of the plot array;
        mainly intended for debug.
    verbose : bool (optional)
        If this exists and is true, progress messages and warnings
        (for non-integer input) are printed.

Return value:
    tuple (`S`, `m`), where:
        S : (`k` * `m`)-by-`N` rank-2 np.array
            where each row is an `N`-tuple of integers in range(1, `k` * `m` + 1).

        m : int, >= 1
            number of bins per parameter in one subspace (i.e. sample slots
            per axis in one box).

            `m` = `n` * (`k` ** (`N` - 1)), but is provided as output for convenience.

**Examples:**

    `N` = 2 dimensions, k = 3 subspaces per axis, `n` = 1 sample per subspace.
    `m` will be `n` * (`k` ** (`N` - 1)) = 1 * 3**(2-1) = 3. Plot the result and show progress messages::

        S,m = sample(2, 3, 1, visualize=True, verbose=True)

    For comparison with the previous example, try this classical Latin hypercube
    that has 9 samples in total, plotting the result. We choose 9, because in
    the previous example, `k` * `m` = 3*3 = 9::

        S,m = sample(2, 1, 9, visualize=True)

**Notes:**

    If `k` = 1, the algorithm reduces to classical Latin hypercube sampling.

    If `N` = 1, the algorithm simply produces a random permutation of range(`k`).

    Let `m` = `n` * (`k` ** (`N` - 1)) denote the number of bins for one variable
    in one subspace. The total number of samples is always exactly `k` * `m'.
    Each component of a sample can take on values 0, 1, ..., (`k` * `m` - 1).
"""
    # sanity check input
    if not isinstance(N, int)  or  N < 1:
        raise ValueError("N must be int >= 1, got %g" % (N))
    if not isinstance(k, int)  or  k < 1:
        raise ValueError("k must be int >= 1, got %g" % (k))
    if not isinstance(n, int)  or  n < 1:
        raise ValueError("n must be int >= 1, got %g" % (n))

    # showing the diagonal implies visualization
    if showdiag:
        visualize = True

    # Discussion.

    # Proof that the following algorithm implements a Sudoku-like LHS method:
    #
    # * We desire two properties: Latin hypercube sampling globally, and equal density
    #   in each subspace.
    # * The independent index vector generation for each parameter guarantees the Latin
    #   hypercube property: some numbers will have been used, and removed from the index
    #   vectors, when the next subspace along the same hyperplane is reached. Thus, the same
    #   indices cannot be used again for any such subspace. This process continues until each
    #   index has been used exactly once.
    # * The equal density property is enforced by the fact that each subspace gets exactly one
    #   sample generated in one run of the loop. The total number of samples is, by design,
    #   divisible by the number of these subspaces. Therefore, each subspace will have the
    #   same sample density.
    #
    # Run time and memory cost:
    #
    # * Exactly k*m samples will be generated. This can be seen from the fact that there are
    #   k*m bins per parameter, and they all get filled by exactly one sample.
    # * Thus, runtime is in O(k*m) = O( k * n*k^(N-1) ) = O( n*k^N ). (This isn't as bad as it
    #   looks. All it's saying is that a linear number of bins gets filled. This is much less
    #   than the total number of bins (k*m)^N - which is why LHS is needed in the first place.
    #   We get a reduction in sample count by the factor (k*m)^(N-1).)
    # * Required memory for the final result is (k*m)*N reals (plus some overhead), where the
    #   N comes from the fact that each N-tuple generated has N elements. Note that the index
    #   vectors also use up k*m*N reals in total (k*N vectors, each with m elements). Thus the
    #   memory cost is 2*k*m*N reals plus overhead.
    # * Note that using a more complicated implementation that frees the elements of the index
    #   vectors as they are used up probably wouldn't help with the memory usage, because many
    #   vector implementations never decrease their storage space even if elements are deleted.
    # * In other programming languages, one might work around this by using linked lists
    #   instead of vectors, and arranging the memory allocations for the elements in a very
    #   special way (i.e. such that the last ones in memory always get deleted first). By
    #   using a linked list for the results, too, and allocating them over the deleted
    #   elements of the index vectors (since they shrink at exactly the same rate the results
    #   grow), one might be able to bring down the memory usage to k*m*N plus overhead.
    # * Finally, note that in practical situations N, k and m are usually small, so the factor
    #   of 2 doesn't really matter.

    # Algorithm.

    # Find necessary number of bins per subspace so that equal nonzero density is possible.
    # A brief analysis shows that in order to exactly fill up all k*m bins for one variable,
    # we must have k*m = n*k^N, i.e...
    m = n * k**(N-1)

    # Create index vectors for each subspace for each parameter. (There are k*N of these.)
    if verbose:
        print('Allocating %d elements for solution...' % (N*k*m))

    I    = np.empty( [N,k,m], dtype=int, order="C" )  # index vectors
    Iidx = np.zeros( [N,k],   dtype=int, order="C" )  # index of first "not yet used" element in each index vector

    # Create random permutations of range(m) so that in the sampling loop
    # we may simply pick the first element from each index vector.
    #
    for i in range(N):
        for j in range(k):
            tmp = np.array( range(m), dtype=int )
            np.random.shuffle(tmp)
            I[i,j,:] = tmp

    if verbose:
        print('Generating sample...')
        print('Looping through %d subspaces.' % (k**N))

    L  = k*m   # number of samples still waiting for placement
    Ns = k**N  # number of subspaces in total (cartesian product
               #         of k subspaces per axis in N dimensions)

    # Start with an empty result set. We will place the generated samples here.
    S = np.empty( [L,N], dtype=int, order="C" )
    out_idx = 0  # index of current output sample in S

    # create views for linear indexing
    I_lin    = np.reshape(I, -1)
    Iidx_lin = np.reshape(Iidx, -1)

    # we will need an array of range(N) several times in the loop...
    rgN = np.arange(N, dtype=int)

    while L > 0:
        # Loop over all subspaces, placing one sample in each.
        for j in range(Ns):  # index subspaces linearly
            # Find, in each dimension, which subspace we are in.
            # Compute the multi-index (vector containing an index in each dimension)
            # for this subspace.
            #
            # Simple example: (N,k,n) = (2,3,1)
            #   =>  pj = 0 0, 1 0, 2 0,  0 1, 1 1, 2 1,  0 2, 1 2, 2 2
            #   when j =  0,   1,   2,    3,   4,   5,    6,   7,   8
            #
            pj = np.array( ( j // (k**rgN) ) % k, dtype=int )

            # Construct one sample point.
            #
            # To do this, we grab the first "not yet used" element in all index vectors
            # (one for each dimension) corresponding to this subspace.
            #
            # Along the dth dimension, we are in the pj[d]th subspace.
            # Hence, in the dth dimension, we want to refer to the vector whose index is pj[d].
            #
            # Hence, we should take
            #  row = d  (effectively, range(N))
            #  col = pj[d]
            #
            # The array Iidx is of the shape [N,k]. NumPy uses C-contiguous ordering
            # by default; last index varies fastest. Hence, the element [row,col] is at
            # k*row + col.
            #
            # This gets us a vector of linear indices into Iidx, where the dth element
            # corresponds to the linear index of the pj[d]th vector.
            #
            i = np.array( k*rgN + pj, dtype=int )

            # Extract the "first unused element" data from Iidx for each of the vectors,
            # to get the actual sample slot numbers (random permutations) stored in I.
            #
            indices = Iidx_lin[i]

            # Indexing: the array I is of shape [N,k,m] and has C storage order.
            #
            idx_first = np.array( k*m*rgN + m*pj + indices, dtype=int )

            s = I_lin[idx_first] # this is our new sample point (vector of length N)
            Iidx_lin[i] += 1     # move to the next element in the selected index vectors

            # Now s contains a sample from (range(m), range(m), ..., range(m)) (N elements).
            # By its construction, the sample conforms globally to the Latin hypercube
            # requirement.

            # Compute the base index along each dimension. In the global numbering
            # which goes 0, 1, ..., (k*m-1) along each axis, the first element
            # of the current subspace is at this multi-index:
            #
            a = pj*m

            # Add the new sample to the result set.
            S[out_idx,:] = a+s
            out_idx += 1

        # We placed exactly Ns samples during the for loop.
        L -= Ns

    # Result visualization (for debug and illustrative purposes)
    #
    if visualize  and  N > 1:
        if verbose:
            print('Plotting...')

        import itertools
        import matplotlib.pyplot as plt

        # if the grid would show more lines than this, the lines are hidden.
        max_major_lines = 5
        max_minor_lines = 15

        major_color = "#a0a0a0"
        minor_color = "#e0e0e0"

        if k*m > 100:
            style = '.'
        else:
            style = 'o' # use circles when a small number of bins

        plt.figure(1)
        plt.clf()

        if N >= 3:
            # We'll make a "pairs" plot (like the pairs() function of the "R"
            # statistics software).

            # generate all pairs of dimensions, make explicit list
            pair_list = list(itertools.combinations(range(N), 2))

            # make final list.
            #
            # We want to populate both sides of the diagonal in the plot,
            # so we need pair_list, plus another copy of it
            # with the first and second components switched in each pair.
            #
            pairs = list(pair_list) # copy
            pairs.extend( tuple(reversed(pair)) for pair in pair_list )

            # Show also the diagonal if requested.
            #
            # This should produce a straight line with no holes onto
            # each subplot that is on the diagonal of the plot array.
            #
            if showdiag:
                pairs.extend( [ (j,j) for j in range(N) ] )
        else: # N == 2:
            pairs = [ (0, 1) ]

        Np = len(pairs)
        for i in range(Np):
            if N >= 3:
                if verbose:
                    print('Subplot %d of %d...' % ((i+1), Np))
                plt.subplot( N,N, N*pairs[i][1] + (pairs[i][0] + 1) )

            # off-diagonal projection? (i.e. a true 2D projection)
            if pairs[i][0] != pairs[i][1]:
                # Plot the points picked by the sample
                plt.plot( S[:,pairs[i][0]], S[:,pairs[i][1]], style)
                axmax = k*m

                # Mark bins (if few enough to fit reasonably on screen)
                #
                if k*m <= max_minor_lines:
                    for j in range(k*m):
                        xy = -0.5 + j
                        plt.plot( [xy, xy], [-0.5, axmax - 0.5], color=minor_color)
                        plt.plot( [-0.5, axmax - 0.5], [xy, xy], color=minor_color)

                # Mark the subspaces onto the figure
                # (if few enough to fit reasonably on screen)
                #
                if k <= max_major_lines:
                    for j in range(k):
                        xy = -0.5 + j*m
                        plt.plot( [xy, xy], [-0.5, axmax - 0.5], color=major_color, linewidth=2.0 )
                        plt.plot( [-0.5, axmax - 0.5], [xy, xy], color=major_color, linewidth=2.0 )

                # Make a box around the area
                plt.plot( [-0.5,         axmax - 0.5], [-0.5,        -0.5],        'k', \
                         linewidth=2.0  )
                plt.plot( [-0.5,         axmax - 0.5], [axmax - 0.5, axmax - 0.5], 'k', \
                         linewidth=2.0  )
                plt.plot( [-0.5,         -0.5],        [-0.5,        axmax - 0.5], 'k', \
                         linewidth=2.0  )
                plt.plot( [axmax - 0.5,  axmax - 0.5],  [-0.5,       axmax - 0.5], 'k', \
                         linewidth=2.0  )

                # Set the axes so that the extreme indices just fit into the view
                plt.axis("equal")
                plt.axis( [-0.5, axmax-0.5, -0.5, axmax-0.5 ] )
            else: # 1D projection
                plt.plot( S[:,pairs[i][0]], np.zeros( [k*m] ), style)
                plt.axis( [-0.5, axmax-0.5, -0.5, 0.5] )

        # Label the variables.
        #
        # We only do this if the diagonal subplots are blank.
        #
        if N >= 3:
            if not showdiag:
                for i in range(N):
                    plt.subplot(N,N, N*i+(i+1))
                    my_label = 'Row: x = var %d' % i
                    plt.text(0.5,0.6, my_label, horizontalalignment="center", fontweight="bold")
                    my_label = 'Col: y = var %d' % i
                    plt.text(0.5,0.4, my_label, horizontalalignment="center", fontweight="bold")
                    plt.axis("off")
        else:
            plt.xlabel('Var 0', fontweight="bold")
            plt.ylabel('Var 1', fontweight="bold")

        if verbose:
            print('Plotting done. Showing figure...')

        # show figures and enter gtk mainloop
        plt.show()

    return (S,m)
