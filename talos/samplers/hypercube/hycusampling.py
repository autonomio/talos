# This Python file uses the following encoding: utf-8
"""
This module provides functions for super-uniform sampling of the unit
hypercube. 'Super-uniform' in this context means that the obtained point
sample is more uniform than a random uniform sample, which is a desirable
property in many applications. After creation, the samples can be
transformed from the unit hypercube to arbitrary cuboids.
"""
import sys
import math
import random
import itertools

import numpy as np

from diversipy.distance import calc_euclidean_dist_matrix
from diversipy.distance import DistanceMatrixFunction



def unitcube(dimension):
    """Shortcut to generate a tuple of bounds of the unit hypercube."""
    assert dimension > 0
    return [0.0] * dimension, [1.0] * dimension



def scaled(points, from_cuboid, to_cuboid):
    """Linear transformation between arbitrary cuboids.

    This function does not check if the points are actually inside
    `from_cuboid`.

    Parameters
    ----------
    points : array_like
        2-D array of points.
    from_cuboid : tuple
        Contains lower and upper boundaries.
    to_cuboid : tuple
        Contains lower and upper boundaries.

    Returns
    -------
    scaled_points : numpy array
        A new array containing the scaled points.

    """
    # sanity checks
    min_bounds_to, max_bounds_to = to_cuboid
    assert len(min_bounds_to) == len(max_bounds_to)
    for min_bound_to, max_bound_to in zip(min_bounds_to, max_bounds_to):
        assert max_bound_to >= min_bound_to
    min_bounds_from, max_bounds_from = from_cuboid
    assert len(min_bounds_from) == len(max_bounds_from)
    for min_bound_from, max_bound_from in zip(min_bounds_from, max_bounds_from):
        assert max_bound_from >= min_bound_from
    # scale
    length_factors = (np.asarray(max_bounds_to) - min_bounds_to)
    length_factors /= (np.asarray(max_bounds_from) - min_bounds_from)
    scaled_points = (np.asarray(points) - min_bounds_from) * length_factors
    scaled_points += min_bounds_to
    return scaled_points



def grid(num_points, dimension):
    """Create conventional grid in unit hypercube.

    Also related to full factorial designs.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
        ``num_points ** (1/dimension)`` must be integer.
    dimension : int
        The dimension of the space.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    points_per_axis = int(num_points ** (1.0 / dimension))
    assert points_per_axis ** dimension == num_points
    possible_values = list(range(points_per_axis))
    divisor = points_per_axis - 1.0
    for i in range(points_per_axis):
        possible_values[i] /= divisor
    points = np.array(list(itertools.product(possible_values, repeat=dimension)))
    return points



def sukharev_grid(num_points, dimension):
    """Create Sukharev grid in unit hypercube.

    Special property of this grid is that points are not placed on the
    boundaries of the hypercube, but at centroids of the `num_points`
    subcells. This design offers optimal results for the covering radius
    regarding distances based on the max-norm.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
        ``num_points ** (1/dimension)`` must be integer.
    dimension : int
        The dimension of the space.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    points_per_axis = int(num_points ** (1.0 / dimension))
    assert points_per_axis ** dimension == num_points
    possible_values = [x + 0.5 for x in range(points_per_axis)]
    divisor = points_per_axis
    for i in range(points_per_axis):
        possible_values[i] /= divisor
    points = np.array(list(itertools.product(possible_values, repeat=dimension)))
    return points



def rank1_design_matrix(num_points, dimension, generator_vector=None):
    """Design matrix for a rank-1 lattice.

    This algorithm is deterministic and has linear run time.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.
    generator_vector : array_like, optional
        An integer vector of length `dimension`.

    Returns
    -------
    design : (`num_points`, `dimension`) numpy array
        Matrix with integers corresponding to the bins of a virtual grid.

    """
    if generator_vector is None:
        generator_vector = np.random.randint(2, num_points, dimension)
    assert len(generator_vector) == dimension
    generator_vector = np.array(generator_vector) % num_points
    points = np.array([i * generator_vector % num_points for i in range(num_points)], dtype="i")
    return points



def korobov_design_matrix(num_points, dimension, generator_param=None):
    """Design matrix for a Korobov lattice.

    This is a special case of the rank-1 lattice. The design has the LHD
    property if ``gcd(num_points, generator_param) == 1``. The algorithm is
    deterministic and has linear run time.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.
    generator_param : int, optional
        An integer used for constructing a generator vector.

    Returns
    -------
    design : (`num_points`, `dimension`) numpy array
        Matrix with integers corresponding to the bins of a virtual grid.

    """
    if generator_param is None:
        generator_param = random.randrange(2, num_points)
    generator_param %= num_points
    assert generator_param > 1
    generator_vector = [1] * dimension
    for i in range(1, dimension):
        generator_vector[i] = (generator_param * generator_vector[i - 1]) % num_points
    return rank1_design_matrix(num_points, dimension, generator_vector)



def random_uniform(num_points, dimension):
    """Syntactic sugar for :func:`numpy.random.rand`."""
    return np.random.rand(num_points, dimension)



def halton(num_points, dimension, skip=0):
    """Generate a Halton point set.

    Quasirandom sequence using the default initialization with the first
    `dimension` prime numbers.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.
    skip : int, optional
        The first `skip` points of the sequence will be left out.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    def is_prime(number):
        number = float(number)
        if number % 2 == 0 and number != 2:
            return False
        for divisor in range(3, int(number ** 0.5) + 1, 2):
            if number % divisor == 0:
                return False
        return True

    def generator(index, base):
        result = 0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = math.floor(i / base)
            f /= base
        return result

    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
             61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
             137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
             199, 211, 223, 227, 229, 233, 239, 241, 251]
    while len(bases) < dimension:
        next_candidate = bases[-1] + 2
        while not is_prime(next_candidate):
            next_candidate += 2
        bases.append(next_candidate)
    points = []
    for i in range(skip, skip + num_points):
        point = [generator(i, bases[j]) for j in range(dimension)]
        points.append(point)
    return np.array(points)



def random_k_means(num_points,
                   dimension,
                   num_steps=None,
                   initial_points=None,
                   dist_matrix_function=None,
                   callback=None):
    """MacQueen's method.

    In its default setup, this algorithm converges to a centroidal Voronoi
    tesselation of the unit hypercube. Further information is given in
    [MacQueen1967]_.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.
    num_steps : int, optional
        The number of iterations to carry out. Default is
        ``100 * num_points``.
    initial_points : array_like, optional
        The point set to improve (if None, a sample is drawn with
        :func:`stratified_sampling`).
    dist_matrix_function : callable, optional
        The function to compute the distances. Default is Euclidean
        distance.
    callback : callable, optional
        If provided, it is called in each iteration with the current point
        set as argument for monitoring progress.

    Returns
    -------
    cluster_centers : (`num_points`, `dimension`) numpy array

    References
    ----------
    .. [MacQueen1967] MacQueen, J. Some methods for classification and
        analysis of multivariate observations. Proceedings of the Fifth
        Berkeley Symposium on Mathematical Statistics and Probability,
        Volume 1: Statistics, pp. 281--297, University of California Press,
        Berkeley, Calif., 1967.
        http://projecteuclid.org/euclid.bsmsp/1200512992.

    """
    # initialization
    if num_steps is None:
        num_steps = 100 * num_points
    if initial_points is None:
        cluster_centers = stratified_sampling(stratify_generalized(num_points,
                                                                   dimension))
    elif len(initial_points) == num_points:
        cluster_centers = np.array(initial_points)
        assert np.all(cluster_centers >= 0.0)
        assert np.all(cluster_centers <= 1.0)
    else:
        raise ValueError("len(initial_points) must be equal to num_points")
    weights = [1.0] * num_points
    if dist_matrix_function is None:
        dist_matrix_function = calc_euclidean_dist_matrix
    # begin iteration
    for _ in range(num_steps):
        if callback is not None:
            callback(cluster_centers)
        random_point = np.random.rand(1, dimension)
        distances = dist_matrix_function(random_point, cluster_centers)
        random_point = random_point.ravel()
        nearest_index = int(np.argmin(distances, axis=1))
        nearest_cluster_center = cluster_centers[nearest_index, :].ravel()
        if hasattr(dist_matrix_function, "max_dists_per_dim") and dist_matrix_function.max_dists_per_dim is not None:
            one_dim_dists = np.abs(nearest_cluster_center - random_point)
            virtual_point = np.array(random_point)
            for j, dist in enumerate(one_dim_dists):
                if dist > 0.5:
                    if nearest_cluster_center[j] < 0.5:
                        virtual_point[j] = 1.0 - random_point[j]
                    else:
                        virtual_point[j] = 1.0 + random_point[j]
        else:
            virtual_point = random_point
        weight = weights[nearest_index]
        cluster_centers[nearest_index, :] = (weight * nearest_cluster_center + virtual_point) / (weight + 1.0)
        cluster_centers[nearest_index, :] %= 1.0
        assert np.all(cluster_centers[nearest_index, :] <= 1.0)
        assert np.all(cluster_centers[nearest_index, :] >= 0.0)
        weights[nearest_index] += 1.0
    return cluster_centers



def maximin_reconstruction(num_points,
                           dimension,
                           num_steps=None,
                           initial_points=None,
                           existing_points=None,
                           use_reflection_edge_correction=False,
                           dist_matrix_function=None,
                           callback=None):
    """Maximize the minimal distance in the unit hypercube with extensions.

    This algorithm carries out a user-specified number of iterations to
    maximize the minimal distance of a point in the set to 1) other points
    in the set, 2) existing (fixed) points, and 3) the boundary of the
    hypercube. Details can be found in [Wessing2015]_.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.
    num_steps : int, optional
        The number of iterations to carry out. Default is
        ``100 * num_points``.
    initial_points : array_like, optional
        The point set to improve (if None, a sample is drawn with
        :func:`stratified_sampling`).
    existing_points : array_like, optional
        Points that cannot be modified anymore, but should be considered in
        the distance computations.
    use_reflection_edge_correction : bool, optional
        If True, selection pressure in boundary regions will be increased by
        considering additional distances to virtual points, which are
        created by mirroring the real points at the boundary.
    dist_matrix_function : callable, optional
        The function to compute the distances. Default is Manhattan distance
        on a torus.
    callback : callable, optional
        If provided, it is called in each iteration with the current point
        set as argument for monitoring progress.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    References
    ----------
    .. [Wessing2015] Wessing, Simon (2015). Two-stage Methods for Multimodal
        Optimization. PhD Thesis, Technische Universität Dortmund.
        http://hdl.handle.net/2003/34148

    """
    def dist_to_bound(point):
        return min(point.min(), (1.0 - point).min())

    # initialization and sanity checks
    assert num_points > 0
    if num_steps is None:
        num_steps = 100 * num_points
    if initial_points is None:
        points = stratified_sampling(stratify_generalized(num_points,
                                                          dimension))
    elif len(initial_points) == num_points:
        points = np.array(initial_points)
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)
    else:
        raise ValueError("len(initial_points) must be equal to num_points")
    if existing_points is None:
        existing_points = []
    existing_points = np.atleast_2d(existing_points)
    if existing_points.size == 0:
        num_existing_points = 0
    else:
        num_existing_points = len(existing_points)
    if dist_matrix_function is None:
        dist_matrix_function = DistanceMatrixFunction(exponent=1,
                                                      max_dists_per_dim=[1.0] * dimension)
    norm_of_one_vector = 2.0 * dist_matrix_function(np.atleast_2d([0.0] * dimension),
                                                    np.atleast_2d([0.5] * dimension))[0, 0]
    remaining_indices = list(range(num_points))
    random.shuffle(remaining_indices)
    removal_candidate_index = remaining_indices.pop()
    removal_candidate = points[removal_candidate_index]
    # initial distance calculations
    distances = dist_matrix_function(np.atleast_2d(removal_candidate), points)[0]
    distances[removal_candidate_index] = np.inf
    current_dist = distances.min()
    if num_existing_points > 0:
        dists_to_existing_points = dist_matrix_function(np.atleast_2d(removal_candidate),
                                                        existing_points)[0]
        current_dist = min(current_dist, dists_to_existing_points.min())
    if use_reflection_edge_correction:
        # compare with 2 * ||1|| * (distance to nearest boundary)
        relaxed_boundary_dist = 2.0 * norm_of_one_vector * dist_to_bound(removal_candidate)
        current_dist = min(current_dist, relaxed_boundary_dist)
    # maximize minimal distance
    for _ in range(num_steps):
        if callback is not None:
            callback(points)
        new_point = np.random.rand(1, dimension)
        new_dist = np.inf
        if use_reflection_edge_correction:
            # compare with 2 * ||1|| * (distance to nearest boundary)
            relaxed_boundary_dist = 2.0 * norm_of_one_vector * dist_to_bound(new_point)
            new_dist = relaxed_boundary_dist
        if new_dist >= current_dist:
            distances = dist_matrix_function(new_point, points)[0]
            distances[removal_candidate_index] = np.inf
            new_dist = min(new_dist, distances.min())
            if new_dist >= current_dist and num_existing_points > 0:
                dists_to_existing_points = dist_matrix_function(new_point, existing_points)[0]
                new_dist = min(new_dist, dists_to_existing_points.min())
        if new_dist >= current_dist:
            # accept new point
            points[removal_candidate_index] = new_point
            current_dist = new_dist
            removal_candidate = new_point
            # removal_candidate_index stays the same, but reset other indices
            remaining_indices = list(range(num_points))
            remaining_indices.remove(removal_candidate_index)
            random.shuffle(remaining_indices)
        else:
            # failed to find better point in this iteration
            if remaining_indices:
                # carry out one attempt to find new removal candidate
                removal_candidate_candidate_index = remaining_indices.pop()
                removal_candidate_candidate = points[removal_candidate_candidate_index]
                # calculate minimal distance
                distances = dist_matrix_function(np.atleast_2d(removal_candidate_candidate), points)[0]
                distances[removal_candidate_candidate_index] = np.inf
                candidate_candidate_dist = distances.min()
                if num_existing_points > 0:
                    dists_to_existing_points = dist_matrix_function(np.atleast_2d(removal_candidate_candidate),
                                                                    existing_points)[0]
                    candidate_candidate_dist = min(candidate_candidate_dist,
                                                   dists_to_existing_points.min())
                if use_reflection_edge_correction:
                    # compare with 2 * ||1|| * (distance to nearest boundary)
                    relaxed_boundary_dist = 2.0 * norm_of_one_vector
                    relaxed_boundary_dist *= dist_to_bound(removal_candidate_candidate)
                    candidate_candidate_dist = min(candidate_candidate_dist,
                                                   relaxed_boundary_dist)
                if candidate_candidate_dist <= current_dist:
                    # found new removal candidate
                    removal_candidate = removal_candidate_candidate
                    current_dist = candidate_candidate_dist
                    removal_candidate_index = removal_candidate_candidate_index
    return points



def stratify_conventional(num_strata, dimension):
    """Stratification of the unit hypercube.

    This algorithm divides the hypercube into `num_points` subcells and
    draws a random uniform point from each cell. Thus, the result is
    stochastic, but more uniform than a random uniform sample. For further
    information see [McKay1979]_.

    Parameters
    ----------
    num_strata : int
        The number of strata to generate.
        ``num_strata ** (1/dimension)`` must be integer.
    dimension : int
        The dimension of the space.

    Returns
    -------
    strata : list of tuple
        As the strata are axis-aligned boxes in this case, each tuple in
        the returned list contains the lower and upper corner of a stratum.

    """
    # shortcuts & initialization
    assert num_strata > 0
    assert dimension > 0
    points_per_axis = int(num_strata ** (1.0 / dimension))
    if points_per_axis ** dimension != num_strata:
        points_per_axis += 1
    assert points_per_axis ** dimension == num_strata
    final_strata = []
    extent = 1.0 / points_per_axis
    possible_values = [float(x) / points_per_axis for x in range(points_per_axis)]
    # calculate grid cells and sample
    for lower_bounds in itertools.product(possible_values, repeat=dimension):
        upper_bounds = [lower + extent for lower in lower_bounds]
        final_strata.append((list(lower_bounds), upper_bounds))
    return final_strata



def stratify_generalized(num_strata,
                         dimension,
                         cuboid=None,
                         detect_special_case=True,
                         avoid_odd_numbers=True):
    """Generalized stratification of the unit hypercube.

    The adjective "generalized" pertains to the fact that the number of
    strata can be chosen arbitrarily, which is not possible with
    :func:`stratify_conventional`. It is guaranteed that all strata have
    volume ``volume(cuboid) / num_strata``, apart from rounding error.

    Parameters
    ----------
    num_strata : int
        The number of strata to generate. An arbitrary number of points
        is possible for this algorithm.
    dimension : int
        The dimension of the search space.
    cuboid : tuple of list, optional
        Optionally specify the hypercube to be sampled in. If None, the
        unit hypercube is chosen.
    detect_special_case : bool, optional
        If True, ``num_points ** (1/dimension)`` is integer, and we are
        sampling the unit cube, use the original stratified sampling in
        :func:`stratify_conventional`.
    avoid_odd_numbers : bool, optional
        If this value is True, splits are chosen so that the resulting
        numbers are even, whenever possible. E.g., if a stratum with six
        points is splitted, it is not split into three and three, but two
        and four points. For more information on this option, see
        [Wessing2018]_.

    Returns
    -------
    strata : list of tuple
        As the strata are axis-aligned boxes in this case, each tuple in
        the returned list contains the lower and upper corner of a stratum.

    References
    ----------
    .. [Wessing2018] Wessing, Simon (2018). Experimental Analysis of a
        Generalized Stratified Sampling Algorithm for Hypercubes. arXiv
        eprint 1705.03809. https://arxiv.org/abs/1705.03809

    """
    # sanity checks
    assert num_strata > 0
    assert dimension > 0
    if cuboid is None:
        cuboid = unitcube(dimension)
    assert len(cuboid[0]) == dimension
    assert len(cuboid[1]) == dimension
    # special case
    if detect_special_case and cuboid == unitcube(dimension):
        points_per_axis = int(num_strata ** (1.0 / dimension))
        is_integer_power = points_per_axis ** dimension == num_strata
        # check because of potential rounding error:
        is_integer_power |= (points_per_axis + 1) ** dimension == num_strata
        if is_integer_power:
            return stratify_conventional(num_strata, dimension)
    # more initialization
    dimensions = list(range(dimension))
    final_strata = []
    remaining_strata = [(num_strata, cuboid)]
    # begin partitioning
    while remaining_strata:
        current_num_points, current_bounds = remaining_strata.pop()
        if current_num_points == 1:
            final_strata.append(current_bounds)
            continue
        current_lower, current_upper = current_bounds
        diffs = [current_upper[i] - current_lower[i] for i in dimensions]
        max_extent = max(diffs)
        max_extent_dims = [i for i in dimensions if diffs[i] == max_extent]
        num1 = int(current_num_points * 0.5)
        do_subtract_one = avoid_odd_numbers and current_num_points >= 6
        do_subtract_one &= num1 % 2 != 0 and current_num_points % 2 == 0
        if do_subtract_one:
            num1 -= 1
        num2 = current_num_points - num1
        if random.random() < 0.5:
            num1, num2 = num2, num1
        split_dim = random.choice(max_extent_dims)
        split_pos = float(num1) / current_num_points
        split_pos = current_lower[split_dim] + max_extent * split_pos
        new_upper = current_upper[:]
        new_upper[split_dim] = split_pos
        new_lower = current_lower[:]
        new_lower[split_dim] = split_pos
        stratum1 = (num1, (current_lower, new_upper))
        stratum2 = (num2, (new_lower, current_upper))
        remaining_strata.extend((stratum1, stratum2))
    return final_strata



def stratified_sampling(strata,
                        bates_param=1,
                        latin="none",
                        matching_init="approx",
                        full_output=False):
    """Stratified sampling with given strata.

    Parameters
    ----------
    strata : list of tuple
        The strata to be sampled. Each tuple in the list must contain the
        lower and upper corner of a stratum.
    bates_param : int, optional
        Each coordinate of a point sampled in a stratum is determined as
        the mean of this number of independent random uniform variables.
        Thus, the coordinates follow the Bates distribution.
    latin : str, optional
        Indicates if and how the point set should be latinized. "none" is
        the fastest option, sampling the points in linear time without
        latinization. "approx" uses a heuristic with runtime
        :math:`O(nN \\log N)` that may produce a slightly imperfect
        latinization. "matching" produces an error-free latinization using
        a maximum cardinality matching algorithm with runtime
        :math:`O(nN^2)`. The union of the strata must be the unit hypercube
        for the second and third option to work.
    matching_init : str, optional
        This is an additional option to the setting ``latin == "matching"``.
        Firstly, the approximative latinization from ``latin == "approx"`` can
        be used as initialization for the matching algorithm, with the
        matching acting as a repair method if necessary. This is the
        recommended choice due to its favorable runtime behavior.
        The other two options use a problem-agnostic greedy initialization.
        "greedy-rand" randomly shuffles the edges in the bipartite
        graph data structure. This order indirectly influences the
        distribution of the point sample, via the deterministic matching
        algorithm. "greedy-det" is the deterministic variant without extras,
        which may produce patterns due to the deterministic nature of the
        algorithm.
    full_output : bool, optional
        Indicates if the indices of points with latin hypercube violations
        are returned in case of latinized sampling.

    Returns
    -------
    points : numpy array
        The sampled points, in corresponding order to `strata`.
    error_indices : set
        Indices of points with violations of the latin hypercube property
        in some dimension. Can only be non-empty for ``latin == "approx"``.

    """
    def bipartite_match(graph, matching=None):
        """Find maximum cardinality matching of a bipartite graph (U, V, E).

        The input format is a dictionary mapping members of U to a list
        of their neighbors in V.  The output is a triple (M, A, B) where M
        is a dictionary mapping members of V to their matches in U, A is
        the part of the maximum independent set in U, and B is the part of
        the MIS in V. The same object may occur in both U and V, and is
        treated as two distinct vertices if this happens.

        Adapted from
        https://github.com/ActiveState/code/tree/master/recipes/Python/123641_HopcroftKarp_bipartite_matching
        Hopcroft-Karp bipartite max-cardinality matching and max independent set
        David Eppstein, UC Irvine, 27 Apr 2002

        """
        if not matching:
            # initialize greedy matching (redundant, but faster than full search)
            matching = {}
            for u in graph:
                for v in graph[u]:
                    if v not in matching:
                        matching[v] = u
                        break
        while True:
            # structure residual graph into layers
            # pred[u] gives the neighbor in the previous layer for u in U
            # preds[v] gives a list of neighbors in the previous layer for v in V
            # unmatched gives a list of unmatched vertices in final layer of V,
            # and is also used as a flag value for pred[u] when u is in the first layer
            preds = {}
            unmatched = []
            pred = dict([(u, unmatched) for u in graph])
            for v in matching:
                del pred[matching[v]]
            layer = list(pred)
            # repeatedly extend layering structure by another pair of layers
            while layer and not unmatched:
                new_layer = {}
                for u in layer:
                    for v in graph[u]:
                        if v not in preds:
                            new_layer.setdefault(v, []).append(u)
                layer = []
                for v in new_layer:
                    preds[v] = new_layer[v]
                    if v in matching:
                        layer.append(matching[v])
                        pred[matching[v]] = v
                    else:
                        unmatched.append(v)
            # did we finish layering without finding any alternating paths?
            if not unmatched:
                unlayered = {}
                for u in graph:
                    for v in graph[u]:
                        if v not in preds:
                            unlayered[v] = None
                return matching, list(pred), list(unlayered)

            # recursively search backward through layers to find alternating paths
            # recursion returns true if found path, false otherwise
            def recurse(v):
                if v in preds:
                    L = preds[v]
                    del preds[v]
                    for u in L:
                        if u in pred:
                            pu = pred[u]
                            del pred[u]
                            if pu is unmatched or recurse(pu):
                                matching[v] = u
                                return 1
                return 0

            for v in unmatched:
                recurse(v)

    # sanity checks & initialization
    assert len(strata) > 0
    assert bates_param > 0
    assert latin != "matching" or matching_init in ("approx", "greedy-det", "greedy-rand")
    num_points = len(strata)
    dimension = len(strata[0][0])
    points = np.empty((num_points, dimension))
    rand_uni = random.uniform
    error_indices = set()
    if latin == "none":
        for dim_index in range(dimension):
            for j, stratum in enumerate(strata):
                low, high = stratum[0][dim_index], stratum[1][dim_index]
                if math.isinf(bates_param):
                    points[j][dim_index] = (low + high) * 0.5
                elif bates_param == 1:
                    points[j][dim_index] = rand_uni(low, high)
                else:
                    uniform_sum = sum(rand_uni(low, high) for _ in range(bates_param))
                    points[j][dim_index] = uniform_sum / bates_param
    else:
        bin_size = 1.0 / num_points
        for dim_index in range(dimension):

            def cog_key(strat_idx):
                """Two times the center of gravity."""
                stratum = strata[strat_idx]
                return stratum[0][dim_index] + stratum[1][dim_index]

            strat_indices = np.random.permutation(num_points).tolist()
            strat_indices.sort(key=cog_key)
            bin_indices = list(range(num_points))
            if latin == "approx":
                # strat_indices has already been sorted above
                pass
            elif latin == "matching":
                dim_graph = {}
                matching = {}
                for bin_idx, strat_idx in zip(bin_indices, strat_indices):
                    stratum = strata[strat_idx]
                    low, high = stratum[0][dim_index], stratum[1][dim_index]
                    low_bin = int(low / bin_size)
                    high_bin = int(math.ceil(high / bin_size))
                    avail_bins = np.arange(low_bin, high_bin)
                    if matching_init == "approx" and bin_idx in avail_bins:
                        matching[bin_idx] = strat_idx
                    elif matching_init == "greedy-rand":
                        np.random.shuffle(avail_bins)
                    dim_graph[strat_idx] = avail_bins
                if len(matching) < num_points:
                    matching, max_ind_set, _ = bipartite_match(dim_graph, matching)
                    assert len(max_ind_set) == 0
                    bin_indices = matching.keys()
                    strat_indices = matching.values()
            else:
                raise Exception("unknown latin hypercube option " + str(latin))
            for bin_idx, strat_idx in zip(bin_indices, strat_indices):
                stratum = strata[strat_idx]
                low, high = stratum[0][dim_index], stratum[1][dim_index]
                low2, high2 = bin_idx * bin_size, (bin_idx + 1) * bin_size
                low, high = max(low, low2), min(high, high2)
                if not high >= low:
                    error_indices.add(strat_idx)
                    # revert to uniform sampling of the stratum in this dim
                    low, high = stratum[0][dim_index], stratum[1][dim_index]
                if math.isinf(bates_param):
                    points[strat_idx][dim_index] = (low + high) * 0.5
                elif bates_param == 1:
                    points[strat_idx][dim_index] = rand_uni(low, high)
                else:
                    uniform_sum = sum(rand_uni(low, high) for _ in range(bates_param))
                    points[strat_idx][dim_index] = uniform_sum / bates_param
    if full_output:
        return points, error_indices
    else:
        return points


def reconstruct_strata_from_points(points, cuboid=None):
    """Partitions the cuboid so that each point has its own hyperbox.

    This partitioning is stochastic (ties are broken randomly). The obtained
    strata will have different volumes. This function can be used to
    calculate an upper bound for the covering radius of arbitrary point
    sets via
    :func:`covering_radius_upper_bound()<diversipy.indicator.covering_radius_upper_bound>`.
    The idea for this approach was introduced in [Wessing2018]_.

    Parameters
    ----------
    points : sequence of sequence
        The sampled points.
    cuboid : tuple of list, optional
        Optionally specify the outer bounds of the partitioning. If None,
        the unit hypercube is chosen.

    Returns
    -------
    strata : list of tuple
        As the strata are axis-aligned boxes in this case, each tuple in
        the returned list contains the lower and upper corner of a stratum.
        The order corresponds to the order of `points`.

    """
    dimension = len(points[0])
    dimensions = list(range(dimension))
    if cuboid is None:
        cuboid = unitcube(dimension)
    assert len(cuboid[0]) == dimension
    assert len(cuboid[1]) == dimension
    for point in points:
        assert np.all(point >= cuboid[0]) and np.all(point <= cuboid[1])
    final_strata = dict()
    remaining_strata = [(points, cuboid)]
    while remaining_strata:
        current_points, current_bounds = remaining_strata.pop()
        if len(current_points) == 1:
            final_strata[tuple(current_points[0])] = current_bounds
            continue
        current_lower, current_upper = current_bounds
        diffs = [current_upper[i] - current_lower[i] for i in dimensions]
        points_diffs = np.max(current_points, axis=0) - np.min(current_points, axis=0)
        combined_diffs = np.array(diffs)
        # filter out dimensions where all point coordinates coincide,
        # because we cannot obtain a non-empty stratum by splitting in this dimension
        combined_diffs[points_diffs == 0] = 0
        max_extent = max(combined_diffs)
        max_extent_dims = [i for i in dimensions if combined_diffs[i] == max_extent]
        split_dim = random.choice(max_extent_dims)
        prelim_split_pos = np.mean(current_points[:, split_dim])
        split_bit_mask = current_points[:, split_dim] < prelim_split_pos
        less_indices = np.where(split_bit_mask)[0]
        greater_equal_indices = np.where(1 - split_bit_mask)[0]
        # try to ensure we have points on either side of the split position by moving it
        if len(less_indices) == 0:
            min_index = np.argmin(current_points[greater_equal_indices, split_dim])
            prelim_split_pos = current_points[min_index, split_dim] + sys.float_info.epsilon
            split_bit_mask = current_points[:, split_dim] < prelim_split_pos
            less_indices = np.where(split_bit_mask)[0]
            greater_equal_indices = np.where(1 - split_bit_mask)[0]
        elif len(greater_equal_indices) == 0:
            max_index = np.argmax(current_points[less_indices, split_dim])
            prelim_split_pos = current_points[max_index, split_dim]
            split_bit_mask = current_points[:, split_dim] < prelim_split_pos
            less_indices = np.where(split_bit_mask)[0]
            greater_equal_indices = np.where(1 - split_bit_mask)[0]
        assert len(less_indices) > 0
        assert len(greater_equal_indices) > 0
        # finally identify nearest neighbors to split position in 1-D projection
        max_index = np.argmax(current_points[less_indices, split_dim])
        max_index = less_indices[max_index]
        min_index = np.argmin(current_points[greater_equal_indices, split_dim])
        min_index = greater_equal_indices[min_index]
        # center split position between these two points
        split_pos = (current_points[max_index, split_dim] + current_points[min_index, split_dim]) * 0.5
        # create new strata
        new_upper = current_upper[:]
        new_upper[split_dim] = split_pos
        new_lower = current_lower[:]
        new_lower[split_dim] = split_pos
        stratum1 = (current_points[less_indices], (current_lower, new_upper))
        stratum2 = (current_points[greater_equal_indices], (new_lower, current_upper))
        remaining_strata.extend((stratum1, stratum2))
    final_strata = [final_strata[tuple(point)] for point in points]
    return final_strata



def lhd_matrix(num_points, dimension):
    """Generate a random latin hypercube design matrix.

    Latin hypercube designs sometimes give an advantage over random uniform
    samples due to their perfect uniformity of one-dimensional projections.
    For further information see [McKay1979]_. This algorithm has linear run
    time.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.

    Returns
    -------
    design : (`num_points`, `dimension`) numpy array
        Matrix with integers corresponding to the bins of a virtual grid.
        Each column consists of a permutation of {0, ..., num_points - 1}.

    References
    ----------
    .. [McKay1979] McKay, M.D.; Beckman, R.J.; Conover, W.J. (May 1979).
        A Comparison of Three Methods for Selecting Values of Input
        Variables in the Analysis of Output from a Computer Code.
        Technometrics 21(2): 239-245.

    """
    design = np.zeros((num_points, dimension), dtype="i")
    permutation = np.random.permutation
    for i in range(dimension):
        design[:, i] = permutation(np.arange(0, num_points))
    return design



def improved_lhd_matrix(num_points,
                        dimension,
                        num_candidates=100,
                        target_value=None,
                        dist_matrix_function=None):
    """Generate an 'improved' latin hypercube design matrix.

    This implementation uses an algorithm with quadratic run time. It is a
    greedy construction heuristic starting with a randomly chosen point.
    In each iteration, a number of random candidates is evaluated by a
    criterion that considers a candidate's distance to the previously chosen
    points. The best point according to this criterion is included in the
    LHD. The concept originally stems from [Beachkofski2002]_. The algorithm
    implemented here was proposed in [Wessing2015]_.

    Parameters
    ----------
    num_points : int
        The number of points to generate.
    dimension : int
        The dimension of the space.
    num_candidates : int, optional
        The number of random candidates considered for every point to be
        added to the LHD.
    target_value : float, optional
        The distance a candidate should ideally have to the already chosen
        points of the LHD.
    dist_matrix_function : callable, optional
        Defines the distance used. Default is Manhattan distance on a torus
        (maximum distance is ``num_points - 1``, well-suited for
        :func:`edge_lhs`).

    Returns
    -------
    design : (`num_points`, `dimension`) numpy array
        Matrix with integers corresponding to the bins of a virtual grid.
        Each column consists of a permutation of {0, ..., num_points - 1}.

    References
    ----------
    .. [Beachkofski2002] Beachkofski, B.; Grandhi, R. (2002). Improved
        Distributed Hypercube Sampling. American Institute of Aeronautics
        and Astronautics Paper 1274.

    """
    # shortcuts & initialization
    dimensions = list(range(dimension))
    permutation = np.random.permutation
    if dist_matrix_function is None:
        max_dists = [num_points - 1.0] * dimension
        dist_matrix_function = DistanceMatrixFunction(exponent=1, max_dists_per_dim=max_dists)
    if target_value is None:
        target_value = np.inf
    design = np.empty((num_points, dimension), dtype="i")
    available_bins = []
    first_point = []
    bins = range(num_points)
    for _ in dimensions:
        dim_bins = list(bins)
        random.shuffle(dim_bins)
        first_point.append(dim_bins.pop())
        available_bins.append(dim_bins)
    design[0, :] = first_point
    # begin choosing other points
    for i in range(1, num_points - 1):
        # create a set of num_candidates random points obeying the LHD property
        available_bins_array = np.array(available_bins).T
        num_available = float(num_points - i)
        duplication_factor = int(math.ceil(num_candidates / num_available))
        candidates = np.vstack([available_bins_array] * duplication_factor)
        for dim in dimensions:
            candidates[:, dim] = permutation(candidates[:, dim])
        # calculate distances to already chosen points
        distances = dist_matrix_function(design[0:i, :], candidates[0:num_candidates, :])
        min_dists = distances.min(axis=0)
        # select best point according to distance criterion
        if not math.isinf(target_value):
            selected_index = np.argmin(np.abs(min_dists - target_value))
        else:
            selected_index = np.argmax(min_dists)
        selected_point = candidates[selected_index, :]
        design[i, :] = selected_point
        # maintain bin data structure
        for value, avail in zip(selected_point, available_bins):
            avail.remove(value)
    last_point = [avail.pop() for avail in available_bins]
    for avail in available_bins:
        assert not avail  # should be empty
    design[-1, :] = last_point
    return design



def has_lhd_property(design_matrix):
    """Check if design matrix has the LHD property.

    It is assumed that counting starts from zero and points are arranged
    row-wise. So, each column must consist of a permutation of
    ``range(num_points)``.

    Parameters
    ----------
    design_matrix : array_like
        2-D array of integer coordinates.

    Returns
    -------
    is_lhd : bool

    """
    design_matrix = np.asarray(design_matrix)
    num_points, dimension = design_matrix.shape
    required_numbers = list(range(num_points))
    for i in range(dimension):
        ith_column = design_matrix[:, i].tolist()
        if required_numbers != sorted(ith_column):
            return False
    return True



def transform_perturbed(design_matrix):
    """Transform a design matrix into a sample in the unit hypercube.

    Applies random perturbations so that each point is distributed randomly
    uniform in its grid cell. This is the variant proposed by [McKay1979]_.
    It is not checked if `design_matrix` is a LHD.

    Parameters
    ----------
    design_matrix : array_like
        Array containing integers to indicate the bins occupied by each
        point.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    points = np.array(design_matrix, dtype="d")
    num_points, num_dimensions = points.shape
    points += np.random.rand(num_points, num_dimensions)
    points /= num_points
    return points



def transform_cell_centered(design_matrix):
    """Transform a design matrix into a sample in the unit hypercube.

    Each point is placed at the centroid of a subcell in the assumed grid
    over the cube. It is not checked if `design_matrix` is a LHD.

    Parameters
    ----------
    design_matrix : array_like
        Array containing integers to indicate the bins occupied by each
        point.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    design_matrix = np.asarray(design_matrix)
    num_points = len(design_matrix)
    points = design_matrix + 0.5
    points /= num_points
    return points



def transform_spread_out(design_matrix):
    """Transform a design matrix into a sample in the unit hypercube.

    The transformation is so that each face of the hypercube is sampled by
    at least one point (exactly one point in the case of LHDs). Use this
    transformation if you want to maximize the minimal distance between
    points in the design. It is not checked if `design_matrix` is a LHD.

    Parameters
    ----------
    design_matrix : array_like
        Array containing integers to indicate the bins occupied by each
        point.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    points = np.array(design_matrix, dtype="d")
    num_points = len(points)
    points /= (num_points - 1.0)
    return points



def transform_anchored(design_matrix):
    """Transform a design matrix into a sample in the unit hypercube.

    This is typically used with rank-1 lattices. The zero-vector is a fix
    point in this transformation. It is not checked if `design_matrix`
    is a LHD. The highest value sampled is ``(num_points - 1) / num_points``.
    You may want to apply :func:`shifted_randomly` afterwards to get
    perturbation.

    Parameters
    ----------
    design_matrix : array_like
        Array containing integers to indicate the bins occupied by each
        point.

    Returns
    -------
    points : (`num_points`, `dimension`) numpy array

    """
    points = np.array(design_matrix, dtype="d")
    num_points = len(points)
    points /= num_points
    return points



def shifted_randomly(points):
    """Cranley-Patterson rotation.

    Parameters
    ----------
    points : array_like
        2-D array of points.

    Returns
    -------
    shifted_points : numpy array
        A new array containing the shifted points.

    References
    ----------
    .. [Cranley1976] R. Cranley; T. N. L. Patterson (1976). Randomization
        of Number Theoretic Methods for Multiple Integration. SIAM Journal
        on Numerical Analysis, vol. 13, no. 6, pp. 904-914.

    """
    _, dim = points.shape
    shift_vector = np.random.rand(dim)
    shifted_points = points + shift_vector
    shifted_points %= 1
    return shifted_points
