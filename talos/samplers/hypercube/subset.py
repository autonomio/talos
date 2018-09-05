# This Python file uses the following encoding: utf-8
"""
This module contains algorithms for the task of subset selection: suppose
you have a set of points in :math:`\\mathbb{R}^n` and want to select a sample
of them distributed as uniform as possible. This problem is related to
clustering, with the difference that when using clustering, you usually want
to retain the structure of the original point set.
"""
import heapq
import random

import numpy as np

from .distance import calc_dists_to_boundary
from .distance import calc_euclidean_dist_matrix


class MinBoundingBox:
    """Data structure containing some information about a cluster of points.

    The box does not store the points themselves, but merely a reference
    to a container of a (possibly) larger set of points and the indices of
    points in this container belonging to this box. Most importantly, the
    box holds cached information about the minimal and maximal coordinates
    of the points and the largest difference between them. This information
    is important for the part-and-select algorithm (PSA) [Salomon2013]_.
    Comparison operators are overloaded to compare by this value for easy
    use in a binary heap.

    """

    def __init__(self, all_points, member_indices):
        self.sort_key = None
        self.dim_index = None
        self.min_bounds = None
        self.max_bounds = None
        self.all_points = all_points
        self.member_indices = member_indices
        # fill in information
        self.update_information()


    def update_information(self):
        """Gather information from the points and save it."""
        all_points = self.all_points
        member_indices = self.member_indices
        dimension = len(all_points[0])
        members_array = np.take(all_points, member_indices, axis=0)
        min_bounds = members_array.min(axis=0)
        max_bounds = members_array.max(axis=0)
        # determine dimension with largest difference
        sort_key = float("inf")
        dim_index = None
        for j in range(dimension):
            neg_diff = min_bounds[j] - max_bounds[j]
            if neg_diff < sort_key:
                sort_key = neg_diff
                dim_index = j
        # save data
        self.sort_key = sort_key
        self.dim_index = dim_index
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds


    def closest_point_index(self, sought_point,
                            candidate_indices,
                            dist_matrix_function=None):
        """Return the index of the point closest to a certain target.

        Parameters
        ----------
        sought_point : array_like
            The target point.
        candidate_indices : array_like
            Indices into the container of all points for the candidates to
            be considered.
        dist_matrix_function : callable, optional
            An arbitrary distance function. Default is Euclidean distance.

        Returns
        -------
        closest_point_index : integer

        """
        all_points = self.all_points
        if dist_matrix_function is None:
            dist_matrix_function = calc_euclidean_dist_matrix
        candidates_array = np.take(all_points, candidate_indices, axis=0)
        distances = dist_matrix_function(np.atleast_2d(sought_point),
                                         candidates_array)
        closest_point_index = candidate_indices[np.argmin(distances)]
        return closest_point_index


    def obtain_representative(self, selection_target="centroid_of_hypercube",
                              tournament_size=0,
                              dist_matrix_function=None):
        """Return a point representing this cluster.

        Parameters
        ----------
        selection_target : string, optional
            Indicates which strategy is used to determine a representative.
            This must be one of ('random_uniform', 'centroid_of_hypercube',
            'center_of_mass', 'max_dist_from_boundary').
        tournament_size : int, optional
            Optionally restrict the candidates for selection to
            ``0 < tournament_size < len(member_indices)`` randomly chosen
            points. Any other value of this parameter leads to consideration
            of all points. Note that a value of 1 leads to a random choice
            among the clusters members, regardless of the target point.
        dist_matrix_function : callable, optional
            An arbitrary distance function. Default is Euclidean distance.
            This choice does not affect the selection strategy
            ``max_dist_from_boundary``.

        Returns
        -------
        representative : array_like

        """
        index = self.obtain_representative_index(selection_target,
                                                 tournament_size,
                                                 dist_matrix_function)
        representative = self.all_points[index]
        return representative


    def obtain_representative_index(self,
                                    selection_target="centroid_of_hypercube",
                                    tournament_size=0,
                                    dist_matrix_function=None):
        """Return the index to a point representing this cluster.

        Parameters
        ----------
        selection_target : string, optional
            Indicates which strategy is used to determine a representative.
            This must be one of ('random_uniform', 'centroid_of_hypercube',
            'center_of_mass', 'max_dist_from_boundary').
        tournament_size : int, optional
            Optionally restrict the candidates for selection to
            ``0 < tournament_size < len(member_indices)`` randomly chosen
            points. Any other value of this parameter leads to consideration
            of all points. Note that a value of 1 leads to a random choice
            among the clusters members, regardless of the target point.
        dist_matrix_function : callable, optional
            An arbitrary distance function. Default is Euclidean distance.
            This choice does not affect the selection strategy
            ``max_dist_from_boundary``.

        Returns
        -------
        representative_index : int

        """
        all_points = self.all_points
        member_indices = self.member_indices
        min_bounds = self.min_bounds
        max_bounds = self.max_bounds
        dimension = len(min_bounds)
        # determine target point
        if selection_target == "random_uniform":
            # random point in the space (!) of the bounding box
            target_point = [random.uniform(min_bounds[i], max_bounds[i]) for i in range(dimension)]
        elif selection_target == "center_of_mass":
            members_array = np.take(all_points, member_indices, axis=0)
            target_point = np.mean(members_array, axis=0)
        elif selection_target == "centroid_of_hypercube" or selection_target == "max_dist_from_boundary":
            target_point = [(max_bounds[i] + min_bounds[i]) * 0.5 for i in range(dimension)]
        else:
            raise ValueError("Unknown target mode '" + selection_target + "'")
        # determine subset of points as candidates
        if tournament_size <= 0 or tournament_size >= len(member_indices):
            candidate_point_indices = member_indices
        else:
            candidate_point_indices = random.sample(member_indices,
                                                    tournament_size)
        # find best of those according to selection criterion
        if selection_target == "max_dist_from_boundary":
            hypercube = (min_bounds, max_bounds)
            candidates_array = np.take(all_points, candidate_point_indices, axis=0)
            dists_to_bound = calc_dists_to_boundary(candidates_array, hypercube)
            max_dist_index = np.argmax(dists_to_bound)
            index = candidate_point_indices[max_dist_index]
        else:
            index = self.closest_point_index(target_point,
                                             candidate_point_indices,
                                             dist_matrix_function)
        return index


    def __lt__(self, other):
        return self.sort_key < other.sort_key


    def __gt__(self, other):
        return self.sort_key > other.sort_key


    def __eq__(self, other):
        return self.sort_key == other.sort_key


    def __le__(self, other):
        return self.sort_key <= other.sort_key


    def __ge__(self, other):
        return self.sort_key >= other.sort_key


    def __ne__(self, other):
        return self.sort_key != other.sort_key



def psa_partition(points, num_clusters, available_points_indices=None):
    """Partition the data set into the given number of clusters.

    The approach was originally proposed in [Salomon2013]_. The
    implementation here is the slightly improved version from
    [Wessing2015]_.

    .. note:: This algorithm fails if there are less than `num_clusters`
        distinct points in the original set.

    Parameters
    ----------
    points : array_like
        2-D data structure holding the points.
    num_clusters : int
        The number of 'clusters' to be identified.
    available_points_indices : array_like, optional
        If desired, the set of considered points can already be restricted
        here by only providing a subset of the indices into `points`.

    Returns
    -------
    clusters : list of MinBoundingBox

    References
    ----------
    .. [Salomon2013] Salomon, Shaul; Avigad, Gideon; Goldvard, Alex;
        Schütze, Oliver (2013). PSA -- A New Scalable Space Partition
        Based Selection Algorithm for MOEAs. EVOLVE -- A Bridge between
        Probability, Set Oriented Numerics, and Evolutionary Computation II,
        Advances in Intelligent Systems and Computing, Springer, Vol. 175,
        pp. 137-151. https://dx.doi.org/10.1007/978-3-642-31519-0_9

    """
    if available_points_indices is None:
        available_points_indices = list(range(len(points)))
    assert num_clusters <= len(available_points_indices)
    assert num_clusters > 0
    clusters = []
    most_dissimilar_cluster = MinBoundingBox(points, available_points_indices)
    while len(clusters) + 1 < num_clusters:
        split_index = most_dissimilar_cluster.dim_index
        lower_bound = most_dissimilar_cluster.min_bounds[split_index]
        upper_bound = most_dissimilar_cluster.max_bounds[split_index]
        split_position = (lower_bound + upper_bound) / 2.0
        indices1 = []
        indices2 = []
        for member_index in most_dissimilar_cluster.member_indices:
            if points[member_index][split_index] < split_position:
                indices1.append(member_index)
            else:
                indices2.append(member_index)
        cluster1 = MinBoundingBox(points, indices1)
        cluster2 = MinBoundingBox(points, indices2)
        heapq.heappush(clusters, cluster1)
        most_dissimilar_cluster = heapq.heappushpop(clusters, cluster2)
    heapq.heappush(clusters, most_dissimilar_cluster)
    return clusters



def psa_select(points,
               num_selected_points,
               available_points_indices=None,
               selection_target="centroid_of_hypercube",
               tournament_size=0,
               dist_matrix_function=None):
    """Combine partitioning and determination of representatives.

    The approach was originally proposed in [Salomon2013]_. The
    implementation here is the slightly improved version from
    [Wessing2015]_.

    Parameters
    ----------
    points : array_like
        2-D data structure holding the points.
    num_selected_points : int
        The number of points to be selected.
    available_points_indices : array_like, optional
        If desired, the set of considered points can already be restricted
        here by only providing a subset of the indices into `points`.
    selection_target : string, optional
        Indicates which strategy is used to determine a representative. This
        must be one of ('random_uniform', 'centroid_of_hypercube',
        'center_of_mass', 'max_dist_from_boundary').
    tournament_size : int, optional
        Optionally restrict the candidates for selection to
        ``0 < tournament_size < len(member_indices)`` randomly chosen
        points. Any other value of this parameter leads to consideration
        of all points. Note that a value of 1 leads to a random choice
        among the clusters members, regardless of the target point.
    dist_matrix_function : callable, optional
        An arbitrary distance function. Default is Euclidean distance.
        This choice does not affect the selection strategy
        ``max_dist_from_boundary``.

    Returns
    -------
    representatives : array_like

    """
    clusters = psa_partition(points, num_selected_points, available_points_indices)
    representatives = []
    for cluster in clusters:
        representative = cluster.obtain_representative(selection_target,
                                                       tournament_size,
                                                       dist_matrix_function)
        representatives.append(representative)
    if isinstance(points, np.ndarray):
        representatives = np.array(representatives)
    return representatives



def select_greedy_maximin(points,
                          num_selected_points,
                          existing_points=None,
                          dist_matrix_function=None,
                          callback=None):
    """Greedily select a subset according to maximin criterion.

    This selection approach corresponds to the indicator
    :func:`separation_dist() <diversipy.indicator.separation_dist>`.

    Parameters
    ----------
    points : array_like
        2-D data structure holding the points.
    num_selected_points : int
        The number of points to be selected.
    existing_points : array_like, optional
        Points that cannot be modified anymore, but should be considered in
        the distance computations.
    dist_matrix_function : callable, optional
        An arbitrary distance function. Default is Euclidean distance.
    callback : callable, optional
        If provided, it is called as ``callback(indices, dists)`` in each
        iteration for monitoring progress. ``indices`` is the current set
        of selected indices and ``dists`` are the current distances. It
        holds that ``indices[-1] == argmax(dists)``.

    Returns
    -------
    selected_points : array_like

    """
    assert num_selected_points <= len(points)
    assert num_selected_points > 0
    points_array = np.asarray(points)
    if existing_points is None or len(existing_points) == 0:
        existing_points = [random.choice(points)]
    existing_points = np.atleast_2d(existing_points)
    if existing_points.size == 0:
        existing_points = np.array([random.choice(points)])
    if dist_matrix_function is None:
        dist_matrix_function = calc_euclidean_dist_matrix
    distances = dist_matrix_function(existing_points, points_array)
    aggregated_dist_criteria = distances.min(axis=0)
    previous_index = np.argmax(aggregated_dist_criteria)
    selected_indices = [previous_index]
    while len(selected_indices) < num_selected_points:
        if callback is not None:
            callback(selected_indices, aggregated_dist_criteria)
        previous_point = np.atleast_2d(points_array[previous_index])
        distances = dist_matrix_function(previous_point, points_array)
        aggregated_dist_criteria = np.minimum(aggregated_dist_criteria,
                                              distances.ravel())
        previous_index = np.argmax(aggregated_dist_criteria)
        selected_indices.append(previous_index)
    if isinstance(points, list):
        return [points[i] for i in selected_indices]
    else:
        return np.take(points_array, selected_indices, axis=0)



def select_greedy_maxisum(points,
                          num_selected_points,
                          existing_points=None,
                          dist_matrix_function=None,
                          callback=None):
    """Greedily select a subset according to maxisum criterion.

    This selection approach corresponds to the indicator
    :func:`sum_of_dists() <diversipy.indicator.sum_of_dists>`.

    .. warning:: This function does not result in a uniformly distributed
        subset of the points, instead it selects extremal points.

    Parameters
    ----------
    points : array_like
        2-D data structure holding the points.
    num_selected_points : int
        The number of points to be selected.
    existing_points : array_like, optional
        Points that cannot be modified anymore, but should be considered in
        the distance computations.
    dist_matrix_function : callable, optional
        An arbitrary distance function. Default is Euclidean distance.
    callback : callable, optional
        If provided, it is called as ``callback(indices, dists)`` in each
        iteration for monitoring progress. ``indices`` is the current set
        of selected indices and ``dists`` are the current distances. It
        holds that ``indices[-1] == argmax(dists)``.

    Returns
    -------
    selected_points : array_like

    """
    assert num_selected_points <= len(points)
    assert num_selected_points > 0
    points_array = np.asarray(points)
    if existing_points is None or len(existing_points) == 0:
        existing_points = [random.choice(points)]
    existing_points = np.atleast_2d(existing_points)
    if existing_points.size == 0:
        existing_points = np.array([random.choice(points)])
    if dist_matrix_function is None:
        dist_matrix_function = calc_euclidean_dist_matrix
    distances = dist_matrix_function(existing_points, points_array)
    aggregated_dist_criteria = distances.sum(axis=0)
    previous_index = np.argmax(aggregated_dist_criteria)
    selected_indices = [previous_index]
    while len(selected_indices) < num_selected_points:
        if callback is not None:
            callback(selected_indices, aggregated_dist_criteria)
        previous_point = np.atleast_2d(points[previous_index])
        distances = dist_matrix_function(previous_point, points_array)
        aggregated_dist_criteria += distances.ravel()
        aggregated_dist_criteria[selected_indices] = -np.inf
        previous_index = np.argmax(aggregated_dist_criteria)
        selected_indices.append(previous_index)
    if isinstance(points, list):
        return [points[i] for i in selected_indices]
    else:
        return np.take(points_array, selected_indices, axis=0)



def select_greedy_energy(points,
                         num_selected_points,
                         existing_points=None,
                         exponent=None,
                         dist_matrix_function=None,
                         callback=None):
    """Greedily select a subset according to potential energy.

    This selection approach uses the Riesz energy as selection criterion,
    which is related to the indicator
    :func:`average_inverse_dist() <diversipy.indicator.average_inverse_dist>`.
    The uniformity of the selected points can be controlled by `exponent`.
    More information about this criterion can for example be found in
    [Hardin2004]_.

    Parameters
    ----------
    points : array_like
        2-D data structure holding the points.
    num_selected_points : int
        The number of points to be selected.
    existing_points : array_like, optional
        Points that cannot be modified anymore, but should be considered in
        the distance computations.
    exponent : int or float, optional
        Parameter controlling the uniformity. Must be >= 0.
        Values greater or equal the dimension lead to uniformity, smaller
        values to a higher density in exterior regions.
    dist_matrix_function : callable, optional
        An arbitrary distance function. Default is Euclidean distance.
    callback : callable, optional
        If provided, it is called as ``callback(indices, dists)`` in each
        iteration for monitoring progress. ``indices`` is the current set
        of selected indices and ``dists`` are the current distances. It
        holds that ``indices[-1] == argmin(dists)``.

    Returns
    -------
    selected_points : array_like

    References
    ----------
    .. [Hardin2004] Hardin, Douglas P.; Saff, Ed B. (2004). Discretizing
        Manifolds via Minimum Energy Points. Notices of the American
        Mathematical Society, Vol. 51, No. 10, pp. 1186-1194.
        http://www.math.vanderbilt.edu/~hardin/papers/HSb2004.pdf

    """
    assert num_selected_points <= len(points)
    assert num_selected_points > 0
    points_array = np.asarray(points)
    if existing_points is None or len(existing_points) == 0:
        existing_points = [random.choice(points)]
    existing_points = np.atleast_2d(existing_points)
    if existing_points.size == 0:
        existing_points = np.array([random.choice(points)])
    if dist_matrix_function is None:
        dist_matrix_function = calc_euclidean_dist_matrix
    _, dimension = points_array.shape
    if exponent is None:
        exponent = dimension + 1
    distances = dist_matrix_function(existing_points, points_array)
    with np.errstate(divide='ignore'):
        if exponent == 0:
            distances = np.log(1.0 / distances)
        else:
            distances = 1.0 / (distances ** exponent)
    aggregated_dist_criteria = distances.sum(axis=0)
    previous_index = np.argmin(aggregated_dist_criteria)
    selected_indices = [previous_index]
    while len(selected_indices) < num_selected_points:
        if callback is not None:
            callback(selected_indices, aggregated_dist_criteria)
        previous_point = np.atleast_2d(points[previous_index])
        distances = dist_matrix_function(previous_point, points_array)
        with np.errstate(divide='ignore'):
            if exponent == 0:
                distances = np.log(1.0 / distances)
            else:
                distances = 1.0 / (distances ** exponent)
        aggregated_dist_criteria += distances.ravel()
        aggregated_dist_criteria[selected_indices] = np.inf
        previous_index = np.argmin(aggregated_dist_criteria)
        selected_indices.append(previous_index)
    if isinstance(points, list):
        return [points[i] for i in selected_indices]
    else:
        return np.take(points_array, selected_indices, axis=0)
