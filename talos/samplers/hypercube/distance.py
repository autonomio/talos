"""
This module provides some basic distance measures and a class to instantiate
arbitrary distance functions based on Minkowski metrics. For more exotic
distance functions (and faster implementations) have a look at
:mod:`scipy.spatial.distance`.
"""
import math

import numpy as np



def calc_dists_to_boundary(points, cuboid=None):
    """Calculate the distance of each point to the boundary of some cuboid.

    This distance is simply the minimum of all differences between
    a point and the lower and upper bounds. This function also checks if all
    calculated distances are larger than zero. If not, some points must be
    located outside the cuboid.

    Parameters
    ----------
    points : array_like
        2-D array of `n` points.
    cuboid : tuple of array_like, optional
        Contains the min and max bounds of the considered cuboid. If
        omitted, the unit hypercube is assumed.

    Returns
    -------
    distances : numpy array
        1-D array of `n` distances

    """
    if cuboid is None:
        dists_to_min_bounds = points.min(axis=1)
        dists_to_max_bounds = (1.0 - points).min(axis=1)
    else:
        min_bounds, max_bounds = cuboid
        dists_to_min_bounds = (points - np.asarray(min_bounds)).min(axis=1)
        dists_to_max_bounds = (np.asarray(max_bounds) - points).min(axis=1)
    distances = np.minimum(dists_to_min_bounds, dists_to_max_bounds)
    assert np.all(distances >= 0.0)  # are all points contained in cuboid?
    return distances



def calc_euclidean_dist_matrix(points1, points2):
    """Calculate Euclidean distance matrix between `points1` and `points2`.

    Generates one column of the matrix at a time.

    Parameters
    ----------
    points1 : array_like
        2-D array of n points.
    points2 : array_like
        2-D array of m points.

    Returns
    -------
    distances : (n,m) numpy array

    """
    num_points1, dim1 = points1.shape
    num_points2, dim2 = points2.shape
    assert dim1 == dim2
    distances = np.zeros((num_points1, num_points2))
    for i in range(dim1):
        diff2 = points1[:, i, None] - points2[:, i]
        diff2 **= 2
        distances += diff2
    # in place sqrt
    np.sqrt(distances, distances)
    return distances



def calc_manhattan_dist_matrix(points1, points2):
    """Calculate Manhattan distance matrix between `points1` and `points2`.

    Generates one column of the matrix at a time.

    Parameters
    ----------
    points1 : array_like
        2-D array of n points.
    points2 : array_like
        2-D array of m points.

    Returns
    -------
    distances : (n,m) numpy array

    """
    num_points1, dim1 = points1.shape
    num_points2, dim2 = points2.shape
    assert dim1 == dim2
    distances = np.zeros((num_points1, num_points2))
    for i in range(dim1):
        diff = np.abs(points1[:, i, None] - points2[:, i])
        distances += diff
    return distances



class DistanceMatrixFunction:
    """General distance function.

    This distance function can handle arbitrary exponents and can optionally
    calculate torus distances. Slightly slower than the specialized
    versions. Special cases ``exponent = 1`` and ``exponent = 2`` correspond
    to Manhattan and Euclidean distance, respectively.

    """
    def __init__(self, exponent=2, max_dists_per_dim=None):
        """Constructor.

        Parameters
        ----------
        exponent : scalar, optional
            The exponent in the distance calculation.
        max_dists_per_dim : array_like, optional
            1-D array of largest possible distance in each dimension.
            Providing these values has the consequence of treating the
            cuboid as a torus. This is useful for eliminating edge effects
            induced by the lack of neighbor points outside the bounds of
            the cuboid.

        """
        self.exponent = exponent
        self.max_dists_per_dim = max_dists_per_dim
        if max_dists_per_dim is not None:
            for max_dist in max_dists_per_dim:
                assert max_dist > 0 or max_dist is None


    def __call__(self, points1, points2):
        """Calculate distance matrix, one column at a time."""
        exponent = self.exponent
        max_dists_per_dim = self.max_dists_per_dim
        num_points1, dim1 = points1.shape
        num_points2, dim2 = points2.shape
        assert dim1 == dim2
        distances = np.zeros((num_points1, num_points2))
        if math.isinf(exponent):
            for i in range(dim1):
                diff = np.abs(points1[:, i, None] - points2[:, i], dtype="d")
                if max_dists_per_dim is not None:
                    np.minimum(max_dists_per_dim[i] - diff, diff, out=diff)
                assert np.all(diff >= 0)
                distances = np.maximum(diff, distances)
        else:
            for i in range(dim1):
                diff = np.abs(points1[:, i, None] - points2[:, i], dtype="d")
                if max_dists_per_dim is not None:
                    np.minimum(max_dists_per_dim[i] - diff, diff, out=diff)
                assert np.all(diff >= 0)
                diff **= exponent
                distances += diff
            # in place root function
            np.power(distances, 1.0 / exponent, out=distances)
        return distances
