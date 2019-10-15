# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:55:19 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numpy as np


def ragged_to_regular(array_list):
    """
    This function takes a list of numpy arrays and turns it into a single
    larger numpy array padding out entries with padding so that the shape is a
    hypercube as required
    """
    join_length = len(array_list)
    # the weird line below is faster than allocating numpy arrays
    dims = list(zip(*[array.shape for array in array_list]))
    max_dims = tuple(max(dim) for dim in dims)
    dtype = array_list[0].dtype
    padded_hypercube = np.zeros((join_length,) + max_dims, dtype=dtype)
    for i in range(join_length):
        multislice = (slice(i, i+1, 1),) + tuple(slice(0, dim[i], 1)
                                                 for dim in dims)
        padded_hypercube[multislice] = array_list[i]
    return padded_hypercube


def trim_zeros(array):
    """
    This function returns the subarray of an array with no faces all equal to
    0.
    """
    multislice = []
    for i in range(array.ndim):
        sum_axes = tuple(j for j in range(array.ndim) if j is not i)
        edges = np.where(np.sum(array, axis=sum_axes) > 0)
        if edges[0].size == 0:
            return np.array([], dtype=array.dtype)
        low = edges[0][0]
        high = edges[0][-1]
        multislice.append(slice(low, high+1, 1))
    return array[tuple(multislice)]


def regular_to_ragged(array):
    """
    Break an array up along its first axes into a list of subarrays trimmed of
    their zeros.
    """
    array_list = []
    for i in range(array.shape[0]):
        array_list.append(trim_zeros(array[i]))
    return array_list
