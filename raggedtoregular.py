# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:55:19 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numbers

import numpy as np


def raggedToRegular(array_list, pad=0):
    """
    This function takes a list of numpy arrays and turns it into a single
    larger numpy array padding out entries with padding so that the shape is a
    hypercube as required
    """

    list_len = len(array_list)
    if isinstance(array_list[0], numbers.Number):
        return np.array(array_list)
    array_ndim = array_list[0].ndim
    dim_lens = np.zeros((array_ndim, list_len))
    for i in range(list_len):
        dim_lens[:, i] = array_list[i].shape
    dim_maxes = dim_lens.max(1)
    padded_hypercube = np.zeros(np.hstack((dim_maxes, list_len)))
    for i in range(list_len):
        pad_tuples_list = []
        for j in range(array_ndim):
            pad_tuples_list.append((0, np.int64(dim_maxes[j]-dim_lens[j, i])))
        pad_tuples = tuple(pad_tuples_list)
        padded_hypercube[..., i] = np.pad(array_list[i], pad_tuples,
                                          'constant', constant_values=pad)
    return padded_hypercube
