# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 13:47:16 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numpy as np


def raggedTo3DRectangle_n(array_list):
    """
    This function takes a list of 2d numpy arrays and turns them into a 3d
    numpy array padded by zeros
    """
    join_length = len(array_list)
    # the weird line below is faster than allocating numpy arrays
    x, y = zip(*[array.shape for array in array_list])
    max_x = max(x)
    max_y = max(y)
    dtype = array_list[0].dtype
    padded_rectangle = np.zeros((join_length, max_x, max_y), dtype=dtype)
    for i in range(join_length):
        padded_rectangle[i, :x[i], :y[i]] = array_list[i]
    return padded_rectangle
