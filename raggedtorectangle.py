# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 13:47:16 2016

@author: Nicholas Sherer
"""

from __future__ import division

import numpy as np


def raggedTo3DRectangle(array_list):
    """
    This function takes a list of 2d numpy arrays and turns them into a 3d
    numpy array padded by zeros
    """
    join_length = len(array_list)
    x = np.zeros(join_length, dtype='int')
    y = np.zeros(join_length, dtype='int')
    for i in range(join_length):
        x[i] = np.shape(array_list[i])[0]
        y[i] = np.shape(array_list[i])[1]
    max_x = max(x)
    max_y = max(y)
    dtype = array_list[0].dtype
    padded_rectangle = np.zeros((max_x, max_y, join_length), dtype=dtype)
    for i in range(join_length):
        padded_rectangle[0:x[i], 0:y[i], i] = array_list[i]
    return padded_rectangle


# BELOW FUNCTION UNFINISHED AND UNTESTED
def _3DRectangleToRagged(array):
    """
    This function takes a 3d numpy array, converts it into a list of 2d numpy
    arrays, then strips these arrays of trailing rows and columns of zero.
    """
    array_list = []
    split_length = array.shape[2]
    for i in range(split_length):
        array_list.append(array[:, :, i])
    return array_list


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
